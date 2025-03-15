import argparse
import os
import time

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.networks import one_hot
from monai.utils import first, set_determinism
from monai.transforms import Compose, Activations, AsDiscrete
from tqdm import tqdm
import logging
import wandb
from dotenv import load_dotenv
load_dotenv()
from get_data import get_data
from get_model import get_model
from get_transforms import get_trainval_transforms
from networks2 import CorticalAwareLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_determinism(seed=0)
params = dict()

params['model_name'] = './pretrained/model.pth'
params['model_type'] = 'CAMST'  # Default to CAMST model
params['sa_type'] = 'parallel'
params['chans_in'] = 2  # Will be updated based on include_thickness
params['chans_out'] = 2
params['feature_size'] = 16
params['project_size'] = 64
params['patch_size'] = [128] * 3
params['num_workers'] = 4
params['seq'] = 't1+t2'
params['samples_per_case'] = 4
params['model_desc_str'] = 'model'

class ModelTrainer:
    def __init__(self, params, device):
        self.params = params
        self.device = device
        self.VAL_AMP = True
        self.val_interval = 1
        self.model = None
        self.train_start_time = time.time()
        wandb.login(key=os.environ["WANDB_API_KEY"])
        self.init_stats()
        self.scaler = torch.cuda.amp.GradScaler()
        self.early_stopping_patience = 15
        self.early_stopping_counter = 0
        self.min_lr = 1e-6

    def init_stats(self):
        self.best_metric = -1
        self.best_metric_epoch = -1
        self.best_metrics_epochs_and_time = [[], [], []]
        self.metric_values = []
        self.metric_values_bg = []
        self.metric_values_fcd = []

    def init_loaders(self, train_data_dir, val_data_dir, include_thickness=False):
        train_transform, val_transform = get_trainval_transforms(self.params)

        train_dict = get_data(train_data_dir, self.params, include_thickness)
        train_ds = CacheDataset(data=train_dict, transform=train_transform, cache_num=4, cache_rate=1,
                                num_workers=self.params['num_workers'])
        self.train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=self.params['num_workers'],
                                       pin_memory=True)

        val_dict = get_data(val_data_dir, self.params, include_thickness)
        val_ds = CacheDataset(data=val_dict, transform=val_transform, cache_num=4, cache_rate=1,
                              num_workers=1)
        self.val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=self.params['num_workers'],
                                     pin_memory=True)

    def get_loss_function(self):
        if self.params['model_type'] == 'CAMST':
            # For CAMST, use the specialized CorticalAwareLoss
            return CorticalAwareLoss(alpha=0.5, beta=0.3)
        else:
            # For other models, use combination of Dice and BCE loss
            return DiceCELoss(
                include_background=False,
                to_onehot_y=True,
                softmax=True,
                lambda_dice=0.5,
                lambda_ce=0.5
            )

    def inference(self, input, thickness_map=None):
        def _compute(input):
            if self.params['model_type'] == 'CAMST' and thickness_map is not None:
                return sliding_window_inference(
                    inputs=input,
                    roi_size=self.params["patch_size"],
                    sw_batch_size=2,
                    predictor=lambda x: self.model(x, thickness_map),
                    overlap=0.5,
                )
            else:
                return sliding_window_inference(
                    inputs=input,
                    roi_size=self.params["patch_size"],
                    sw_batch_size=2,
                    predictor=self.model,
                    overlap=0.5,
                )

        if self.VAL_AMP:
            with torch.amp.autocast("cuda"):
                return _compute(input)
        else:
            return _compute(input)

    def validate(self, epoch):
        post_trans = Compose([Activations(softmax=True), AsDiscrete(threshold=0.5)])
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
        new_best = False

        self.model.eval()
        val_loss = 0
        num_val_batches = 0

        with torch.no_grad():
            for val_data in tqdm(self.val_loader, desc="validation", dynamic_ncols=True):
                val_inputs, val_labels = (
                    val_data["image"].to(self.device, dtype=torch.float32),
                    val_data["label"].to(self.device, dtype=torch.float32),
                )
                
                # Extract thickness map for CAMST model
                thickness_map = None
                if self.params['model_type'] == 'CAMST' and val_inputs.shape[1] > 2:
                    # Extract thickness map (third channel)
                    thickness_map = val_inputs[:, 2:3]
                    # Remove thickness map from inputs for models that don't expect it
                    val_inputs = val_inputs[:, :2]
                
                val_outputs = self.inference(val_inputs, thickness_map)
                
                # Calculate validation loss
                if self.params['model_type'] == 'CAMST':
                    val_loss += self.loss_function(val_outputs, val_labels, thickness_map).item()
                else:
                    val_loss += self.loss_function(val_outputs, val_labels).item()
                
                num_val_batches += 1
                
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                val_labels = one_hot(val_labels, num_classes=2)
                dice_metric(y_pred=val_outputs, y=val_labels)
                dice_metric_batch(y_pred=val_outputs, y=val_labels)

            metric = dice_metric.aggregate().item()
            self.metric_values.append(metric)
            metric_batch = dice_metric_batch.aggregate()
            metric_bg = metric_batch[0].item()
            self.metric_values_bg.append(metric_bg)
            metric_fcd = metric_batch[1].item()
            self.metric_values_fcd.append(metric_fcd)
            
            avg_val_loss = val_loss / num_val_batches
            
            dice_metric.reset()
            dice_metric_batch.reset()

            if metric > self.best_metric:
                self.best_metric = metric
                self.best_metric_epoch = epoch + 1
                self.best_metrics_epochs_and_time[0].append(self.best_metric)
                self.best_metrics_epochs_and_time[1].append(self.best_metric_epoch)
                self.best_metrics_epochs_and_time[2].append(time.time() - self.train_start_time)
                new_best = True
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f" bg: {metric_bg:.4f} fcd: {metric_fcd:.4f}"
                f"\nbest mean dice: {self.best_metric:.4f}"
                f" at epoch: {self.best_metric_epoch}"
                f"\nvalidation loss: {avg_val_loss:.4f}",
                flush=True
            )
            return new_best, metric, metric_bg, metric_fcd, avg_val_loss

    def train(self, train_data_dir, val_data_dir, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
        # Update number of input channels based on thickness map inclusion
        include_thickness = self.params['model_type'] == 'CAMST'
        if include_thickness:
            self.params['chans_in'] = 3  # T1, FLAIR, and thickness map
        
        self.init_loaders(train_data_dir, val_data_dir, include_thickness)
        self.model, self.params = get_model(self.params)
        self.model.to(self.device)
        self.init_stats()

        # Initialize loss function
        self.loss_function = self.get_loss_function()

        # Load existing model weights if available
        model_path = os.path.join(save_dir, "best_metric_model.pth")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded existing model weights from {model_path}")

        # Initialize Weights & Biases with more detailed config
        wandb_config = {
            **self.params,
            "optimizer": "Adam",
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "loss_function": self.loss_function.__class__.__name__,
            "max_epochs": 300,
            "early_stopping_patience": self.early_stopping_patience,
        }
        wandb.init(project="fcd_detection", name=f"{self.params['model_type']}_Training", config=wandb_config)
        wandb.watch(self.model, log="all")

        # Create a detailed log file
        log_file = os.path.join(save_dir, "training_log.txt")
        with open(log_file, "w") as log:
            log.write("Epoch,Train_Loss,Val_Loss,Val_Dice,Val_Dice_BG,Val_Dice_FCD,Learning_Rate,Time\n")

        max_epochs = 300
        optimizer = torch.optim.AdamW(self.model.parameters(), 1e-4, weight_decay=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, min_lr=self.min_lr
        )

        # enable cuDNN benchmark
        torch.backends.cudnn.benchmark = True

        epoch_loss_values = []

        self.train_start_time = time.time()
        for epoch in range(max_epochs):
            epoch_start = time.time()
            print("-" * 10)
            print(f"epoch {epoch + 1}/{max_epochs}", flush=True)
            self.model.train()
            epoch_loss = 0
            step = 0
            train_iterator = tqdm(self.train_loader, dynamic_ncols=True)
            
            for batch_data in train_iterator:
                step += 1
                inputs, labels = (
                    batch_data["image"].to(self.device, dtype=torch.float32),
                    batch_data["label"].to(self.device, dtype=torch.float32),
                )
                
                optimizer.zero_grad()
                
                with torch.amp.autocast("cuda"):
                    if self.params['model_type'] == 'CAMST':
                        # Extract thickness map (third channel)
                        thickness_map = inputs[:, 2:3]
                        # Remove thickness map from inputs
                        model_inputs = inputs[:, :2]
                        outputs = self.model(model_inputs, thickness_map)
                        loss = self.loss_function(outputs, labels, thickness_map)
                    else:
                        outputs = self.model(inputs)
                        loss = self.loss_function(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                
                epoch_loss += loss.item()
                train_iterator.set_description(f"train_loss: {loss.item():.4f}")

            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)

            # Validation Step
            val_dice, val_dice_bg, val_dice_fcd, val_loss = None, None, None, None
            if (epoch + 1) % self.val_interval == 0:
                new_best, val_dice, val_dice_bg, val_dice_fcd, val_loss = self.validate(epoch)
                if new_best:
                    torch.save(self.model.state_dict(), model_path)
                    print("saved new best metric model", flush=True)
                
                # Update learning rate based on validation metric
                lr_scheduler.step(val_dice)
                current_lr = optimizer.param_groups[0]['lr']
                
                # Early stopping check
                if self.early_stopping_counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
                
                # Learning rate minimum check
                if current_lr <= self.min_lr:
                    print(f"Learning rate {current_lr} below minimum threshold, stopping training")
                    break

            elapsed_time = time.time() - epoch_start

            # Log results to wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "val_loss": val_loss if val_loss is not None else 0,
                "val_dice": val_dice if val_dice is not None else 0,
                "val_dice_bg": val_dice_bg if val_dice_bg is not None else 0,
                "val_dice_fcd": val_dice_fcd if val_dice_fcd is not None else 0,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch_time": elapsed_time
            })

            # Write to log file
            with open(log_file, "a") as log:
                log_message = f"{epoch + 1},{epoch_loss:.4f},"
                if val_loss is not None:
                    log_message += f"{val_loss:.4f},{val_dice:.4f},{val_dice_bg:.4f},{val_dice_fcd:.4f},"
                else:
                    log_message += ",,,,,"
                log_message += f"{optimizer.param_groups[0]['lr']:.6f},{elapsed_time:.4f}"
                log.write(log_message + "\n")

        total_time = time.time() - self.train_start_time
        print(f"Training completed, total time: {total_time:.2f} seconds")
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Model for FCD Detection.')
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Path to the train dataset directory.')
    parser.add_argument('--val_dir', type=str, required=True,
                        help='Path to the validation dataset directory.')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path to the model output directory.')
    parser.add_argument('--model_type', type=str, default='CAMST',
                        help='Model type to use (CAMST, MS_DSA_NET, etc.)')

    args = parser.parse_args()
    params['model_type'] = args.model_type
    
    model_trainer = ModelTrainer(params, device)
    model_trainer.train(args.train_dir, args.val_dir, args.save_dir)
