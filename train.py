import argparse
import os
import time
from datetime import datetime
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss, DiceCELoss, DiceFocalLoss
from monai.metrics import (
    DiceMetric,
    MeanIoU,
    HausdorffDistanceMetric,
    SurfaceDistanceMetric,
    ConfusionMatrixMetric,
)
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
import numpy as np
import torch
from scipy.ndimage import binary_dilation, label
import skimage.measure
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_determinism(seed=0)
params = dict()

params['model_name'] = './pretrained/model.pth'
params['model_type'] = 'MS_DSA_NET'  # Default to MS_DSA_NET
params['sa_type'] = 'parallel'
params['chans_in'] = 2
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
        self.scaler = torch.amp.GradScaler('cuda')
        self.early_stopping_patience = 15
        self.early_stopping_counter = 0
        self.min_lr = 1e-6
        self.init_metrics()

    def init_metrics(self):
        """Initialize all metrics for tracking"""
        # Single Dice metric for both background and FCD
        self.dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
        
        # Single IoU metric for both background and FCD
        self.iou_metric = MeanIoU(include_background=True, reduction="mean_batch")
        
        # Distance metrics
        self.hausdorff_metric = HausdorffDistanceMetric(include_background=False, percentile=95.0)
        self.surface_distance_metric = SurfaceDistanceMetric(include_background=False)
        
        # Confusion matrix metrics
        self.confusion_matrix = ConfusionMatrixMetric(
            include_background=False,
            metric_name=["sensitivity", "specificity", "precision", "f1_score"]
        )
        

    def init_stats(self):
        """Initialize statistics tracking"""
        self.best_val_loss = float('inf')
        self.best_val_loss_epoch = -1
        self.metric_values = {
            'dice_bg': [],
            'dice_fcd': [],
            'iou_bg': [],
            'iou_fcd': [],
            'hausdorff': [],
            'surface_distance': [],
            'sensitivity': [],
            'specificity': [],
            'precision': [],
            'f1_score': [],
        }

    def init_loaders(self, train_data_dir, val_data_dir):
        train_transform, val_transform = get_trainval_transforms(self.params)

        train_dict = get_data(train_data_dir, self.params)
        train_ds = CacheDataset(data=train_dict, transform=train_transform, cache_num=4, cache_rate=1,
                                num_workers=self.params['num_workers'])
        self.train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=self.params['num_workers'],
                                       pin_memory=True)

        val_dict = get_data(val_data_dir, self.params)
        val_ds = CacheDataset(data=val_dict, transform=val_transform, cache_num=4, cache_rate=1,
                              num_workers=1)
        self.val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=self.params['num_workers'],
                                     pin_memory=True)

    def get_loss_function(self):
        #return DiceCELoss(
        #    include_background=False,
        #    to_onehot_y=True,
        #    softmax=True,
        #    lambda_dice=0.8,
        #    lambda_ce=0.2
        #)
        return DiceFocalLoss(
            include_background=False,  
            to_onehot_y=True,  
            sigmoid=True,  
            gamma=2.0  
        )    

    def inference(self, input):
        def _compute(input):
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


    def post_process(self, predictions, threshold=0.5, min_region_size=50):
        # Step 1: Thresholding to get binary mask
        fcd_predictions = predictions[0, 1, ...]  # Extract FCD channel
        binary_mask = (fcd_predictions > threshold).float()

        # Step 2: Dilation (expand the FCD region a bit)
        dilated_mask = binary_dilation(binary_mask.cpu().numpy(), structure=np.ones((3, 3, 3)))

        # Step 3: Connected component analysis (remove small regions)
        labeled_mask, num_labels = skimage.measure.label(dilated_mask, return_num=True)
        
        # Step 4: Remove small connected components (based on min_region_size)
        for region in range(1, num_labels + 1):
            region_mask = labeled_mask == region
            if np.sum(region_mask) < min_region_size:
                dilated_mask[region_mask] = 0  # Remove small regions

        # Step 5: Convert back to tensor and restore original shape
        processed_fcd_mask = torch.tensor(dilated_mask, dtype=torch.float32, device=predictions.device)

        # Reconstruct output with same shape as `predictions`
        processed_output = predictions.clone()  # Keep original structure
        processed_output[0, 1, ...] = processed_fcd_mask  # Replace only FCD channel

        return processed_output

    def validate(self, epoch):
        """Enhanced validation with comprehensive metrics"""
        post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
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
                
                val_outputs = self.inference(val_inputs)
                val_loss += self.loss_function(val_outputs, val_labels).item()
                num_val_batches += 1

                decoded_outputs = decollate_batch(val_outputs)  # Break into individual elements
                transformed_outputs = [post_trans(i) for i in decoded_outputs]  # Apply transformation
                val_outputs = torch.cat(transformed_outputs, dim=0).unsqueeze(0)  # Reassemble   
                post_processed_outputs = self.post_process(val_outputs)
             

                val_labels = one_hot(val_labels, num_classes=2)
                
                # Update all metrics 
                self.dice_metric(y_pred=post_processed_outputs, y=val_labels)
                self.iou_metric(y_pred=val_outputs, y=val_labels)
                self.hausdorff_metric(y_pred=val_outputs, y=val_labels)
                self.surface_distance_metric(y_pred=val_outputs, y=val_labels)
                self.confusion_matrix(y_pred=val_outputs, y=val_labels)

            # Aggregate all metrics
            dice_metrics = self.dice_metric.aggregate()
            iou_metrics = self.iou_metric.aggregate()
            hausdorff = self.hausdorff_metric.aggregate().item()
            surface_distance = self.surface_distance_metric.aggregate().item()
            
            # Aggregate confusion matrix once
            cm_metrics = self.confusion_matrix.aggregate()
            sensitivity = cm_metrics[0].item()
            specificity = cm_metrics[1].item()
            precision = cm_metrics[2].item()
            f1_score = cm_metrics[3].item()
            
            metrics = {
                'dice_bg': dice_metrics[0].item(),
                'dice_fcd': dice_metrics[1].item(),
                'iou_bg': iou_metrics[0].item(),
                'iou_fcd': iou_metrics[1].item(),
                'hausdorff': hausdorff,
                'surface_distance': surface_distance,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'precision': precision,
                'f1_score': f1_score,
            }
            
            avg_val_loss = val_loss / num_val_batches
            
            # Update metric history
            for metric_name, value in metrics.items():
                self.metric_values[metric_name].append(value)
            
            # Reset all metrics
            self.dice_metric.reset()
            self.iou_metric.reset()
            self.hausdorff_metric.reset()
            self.surface_distance_metric.reset()
            self.confusion_matrix.reset()
            
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.best_val_loss_epoch = epoch + 1
                new_best = True
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

            # Print comprehensive metrics
            print(
                f"current epoch: {epoch + 1} validation loss: {avg_val_loss:.4f}"
                f"\nDice - BG: {metrics['dice_bg']:.4f}, FCD: {metrics['dice_fcd']:.4f}"
                f"\nIoU - BG: {metrics['iou_bg']:.4f}, FCD: {metrics['iou_fcd']:.4f}"
                f"\nHausdorff Distance: {metrics['hausdorff']:.4f}"
                f"\nSurface Distance: {metrics['surface_distance']:.4f}"
                f"\nSensitivity: {metrics['sensitivity']:.4f}"
                f"\nSpecificity: {metrics['specificity']:.4f}"
                f"\nPrecision: {metrics['precision']:.4f}"
                f"\nF1 Score: {metrics['f1_score']:.4f}"
                f"\nbest validation loss: {self.best_val_loss:.4f}"
                f" at epoch: {self.best_val_loss_epoch}",
                flush=True
            )
            return new_best, metrics, avg_val_loss

    def train(self, train_data_dir, val_data_dir, save_dir):
        
        self.init_loaders(train_data_dir, val_data_dir)
        self.model, self.params = get_model(self.params)
        save_dir = os.path.join(save_dir, self.params['model_desc_str'])
        os.makedirs(save_dir, exist_ok=True)

        self.model.to(self.device)
        self.init_stats()

        # Initialize loss function
        self.loss_function = self.get_loss_function()


        latest_model_path = os.path.join(save_dir, "latest_model.pth")
        best_model_path = os.path.join(save_dir, "best_model.pth")
        # Load existing model weights if available
        if os.path.exists(latest_model_path):
            self.model.load_state_dict(torch.load(latest_model_path, map_location=self.device))
            print(f"Loaded existing model weights from {latest_model_path}")

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
            optimizer, mode='min', factor=0.5, patience=5, min_lr=self.min_lr
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
            val_metrics, val_loss = {}, None
            if (epoch + 1) % self.val_interval == 0:
                new_best, val_metrics, val_loss = self.validate(epoch)
                if new_best:
                    torch.save(self.model.state_dict(), best_model_path)
                    print("saved new best metric model", flush=True)
                
                # Update learning rate based on validation metric
                lr_scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]['lr']
                
                # Early stopping check
                if self.early_stopping_counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
                
                # Learning rate minimum check
                if current_lr <= self.min_lr:
                    print(f"Learning rate {current_lr} below minimum threshold, stopping training")
                    break
            #save the latest model until now
            torch.save(self.model.state_dict(), latest_model_path)
            
            elapsed_time = time.time() - epoch_start

            # Log results to wandb with all metrics
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "val_loss": val_loss if val_loss is not None else 0,
                **({f"val_{k}": v for k, v in val_metrics.items()} if val_metrics is not None else {}),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch_time": elapsed_time
            })

            # Write to log file with all metrics
            with open(log_file, "a") as log:
                log_message = f"{epoch + 1},{epoch_loss:.4f},"
                if val_loss is not None:
                    log_message += f"{val_loss:.4f},"
                    for metric_name in ['dice_bg', 'dice_fcd', 'iou_bg', 'iou_fcd', 
                                      'hausdorff', 'surface_distance', 'sensitivity', 
                                      'specificity', 'precision', 'f1_score']:
                        log_message += f"{val_metrics[metric_name]:.4f},"
                else:
                    log_message += ",,,,,,,,,,,,,"
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
    parser.add_argument('--model_type', type=str, default='MS_DSA_NET',
                        help='Model type to use (default: MS_DSA_NET)')

    args = parser.parse_args()
    params['model_type'] = args.model_type
    
    
    model_trainer = ModelTrainer(params, device)
    model_trainer.train(args.train_dir, args.val_dir, os.path.join(args.save_dir, datetime.now().strftime("%Y-%m-%d")))
