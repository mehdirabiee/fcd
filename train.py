import os
import time
from datetime import datetime
from typing import List
import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

import torch.multiprocessing as mp

from monai.data import Dataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference

from monai.transforms import Compose, Activations, AsDiscrete, EnsureType
from monai.utils.misc import ensure_tuple_rep
from tqdm import tqdm
import wandb
from dotenv import load_dotenv


from get_loss import CombinedLoss
from utils import post_process_segment
from metrics import calculate_voxel_level_metrics, calculate_lesion_wise_metrics
from train_utils import get_optimizer, initialize_weights, seed_torch, validate_gpu_ids
from train_cli_utils import parse_args, parse_kwargs
from config import get_default_params
from get_data import read_split_file, get_data
from get_model import get_model
from get_transforms import FCDTrainTransform
from contextlib import nullcontext
from codecarbon import EmissionsTracker

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system') #'file_system', 'file_descriptor'
if mp.get_start_method(allow_none=True) != "fork": #fork, spawn, forkserver
    mp.set_start_method("fork", force=True)

class ModelTrainer:
    latest_model_filename = "latest_model.pth"
    best_model_filename = "best_model.pth"
    def __init__(self, params, device):
        self.params = params
        seed_torch(self.params['seed'], deterministic=self.params['deterministic'])
        self.device = device
        
        self.val_interval = 1
        self.model = None
        self.transforms = FCDTrainTransform(self.params)
        self.train_start_time = time.time()
        if os.environ.get("WANDB_MODE") != "offline":
            wandb.login(key=os.environ["WANDB_API_KEY"])        
        self.init_stats()
        self.early_stopping_patience = params.get('early_stopping_patience', 25)
        self.early_stopping_counter = 0
        self.ema_val_loss = None
        self.min_lr = params.get('min_lr', 1e-6)
        self.loss_function = CombinedLoss(self.params, self.device)
        self.model, self.params = get_model(self.params)
        self.model.to(self.device)
        self.model.apply(initialize_weights)

    def init_stats(self):
        """Initialize statistics tracking"""
        self.best_val_loss = float('inf')
        self.best_ema_val_loss = float('inf')
        self.best_val_loss_epoch = -1
        self.best_ema_val_loss_epoch = -1
        self.ema_val_loss = None
        self.early_stopping_counter = 0
        self.log_keys = None

    def init_loaders(self, data_dir, train_subjects, val_subjects):
        train_transform, val_transform = self.transforms.get_transforms()

        train_dict = get_data(data_dir, self.params, train_subjects)
        train_ds = Dataset(data=train_dict, transform=train_transform)
        #train_ds = CacheDataset(data=train_dict, transform=train_transform, num_workers=self.params['num_workers'])
        self.train_loader = DataLoader(
            train_ds, 
            batch_size=self.params['batch_size'], 
            shuffle=True,
            num_workers=self.params['num_workers'],
            pin_memory=True,
            persistent_workers=False
        )        

        val_dict = get_data(data_dir, self.params, val_subjects)
        val_ds = Dataset(data=val_dict, transform=val_transform)
        #val_ds = CacheDataset(data=val_dict, transform=val_transform, num_workers=self.params['num_workers'])
        self.val_loader = DataLoader(
            val_ds, 
            batch_size=1, 
            shuffle=False,
            num_workers=0, #self.params['num_workers'],
            pin_memory=False,
            persistent_workers=False
        )
        

    def compute_loss(self, pred_mask, true_mask, thickness_map=None):
        """
        Computes the main loss (Dice, DiceCE, or DiceFocal) and optionally adds TV loss.
        
        Args:
            pred_mask (Tensor): Model prediction of shape (B, C, D, H, W)
            true_mask (Tensor): Ground truth mask of shape (B, C, D, H, W)
            thickness_map (Tensor, optional): Cortical thickness map of shape (B, 1, D, H, W) if using cortical aware loss.
        Returns:
            loss (Tensor): Total loss combining main loss and optional TV loss or graph_laplacian loss.
        """
        loss = self.loss_function(pred_mask, true_mask, thickness_map)
        return loss

    def save_model(self, path, optimizer=None, lr_scheduler=None, scaler=None, epoch=None):
        """Save model, optimizer state, lr scheduler state, scaler, and epoch number."""
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),  
            'epoch': epoch
        }
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if lr_scheduler:
            checkpoint['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
        if scaler:
            checkpoint['scaler_state_dict'] = scaler.state_dict()

        torch.save(checkpoint, path)

    def load_model(self, path, optimizer=None, lr_scheduler=None, scaler=None):
        """Load model, optimizer state, lr scheduler state, scaler, and epoch number."""
        checkpoint = torch.load(path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if lr_scheduler and 'lr_scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        if scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.model.to(self.device)
        return checkpoint.get('epoch', None)

    def inference(self, input):
        def _compute(input):
            def _custom_predictor(x):
                y = self.model(x)
                if isinstance(y, (tuple,List)):
                    y = y[0]
                return y

            return sliding_window_inference(
                inputs=input,
                roi_size=self.params["patch_size"],
                sw_batch_size=2,
                predictor=_custom_predictor,
                overlap=0.25,
            )

        with torch.amp.autocast("cuda", enabled=self.params['use_amp']): 
            return _compute(input)

    def post_process(self, predictions, threshold=0.5):
        # Step 1: Thresholding to get binary mask
        n_pred_ch = predictions.shape[1]
        fcd_channel_index = 0 if n_pred_ch == 1 else 1
        fcd_predictions = predictions[0, fcd_channel_index, ...]  # Extract FCD channel
        binary_mask = (fcd_predictions > threshold).float()
        binary_mask = binary_mask.cpu().numpy()
        output_mask, output_label = post_process_segment(binary_mask, self.params['min_region_size'])
        #Convert back to tensor and restore original shape
        processed_fcd_mask = torch.tensor(output_mask, dtype=torch.float32, device=predictions.device)

        # Reconstruct output with same shape as `predictions`
        processed_output = predictions.clone()  # Keep original structure
        processed_output[0, fcd_channel_index, ...] = processed_fcd_mask  # Replace only FCD channel

        return processed_output

    def evaluate(self, data_loader, post_process=True, compute_lesion_level_metrics=False, desc="validation", include_hd95=False):
        post_trans = Compose([Activations(softmax=self.params['softmax'], sigmoid=self.params['sigmoid']), AsDiscrete(threshold=0.5), EnsureType()])

        self.model.eval()
        val_loss = 0
        num_val_batches = 0
        
        # Collect all predictions and labels for batch evaluation
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for val_data in tqdm(data_loader, desc=desc, dynamic_ncols=True):
                val_inputs, val_labels = (
                    val_data["image"].to(self.device, dtype=torch.float32),
                    val_data["label"].to(self.device, dtype=torch.float32),
                )
                
                val_outputs = self.inference(val_inputs)
                n_pred_ch = val_outputs.shape[1]
                fcd_channel_index = 0 if n_pred_ch == 1 else 1

                val_loss += self.compute_loss(val_outputs, val_labels).item()
                num_val_batches += 1

                decoded_outputs = decollate_batch(val_outputs)
                transformed_outputs = [post_trans(i) for i in decoded_outputs]
                val_outputs = torch.cat(transformed_outputs, dim=0).unsqueeze(0)
                post_processed_outputs = self.post_process(val_outputs) if post_process else val_outputs
                
                # Store predictions and labels for batch evaluation
                all_predictions.append(post_processed_outputs[0, fcd_channel_index])  # Take FCD channel
                all_labels.append(val_labels[0, 0])  # Take first channel as it's binary


        metrics_voxel, metrics_subject, metrics_lesion = {}, {}, {}
        metrics_voxel = calculate_voxel_level_metrics(all_predictions, all_labels, compute_hd95=include_hd95, average_across_subjects=False)

        if compute_lesion_level_metrics:
            metrics_lesion = calculate_lesion_wise_metrics(all_predictions, all_labels)  
            #metrics_subject = calculate_subject_level_metrics(all_predictions, all_labels)
        metrics = {**metrics_voxel, **metrics_subject, **metrics_lesion}

        val_loss = val_loss / num_val_batches


        # Print comprehensive metrics
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}", flush=True)

        return val_loss, metrics

    def test(self, data_dir, test_subjects, post_process=True):
        if len(test_subjects) == 0:
            print("No test subjects provided, skipping testing.")
            return {}
        _, val_transform = self.transforms.get_transforms()

        test_dict = get_data(data_dir, self.params, test_subjects)
        #test_ds = CacheDataset(data=test_dict, transform=val_transform, num_workers=self.params['num_workers'])
        test_ds = Dataset(data=test_dict, transform=val_transform)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False,persistent_workers=False)

        val_loss, metrics = self.evaluate(test_loader, post_process=post_process, compute_lesion_level_metrics=True, include_hd95=True, desc="test"+("_postprocess" if post_process else ""))
        print(','.join([f"{k}" for k, _ in metrics.items()])+',', flush=True)
        print(','.join([f"{v:.4f}" for _, v in metrics.items()])+',', flush=True)
        return metrics

    def validate(self, epoch):
        avg_val_loss, metrics = self.evaluate(self.val_loader, post_process=False, compute_lesion_level_metrics=False, include_hd95=False, desc="validation")

        new_best = False
        if self.ema_val_loss is None:
            self.ema_val_loss = avg_val_loss
        else:
            # Exponential moving average for validation loss
            alpha = self.params['val_loss_ema_alpha']
            self.ema_val_loss = (1 - alpha) * avg_val_loss + alpha * self.ema_val_loss
        
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.best_val_loss_epoch = epoch + 1
            new_best = True

        if self.ema_val_loss < self.best_ema_val_loss:
            self.best_ema_val_loss = self.ema_val_loss
            self.best_ema_val_loss_epoch = epoch + 1
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1

        print(
            f"current epoch: {epoch + 1} validation loss: {avg_val_loss:.4f}, ema_val_loss: {self.ema_val_loss:.4f}"
            f"\nbest validation loss: {self.best_val_loss:.4f}"
            f" at epoch: {self.best_val_loss_epoch}",
            f"\nbest ema_val_loss: {self.best_ema_val_loss:.4f}"
            f" at epoch: {self.best_ema_val_loss_epoch}",
            flush=True)
        return new_best, metrics, avg_val_loss

    def log_metrics(self, epoch, train_loss, val_loss, ema_val_loss, val_metrics, optimizer, elapsed_time, csv_path=None):
        """Log metrics to wandb and optionally to CSV."""
        values_dict = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss if val_loss is not None else 0,
            "ema_val_loss": ema_val_loss if ema_val_loss is not None else 0,
            **({f"val_{k}": v for k, v in val_metrics.items()} if val_metrics is not None else {}),
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch_time": elapsed_time
        }
        """Log metrics to wandb."""
        wandb.log(values_dict)
        if csv_path:
            if epoch == 0 or self.log_keys is None or not os.path.exists(csv_path):
                # Write header if it's the first epoch
                with open(csv_path, 'w') as f:
                    f.write(','.join(values_dict.keys()) + '\n')
                self.log_keys = values_dict.keys()
            with open(csv_path, 'a') as f:
                log_values = [str(values_dict[k]) for k in self.log_keys]
                f.write(','.join(log_values) + '\n')

    def train(self, data_dir, train_subjects, val_subjects, save_dir, test_subjects=[], resume=False):
        if len(train_subjects) == 0 or len(val_subjects) == 0:
            raise ValueError("Train and validation subject lists must be non-empty.")
        self.init_loaders(data_dir, train_subjects, val_subjects)
        os.makedirs(save_dir, exist_ok=True)

        self.init_stats()

        latest_model_path = os.path.join(save_dir, self.latest_model_filename)
        best_model_path = os.path.join(save_dir, self.best_model_filename)
        log_file_path = os.path.join(save_dir, "training_log.csv")

        max_epochs = self.params.get('max_epochs', 300)

        optimizer = get_optimizer(self.model, self.params)

        warmup_epochs = self.params.get('warmup_epochs', 10)
        min_lr = self.params.get('min_lr', 1e-6)
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        anneal_scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs - warmup_epochs, eta_min=min_lr)
        lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, anneal_scheduler], milestones=[warmup_epochs])        
        scaler = torch.amp.GradScaler('cuda', enabled=self.params['use_amp'])

        wandb_config = {
            **self.params,
            "optimizer": "AdamW",
            "early_stopping_patience": self.early_stopping_patience,
        }
        save_dir_name = os.path.basename(save_dir)
        wandb.init(project=self.params["wandb_project"], name=f"{self.params['model_type']}_{save_dir_name}", config=wandb_config)
        wandb.watch(self.model, log="all")


        current_epoch = 0
        if resume and os.path.exists(latest_model_path):
            current_epoch = self.load_model(latest_model_path, optimizer=optimizer, lr_scheduler=lr_scheduler, scaler=scaler)
            if not current_epoch:
                current_epoch = 0
            for _ in range(current_epoch):
                lr_scheduler.step()
            print(f"Loaded existing model weights from {latest_model_path}")
        gradient_accumulation_steps = self.params.get("gradient_accumulation_steps", 1)
        self.train_start_time = time.time()
        for epoch in range(current_epoch, max_epochs):
            epoch_start = time.time()
            
            print("-" * 10)
            print(f"epoch {epoch + 1}/{max_epochs}", flush=True)

            self.model.train()

            if self.transforms.has_gradual_prob():
                self.transforms.set_prob(epoch, max_epochs)
            epoch_loss = 0
            step = 0
            
            train_iterator = tqdm(self.train_loader, dynamic_ncols=True) 

            for batch_data in train_iterator:
                step += 1
                inputs, labels = (
                    batch_data["image"].to(self.device, dtype=torch.float32),
                    batch_data["label"].to(self.device, dtype=torch.float32),
                )
                
                
                with torch.amp.autocast("cuda", enabled=self.params['use_amp']):
                    outputs = self.model(inputs)
                    loss_vae = 0
                    if isinstance(outputs, (tuple, list)):
                        if self.params['model_returns_vaeloss']:
                            loss_vae = outputs[1]
                        outputs = outputs[0]
                    loss = self.compute_loss(outputs, labels) + self.params['loss_vae_weight'] * loss_vae
                
                epoch_loss += loss.item()
                
                loss = loss / gradient_accumulation_steps 
                scaler.scale(loss).backward()
                
                if step % gradient_accumulation_steps == 0 or step == len(self.train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                train_iterator.set_description(f"train_loss: {(epoch_loss/step):.4f}")

            epoch_loss /= step
            lr_scheduler.step()

            val_metrics, val_loss = {}, None
            if (epoch + 1) % self.val_interval == 0:
                new_best, val_metrics, val_loss = self.validate(epoch)
                if new_best:
                    self.save_model(best_model_path, optimizer, lr_scheduler, scaler, epoch)
                    print("saved new best metric model", flush=True)
                stop_flag = epoch >= self.params["min_epochs"] and (self.early_stopping_counter >= self.early_stopping_patience or optimizer.param_groups[0]['lr'] <= self.min_lr)


                if stop_flag:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break

            if self.params['keep_latest_model']:
                self.save_model(latest_model_path, optimizer, lr_scheduler, scaler, epoch)

            elapsed_time = time.time() - epoch_start

            self.log_metrics(epoch, epoch_loss, val_loss, self.ema_val_loss, val_metrics, optimizer, elapsed_time, csv_path=log_file_path)


        total_time = time.time() - self.train_start_time
        print(f"Training completed, total time: {total_time:.2f} seconds")

        if len(test_subjects) > 0:
            self.load_model(best_model_path)
            self.test(data_dir, test_subjects, post_process=False)
            self.test(data_dir, test_subjects, post_process=True)

        wandb.finish()

def main():
    load_dotenv()    
    params = get_default_params()
    args = parse_args(default_params=params)
    params['model_type'] = args.model_type
    
    if args.kwargs:
        params = parse_kwargs(params, args.kwargs)
    
    _, params = get_model(params, return_model=False)
    params['chans_in'] = len(params['seq'].split('+'))
    
    params['patch_size'] = ensure_tuple_rep(params['patch_size'], 3)


    enable_emission_tracking = args.emission_tracking
    
    gpu_indices = validate_gpu_ids(args.gpus)
    
    device = torch.device(f'cuda:{gpu_indices[0]}')
    trainer = ModelTrainer(params, device)
    if args.checkpoint_path:
        trainer.load_model(args.checkpoint_path)

    split_dict = read_split_file(args.split_file)
    requested_splits = {s.lower() for s in args.splits}

    if "train" in requested_splits:
        train_subjects = split_dict.get("train", [])
        val_subjects = split_dict.get("val", [])
        test_subjects = split_dict.get("test", []) if "test" in requested_splits else []
        
        if args.resume:
            save_dir = args.save_dir
        else: 
            timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            if args.prefix and args.prefix != '':
                timestamp = f"{args.prefix}_{timestamp}"
            save_dir = os.path.join(args.save_dir, params['model_type'], timestamp)

        os.makedirs(save_dir, exist_ok=True)
        tracker = None
        context = EmissionsTracker(log_level="critical", project_name="fcd_detection", output_dir=save_dir, output_file="train_emission.csv", save_to_file=True) if enable_emission_tracking else nullcontext()
        with context as tracker:
            trainer.train(args.data_dir, train_subjects, val_subjects, save_dir, test_subjects, resume=args.resume)
        if enable_emission_tracking and tracker is not None:
            print(f"\nCarbon emissions from computation: {tracker.final_emissions * 1000:.4f} g CO2eq")

    elif "test" in requested_splits:
        test_subjects = split_dict.get("test", [])
        model_dir = os.path.dirname(args.checkpoint_path)
        tracker = None
        context = EmissionsTracker(log_level="critical", project_name="fcd_detection", output_dir=model_dir, output_file=f"test_emission_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.csv", save_to_file=True) if enable_emission_tracking else nullcontext()
        with context as tracker:
            trainer.test(args.data_dir, test_subjects, post_process=False)
            trainer.test(args.data_dir, test_subjects, post_process=True)
        if enable_emission_tracking and tracker is not None:
            print(f"\nCarbon emissions from computation: {tracker.final_emissions * 1000:.4f} g CO2eq")

if __name__ == '__main__':
    main()