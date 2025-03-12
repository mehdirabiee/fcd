import argparse
import os
import time

import torch
from matplotlib import pyplot as plt
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks import one_hot
from monai.utils import first, set_determinism
from monai.transforms import Compose, Activations, AsDiscrete
from tqdm import tqdm

from get_data import get_data
from get_model import get_model
from get_transforms import get_trainval_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_determinism(seed=0)
params = dict()

params['model_name'] = './pretrained/model.pth'
params['model_type'] = 'MS_DSA_NET'
params['sa_type'] = 'parallel'
params['chans_in'] = 2
params['chans_out'] = 2
params['feature_size'] = 16
params['project_size'] = 64  #dsa projection size
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
        self.init_stats()


    @staticmethod
    def visualize_data(data_loader):
        val_data_example = first(data_loader)
        print(f"image shape: {val_data_example['image'].shape}")
        print(f"label shape: {val_data_example['label'].shape}")
        plt.figure("image", (24, 6))
        for i in range(2):
            plt.subplot(1, 3, i + 1)
            plt.title(f"image channel {i}")
            plt.imshow(val_data_example["image"][0, i, :, :, 60].detach().cpu(), cmap="gray")
        plt.subplot(1, 3, 3)
        plt.title(f"label")
        plt.imshow(val_data_example["label"][0, 0, :, :, 60].detach().cpu())
        plt.show()

    def init_loaders(self, train_data_dir, val_data_dir):
        train_transform, val_transform = get_trainval_transforms(self.params)

        train_dict = get_data(train_data_dir, self.params)
        train_ds = CacheDataset(data=train_dict, transform=train_transform, cache_num=4, cache_rate=1,
                                num_workers=self.params['num_workers'])
        self.train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=self.params['num_workers'],
                                       pin_memory=True)

        # visualize_data(train_loader)

        val_dict = get_data(val_data_dir, self.params)
        val_ds = CacheDataset(data=val_dict, transform=val_transform, cache_num=4, cache_rate=1,
                              num_workers=1)
        self.val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=self.params['num_workers'],
                                     pin_memory=True)

    def init_stats(self):
        self.best_metric = -1
        self.best_metric_epoch = -1
        self.best_metrics_epochs_and_time = [[], [], []]
        self.metric_values = []
        self.metric_values_bg = []
        self.metric_values_fcd = []

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

    def validate(self, epoch):
        post_trans = Compose([Activations(softmax=True), AsDiscrete(threshold=0.5)])
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
        new_best = False

        self.model.eval()
        with torch.no_grad():
            for val_data in tqdm(self.val_loader, desc="validation", dynamic_ncols=True):
                val_inputs, val_labels = (
                    val_data["image"].to(self.device, dtype=torch.float32),
                    val_data["label"].to(self.device, dtype=torch.float32),
                )
                val_outputs = self.inference(val_inputs)
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
            dice_metric.reset()
            dice_metric_batch.reset()

            if metric > self.best_metric:
                self.best_metric = metric
                self.best_metric_epoch = epoch + 1
                self.best_metrics_epochs_and_time[0].append(self.best_metric)
                self.best_metrics_epochs_and_time[1].append(self.best_metric_epoch)
                self.best_metrics_epochs_and_time[2].append(time.time() - self.train_start_time)
                new_best = True

            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f" bg: {metric_bg:.4f} fcd: {metric_fcd:.4f}"
                f"\nbest mean dice: {self.best_metric:.4f}"
                f" at epoch: {self.best_metric_epoch}",
                flush=True
            )
            return new_best

    def train(self, train_data_dir, val_data_dir, save_dir):

        self.init_loaders(train_data_dir, val_data_dir)
        self.model, self.params = get_model(self.params)
        self.model.to(self.device)
        self.init_stats()

        max_epochs = 300
        loss_function = DiceLoss(include_background=False, smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=True, sigmoid=False, softmax=True)
        optimizer = torch.optim.Adam(self.model.parameters(), 1e-4, weight_decay=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

        # use amp to accelerate training
        scaler = torch.amp.GradScaler("cuda")
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
                    batch_data["label"].to(self.device, dtype= torch.float32),
                )
                #labels = one_hot(labels, num_classes=2)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda"):
                    outputs = self.model(inputs)
                    loss = loss_function(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
                train_iterator.set_description(f"train_loss: {loss.item():.4f}")
            lr_scheduler.step()
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}, time: {(time.time() - epoch_start):.4f}", flush=True)

            if (epoch + 1) % self.val_interval == 0:
                if self.validate(epoch):
                    torch.save(self.model.state_dict(), os.path.join(save_dir, "best_metric_model.pth"))
                    print("saved new best metric model", flush=True)

        total_time = time.time() - self.train_start_time
        print(f"train completed, total time: {total_time}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Model for FCD Detection.')
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Path to the train dataset directory.')
    parser.add_argument('--val_dir', type=str, required=True,
                        help='Path to the validation dataset directory.')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path to the model output directory.')

    args = parser.parse_args()
    train_data_dir =  args.train_dir #'/mnt/d/dimes/dataset/ds004199_fsl'
    val_data_dir =  args.val_dir #'/mnt/d/dimes/dataset/ds004199_fsl'
    save_dir = args.save_dir #'/mnt/d/dimes/dataset/model'

    model_trainer = ModelTrainer(params, device)
    model_trainer.train(train_data_dir, val_data_dir, save_dir)
