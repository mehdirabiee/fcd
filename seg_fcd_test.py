import os
import shutil
import tempfile
import glob
import random
import datetime

from monai.visualize import matshow3d
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device_ids = [0]
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import nibabel as nib
import torch

from monai.inferers import sliding_window_inference
from get_data import get_data
from get_model import get_model
from get_transforms import get_test_transforms, get_trainval_transforms

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)
from monai.transforms import (
    SaveImaged, LoadImage, AsDiscrete, LoadImaged,
)
from preprocess_data import preprocess_dataset_fsl

#==================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
params['samples_per_case'] = 8
params['model_desc_str'] = 'model'

def inspect_nifti_file_monai(data_file):
    image, meta = LoadImage(image_only=False)(data_file)

    #matshow3d(volume=image, title='3D Image Visualization', cmap='gray')
    #plt.show()

    # Get the affine transformation matrix
    affine = meta['affine'].numpy()
    print("Affine Matrix:\n", affine)

    # Convert to a NumPy array
    image_array = image.numpy()

    # Get the middle slices for each view
    axial_slice = image_array[:, :, image_array.shape[2] // 2]  # Middle slice along Z-axis
    coronal_slice = image_array[:, image_array.shape[1] // 2, :]  # Middle slice along Y-axis
    sagittal_slice = image_array[image_array.shape[0] // 2, :, :]  # Middle slice along X-axis

    # Plot the slices
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Axial view (looking from top-down)
    axes[0].imshow(axial_slice, cmap="gray")
    axes[0].set_title("Axial View (Top-Down)")
    axes[0].axis("off")

    # Coronal view (looking from front)
    axes[1].imshow(coronal_slice, cmap="gray")
    axes[1].set_title("Coronal View (Front)")
    axes[1].axis("off")

    # Sagittal view (looking from the side)
    axes[2].imshow(sagittal_slice, cmap="gray")
    axes[2].set_title("Sagittal View (Side)")
    axes[2].axis("off")

    plt.show()

def inspect_nifti_file_nib(data_file):
    # Load the NIfTI file
    nii_img = nib.load(data_file)

    # Get the affine transformation matrix
    affine = nii_img.affine
    print("Affine Matrix:\n", affine)

    # Check header metadata
    header = nii_img.header
    print("Header Info:\n", header)

    # Convert to a NumPy array
    image_array = nii_img.get_fdata(dtype=np.float32)

    # Get the middle slices for each view
    axial_slice = image_array[:, :, image_array.shape[2] // 2]  # Middle slice along Z-axis
    coronal_slice = image_array[:, image_array.shape[1] // 2, :]  # Middle slice along Y-axis
    sagittal_slice = image_array[image_array.shape[0] // 2, :, :]  # Middle slice along X-axis

    # Plot the slices
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Axial view (looking from top-down)
    axes[0].imshow(axial_slice, cmap="gray")
    axes[0].set_title("Axial View (Top-Down)")
    axes[0].axis("off")

    # Coronal view (looking from front)
    axes[1].imshow(coronal_slice, cmap="gray")
    axes[1].set_title("Coronal View (Front)")
    axes[1].axis("off")

    # Sagittal view (looking from the side)
    axes[2].imshow(sagittal_slice, cmap="gray")
    axes[2].set_title("Sagittal View (Side)")
    axes[2].axis("off")

    plt.show()

def evaluate(data_dir, save_dir, params):
    model, params = get_model(params)
    #save_dir = os.path.join(save_dir, params["model_desc_str"])
    os.makedirs(save_dir, exist_ok=True)

    pretrain = params['model_name']
    if os.path.exists(pretrain):
        model.load_state_dict(torch.load(pretrain))
        print(f'Pretrained model {pretrain} loaded')
    else:
        print('No pretrained model found')

    model.to(device=device)
    model.eval()

    # Setup the transformations
    test_transform, post_transform = get_test_transforms(params)

    # Load the data
    test_dict = get_data(data_dir, params)
    test_ds = CacheDataset(data=test_dict, transform=test_transform, cache_num=4, cache_rate=1,
                           num_workers=params['num_workers'])
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=params['num_workers'], pin_memory=True)

    # Initialize metrics
    dice_metric = DiceMetric(include_background=False)
    meaniou_metric = MeanIoU(include_background=False)

    epoch_iterator_val = tqdm(test_loader, desc=f"Test (0 / {len(test_loader)} Steps)", dynamic_ncols=True)
    metrics = dict()
    with torch.no_grad():
        for idx, batch in enumerate(epoch_iterator_val, 1):
            # Get the affine transformation and image name
            original_affine = batch["image_meta_dict"]["affine"][0].numpy()
            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-2]
            img_name = img_name.split('.')[0]

            val_inputs = batch["image"].to(device)

            # Perform inference
            val_outputs = sliding_window_inference(val_inputs, params['patch_size'], 2, model, overlap=0.25)
            batch['pred'] = val_outputs

            # Load the label images (using the label filenames in batch['label'] if available)
            if "label" in batch:  # Only process label if it exists
                label_files = batch["label"]
                labels = [LoadImaged(keys=["label"])({"label": label_file})["label"] for label_file in label_files]
                labels = torch.stack(labels).to(device)
            else:
                labels = None

            # Post-processing for the predictions
            output_dir = os.path.join(save_dir, img_name)
            os.makedirs(output_dir, exist_ok=True)

            batch = [post_transform(i) for i in decollate_batch(batch)]
            # Save the prediction result
            SaveImaged(keys="pred", output_dir=output_dir, output_postfix="seg", resample=False, separate_folder=False)(batch[0])

            # Compute the Dice score if label is available
            if labels is not None:
                pred = batch[0]['pred']  # Get the prediction

                # Update metrics

                # Handle case when ground truth is all zeros (edge case)
                if labels.sum() == 0:  # If ground truth is all zeros
                    if pred.sum() == 0:  # If prediction is also all zeros
                        dice_score = 1.0  # Perfect match
                        iou = 1.0
                    else:
                        dice_score = 0.0  # No match, false positives
                        iou = 0.0
                else:
                    dice_metric(pred, labels)
                    meaniou_metric(pred, labels)

                    dice_score = dice_metric.aggregate().item()
                    iou = meaniou_metric.aggregate().item()

                # Store metrics
                metrics[img_name] = {
                    'dice': dice_score,
                    'iou': iou
                }

                epoch_iterator_val.set_description(f"Test ({idx} / {len(test_loader)} Steps)")
                # Reset metrics
                dice_metric.reset()
                meaniou_metric.reset()

    # Print final metrics
    print(f"Final metrics: {metrics}", flush=True)
    avg_metrics = {
        "dice": np.mean([m['dice'] for m in metrics.values()]),
        "iou": np.mean([m['iou'] for m in metrics.values()])
    }
    print(f"Average Dice: {avg_metrics['dice']:.4f}, Average IOU: {avg_metrics['iou']:.4f}", flush=True)

def test(data_dir, save_dir, params):
    model, params = get_model(params)
    #save_dir = os.path.join(save_dir, params["model_desc_str"])
    os.makedirs(save_dir, exist_ok=True)

    pretrain = params['model_name']
    if os.path.exists(pretrain):
        model.load_state_dict(torch.load(pretrain))
        print('pretrained model ' + pretrain + ' loaded')
    else:
        print('no pretrained model found')

    model.to(device=device)
    model.eval()

    test_transform, post_transform = get_test_transforms(params)
    test_dict = get_data(data_dir, params)
    test_ds = CacheDataset(data=test_dict, transform=test_transform, cache_num=4, cache_rate=1,
                           num_workers=params['num_workers'])
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=params['num_workers'], pin_memory=True)

    epoch_iterator_val = tqdm(test_loader, desc="test (X / X Steps) (dice=X.X)", dynamic_ncols=True)

    metrics = dict()
    with torch.no_grad():
        idx = 0
        for batch in epoch_iterator_val:
            idx += 1

            original_affine = batch["image_meta_dict"]["affine"][0].numpy()
            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-2]
            img_name = img_name.split('.')[0]

            val_inputs = batch["image"].to(device)
            val_outputs = sliding_window_inference(val_inputs, params['patch_size'], 2, model, overlap=0.25)
            batch['pred'] = val_outputs

            output_dir = os.path.join(save_dir, img_name)
            os.makedirs(output_dir, exist_ok=True)

            batch = [post_transform(i) for i in decollate_batch(batch)]
            #val_preds = val_inputs['pred'].cpu().numpy().astype(np.uint8)
            output_name = os.path.join(output_dir, 'seg.nii.gz')
            SaveImaged(keys="pred", output_dir=output_dir, output_postfix="seg", resample=False, separate_folder=False)(
                batch[0])


if __name__ == '__main__':
    data_dir = '/mnt/d/dimes/dataset/ds004199/'
    preprocessed_data_dir = '/mnt/d/dimes/dataset/exp'
    save_dir = '/mnt/d/dimes/dataset/ds004199_out'

    #inspect_nifti_file_monai('/mnt/d/dimes/dataset/ds004199_fsl/sub-00001/t1_reg.nii.gz')

    evaluate(preprocessed_data_dir, save_dir, params)
