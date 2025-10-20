import os
import argparse


from monai.metrics import DiceMetric, MeanIoU

from config import get_default_params

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device_ids = [0]
from tqdm import tqdm
import numpy as np
import torch

from monai.inferers import sliding_window_inference
from get_data import get_data
from get_model import get_model
from get_transforms import get_test_transforms
from utils import post_process_segment


from monai.data import (
    DataLoader,
    CacheDataset,
    decollate_batch,
)
from monai.transforms import (
    Compose, SaveImaged, LoadImaged,
)
from preprocess_data import preprocess_dataset_fsl

#==================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def inference(input,  model, params):
    def _compute(input):
        def _custom_predictor(x):
            y = model(x)
            if isinstance(y, tuple) and len(y) == 2:
                y = y[0]
            return y

        return sliding_window_inference(
            inputs=input,
            roi_size=params["patch_size"],
            sw_batch_size=2,
            predictor=_custom_predictor,
            overlap=0.25,
        )

    with torch.amp.autocast("cuda", enabled=params['use_amp']): 
        return _compute(input)

def post_process(predictions, min_region_size=50, threshold=0.5):
    # Step 1: Thresholding to get binary mask
    n_pred_ch = predictions.shape[1]
    fcd_channel_index = 0 if n_pred_ch == 1 else 1
    fcd_predictions = predictions[0, fcd_channel_index, ...]  # Extract FCD channel
    binary_mask = (fcd_predictions > threshold).float()
    binary_mask = binary_mask.cpu().numpy()
    output_mask, output_label = post_process_segment(binary_mask, min_region_size)
    #Convert back to tensor and restore original shape
    processed_fcd_mask = torch.tensor(output_mask, dtype=torch.float32, device=predictions.device)

    # Reconstruct output with same shape as `predictions`
    processed_output = predictions.clone()  # Keep original structure
    processed_output[0, fcd_channel_index, ...] = processed_fcd_mask  # Replace only FCD channel

    return processed_output

def evaluate(data_dir, save_dir, checkpoint_path, params, preprocess, postprocess=True):
    model, params = get_model(params)
    #save_dir = os.path.join(save_dir, params["model_desc_str"])
    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print('pretrained model ' + checkpoint_path + ' loaded')
    else:
        print('no pretrained model found')

    model.to(device=device)
    model.eval()

    preprocessed_data_dir = data_dir
    if preprocess:
        preprocessed_data_dir = os.path.join(save_dir, 'preprocessed')
        preprocess_dataset_fsl(data_dir, preprocessed_data_dir, delete_intermediate_files=True)


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

    metrics = dict()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="test", dynamic_ncols=True):
            # Get the affine transformation and image name
            original_affine = batch["image_meta_dict"]["affine"][0].numpy()
            pixdim = batch['image_meta_dict']['pixdim'][0].cpu().numpy()
            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-2]
            img_name = img_name.split('.')[0]

            val_inputs = batch["image"].to(device)

            # Perform inference
            val_outputs = inference(val_inputs, model, params)
            batch['pred'] = val_outputs

            # Load the label images (using the label filenames in batch['label'] if available)
            if "label" in batch:  # Only process label if it exists
                label_pipeline = Compose([
                    LoadImaged(keys=["label"]),
                    #Spacingd(keys=["label"], pixdim=pixdim[1:4], mode="nearest"),
                ])

                labels = []
                for label_file in batch["label"]:
                    sample = {"label": label_file}
                    label = label_pipeline(sample)["label"]
                    labels.append(label)
                

                labels = torch.stack(labels).to(device)
                
            else:
                labels = None

            # Post-processing for the predictions
            output_dir = os.path.join(save_dir, img_name)
            os.makedirs(output_dir, exist_ok=True)

            batch = [post_transform(i) for i in decollate_batch(batch)]
            if postprocess:
                val_outputs = batch[0]['pred'].unsqueeze(0)
                val_outputs = post_process(val_outputs, min_region_size=params['min_region_size'], threshold=0.5)
                batch[0]['pred'] = val_outputs.squeeze(0)
            # Save the prediction result
            SaveImaged(keys="pred", output_dir=output_dir, output_postfix="seg", resample=False, separate_folder=False, print_log=False)(batch[0])

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

                # Reset metrics
                dice_metric.reset()
                meaniou_metric.reset()

    # Print final metrics
    print("Subject, Dice, IOU")
    for img_name in metrics.keys():
        print(f"{img_name}, {metrics[img_name]['dice']:.4f}, {metrics[img_name]['iou']:.4f}")
    avg_metrics = {
        "dice": np.mean([m['dice'] for m in metrics.values()]),
        "iou": np.mean([m['iou'] for m in metrics.values()])
    }
    print(f"Average Dice: {avg_metrics['dice']:.4f}, Average IOU: {avg_metrics['iou']:.4f}")

def test(data_dir, save_dir, checkpoint_path, preprocess, params):
    model, params = get_model(params)
    #save_dir = os.path.join(save_dir, params["model_desc_str"])
    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print('pretrained model ' + checkpoint_path + ' loaded')
    else:
        print('no pretrained model found')

    model.to(device=device)
    model.eval()

    preprocessed_data_dir = data_dir
    if preprocess:
        preprocessed_data_dir = os.path.join(save_dir, 'preprocessed')
        preprocess_dataset_fsl(data_dir, preprocessed_data_dir, delete_intermediate_files=True)

    test_transform, post_transform = get_test_transforms(params)
    test_dict = get_data(preprocessed_data_dir, params)
    test_ds = CacheDataset(data=test_dict, transform=test_transform, cache_num=4, cache_rate=1,
                           num_workers=params['num_workers'])
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=params['num_workers'], pin_memory=True)


    with torch.no_grad():
        idx = 0
        for batch in tqdm(test_loader, desc="test", dynamic_ncols=True):
            idx += 1

            original_affine = batch["image_meta_dict"]["affine"][0].numpy()
            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-2]
            img_name = img_name.split('.')[0]

            val_inputs = batch["image"].to(device)
            val_outputs = inference(val_inputs, model, params)
            #val_outputs = sliding_window_inference(val_inputs, params['patch_size'], 2, model, overlap=0.25)
            batch['pred'] = val_outputs

            output_dir = os.path.join(save_dir, img_name)
            os.makedirs(output_dir, exist_ok=True)

            batch = [post_transform(i) for i in decollate_batch(batch)]
            #val_preds = val_inputs['pred'].cpu().numpy().astype(np.uint8)
            output_name = os.path.join(output_dir, 'seg.nii.gz')
            SaveImaged(keys="pred", output_dir=output_dir, output_postfix="seg", resample=False, separate_folder=False)(
                batch[0])


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_dir', '-d', type=str, required=True, help='data directory')
    argparser.add_argument('--save_dir', '-s', type=str, required=True, help='save directory')
    argparser.add_argument('--checkpoint_path', type=str, required=True, help='model checkpoint path')
    argparser.add_argument('--preprocess', action='store_true', help='whether to preprocess the dataset')
    argparser.add_argument('--postprocess', action='store_true', help='whether to postprocess the dataset')
    args = argparser.parse_args()
    params = get_default_params()
    params['min_region_size'] = -1
    evaluate(args.data_dir, args.save_dir, args.checkpoint_path, params, preprocess=args.preprocess, postprocess=args.postprocess)

    #data_dir = '/mnt/d/dimes/dataset/ds004199/'
    #preprocessed_data_dir = '/mnt/d/dimes/dataset/exp'
    #save_dir = '/mnt/d/dimes/dataset/ds004199_out'
    ##inspect_nifti_file_monai('/mnt/d/dimes/dataset/ds004199_fsl/sub-00001/t1_reg.nii.gz')
    #evaluate(preprocessed_data_dir, save_dir, params)

# run: python seg_fcd_test.py --data_dir /mnt/d/dimes/dataset/ds004199_fsl/test/ --save_dir /mnt/d/dimes/dataset/res/ --checkpoint_path /mnt/d/dimes/dataset/model/MS_DSA_NET/2025-04-14-20-09-05/best_model.pth [--preprocess] [--postprocess]
