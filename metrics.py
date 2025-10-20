from monai.metrics import (
    DiceMetric,
    HausdorffDistanceMetric,
    ConfusionMatrixMetric,
    ROCAUCMetric
)
import numpy as np
import torch
import scipy
import scipy.ndimage

import cc3d
from collections import namedtuple


from utils import evaluate_fp


from brats import compute_surface_distances, compute_robust_hausdorff
from brats import dice, get_GTseg_combinedByDilation





def calculate_subject_level_metrics(predictions, labels):
    """
    Calculate subject-level metrics (sSens and nFPC) using torch tensors.
    
    Args:
        predictions: List of prediction maps for each subject (torch tensors)
        labels: List of ground truth maps for each subject (torch tensors)
        
    Returns:
        dict: Dictionary containing sSens and nFPC metrics
    """
    TPs = 0
    FNs = 0
    total_FPC = 0
    
    
    for pred, label in zip(predictions, labels):
        # Convert to binary
        pred_binary = (pred > 0).float()
        label_binary = (label > 0).float()
        
        # Check if there's any lesion in ground truth
        if torch.sum(label_binary) > 0:
            # Calculate overlap using tensor operations
            intersection = torch.logical_and(pred_binary > 0, label_binary > 0).sum()
            
            if intersection > 0:  # If there's any overlap
                TPs += 1
            else:
                FNs += 1
        
        # Calculate false positive clusters
        if torch.sum(pred_binary) > 0:
            # For connected components analysis, we need to use numpy
            pred_np = pred_binary.cpu().numpy()
            label_np = label_binary.cpu().numpy()
            labeled_pred, _ = scipy.ndimage.label(pred_np)
            total_FPC += evaluate_fp(labeled_pred, label_np)
    
    # Calculate metrics
    sSens = TPs / (TPs + FNs) if (TPs + FNs) > 0 else 0
    nFPC = total_FPC / len(predictions)
    
    return {
        'sSens': sSens,
        'nFPC': nFPC
    }

def _compute_metrics(y_pred, y_true, compute_roc_auc=False, compute_hd95=False):
    """
    Compute metrics for a single subject or a batch.
    Assumes y_pred and y_true are tensors of shape [B, C, ...]
    """
    pred_bin = (y_pred > 0.5).float()
    true_bin = (y_true > 0.5).float()

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    cm_metric = ConfusionMatrixMetric(
        include_background=False,
        metric_name=["precision", "sensitivity", "specificity", "f1 score"],
        reduction="mean"
    )


    dice_metric(y_pred=pred_bin, y=true_bin)
    cm_metric(y_pred=pred_bin, y=true_bin)

    dice_value = dice_metric.aggregate().item()
    cm_values = cm_metric.aggregate()
    precision = cm_values[0].item()
    sensitivity = cm_values[1].item()
    specificity = cm_values[2].item()
    f1 = cm_values[3].item()


    metrics = {
        'Prec': precision,
        'Sens': sensitivity,
        #'Spec': specificity,
        'F1': f1,
        'DC': dice_value,
    }

    if compute_roc_auc:
        roc_auc_metric = ROCAUCMetric()
        roc_auc_metric(y_pred=pred_bin.flatten(), y=true_bin.flatten())
        roc_auc = roc_auc_metric.aggregate().item()
        metrics['ROC_AUC'] = roc_auc

    if compute_hd95:
        hd95_metric = HausdorffDistanceMetric(
            include_background=False,
            percentile=95,
            reduction="mean",
        )
        hd95_metric(y_pred=pred_bin, y=true_bin)
        hd95_value = hd95_metric.aggregate().item()
        metrics['HD95'] = hd95_value


    return metrics

def calculate_voxel_level_metrics(predictions, labels, compute_roc_auc=False, compute_hd95=False, average_across_subjects=False):
    """
    Calculate voxel-level metrics using torch tensors and MONAI APIs.
    
    Args:
        predictions: List of prediction maps for each subject (torch tensors)
        labels: List of ground truth maps for each subject (torch tensors)
        compute_roc_auc (bool): Whether to compute ROC AUC
        average_across_subjects (bool): Whether to compute per-subject metrics and average them
    
    Returns:
        dict: Dictionary containing averaged metrics
    """
    if average_across_subjects:
        all_metrics = []
        for pred, label in zip(predictions, labels):
            pred = pred.unsqueeze(0).unsqueeze(0) if len(pred.shape) == 3 else pred
            label = label.unsqueeze(0).unsqueeze(0) if len(label.shape) == 3 else label
            subj_metrics = _compute_metrics(pred, label, compute_roc_auc)
            all_metrics.append(subj_metrics)

        # Average metrics across subjects
        final_metrics = {}
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics if m[key] is not None]
            if values:  # In case ROC_AUC is disabled and None
                final_metrics[key] = sum(values) / len(values)
        return final_metrics
    else:
        # Global evaluation
        all_preds = torch.cat([pred.unsqueeze(0).unsqueeze(0) if len(pred.shape) == 3 else pred for pred in predictions])
        all_labels = torch.cat([label.unsqueeze(0).unsqueeze(0) if len(label.shape) == 3 else label for label in labels])
        return _compute_metrics(all_preds, all_labels, compute_roc_auc, compute_hd95)

def calculate_lesion_wise_metrics(
    predictions, 
    labels, 
    dilation_factor=3, 
    voxel_spacing=(1.0, 1.0, 1.0), 
    lesion_volume_thresh=0,
    penalty_distance=374
):
    """
    Compute BraTS-style lesion-wise metrics for binary segmentation.

    Returns both macro (average per subject) and micro (pooled across dataset) metrics.
    Reports both matched-only and penalized (with FP penalties) Dice/HD95.
    
    Parameters
    ----------
    predictions : list of tensors
    labels : list of tensors
    dilation_factor : int, optional
        Number of dilation iterations for GT matching.
    voxel_spacing : tuple of float, optional
        Physical voxel spacing (x,y,z).
    lesion_volume_thresh : int, optional
        Minimum lesion volume to include in evaluation.
    penalty_distance : float, optional
        Distance penalty assigned to false positive lesions in HD95.
    """
    
    LesionMetric = namedtuple("LesionMetric", ["pred_ids", "gt_id", "gt_volume", "dice_score", "hd95"])
    
    results = []
    micro_tp, micro_fp, micro_fn = 0, 0, 0
    micro_dice_matched, micro_dice_penalized = [], []
    micro_hd95_matched, micro_hd95_penalized = [], []
    sx, sy, sz = voxel_spacing

    # --- subject-level counters ---
    subject_tp_count = 0
    subject_fn_count = 0

    for pred_t, gt_t in zip(predictions, labels):
        # --- Binarize ---
        pred = (pred_t.detach().cpu().numpy() > 0.5).astype(np.uint8)
        gt   = (gt_t.detach().cpu().numpy() > 0.5).astype(np.uint8)

        # Connected components
        dilation_struct = scipy.ndimage.generate_binary_structure(3, 2)
        gt_cc = cc3d.connected_components(gt, connectivity=26)
        pred_cc = cc3d.connected_components(pred, connectivity=26)

        # Dilated GT for lesion matching
        gt_dil = scipy.ndimage.binary_dilation(gt, structure=dilation_struct, iterations=dilation_factor)
        gt_dil_cc = cc3d.connected_components(gt_dil, connectivity=26)
        gt_combined = get_GTseg_combinedByDilation(gt_dil_cc, gt_cc)

        tp_ids, fn_ids, fp_ids = [], [], []
        lesion_metrics = []

        # --- Loop over GT lesions ---
        for gtcomp in range(1, np.max(gt_combined) + 1):
            gt_mask = (gt_combined == gtcomp).astype(np.uint8)
            gt_mask_dil = scipy.ndimage.binary_dilation(gt_mask, structure=dilation_struct, iterations=dilation_factor)
            gt_volume = np.sum(gt_mask) * sx * sy * sz

            # Overlapping predicted lesions
            overlapping_pred = np.unique(pred_cc * gt_mask_dil)
            overlapping_pred = overlapping_pred[overlapping_pred != 0]

            if len(overlapping_pred) > 0:
                tp_ids.extend(overlapping_pred.tolist())
                # Restrict prediction mask to overlapping lesions only
                pred_iso = np.copy(pred_cc)
                pred_iso[np.isin(pred_iso, overlapping_pred, invert=True)] = 0
                pred_iso[np.isin(pred_iso, overlapping_pred)] = 1

                dice_score = dice(pred_iso, gt_mask)
                sd = compute_surface_distances(gt_mask, pred_iso, (sx, sy, sz))
                hd95 = compute_robust_hausdorff(sd, 95)
            else:
                fn_ids.append(gtcomp)
                dice_score = np.nan
                hd95 = np.nan


            lesion_metrics.append(
                LesionMetric(
                    pred_ids=overlapping_pred.tolist(),
                    gt_id=gtcomp,
                    gt_volume=gt_volume,
                    dice_score=dice_score,
                    hd95=hd95,
                )
            )

        # False positives = predicted lesions that never matched
        fp_ids = np.unique(pred_cc[np.isin(pred_cc, tp_ids + [0], invert=True)])

        # Filter by lesion volume threshold
        if lesion_volume_thresh > 0:
            lesion_metrics = [m for m in lesion_metrics if m.gt_volume > lesion_volume_thresh]

        # --- Dice & HD95 (macro per subject) ---
        matched_lesions = [m for m in lesion_metrics if not np.isnan(m.dice_score) and not np.isnan(m.hd95)]
        if len(matched_lesions) > 0:
            dice_matched = np.mean([m.dice_score for m in matched_lesions])
            hd95_matched = np.mean([m.hd95 for m in matched_lesions])
        else:
            dice_matched, hd95_matched = np.nan, np.nan

        # Count FN lesions that were above volume threshold
        fn_count = np.sum([1 for m in lesion_metrics if np.isnan(m.dice_score)])

        # Penalized Dice & HD95 (BraTS-style, include FPs and FNs)
        if len(lesion_metrics) > 0 or len(fp_ids) > 0 or fn_count > 0:
            # Dice: treat NaN as 0
            dice_penalized = (
                np.sum([0.0 if np.isnan(m.dice_score) else m.dice_score for m in lesion_metrics])
                / (len(lesion_metrics) + len(fp_ids) + fn_count)
            )
            # HD95: treat NaN as penalty_distance
            hd95_penalized = (
                np.sum([penalty_distance if np.isnan(m.hd95) else m.hd95 for m in lesion_metrics])
                + len(fp_ids) * penalty_distance
            ) / (len(lesion_metrics) + len(fp_ids) + fn_count)
        else:
            dice_penalized, hd95_penalized = 1.0, 0.0

        # Subject-level sensitivity/precision/F1
        tp, fp, fn = len(tp_ids), len(fp_ids), len(fn_ids)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        f1   = 2 * sens * prec / (sens + prec) if (sens + prec) > 0 else 0.0

        # --- Store subject-level (macro) results ---
        results.append({
            'Lesion_FP': fp,
            'Lesion_FN': fn,
            #'Lesion_Prec_macro': prec,
            #'Lesion_Sens_macro': sens,
            #'Lesion_F1_macro': f1,
            #'Lesion_Dice_macro': dice_matched,
            #'Lesion_Dice_penalized_macro': dice_penalized,
            #'Lesion_HD95_macro': hd95_matched,
            #'Lesion_HD95_penalized_macro': hd95_penalized,
        })

        # --- Update micro statistics ---
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        micro_dice_matched.extend([m.dice_score for m in lesion_metrics])
        micro_hd95_matched.extend([m.hd95 for m in lesion_metrics])
        # Penalized includes FP penalties
        micro_dice_penalized.extend([m.dice_score for m in lesion_metrics] + [0] * len(fp_ids))
        micro_hd95_penalized.extend([m.hd95 for m in lesion_metrics] + [penalty_distance] * len(fp_ids))

        # --- Subject-level sensitivity (sSens) ---
        if np.sum(gt) > 0:  # subject has at least one GT lesion
            if tp > 0:  # detected at least one
                subject_tp_count += 1
            else:
                subject_fn_count += 1

    # --- Macro (average per subject) ---
    macro_results = {k: np.nanmean([r[k] for r in results]) for k in results[0]}

    # --- Micro (pooled across dataset) ---
    lesion_sens_micro = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 1.0
    lesion_prec_micro = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 1.0
    lesion_f1_micro = (
        2 * lesion_sens_micro * lesion_prec_micro / (lesion_sens_micro + lesion_prec_micro)
        if (lesion_sens_micro + lesion_prec_micro) > 0 else 0.0
    )

    lesion_dice_matched_micro = np.nanmean(micro_dice_matched) if micro_dice_matched else np.nan
    lesion_hd95_matched_micro = np.nanmean(micro_hd95_matched) if micro_hd95_matched else np.nan

    micro_dice_penalized_corrected = [0.0 if np.isnan(d) else d for d in micro_dice_penalized]
    lesion_dice_penalized_micro = np.mean(micro_dice_penalized) if micro_dice_penalized else 1.0

    micro_hd95_penalized_corrected = [penalty_distance if np.isnan(d) else d for d in micro_hd95_penalized]
    lesion_hd95_penalized_micro = np.mean(micro_hd95_penalized) if micro_hd95_penalized else 0.0

    # --- Subject-level sensitivity ---
    sSens = subject_tp_count / (subject_tp_count + subject_fn_count) if (subject_tp_count + subject_fn_count) > 0 else np.nan

    # --- Final combined results ---
    final_results = {
        **macro_results,
        "Lesion_Prec": lesion_prec_micro,
        "Lesion_Sens": lesion_sens_micro,
        "Lesion_F1": lesion_f1_micro,
        "Lesion_Dice": lesion_dice_matched_micro,
        #"Lesion_Dice_penalized": lesion_dice_penalized_micro,
        "Lesion_HD95": lesion_hd95_matched_micro,
        #"Lesion_HD95_penalized": lesion_hd95_penalized_micro,
        "sSens": sSens,  # subject-level sensitivity
    }
    return final_results
