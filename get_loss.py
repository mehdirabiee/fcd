from __future__ import annotations

from pyparsing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss, DiceLoss, GeneralizedDiceLoss, DiceFocalLoss, GeneralizedDiceFocalLoss


class CombinedLoss(nn.Module):
    """
    Custom loss function that combines multiple loss functions.
    """
    def __init__(self, params:dict, device) -> None:
        super().__init__()
        self.params = params
        self.device = device
        self.main_loss_func = get_loss_function_from_params(params, device=device)
        self.tv_loss_weight = params.get('tv_loss_weight', 0.0)
        self.boundaryloss_weight = params.get('boundaryloss_weight', 0.0)
        self.caloss_weight = params.get('caloss_weight', 0.0)


    def forward(self, pred: torch.Tensor, target: torch.Tensor, thickness_map: Optional[torch.Tensor]=None) -> torch.Tensor:
        total_loss = 0.0
        main_loss = self.main_loss_func(pred, target) if self.main_loss_func is not None else 0.0
        total_loss += main_loss
        if self.tv_loss_weight > 0:
            tv_loss_norm = 2 if self.params['tv_loss_norm'] == 'l2' else 1
            tv_loss = compute_total_variation_loss(pred, target, norm=tv_loss_norm, sigmoid=self.params['sigmoid'], 
                                                   softmax=self.params['softmax'], exclude_borders=self.params['tvloss_exclude_borders'])
            total_loss += self.tv_loss_weight * tv_loss
        if self.boundaryloss_weight > 0:
            boundary_loss = compute_boundary_loss(pred, target)
            total_loss += self.boundaryloss_weight * boundary_loss
        if self.caloss_weight > 0 and thickness_map is not None:
            caloss = compute_cortical_boundary_loss(pred, thickness_map)
            total_loss += self.caloss_weight * caloss
        return total_loss


def get_loss_function_from_params(params, device):
    """Initialize loss function based on params"""
    loss_type = params.get('loss', 'DiceLoss')
    
    common_kwargs = {
        'include_background': False,
        'smooth_nr': 1e-5,
        'smooth_dr': 1e-5,
        'to_onehot_y': params['chans_out'] > 1,
        'sigmoid': params['sigmoid'],
        'softmax': params['softmax'],
        'batch': True,
    }

    if loss_type == 'DiceLoss':
        loss_function = DiceLoss(**common_kwargs, squared_pred=params['square_pred'], jaccard=params['jaccard'])
    elif loss_type == 'DiceCELoss':
        loss_function = DiceCELoss(
            **common_kwargs,
            squared_pred=params['square_pred'],
            jaccard=params['jaccard'],
            lambda_dice=params['lambda_dice'],
            lambda_ce=params['lambda_ce'],
            weight=torch.tensor([
                params['ce_background_weight'],
                params['ce_fcd_weight']
            ], dtype=torch.float32, device=device)
        )
    elif loss_type == 'DiceFocalLoss':
        loss_function = DiceFocalLoss(
            **common_kwargs,
            squared_pred=params['square_pred'],
            jaccard=params['jaccard'],
            lambda_dice=params['lambda_dice'],
            lambda_focal=params['lambda_focal'],
            gamma=params['gamma_focal']
        )
    elif loss_type == 'GeneralizedDiceLoss':
        common_kwargs['include_background'] = True
        loss_function = GeneralizedDiceLoss(
            **common_kwargs,
            w_type=params['gdice_wtype']
        )
    elif loss_type == 'GeneralizedDiceFocalLoss':
        common_kwargs['include_background'] = True
        loss_function = GeneralizedDiceFocalLoss(
            **common_kwargs,
            lambda_gdl=params['lambda_dice'],
            lambda_focal=params['lambda_focal'],
            gamma=params['gamma_focal'],
            w_type=params['gdice_wtype']
        )
    else:
        loss_function = None
    
    return loss_function


def dilate_mask(mask, kernel_size=3, iterations=1):
    """
    mask: (B, 1, H, W, D) binary tensor
    kernel_size: size of dilation kernel
    iterations: how many times to apply
    """
    # Create dilation kernel
    device = mask.device
    kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), device=device)
    
    dilated = mask
    for _ in range(iterations):
        dilated = (F.conv3d(dilated.float(), kernel, padding=kernel_size//2) > 0).float()
    return dilated


def compute_total_variation_loss(pred, gt, norm=2, sigmoid=False, softmax=False, exclude_borders=True):
    """
    Computes the Total Variation (TV) loss to encourage spatial smoothness in the predicted mask.
    Applies TV loss only on the FCD class channel (index 1) in multi-class settings.
    
    Args:
        pred (Tensor): Predicted mask of shape (B, C, D, H, W)

    Returns:
        tv_loss (Tensor): Scalar tensor representing total variation loss.
    """
    n_pred_ch = pred.shape[1]

    # Apply activation if specified
    if sigmoid:
        pred = torch.sigmoid(pred)
    if softmax and n_pred_ch > 1:
        pred = torch.softmax(pred, dim=1)        

    # If multi-channel, use only the FCD class (assumed at index 1)
    if n_pred_ch > 1:
        fcd_channel_index = 1
        pred = pred[:, fcd_channel_index:fcd_channel_index+1, ...]

    if exclude_borders:
        dilated = dilate_mask(gt, kernel_size=3, iterations=2)
        eroded  = 1 - dilate_mask(1 - gt, kernel_size=3, iterations=2)

        border = dilated - eroded   # voxels near lesion boundary
        border = (border > 0).float()

        ignore_mask = border   # 1 at border, 0 elsewhere
        tv_mask = 1 - ignore_mask

        pred = pred * tv_mask    

    # Compute variation along each spatial axis
    if norm == 1:
        tv_z = torch.abs(pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]).mean()
        tv_y = torch.abs(pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]).mean()
        tv_x = torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]).mean()
    elif norm == 2:
        epsilon = 1e-10
        tv_z = torch.sqrt(torch.mean((pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]) ** 2) + epsilon)
        tv_y = torch.sqrt(torch.mean((pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]) ** 2) + epsilon)
        tv_x = torch.sqrt(torch.mean((pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]) ** 2) + epsilon)
    else:
        raise ValueError("Unsupported norm type. Use 1 or 2.")
    
    return tv_z + tv_y + tv_x

def compute_boundary_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Compute gradients as a proxy for boundaries
    pred_grad = torch.gradient(pred, dim=(2,3,4))
    target_grad = torch.gradient(target, dim=(2,3,4))
    
    # Compute L1 loss between gradients
    boundary_loss = sum(
        torch.mean(torch.abs(pg - tg))
        for pg, tg in zip(pred_grad, target_grad)
    ) / 3.0
    
    return boundary_loss

def compute_cortical_boundary_loss(pred: torch.Tensor, thickness_map: torch.Tensor) -> torch.Tensor:
    # Encourage predictions to align with cortical boundaries
    thickness_grad = torch.gradient(thickness_map, dim=(2,3,4))
    pred_grad = torch.gradient(pred, dim=(2,3,4))
    
    consistency_loss = sum(
        torch.mean(torch.abs(pg * tg))
        for pg, tg in zip(pred_grad, thickness_grad)
    ) / 3.0
    
    return consistency_loss