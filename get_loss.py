from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any


import torch
import torch.nn as nn
from monai.losses import DiceCELoss, DiceLoss, GeneralizedDiceLoss
from monai.utils import DiceCEReduction, LossReduction, Weight
from utils2 import WarmupCosineSchedule, LinearWarmupCosineAnnealingLR

#generalized_dice_loss and ce loss
class gDiceCELoss(nn.Module):
    def __init__(
        self, 
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        w_type: Weight | str = Weight.SQUARE,
        reduction: LossReduction | str = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        ce_weight: torch.Tensor | None = None, #``ce_weight`` and ``lambda_ce`` are only used for cross entropy loss.
        lambda_ce: float = 1.0,
        batch: bool = False,
    ):
        super().__init__()
        self.gDiceLoss = GeneralizedDiceLoss(
            include_background=include_background,
            to_onehot_y= to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax, 
            other_act=other_act,
            w_type=w_type,
            reduction=reduction,
            batch=batch,
        )
        self.ceLoss = nn.CrossEntropyLoss(weight=ce_weight, reduction=reduction)
        self.lambda_ce = lambda_ce
    
    def ce(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute CrossEntropy loss for the input and target.
        Will remove the channel dim according to PyTorch CrossEntropyLoss:
        https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?#torch.nn.CrossEntropyLoss.

        """
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch != n_target_ch and n_target_ch == 1:
            target = torch.squeeze(target, dim=1)
            target = target.long()
        elif self.old_pt_ver:
            warnings.warn(
                f"Multichannel targets are not supported in this older Pytorch version {torch.__version__}. "
                "Using argmax (as a workaround) to convert target to a single channel."
            )
            target = torch.argmax(target, dim=1)
        elif not torch.is_floating_point(target):
            target = target.to(dtype=input.dtype)

        return self.ceLoss(input, target)  # type: ignore[no-any-return]

    def forward(self, input, target):
        loss_gDice = self.gDiceLoss(input, target)
        loss_CE = self.ce(input, target)
        return loss_gDice + self.lambda_ce*loss_CE

def get_loss(params):
    if params['loss_type'] == 'Dice':
        loss_func = DiceLoss(to_onehot_y=True, include_background=params['include_background'], softmax=True)
    elif params['loss_type'] == 'gDiceCE':
        loss_func = gDiceCELoss(
            include_background=params['include_background'],
            to_onehot_y=True,
            softmax=True,
            w_type=Weight.SQUARE,
        )
    else:
        loss_func = DiceCELoss(to_onehot_y=True, include_background=params['include_background'], softmax=True)
    
    return loss_func

def get_lrschedule(optimizer, params):
    if params['lrschedule'] == 'warmup_cosine':
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=params['warmup_epochs'], max_epochs=params['epochs']
        )
       
    elif params['lrschedule'] == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=5, mode='max', min_lr=1e-6, threshold=0.0001)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['epochs'])
    
    return scheduler

