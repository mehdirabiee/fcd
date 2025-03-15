from __future__ import annotations

from collections.abc import Sequence
from typing import Union, Tuple

import torch
import torch.nn as nn
import numpy as np

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.networks.layers.utils import get_norm_layer, get_act_layer
from monai.utils import ensure_tuple_rep

from .conv_blocks import UnetrBasicBlock, UnetrUpBlock, TransformerBlock

class CorticalAwareAttention(nn.Module):
    """
    Cortical-Aware Attention module that integrates anatomical priors with image features.
    Combines cortical thickness information and boundary sensitivity.
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        feature_size: int,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        norm_name: str = "instance",
    ) -> None:
        super().__init__()
        
        self.thickness_encoder = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=1,  # Cortical thickness map
                out_channels=feature_size,
                strides=1,
                kernel_size=3,
                norm=norm_name,
                act=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            ),
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=feature_size,
                out_channels=feature_size,
                strides=1,
                kernel_size=1,
                norm=norm_name,
                act=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            ),
        )

        self.boundary_attention = nn.MultiheadAttention(
            embed_dim=feature_size,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(feature_size * 2, feature_size),
            nn.LayerNorm(feature_size),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable scaling factor

    def forward(self, x: torch.Tensor, thickness_map: torch.Tensor) -> torch.Tensor:
        # Encode thickness information
        thick_features = self.thickness_encoder(thickness_map)
        
        B, C, H, W, D = x.shape
        
        # Reshape for attention
        x_flat = x.reshape(B, C, -1).permute(0, 2, 1)  # B, HWD, C
        thick_flat = thick_features.reshape(B, C, -1).permute(0, 2, 1)  # B, HWD, C
        
        # Apply boundary-sensitive attention
        attn_out, _ = self.boundary_attention(x_flat, thick_flat, thick_flat)
        
        # Fuse features
        fused = self.fusion(torch.cat([x_flat, attn_out], dim=-1))
        
        # Residual connection with learnable scale
        output = x_flat + self.gamma * fused
        
        # Reshape back
        output = output.permute(0, 2, 1).reshape(B, C, H, W, D)
        
        return output

class CAMST(nn.Module):
    """
    Cortical-Aware Multi-Scale Transformer (CAMST) for FCD Detection.
    Combines hierarchical feature extraction with cortical-specific attention mechanisms.
    """
    def __init__(
        self,
        img_size: Union[Sequence[int], int],
        in_channels: int,
        out_channels: int,
        feature_size: int = 48,
        num_heads: int = 8,
        norm_name: str = "instance",
        spatial_dims: int = 3,
        dropout_rate: float = 0.1,
        attention_levels: int = 3,  # Number of levels to apply cortical attention
    ) -> None:
        super().__init__()
        
        self.name = "CAMST"
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.feature_size = feature_size
        self.attention_levels = attention_levels

        # Initial feature extraction
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        # Encoder stages with increasing feature dimensions
        self.encoders = nn.ModuleList()
        self.down_samplers = nn.ModuleList()
        self.cortical_attentions = nn.ModuleList()
        
        curr_size = feature_size
        
        for i in range(attention_levels):
            # Double the features at each level
            next_size = curr_size * 2
            
            # Down-sampling
            self.down_samplers.append(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=curr_size,
                    out_channels=next_size,
                    strides=2,
                    kernel_size=2,
                    norm=norm_name,
                    act=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
                )
            )
            
            # Feature extraction
            self.encoders.append(
                UnetrBasicBlock(
                    spatial_dims=spatial_dims,
                    in_channels=next_size,
                    out_channels=next_size,
                    kernel_size=3,
                    stride=1,
                    norm_name=norm_name,
                    res_block=True,
                )
            )
            
            # Cortical-aware attention
            self.cortical_attentions.append(
                CorticalAwareAttention(
                    spatial_dims=spatial_dims,
                    in_channels=next_size,
                    feature_size=next_size,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    norm_name=norm_name,
                )
            )
            
            curr_size = next_size

        # Decoder stages
        self.decoders = nn.ModuleList()
        
        for i in range(attention_levels):
            self.decoders.append(
                UnetrUpBlock(
                    spatial_dims=spatial_dims,
                    in_channels=curr_size,
                    out_channels=curr_size // 2,
                    kernel_size=3,
                    upsample_kernel_size=2,
                    norm_name=norm_name,
                    res_block=True,
                )
            )
            curr_size //= 2

        # Final output layer
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor, thickness_maps: torch.Tensor = None) -> torch.Tensor:
        if thickness_maps is None:
            # If no thickness maps provided, create dummy ones
            thickness_maps = torch.zeros_like(x[:, :1])

        # Initial feature extraction
        enc1 = self.encoder1(x)
        
        # Encoder path with cortical attention
        features = [enc1]
        curr_feature = enc1
        
        for i in range(self.attention_levels):
            # Down-sampling
            curr_feature = self.down_samplers[i](curr_feature)
            
            # Feature extraction
            curr_feature = self.encoders[i](curr_feature)
            
            # Apply cortical-aware attention
            # Downsample thickness maps to match current feature size
            curr_thickness = torch.nn.functional.interpolate(
                thickness_maps,
                size=curr_feature.shape[2:],
                mode='trilinear',
                align_corners=False
            )
            curr_feature = self.cortical_attentions[i](curr_feature, curr_thickness)
            
            features.append(curr_feature)

        # Decoder path
        for i in range(self.attention_levels):
            curr_feature = self.decoders[i](
                curr_feature,
                features[-(i+2)]  # Skip connections
            )

        # Final output
        out = self.out(curr_feature)
        
        return out

class CorticalAwareLoss(nn.Module):
    """
    Custom loss function that combines segmentation accuracy with anatomical priors.
    """
    def __init__(
        self,
        alpha: float = 0.5,  # Weight for anatomical consistency
        beta: float = 0.3,   # Weight for boundary preservation
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
        self.dice_loss = DiceLoss(
            include_background=False,
            sigmoid=True,
            smooth_nr=1e-5,
            smooth_dr=1e-5,
        )

    def compute_boundary_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute gradients as a proxy for boundaries
        pred_grad = torch.gradient(pred, dim=(2,3,4))
        target_grad = torch.gradient(target, dim=(2,3,4))
        
        # Compute L1 loss between gradients
        boundary_loss = sum(
            torch.mean(torch.abs(pg - tg))
            for pg, tg in zip(pred_grad, target_grad)
        ) / 3.0
        
        return boundary_loss

    def compute_anatomical_consistency(
        self,
        pred: torch.Tensor,
        thickness_map: torch.Tensor
    ) -> torch.Tensor:
        # Encourage predictions to align with cortical boundaries
        thickness_grad = torch.gradient(thickness_map, dim=(2,3,4))
        pred_grad = torch.gradient(pred, dim=(2,3,4))
        
        consistency_loss = sum(
            torch.mean(torch.abs(pg * tg))
            for pg, tg in zip(pred_grad, thickness_grad)
        ) / 3.0
        
        return consistency_loss

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        thickness_map: torch.Tensor
    ) -> torch.Tensor:
        # Main segmentation loss
        seg_loss = self.dice_loss(pred, target)
        
        # Boundary preservation loss
        boundary_loss = self.compute_boundary_loss(pred, target)
        
        # Anatomical consistency loss
        anatomical_loss = self.compute_anatomical_consistency(pred, thickness_map)
        
        # Combine losses
        total_loss = seg_loss + self.alpha * anatomical_loss + self.beta * boundary_loss
        
        return total_loss 