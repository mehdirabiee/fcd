import os
import torch
from thop import profile, clever_format
from networks2 import MS_DSA_NET, CAMST
from monai.networks.nets import UNETR, SwinUNETR, UNEST, DynUNet, SegResNet, VNet
from monai.networks.layers.factories import Norm,Act

def get_model(params):

    if params['patch_size'][0] == params['patch_size'][1] and params['patch_size'][0] == params['patch_size'][2]:
        str_ps = 'ps{}'.format(params['patch_size'][0]) 
    else:
        str_ps = 'ps{}x{}x{}'.format(params['patch_size'][0], params['patch_size'][1], params['patch_size'][2])


    if 'MS_DSA_NET' == params['model_type']:
        # MS-DSA-NET: Multi-Scale Dual Stream Attention Network
        # Reference: "Zhang et al., Focal Cortical Dysplasia Lesion Segmentation 
        # Using Multiscale Transformer" (2023)
        # Key features:
        # - Specifically designed for FCD lesion segmentation in MRI
        # - Multi-scale feature extraction with transformer blocks
        # - Dual stream attention mechanism for better feature representation
        # - Combines spatial and channel attention for capturing lesion characteristics
        # - Efficient processing of 3D MRI volumes with position embedding
        model = MS_DSA_NET(
                spatial_dims=3,                # 3D input for volumetric segmentation
                in_channels=params['chans_in'],
                out_channels=params['chans_out'],
                img_size=params['patch_size'],
                feature_size=params['feature_size'],
                pos_embed=True,                # Enable position embedding for spatial context
                project_size=params['project_size'],
                sa_type=params['sa_type'],     # Type of self-attention mechanism
                norm_name='instance',          # Instance normalization for stable training
                act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
                res_block=True,                # Enable residual connections
                bias=False,                    # Disable bias for better generalization
                dropout_rate=0.1,              # Dropout for regularization
            ) 
        model_desc_str = '{}_{}_fs{}'.format(model.name, str_ps, params['feature_size'])

    elif 'UNETR' == params['model_type']:
        # UNETR: UNet-like architecture with Transformer encoder
        # Reference: "Hatamizadeh et al., UNETR: Transformers for 3D Medical Image Segmentation
        # https://arxiv.org/abs/2103.10504
        # Key features: Combines CNN decoder with pure transformer encoder for 3D medical imaging
        model = UNETR(
            in_channels=params['chans_in'],
            out_channels=params['chans_out'],
            img_size=params['patch_size'],
            feature_size=params['feature_size'],
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed='conv',
            norm_name='instance',
            res_block=True,
            dropout_rate=0.1,
        )
        model_desc_str = 'UNETR_{}_fs{}'.format(str_ps, params['feature_size'])

    elif 'SwinUNETR' == params['model_type']:
        # SwinUNETR: Swin Transformers for Medical Image Segmentation
        # Reference: "Hatamizadeh et al., Swin UNETR: Swin Transformers for Semantic
        # Segmentation of Brain Tumors in MRI Images" https://arxiv.org/abs/2201.01266
        # Key features: 
        # - Hierarchical Swin Transformer blocks for encoding MRI features
        # - Shifted windowing scheme for efficient self-attention computation
        # - Skip connections between encoder and decoder for fine-grained details
        # - Patch merging and patch expanding layers for resolution adjustment
        # - Specifically designed for 3D medical image segmentation tasks
        model = SwinUNETR(
            img_size=params['patch_size'],
            in_channels=params['chans_in'],
            out_channels=params['chans_out'],
            feature_size=params['feature_size'],
            use_checkpoint=True,  # Enable gradient checkpointing to save memory
            spatial_dims=3,       # 3D input for volumetric segmentation
        )
        model_desc_str = 'SwinUNETR_{}_fs{}'.format(str_ps, params['feature_size'])

    elif 'UNEST' == params['model_type']:
        # UNEST: UNet-like architecture with Nested Hierarchical Transformer
        # Reference: "Zhang et al., Nested Hierarchical Transformer: Towards Accurate,
        # Data-Efficient and Interpretable Visual Understanding"
        # https://arxiv.org/abs/2105.12723
        # Key features: Hierarchical transformer with nested attention for better feature extraction
        model = UNEST(
            in_channels=params['chans_in'],
            out_channels=params['chans_out'],
            img_size=params['patch_size'],
            feature_size=params['feature_size'],
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            norm_name='instance',
            res_block=True,
            dropout_rate=0.1,
        )
        model_desc_str = 'UNEST_{}_fs{}'.format(str_ps, params['feature_size'])

    elif 'DynUNet' == params['model_type']:
        # DynUNet: Dynamic UNet for variable input sizes
        # Reference: "nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation"
        # https://arxiv.org/abs/1809.10486
        # Key features: Automatically adapts architecture based on input size,
        # includes deep supervision for better gradient flow
        sizes = params['patch_size']
        strides = [2, 2, 2, 2, 2]  # 5 levels of downsampling
        kernel_sizes = []
        for i in range(5):
            kernel_size = [min(s // (2**i), 3) for s in sizes]
            kernel_sizes.append(kernel_size)
            
        model = DynUNet(
            spatial_dims=3,
            in_channels=params['chans_in'],
            out_channels=params['chans_out'],
            kernel_size=kernel_sizes,
            strides=strides,
            upsample_kernel_size=strides,
            norm_name='instance',
            deep_supervision=True,
            deep_supr_num=2,
        )
        model_desc_str = 'DynUNet_{}_fs{}'.format(str_ps, params['feature_size'])

    elif 'SegResNet' == params['model_type']:
        # SegResNet: 3D ResNet-based architecture for volumetric segmentation
        # Reference: "Myronenko A., 3D MRI Brain Tumor Segmentation Using Autoencoder Regularization"
        # https://arxiv.org/abs/1810.11654
        # Key features: Deep residual connections and variational autoencoder regularization
        model = SegResNet(
            spatial_dims=3,
            in_channels=params['chans_in'],
            out_channels=params['chans_out'],
            init_filters=params['feature_size'],
            dropout_prob=0.1,
            norm='INSTANCE',
            use_conv_final=True,
        )
        model_desc_str = 'SegResNet_{}_fs{}'.format(str_ps, params['feature_size'])

    elif 'VNet' == params['model_type']:
        # VNet: Volumetric Medical Image Segmentation
        # Reference: "Milletari et al., V-Net: Fully Convolutional Neural Networks for
        # Volumetric Medical Image Segmentation" https://arxiv.org/abs/1606.04797
        # Key features: Volumetric convolutions, residual connections, and dice loss optimization
        model = VNet(
            spatial_dims=3,
            in_channels=params['chans_in'],
            out_channels=params['chans_out'],
            act=('prelu', {'init': 0.2}),
            dropout_prob=0.1,
            dropout_dim=3,
        )
        model_desc_str = 'VNet_{}_fs{}'.format(str_ps, params['feature_size'])

    elif 'CAMST' == params['model_type']:
        # CAMST: Cortical-Aware Multi-Scale Transformer
        # A novel architecture specifically designed for FCD detection that combines:
        # - Cortical thickness-aware attention mechanisms
        # - Multi-scale feature processing
        # - Anatomically-guided feature extraction
        # Key features:
        # - Integration of cortical thickness information
        # - Boundary-sensitive attention modules
        # - Hierarchical feature processing with anatomical priors
        # - Efficient handling of 3D MRI volumes
        model = CAMST(
            img_size=params['patch_size'],
            in_channels=params['chans_in'],
            out_channels=params['chans_out'],
            feature_size=params['feature_size'],
            num_heads=8,                # Multi-head attention for better feature relationships
            norm_name='instance',       # Instance normalization for stable training
            spatial_dims=3,             # 3D input for volumetric segmentation
            dropout_rate=0.1,           # Dropout for regularization
            attention_levels=3,         # Number of levels for multi-scale processing
        )
        model_desc_str = 'CAMST_{}_fs{}'.format(str_ps, params['feature_size'])

    params['model_desc_str'] = model_desc_str
    return model, params