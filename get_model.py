import torch
from thop import profile, clever_format
from networks import MS_DSA_NET, BaseUNet, UNETR_PP, SegResNet_DSA, SegResNetVAE_DSA, MS_DSA_NET_PS
from monai.networks.nets import UNETR, SwinUNETR, SegResNet, SegResNetVAE, VNet, UNet
from monai.networks.layers.factories import Norm,Act



def get_model(params, return_model=True):
    model = None
    params["model_returns_vaeloss"] = False
    if params['model_type'].lower() == 'ms_dsa_net':
        # MS-DSA-NET: Multi-Scale Dual Stream Attention Network
        # Reference: "Zhang et al., Focal Cortical Dysplasia Lesion Segmentation 
        # Using Multiscale Transformer" (2023)
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
            ) if return_model else None
    
    elif params['model_type'].lower() == 'ms_dsa_net_ps':
        model = MS_DSA_NET_PS(
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
                upsample_mode="pixelshuffle", 
                interpolate_mode= "linear",
            ) if return_model else None
    
    elif params['model_type'].lower() == "baseunet":
        model = BaseUNet(
            spatial_dims= 3,
            in_channels=params['chans_in'],
            out_channels=params['chans_out'],
            feature_size=params['feature_size'],
            norm_name='instance',          # Instance normalization for stable training
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            res_block=True,                # Enable residual connections
            bias=False,                    # Disable bias for better generalization
            depth=6,
        ) if return_model else None

    elif params['model_type'].lower() == 'unet':
        # UNet: Classic UNet architecture for image segmentation
        # Reference: "Ronneberger et al., U-Net: Convolutional Networks for Biomedical Image Segmentation"
        # https://arxiv.org/abs/1505.04597
        # Key features: Encoder-decoder structure with skip connections
        model = UNet(
            spatial_dims=3,
            in_channels=params['chans_in'],
            out_channels=params['chans_out'],
            channels=[16, 32, 64, 128, 256, 512], #[params['feature_size'], params['feature_size]*2, ...] ,
            strides=[2, 2, 2, 2, 2],
            num_res_units=2,
            norm=Norm.INSTANCE,
            act=Act.PRELU,
            dropout=0.1,
        ) if return_model else None

    elif params['model_type'].lower() == 'vnet':
        # VNet: Volumetric Medical Image Segmentation
        # Reference: "Milletari et al., V-Net: Fully Convolutional Neural Networks for
        # Volumetric Medical Image Segmentation" https://arxiv.org/abs/1606.04797
        # Key features: Volumetric convolutions, residual connections, and dice loss optimization
        model = VNet(
            spatial_dims=3,
            in_channels=params['chans_in'],
            out_channels=params['chans_out'],
            act=('prelu', {'init': 0.2}),
            dropout_prob_down = 0.5,
            dropout_prob_up = (0.5, 0.5),
            dropout_dim=3,
        ) if return_model else None
    
    elif params['model_type'].lower() == 'unetr':
        # UNETR: UNet-like architecture with Transformer encoder
        # Reference: "Hatamizadeh et al., UNETR: Transformers for 3D Medical Image Segmentation
        # https://arxiv.org/abs/2103.10504
        model = UNETR(
            in_channels=params['chans_in'],
            out_channels=params['chans_out'],
            img_size=params['patch_size'],
            feature_size=params['feature_size'],
            hidden_size=768, #default=768,
            mlp_dim=1024, #default=3072,
            num_heads=12,
            proj_type='conv',
            norm_name='instance',
            res_block=True,
            dropout_rate=0.1,
        ) if return_model else None

    elif params['model_type'].lower() == 'unetrpp':
        # UNETR++: UNETR with additional transformer blocks
        # Reference: "Shaker et al., UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
        # https://arxiv.org/abs/2201.01266
        model = UNETR_PP(
            in_channels=params['chans_in'],
            out_channels=params['chans_out'],
            feature_size=params['feature_size'],
            hidden_size=256,
            num_heads=4,
            depths=[3, 3, 3, 3],
            dims= [32, 64, 128, 256],
            norm_name='instance',
            do_ds=False,
            dropout_rate=0.1,
            pos_embed='conv',
        ) if return_model else None

    elif params['model_type'].lower() == 'swinunetr':
        # SwinUNETR: Swin Transformers for Medical Image Segmentation
        # Reference: "Hatamizadeh et al., Swin UNETR: Swin Transformers for Semantic
        # Segmentation of Brain Tumors in MRI Images" https://arxiv.org/abs/2201.01266
        model = SwinUNETR(
            img_size=params['patch_size'],
            in_channels=params['chans_in'],
            out_channels=params['chans_out'],
            feature_size=24,
            use_checkpoint=True,  # Enable gradient checkpointing to save memory
            spatial_dims=3,       # 3D input for volumetric segmentation
        ) if return_model else None

    elif params['model_type'].lower() == 'segresnet':
        # SegResNet: 3D ResNet-based architecture for volumetric segmentation
        # Reference: "Myronenko A., 3D MRI Brain Tumor Segmentation Using Autoencoder Regularization"
        # https://arxiv.org/abs/1810.11654
        blocks_down = (1, 2, 2, 4) if not params['segresnet_deeper'] else (1, 2, 2, 4, 4)
        blocks_up = (1, 1, 1) if not params['segresnet_deeper'] else (2, 2, 2, 2)
        model = SegResNet(
            spatial_dims=3,
            in_channels=params['chans_in'],
            out_channels=params['chans_out'],
            init_filters=params['feature_size'],
            dropout_prob=0.1,
            act=('RELU', {'inplace': True}),
            norm='INSTANCE', #default: ("GROUP", {"num_groups": 8})
            use_conv_final=True,
            upsample_mode= params['segresnet_upsample_mode'], # default: "nontrainable"
            blocks_down = blocks_down,
            blocks_up = blocks_up,
        ) if return_model else None

    elif params['model_type'].lower() == 'segresnetvae':
        # SegResNetVAE: 3D ResNet-based architecture with variational autoencoder
        # Reference: "Myronenko A., 3D MRI Brain Tumor Segmentation Using Autoencoder Regularization"
        # https://arxiv.org/abs/1810.11654
        blocks_down = (1, 2, 2, 4) if not params['segresnet_deeper'] else (1, 2, 2, 4, 4)
        blocks_up = (1, 1, 1) if not params['segresnet_deeper'] else (2, 2, 2, 2)
        model = SegResNetVAE(
            input_image_size=params['patch_size'],
            vae_estimate_std=False, #default = False
            vae_default_std=0.3, #default = 0.3
            vae_nz = 256, #default=256,
            spatial_dims=3,
            in_channels=params['chans_in'],
            out_channels=params['chans_out'],
            init_filters=params['feature_size'],
            dropout_prob=0.1,
            norm='INSTANCE',
            use_conv_final=True,
            upsample_mode= params['segresnet_upsample_mode'], # default: "nontrainable"
            blocks_down = blocks_down,
            blocks_up = blocks_up,
        ) if return_model else None
        params["model_returns_vaeloss"] = True

    elif params['model_type'].lower() == 'segresnet_dsa':
        blocks_down = (1, 2, 2, 4) if not params['segresnet_deeper'] else (1, 2, 2, 4, 4)
        blocks_up = (1, 1, 1) if not params['segresnet_deeper'] else (2, 2, 2, 2)
        dsa_start_level = len(blocks_down) - 2
        model = SegResNet_DSA(
            spatial_dims=3,
            in_channels=params['chans_in'],
            out_channels=params['chans_out'],
            init_filters= params['feature_size'], #default= 8
            dropout_prob=0.1,
            norm='INSTANCE',
            use_conv_final=True,
            upsample_mode= params['segresnet_upsample_mode'],
            blocks_down = blocks_down,
            blocks_up = blocks_up,
            dsa_img_size=params['patch_size'],
            dsa_project_size=params['project_size'],
            dsa_num_heads=4,
            dsa_pos_embed=True,
            dsa_dropout_rate=0.1,
            dsa_sa_type=params['sa_type'],
            dsa_bias=False,
            dsa_num_layers=3, #default = 3
            dsa_start_level=dsa_start_level #default = 2
        ) if return_model else None

    elif params['model_type'].lower() == 'segresnetvae_dsa':
        blocks_down = (1, 2, 2, 4) if not params['segresnet_deeper'] else (1, 2, 2, 4, 4)
        blocks_up = (1, 1, 1) if not params['segresnet_deeper'] else (2, 2, 2, 2)
        dsa_start_level = len(blocks_down) - 2
        model = SegResNetVAE_DSA(
            input_image_size=params['patch_size'],
            vae_estimate_std=False, #default = False
            vae_default_std=0.3, #default = 0.3
            vae_nz = 256, #default=256,
            spatial_dims=3,
            in_channels=params['chans_in'],
            out_channels=params['chans_out'],
            init_filters= params['feature_size'], #default= 8
            dropout_prob=0.1,
            norm='INSTANCE',
            use_conv_final=True,
            upsample_mode= params['segresnet_upsample_mode'],
            blocks_down = blocks_down,
            blocks_up = blocks_up,
            dsa_img_size=params['patch_size'],
            dsa_project_size=params['project_size'],
            dsa_num_heads=4,
            dsa_pos_embed=True,
            dsa_dropout_rate=0.1,
            dsa_sa_type=params['sa_type'],
            dsa_bias=False,
            dsa_num_layers=3, #default = 3
            dsa_start_level=dsa_start_level
        ) if return_model else None
        params["model_returns_vaeloss"] = True

    if model is not None:
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {num_trainable_params}")
    return model, params

def get_model_flops(model, params):
    """
    Calculate FLOPs and parameters for the given model.
    """
    input_tensor = torch.randn((params['batch_size'], params['chans_in'], *params['patch_size']), device='cuda')
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print(f"FLOPs: {flops}, Parameters: {params}")
    return flops, params

def get_model_flops_fvcore(model, params):
    """
    Calculate FLOPs and parameters for the given model using fvcore.
    """
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    input_tensor = torch.randn((params['batch_size'], params['chans_in'], *params['patch_size']), device='cuda')
    flops = FlopCountAnalysis(model, input_tensor)
    #params_count = parameter_count_table(model)
    #print(f"FLOPs: {flops.total()}, Parameters: {params_count}")
    print(f"FLOPs: {flops.total()}")
    return flops

if __name__ == "__main__":
    from config import get_default_params
    import argparse
    from monai.utils.misc import ensure_tuple_rep
    from train_cli_utils import parse_kwargs

    params = get_default_params()
    params['patch_size'] = ensure_tuple_rep(params['patch_size'], 3)
    parser = argparse.ArgumentParser(description='Get Model')
    parser.add_argument('--model_type', type=str, default=params['model_type'], help=f'Model type to use (default: {params["model_type"]})')
    parser.add_argument('--kwargs', nargs='*', help='key=value pairs to override params')

    args = parser.parse_args()
    params.update({'model_type': args.model_type})
    if args.kwargs:
        params = parse_kwargs(params, args.kwargs)

    model, params = get_model(params)
    model.to('cuda')

    #get_model_flops(model, params)
    get_model_flops_fvcore(model, params)
    #print(model)
