from __future__ import annotations

from collections.abc import Sequence
from typing import Union, Tuple

import torch
import torch.nn as nn
import numpy as np
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.utils import ensure_tuple_rep
from monai.networks.blocks.dynunet_block import get_conv_layer
from monai.networks.layers.factories import Conv
from monai.networks.layers.utils import get_norm_layer
from .conv_blocks import UnetrBasicBlock, UnetrUpBlock, GeneralUnetrUpBlock
from .conv_blocks import TransformerBlock

from monai.utils import InterpolateMode, UpsampleMode


class BaseUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        feature_size: int = 16,
        norm_name: Union[Tuple, str] = "instance",
        act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        spatial_dims: int = 3,
        res_block: bool = False,
        bias: bool = True,
        depth: int = 5,   # make depth configurable
    ) -> None:
        super().__init__()
        self.name = "BaseUNet"
        self.depth = depth

        # ---------- Encoder ----------
        self.encoders = nn.ModuleList()
        chans_in = in_channels
        chans_out = feature_size
        for i in range(depth):
            layer_en = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=chans_in,
                out_channels=chans_out,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                act_name=act_name,
                res_block=res_block,
                bias=bias,
            )
            self.encoders.append(layer_en)

            if i != depth - 1:
                chans_in = chans_out
                chans_out = chans_in * 2

        # ---------- Decoder ----------
        self.decoders = nn.ModuleList()
        chans_in = chans_out
        chans_out = chans_in // 2
        for i in range(depth - 1):
            layer_de = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=chans_in,
                out_channels=chans_out,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                act_name=act_name,
                res_block=res_block,
                bias=bias,
            )
            self.decoders.append(layer_de)

            if i != depth - 2:
                chans_in = chans_out
                chans_out = chans_in // 2

        # Final conv
        self.final_conv = Conv["conv", spatial_dims](chans_out, out_channels, kernel_size=1)

    def forward(self, x):
        # ---------- Encoder path ----------
        enc_features = []
        out = x
        for i, enc in enumerate(self.encoders):
            out = enc(out)
            enc_features.append(out)
            if i != self.depth - 1:  # no pooling after last encoder
                out = torch.max_pool3d(out, 2, 2)

        # ---------- Decoder path ----------
        dec_out = out
        for i, dec in enumerate(self.decoders):
            skip = enc_features[-(i + 2)]  # take skip connection from encoder
            dec_out = dec(dec_out, skip)

        # ---------- Final conv ----------
        return self.final_conv(dec_out)
    
#from 1/4 to perform transformer
class MS_DSA_NET(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Sequence[int] | int,
        feature_size: int = 16,
        project_size: int = 64,
        num_heads: int = 4,
        pos_embed: bool = True,
        norm_name: Union[Tuple, str] = "instance",
        act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout_rate: float = 0.0,
        do_ds=True,
        spatial_dims:int = 3,
        sa_type = 'parallel',
        res_block = True,
        bias: bool = False,
    ) -> None:

        super().__init__()
        self.name = 'MS_DSA_NET'
        self.do_ds = do_ds
        self.num_classes = out_channels
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        self.img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.num_layers = 3 # of each level: default 3
        self.proj_size = project_size #projection to lower dim
        self.upsample_kernel_size = 2
        self.res_block = res_block

        #[16,D,H,W]
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            act_name=act_name,
            res_block=self.res_block,
            bias=bias,
        )

        #[32,D/2,H/2,W/2]
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size*2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            act_name=act_name,
            res_block=self.res_block,
            bias=bias,
        )

        #[64,D/4,H/4,W/4]
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size*2,
            out_channels=feature_size*4,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            act_name=act_name,
            res_block=self.res_block,
            bias=bias,
        )

        #[128,D/8,H/8,W/8]
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size*4,
            out_channels=feature_size*8,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            act_name=act_name,
            res_block=self.res_block,
            bias=bias,
        )

        #[256,D/16,H/16,W/16]
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size*8,
            out_channels=feature_size*16,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            act_name=act_name,
            res_block=self.res_block,
            bias=bias,
        )

        #[512,D/32,H/32,W/32]
        self.encoder6 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size*16,
            out_channels=feature_size*32,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            act_name=act_name,
            res_block=self.res_block,
            bias=bias,
        )

        self.patch_embedding3 = nn.Sequential(
            get_conv_layer(spatial_dims, feature_size*4, feature_size*2, kernel_size=1, stride=1, dropout=0.0, conv_only=True, bias=bias),
            get_norm_layer(name=("group", {"num_groups": feature_size*1}), channels=feature_size*2),
        )

        fs = tuple(img_d // p_d for img_d, p_d in zip(self.img_size, [4]*3))
        input_size = np.prod(fs)
        self.trans3 = nn.ModuleList(
            [
                TransformerBlock(
                    input_size = input_size,
                    hidden_size = feature_size*2,
                    proj_size = self.proj_size,
                    num_heads = 4,
                    dropout_rate=dropout_rate,
                    pos_embed=pos_embed,
                    sa_type=sa_type,
                ) for i in range(self.num_layers)
            ]
        )
 
        self.patch_embedding4 = nn.Sequential(
            get_conv_layer(spatial_dims, feature_size*8, feature_size*4, kernel_size=1, stride=1, dropout=0.0, conv_only=True, bias=bias),
            get_norm_layer(name=("group", {"num_groups": feature_size*2}), channels=feature_size*4),
        )

        fs = tuple(img_d // p_d for img_d, p_d in zip(self.img_size, [8]*3))
        self.trans4 = nn.ModuleList(
            [
                TransformerBlock(
                    input_size = np.prod(fs),
                    hidden_size = feature_size*4,
                    proj_size = self.proj_size,
                    num_heads = 4,
                    dropout_rate=dropout_rate,
                    pos_embed=pos_embed,
                    sa_type=sa_type,
                ) for i in range(self.num_layers)
            ]
        )

        self.patch_embedding5 = nn.Sequential(
            get_conv_layer(spatial_dims, feature_size*16, feature_size*8, kernel_size=1, stride=1, dropout=0.0, conv_only=True, bias=bias),
            get_norm_layer(name=("group", {"num_groups": feature_size*4}), channels=feature_size*8),
        )
        fs = tuple(img_d // p_d for img_d, p_d in zip(self.img_size, [16]*3))
        self.trans5 = nn.ModuleList(
            [
                TransformerBlock(
                    input_size = np.prod(fs),
                    hidden_size = feature_size*8,
                    proj_size = self.proj_size,
                    num_heads = 4,
                    dropout_rate=dropout_rate,
                    pos_embed=pos_embed,
                    sa_type=sa_type,

                ) for i in range(self.num_layers)
            ]
        )

        self.patch_embedding6 = nn.Sequential(
            get_conv_layer(spatial_dims, feature_size*32, feature_size*16, kernel_size=1, stride=1, dropout=0.0, conv_only=True, bias=bias),
            get_norm_layer(name=("group", {"num_groups": feature_size*8}), channels=feature_size*16),
        )

        fs = tuple(img_d // p_d for img_d, p_d in zip(self.img_size, [32]*3))
        self.trans6 = nn.ModuleList(
            [
                TransformerBlock(
                    input_size = np.prod(fs),
                    hidden_size = feature_size*16,
                    proj_size = 32,
                    num_heads = 4,
                    dropout_rate=dropout_rate,
                    pos_embed=pos_embed,
                    sa_type=sa_type,

                ) for i in range(self.num_layers)
            ]
        )

        #2x upsample trans6 and concat with trans5
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size*16,
            out_channels=feature_size*8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            act_name = act_name,
            res_block=self.res_block,
            bias=bias,
        )

        #2x upsample decoder5 and concat with trans4
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size*8,
            out_channels=feature_size*4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            act_name=act_name,
            res_block=self.res_block,
            bias=bias,
        )

        #2x upsample decoder4 and concat with trans3
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size*4,
            out_channels=feature_size*2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            act_name=act_name,
            res_block=self.res_block,
            bias=bias,
        )

        # 2x upsample decoder3 and concat with encoder2
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size*2,
            out_channels=feature_size*2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            act_name=act_name,
            res_block=self.res_block,
            bias=bias,
        )

        # 2x upsample decoder2 and concat with encoder1
        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size*2,
            out_channels=feature_size*1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            act_name=act_name,
            res_block=self.res_block,
            bias=bias,
        )
        self.features_dim = feature_size * 32 #self.encoder6.layer.out_channels
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            #trunc_normal_(m.weight, std=.02)
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):

        x1 = self.encoder1(x)
        x2 = self.encoder2(torch.max_pool3d(x1, 2, 2))
        x3 = self.encoder3(torch.max_pool3d(x2, 2, 2))
        x4 = self.encoder4(torch.max_pool3d(x3, 2, 2))
        x5 = self.encoder5(torch.max_pool3d(x4, 2, 2))
        x6 = self.encoder6(torch.max_pool3d(x5, 2, 2))

        t3 = self.patch_embedding3(x3)
        t4 = self.patch_embedding4(x4)
        t5 = self.patch_embedding5(x5)
        t6 = self.patch_embedding6(x6)

        for blk in self.trans3:
            t3 = blk(t3)
        for blk in self.trans4:
            t4 = blk(t4)
        for blk in self.trans5:
            t5 = blk(t5)
        for blk in self.trans6:
            t6 = blk(t6)

        #decoding part
        y5 = self.decoder5(t6,t5)
        y4 = self.decoder4(y5,t4)
        y3 = self.decoder3(y4,t3)
        y2 = self.decoder2(y3,x2)
        y1 = self.decoder1(y2,x1)

        y = self.out(y1)
        bottleneck_feat = x6
        return y

class MS_DSA_NET_PS(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Sequence[int] | int,
        feature_size: int = 16,
        project_size: int = 64,
        num_heads: int = 4,
        pos_embed: bool = True,
        norm_name: Union[Tuple, str] = "instance",
        act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout_rate: float = 0.0,
        do_ds=True,
        spatial_dims:int = 3,
        sa_type = 'parallel',
        res_block = True,
        bias: bool = False,
        upsample_mode: UpsampleMode | str = UpsampleMode.PIXELSHUFFLE,
        interpolate_mode: InterpolateMode = InterpolateMode.LINEAR,
    ) -> None:

        super().__init__()
        self.name = 'MS_DSA_NET_PS'
        self.do_ds = do_ds
        self.num_classes = out_channels
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.interpolate_mode = InterpolateMode(interpolate_mode)
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        self.img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.num_layers = 3 # of each level: default 3
        self.proj_size = project_size #projection to lower dim
        self.upsample_kernel_size = 2
        self.res_block = res_block

        #[16,D,H,W]
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            act_name=act_name,
            res_block=self.res_block,
            bias=bias,
        )

        #[32,D/2,H/2,W/2]
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size*2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            act_name=act_name,
            res_block=self.res_block,
            bias=bias,
        )

        #[64,D/4,H/4,W/4]
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size*2,
            out_channels=feature_size*4,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            act_name=act_name,
            res_block=self.res_block,
            bias=bias,
        )

        #[128,D/8,H/8,W/8]
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size*4,
            out_channels=feature_size*8,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            act_name=act_name,
            res_block=self.res_block,
            bias=bias,
        )

        #[256,D/16,H/16,W/16]
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size*8,
            out_channels=feature_size*16,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            act_name=act_name,
            res_block=self.res_block,
            bias=bias,
        )

        #[512,D/32,H/32,W/32]
        self.encoder6 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size*16,
            out_channels=feature_size*32,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            act_name=act_name,
            res_block=self.res_block,
            bias=bias,
        )

        self.patch_embedding3 = nn.Sequential(
            get_conv_layer(spatial_dims, feature_size*4, feature_size*2, kernel_size=1, stride=1, dropout=0.0, conv_only=True, bias=bias),
            get_norm_layer(name=("group", {"num_groups": feature_size*1}), channels=feature_size*2),
        )

        fs = tuple(img_d // p_d for img_d, p_d in zip(self.img_size, [4]*3))
        input_size = np.prod(fs)
        self.trans3 = nn.ModuleList(
            [
                TransformerBlock(
                    input_size = input_size,
                    hidden_size = feature_size*2,
                    proj_size = self.proj_size,
                    num_heads = 4,
                    dropout_rate=dropout_rate,
                    pos_embed=pos_embed,
                    sa_type=sa_type,
                ) for i in range(self.num_layers)
            ]
        )
 
        self.patch_embedding4 = nn.Sequential(
            get_conv_layer(spatial_dims, feature_size*8, feature_size*4, kernel_size=1, stride=1, dropout=0.0, conv_only=True, bias=bias),
            get_norm_layer(name=("group", {"num_groups": feature_size*2}), channels=feature_size*4),
        )

        fs = tuple(img_d // p_d for img_d, p_d in zip(self.img_size, [8]*3))
        self.trans4 = nn.ModuleList(
            [
                TransformerBlock(
                    input_size = np.prod(fs),
                    hidden_size = feature_size*4,
                    proj_size = self.proj_size,
                    num_heads = 4,
                    dropout_rate=dropout_rate,
                    pos_embed=pos_embed,
                    sa_type=sa_type,
                ) for i in range(self.num_layers)
            ]
        )

        self.patch_embedding5 = nn.Sequential(
            get_conv_layer(spatial_dims, feature_size*16, feature_size*8, kernel_size=1, stride=1, dropout=0.0, conv_only=True, bias=bias),
            get_norm_layer(name=("group", {"num_groups": feature_size*4}), channels=feature_size*8),
        )
        fs = tuple(img_d // p_d for img_d, p_d in zip(self.img_size, [16]*3))
        self.trans5 = nn.ModuleList(
            [
                TransformerBlock(
                    input_size = np.prod(fs),
                    hidden_size = feature_size*8,
                    proj_size = self.proj_size,
                    num_heads = 4,
                    dropout_rate=dropout_rate,
                    pos_embed=pos_embed,
                    sa_type=sa_type,

                ) for i in range(self.num_layers)
            ]
        )

        self.patch_embedding6 = nn.Sequential(
            get_conv_layer(spatial_dims, feature_size*32, feature_size*16, kernel_size=1, stride=1, dropout=0.0, conv_only=True, bias=bias),
            get_norm_layer(name=("group", {"num_groups": feature_size*8}), channels=feature_size*16),
        )

        fs = tuple(img_d // p_d for img_d, p_d in zip(self.img_size, [32]*3))
        self.trans6 = nn.ModuleList(
            [
                TransformerBlock(
                    input_size = np.prod(fs),
                    hidden_size = feature_size*16,
                    proj_size = 32,
                    num_heads = 4,
                    dropout_rate=dropout_rate,
                    pos_embed=pos_embed,
                    sa_type=sa_type,

                ) for i in range(self.num_layers)
            ]
        )

        #2x upsample trans6 and concat with trans5
        self.decoder5 = GeneralUnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size*16,
            out_channels=feature_size*8,
            kernel_size=3,
            norm_name=norm_name,
            act_name = act_name,
            res_block=self.res_block,
            bias=bias,
            upsample_mode=self.upsample_mode,
            interpolate_mode=self.interpolate_mode,
            scale_factor=2,
        )

        #2x upsample decoder5 and concat with trans4
        self.decoder4 = GeneralUnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size*8,
            out_channels=feature_size*4,
            kernel_size=3,
            norm_name=norm_name,
            act_name=act_name,
            res_block=self.res_block,
            bias=bias,
            upsample_mode=self.upsample_mode,
            interpolate_mode=self.interpolate_mode,
            scale_factor=2,
        )

        #2x upsample decoder4 and concat with trans3
        self.decoder3 = GeneralUnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size*4,
            out_channels=feature_size*2,
            kernel_size=3,
            norm_name=norm_name,
            act_name=act_name,
            res_block=self.res_block,
            bias=bias,
            upsample_mode=self.upsample_mode,
            interpolate_mode=self.interpolate_mode,
            scale_factor=2,
        )

        # 2x upsample decoder3 and concat with encoder2
        self.decoder2 = GeneralUnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size*2,
            out_channels=feature_size*2,
            kernel_size=3,
            norm_name=norm_name,
            act_name=act_name,
            res_block=self.res_block,
            bias=bias,
            upsample_mode=self.upsample_mode,
            interpolate_mode=self.interpolate_mode,
            scale_factor=2,
        )

        # 2x upsample decoder2 and concat with encoder1
        self.decoder1 = GeneralUnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size*2,
            out_channels=feature_size*1,
            kernel_size=3,
            norm_name=norm_name,
            act_name=act_name,
            res_block=self.res_block,
            bias=bias,
            upsample_mode=self.upsample_mode,
            interpolate_mode=self.interpolate_mode,
            scale_factor=2,
        )
        self.features_dim = feature_size * 32 #self.encoder6.layer.out_channels
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            #trunc_normal_(m.weight, std=.02)
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):

        x1 = self.encoder1(x)
        x2 = self.encoder2(torch.max_pool3d(x1, 2, 2))
        x3 = self.encoder3(torch.max_pool3d(x2, 2, 2))
        x4 = self.encoder4(torch.max_pool3d(x3, 2, 2))
        x5 = self.encoder5(torch.max_pool3d(x4, 2, 2))
        x6 = self.encoder6(torch.max_pool3d(x5, 2, 2))

        t3 = self.patch_embedding3(x3)
        t4 = self.patch_embedding4(x4)
        t5 = self.patch_embedding5(x5)
        t6 = self.patch_embedding6(x6)

        for blk in self.trans3:
            t3 = blk(t3)
        for blk in self.trans4:
            t4 = blk(t4)
        for blk in self.trans5:
            t5 = blk(t5)
        for blk in self.trans6:
            t6 = blk(t6)

        #decoding part
        y5 = self.decoder5(t6,t5)
        y4 = self.decoder4(y5,t4)
        y3 = self.decoder3(y4,t3)
        y2 = self.decoder2(y3,x2)
        y1 = self.decoder1(y2,x1)

        y = self.out(y1)
        bottleneck_feat = x6
        return y

