from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np
import torch
import torch.nn as nn
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.networks.blocks.dynunet_block import get_conv_layer
from monai.networks.blocks.mlp import MLPBlock

class TransformerBlock(nn.Module):

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
            sa_type = 'parallel',
            norm_name = 'batch',
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")
        self.sa_type = sa_type
        self.norm = nn.LayerNorm(hidden_size)
        #self.norm2 = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)

        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name='batch') #norm_name
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

        self.dsa = DSA(input_size=input_size, hidden_size=hidden_size, proj_size=proj_size, num_heads=num_heads, 
                channel_attn_drop=dropout_rate,
                spatial_attn_drop=dropout_rate,
                sa_type=sa_type,
            )

    def forward(self, x):
        B, C, H, W, D = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed

        x = x + self.gamma *self.dsa(self.norm(x))
        #x = self.norm2(x)

        x = x.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        x = x + self.conv8( self.conv51(x) )  

        '''
        if self.sa_type == 'serial': #vit spatial + se
            y = self.squeeze(x)
            y = self.excitation(y)
            x = x * y
        '''

        return x

class TransformerBlockDSA(nn.Module):

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
            sa_type = 'parallel',
            norm_name = 'batch',
    ) -> None:

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))
        
        #self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.dsa = DSA(input_size=input_size, hidden_size=hidden_size, proj_size=proj_size, num_heads=num_heads, 
            channel_attn_drop=dropout_rate,
            spatial_attn_drop=dropout_rate,
            sa_type=sa_type,
        )

        self.mlp = MLPBlock(hidden_size=hidden_size, mlp_dim=hidden_size*4, dropout_rate= dropout_rate)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        B, C, H, W, D = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed

        x = x + self.dsa(self.norm1(x))
        x = x + self.mlp( self.norm2(x) )

        x = x.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        return x

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

class CrossAttentionBlock(nn.Module):
    def  __init__(self, input_size, hidden_size, proj_size, num_heads=4, qkv_bias=False, drop_rate=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Linear(hidden_size, hidden_size * 1, bias=qkv_bias)
        self.kv = nn.Linear(hidden_size, hidden_size * 2, bias=qkv_bias)

        self.EF = nn.Parameter(init_(torch.zeros(input_size, proj_size)))

        self.attn_drop = nn.Dropout(drop_rate)

        self.mlp = MLPBlock(hidden_size=hidden_size, mlp_dim=hidden_size*4, dropout_rate= drop_rate)
        self.norm = nn.LayerNorm(hidden_size)

    #x from encoder , y from decoder
    def forward(self, x, y): 

        B, C, H, W, D = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)
        y = y.reshape(B, C, H * W * D).permute(0, 2, 1)

        B, N, C = x.shape

        q = self.q(x).reshape(B, N, 1, self.num_heads, C // self.num_heads)
        q = q.permute(2, 0, 3, 1, 4)
        q = q[0]

        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        q = q.transpose(-2, -1) #B,h,c,N
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        proj_e_f = lambda args: torch.einsum('bhdn,nk->bhdk', *args)
        kp, vp = map(proj_e_f, zip((k, v), (self.EF, self.EF)))

        q = torch.nn.functional.normalize(q, dim=-1)
        
        attn = (q.permute(0, 1, 3, 2) @ kp) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        o = (attn @ vp.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)

        y = y + self.mlp( self.norm(o) )
        y = y.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3) 

        return y

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}

#dual-self-attention
class DSA(nn.Module):
    def __init__(self, input_size, hidden_size, proj_size, num_heads=4, qkv_bias=False,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1, sa_type='parallel'): #parallel, serial, spatial, and channel
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.sa_type = sa_type

        self.num = 4 if sa_type == 'parallel' else 3
     
        # qkvv are 4 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qkvv = nn.Linear(hidden_size, hidden_size * self.num, bias=qkv_bias)

        # E and F are projection matrices with shared weights used in spatial attention module to project
        # keys and values from HWD-dimension to P-dimension
        self.EF = nn.Parameter(init_(torch.zeros(input_size, proj_size)))
        self.input_size = input_size
        self.proj_size = proj_size

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

    def forward_spatial(self,x):

        B, N, C = x.shape

        qkvv = self.qkvv(x).reshape(B, N, self.num, self.num_heads, C // self.num_heads)
        qkvv = qkvv.permute(2, 0, 3, 1, 4)
        q_shared, k_shared, v_SA = qkvv[0], qkvv[1], qkvv[2]

        q_shared = q_shared.transpose(-2, -1) #B,h,c,N
        k_shared = k_shared.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        proj_e_f = lambda args: torch.einsum('bhdn,nk->bhdk', *args)
        k_shared_projected, v_SA_projected = map(proj_e_f, zip((k_shared, v_SA), (self.EF, self.EF)))

        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)
        
        attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_shared_projected) * self.temperature2 #self.scale
        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)
        x_SA = (attn_SA @ v_SA_projected.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)
        return x_SA

    def forward_channel(self,x):
        B, N, C = x.shape

        qkvv = self.qkvv(x).reshape(B, N, self.num, self.num_heads, C // self.num_heads)
        qkvv = qkvv.permute(2, 0, 3, 1, 4)
        q_shared, k_shared, v_CA = qkvv[0], qkvv[1], qkvv[2]

        q_shared = q_shared.transpose(-2, -1) #B,h,c,N
        k_shared = k_shared.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)

        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature
        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)
        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)

        return x_CA

    def forward_serial(self, x):
        B, N, C = x.shape

        qkvv = self.qkvv(x).reshape(B, N, self.num, self.num_heads, C // self.num_heads)
        qkvv = qkvv.permute(2, 0, 3, 1, 4)
        q_shared, k_shared, v_SA = qkvv[0], qkvv[1], qkvv[2] #B,h,N,c

        #B,h,c,N: C=hc
        q_shared = q_shared.transpose(-2, -1) 
        k_shared = k_shared.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        proj_e_f = lambda args: torch.einsum('bhdn,nk->bhdk', *args)
        k_shared_projected, v_SA_projected = map(proj_e_f, zip((k_shared, v_SA), (self.EF, self.EF))) #cxp

        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

        # Nxc x cxp = Nxp
        attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_shared_projected) * self.temperature2
        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)
        #x_SA = (attn_SA @ v_SA_projected.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)
        #         Nxp x pxc = Nxc
        x_SA = (attn_SA @ v_SA_projected.transpose(-2, -1)) 

        #           cxN     x   Nxc = cxc
        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature 
        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)
        #cxc x cxN = cxN
        x_CA = (attn_CA @ x_SA.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)

        return x_CA
        

    def forward(self, x):

        if self.sa_type == 'spatial':
            return self.forward_spatial(x)
        
        if self.sa_type == 'channel':
            return self.forward_channel(x)

        if self.sa_type == 'serial':
            return self.forward_serial(x)

        B, N, C = x.shape

        qkvv = self.qkvv(x).reshape(B, N, 4, self.num_heads, C // self.num_heads)
        qkvv = qkvv.permute(2, 0, 3, 1, 4)
        q_shared, k_shared, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3] #B,h,N,c

        q_shared = q_shared.transpose(-2, -1) #B,h,c,N
        k_shared = k_shared.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        proj_e_f = lambda args: torch.einsum('bhdn,nk->bhdk', *args)
        k_shared_projected, v_SA_projected = map(proj_e_f, zip((k_shared, v_SA), (self.EF, self.EF)))
            
        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature
        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)
        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)

        attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_shared_projected) * self.temperature2
        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)
        x_SA = (attn_SA @ v_SA_projected.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)

        return x_CA + x_SA

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature', 'temperature2'}


class UnetResBlock(nn.Module):
    """
    A skip-connection based module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        norm_name: tuple | str,
        act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: tuple | str | float | None = None,
        bias: bool = False,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
            bias=bias,
        )
        self.conv2 = get_conv_layer(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
            bias=bias,
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True
        if self.downsample:
            self.conv3 = get_conv_layer(
                spatial_dims,
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                dropout=dropout,
                act=None,
                norm=None,
                conv_only=False,
                bias=bias,
            )
            self.norm3 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if hasattr(self, "conv3"):
            residual = self.conv3(residual)
        if hasattr(self, "norm3"):
            residual = self.norm3(residual)
        out += residual
        out = self.lrelu(out)
        return out


class UnetBasicBlock(nn.Module):
    """
    A CNN module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        norm_name: tuple | str,
        act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: tuple | str | float | None = None,
        bias: bool = False,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
            bias = bias,
        )
        self.conv2 = get_conv_layer(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
            bias = bias,
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, inp):
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.lrelu(out)
        return out

#up+ cat/sum +3xdsa
class DsaUpBlock(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        upsample_kernel_size: Sequence[int] | int,
        norm_name: tuple | str,
        act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        bias:bool = False,
        fuse: str ='cat', #'sum','cat'
        out_size: int = 0, #spatial size of decoder
        proj_size: int = 64,
        drop_rate: float = 0,
    ) -> None:

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
            bias=bias,
        )
        self.fuse = fuse
        self.depth = 3
        self.num_heads = 4

        if fuse == 'cat':
            chans_fuse = out_channels + out_channels
            stage_blocks = []
            stage_blocks.append(
                UnetResBlock(
                    spatial_dims,
                    chans_fuse,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    norm_name=norm_name,
                    act_name=act_name,
                    bias=bias,
                )
            )
            for j in range(self.depth):
                stage_blocks.append(TransformerBlock(input_size=out_size, hidden_size=out_channels,
                    proj_size=proj_size, num_heads=self.num_heads, dropout_rate=drop_rate, pos_embed=True))
            self.conv_block = nn.Sequential(*stage_blocks)

        elif fuse == 'cross':
            self.cross_block = CrossAttentionBlock(input_size=out_size, hidden_size=out_channels,
                    proj_size=proj_size, num_heads=self.num_heads, drop_rate=drop_rate)
        else:
            chans_fuse = out_channels
            stage_blocks = []
            for j in range(self.depth):
                stage_blocks.append(TransformerBlock(input_size=out_size, hidden_size= chans_fuse,
                    proj_size=proj_size, num_heads=self.num_heads, dropout_rate=drop_rate, pos_embed=True))
            self.conv_block = nn.Sequential(*stage_blocks)
       
    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        if self.fuse == 'cat':
            out = torch.cat((out, skip), dim=1)
            out = self.conv_block(out)
        elif self.fuse == 'cross':
            out = self.cross_block(skip, out)
        else:
            out = out + skip
            out = self.conv_block(out)
        
        return out

class UnetrUpBlock(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        upsample_kernel_size: Sequence[int] | int,
        norm_name: tuple | str,
        act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        res_block: bool = False,
        bias:bool = False,
        fuse: str ='cat', #'sum','cat'
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
            bias=bias,
        )
        self.fuse = fuse

        if fuse == 'cat':
            chans_fuse = out_channels + out_channels
        else:
            chans_fuse = out_channels
        
        if res_block:
            
            self.conv_block = UnetResBlock(
                spatial_dims,
                chans_fuse,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
                act_name=act_name,
                bias=bias,
            )
        else:
            self.conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims,
                chans_fuse,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
                act_name=act_name,
                bias = bias,
            )            

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        if self.fuse == 'cat':
            out = torch.cat((out, skip), dim=1)
        else:
            out = out + skip
        out = self.conv_block(out)
        return out



class UnetrBasicBlock(nn.Module):
    """
    A CNN module that can be used for UNETR, based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        norm_name: tuple | str,
        act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        res_block: bool = False,
        bias: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            stride: convolution stride.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()

        if res_block:
            self.layer = UnetResBlock(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_name=norm_name,
                act_name=act_name,
                bias=bias,
            )
        else:
            self.layer = UnetBasicBlock(  # type: ignore
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_name=norm_name,
                act_name=act_name,
                bias=bias,
            )

    def forward(self, inp):
        return self.layer(inp)


class AttentionBlock(nn.Module):
    def __init__(self, spatial_dims: int, f_int: int, f_g: int, f_l: int, dropout=0.0, bias=False):
        super().__init__()
        self.W_g = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_g,
                out_channels=f_int,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
                bias=bias,
            ),
            Norm[Norm.BATCH, spatial_dims](f_int),
        )

        self.W_x = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_l,
                out_channels=f_int,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
                bias=bias,
            ),
            Norm[Norm.BATCH, spatial_dims](f_int),
        )

        self.psi = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_int,
                out_channels=1,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[Norm.BATCH, spatial_dims](1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU()

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi: torch.Tensor = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

#attention gate and upsample
class AgUpBlock(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        upsample_kernel_size: Sequence[int] | int,
        norm_name: tuple | str,
        res_block: bool = False,
        bias: bool = False,
        fuse: str = 'sum',
    ) -> None:

        super().__init__()
        self.fuse = fuse
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
            bias=bias,
        )
        self.ag = AttentionBlock(
            spatial_dims = spatial_dims,
            f_int = out_channels // 2,
            f_g = out_channels,
            f_l = out_channels,
            bias=bias,
        )

        fuse_in = out_channels if self.fuse == 'sum' else out_channels*2
        if res_block:
            self.conv_block = UnetResBlock(
                spatial_dims,
                fuse_in,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
                bias=bias,
            )
        else:
            self.conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims,
                fuse_in,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
                bias=bias,
            )

    def forward(self, inp, skip):
        out = self.transp_conv(inp)
        skip = self.ag(out, skip) 
        if self.fuse == 'sum':
            out = out + skip
        else:
            out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out
