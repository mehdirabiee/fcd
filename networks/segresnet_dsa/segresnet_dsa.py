from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from monai.utils import ensure_tuple_rep
from monai.networks.blocks.dynunet_block import get_conv_layer
from monai.networks.blocks import UpSample
from monai.networks.layers.utils import get_norm_layer, get_act_layer
from .conv_blocks import TransformerBlock

from monai.networks.blocks.upsample import UpSample
from monai.utils import InterpolateMode, UpsampleMode

from monai.networks.layers.factories import Dropout
from monai.networks.blocks.segresnet_block import ResBlock, get_upsample_layer
from monai.networks.blocks.segresnet_block import get_conv_layer as get_conv_layer_segresnet


class SegResNet_DSA(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 8,
        in_channels: int = 1,
        out_channels: int = 2,
        dropout_prob: float | None = None,
        act: tuple | str = ("RELU", {"inplace": True}),
        norm: tuple | str = ("GROUP", {"num_groups": 8}),
        norm_name: str = "",
        num_groups: int = 8,
        use_conv_final: bool = True,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        upsample_mode: UpsampleMode | str = UpsampleMode.PIXELSHUFFLE,
        interpolate_mode: InterpolateMode = InterpolateMode.LINEAR,
        dsa_img_size: Sequence[int] | int = 128,
        dsa_project_size: int = 64,
        dsa_num_heads: int = 4,
        dsa_pos_embed: bool = True,
        dsa_dropout_rate: float = 0.0,
        dsa_sa_type = 'parallel',
        dsa_bias: bool = False,
        dsa_num_layers: int = 3,
        dsa_start_level: int = 3,
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims` can only be 2 or 3.")

        self.dsa_img_size = ensure_tuple_rep(dsa_img_size, spatial_dims)
        self.dsa_project_size = dsa_project_size
        self.dsa_num_heads = dsa_num_heads
        self.dsa_pos_embed = dsa_pos_embed
        self.dsa_dropout_rate = dsa_dropout_rate
        self.dsa_sa_type = dsa_sa_type
        self.dsa_bias = dsa_bias
        self.dsa_num_layers = dsa_num_layers
        self.dsa_start_level = dsa_start_level

        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.act = act  # input options
        self.act_mod = get_act_layer(act)

        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
            norm = ("group", {"num_groups": num_groups})
        self.norm = norm
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.interpolate_mode = InterpolateMode(interpolate_mode)
        self.use_conv_final = use_conv_final
        self.convInit = get_conv_layer_segresnet(spatial_dims, in_channels, init_filters)
        self.down_layers = self._make_down_layers()
        self.up_layers, self.up_samples = self._make_up_layers()
        self.patch_embeddings, self.transformer_layers = self._make_transformer_layers()
        self.conv_final = self._make_final_conv(out_channels)

        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

    def _make_down_layers(self):
        down_layers = nn.ModuleList()
        blocks_down, spatial_dims, filters, norm = (self.blocks_down, self.spatial_dims, self.init_filters, self.norm)
        for i, item in enumerate(blocks_down):
            layer_in_channels = filters * 2**i
            pre_conv = (
                get_conv_layer_segresnet(spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2)
                if i > 0
                else nn.Identity()
            )
            down_layer = nn.Sequential(
                pre_conv, *[ResBlock(spatial_dims, layer_in_channels, norm=norm, act=self.act) for _ in range(item)]
            )
            down_layers.append(down_layer)
        return down_layers

    def _make_up_layers(self):
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters, norm = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters,
            self.norm,
        )
        interpolate_mode = self.interpolate_mode
        n_up = len(blocks_up)
        for i in range(n_up):
            corresponding_down_layer_index = n_up - i 
            sample_in_channels = filters * 2 ** (n_up - i)
            up_layers.append(
                nn.Sequential(
                    *[
                        ResBlock(spatial_dims, sample_in_channels // 2, norm=norm, act=self.act)
                        for _ in range(blocks_up[i])
                    ]
                )
            )
            up_samples.append(
                nn.Sequential(
                    *[
                        get_conv_layer_segresnet(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                        UpSample(spatial_dims=spatial_dims, 
                                in_channels=sample_in_channels // 2,
                                out_channels=sample_in_channels // 2,
                                scale_factor=2,
                                mode = upsample_mode,
                                #mode=upsample_mode if corresponding_down_layer_index < self.dsa_start_level else "deconv",
                                interp_mode=interpolate_mode,
                                align_corners=False,
                            )                        
                    ]
                )
            )
        return up_layers, up_samples

    def _make_transformer_layers(self):
        patch_embeddings = nn.ModuleList()
        transformer_layers = nn.ModuleList()

        n_down = len(self.blocks_down)
        for i in range(self.dsa_start_level, n_down):
            down_layer_n_channels = self.init_filters * (2**i)
            down_layer_image_size = tuple(img_d // (2**i) for img_d in self.dsa_img_size)
            #trans_n_channels = down_layer_n_channels // 2
            #patch_embedding = nn.Sequential(
            #    get_conv_layer(self.spatial_dims, down_layer_n_channels, trans_n_channels, kernel_size=1, stride=1, dropout=0.0, conv_only=True, bias=self.dsa_bias),
            #    get_norm_layer(name=("group", {"num_groups": trans_n_channels//2}), channels=trans_n_channels),
            #)

            trans_n_channels = down_layer_n_channels
            patch_embedding = nn.Identity()
            #patch_embedding = get_norm_layer(name=("group", {"num_groups": trans_n_channels//2}), channels=trans_n_channels)


            input_size = np.prod(down_layer_image_size)
            trans = nn.ModuleList(
                [
                    *[
                    TransformerBlock(
                        input_size = input_size,
                        hidden_size = trans_n_channels,
                        proj_size = self.dsa_project_size,
                        num_heads = 4,
                        dropout_rate=self.dsa_dropout_rate,
                        pos_embed=self.dsa_pos_embed,
                        sa_type=self.dsa_sa_type,
                    ) for _ in range(self.dsa_num_layers)
                    ],
                    #get_conv_layer(self.spatial_dims, trans_n_channels, down_layer_n_channels, kernel_size=1, stride=1, dropout=0.0, conv_only=True, bias=self.dsa_bias)
                ]
            )
            patch_embeddings.append(patch_embedding)
            transformer_layers.append(trans)

        return patch_embeddings, transformer_layers

    def _make_final_conv(self, out_channels: int):
        return nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.act_mod,
            get_conv_layer_segresnet(self.spatial_dims, self.init_filters, out_channels, kernel_size=1, bias=True),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.convInit(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)
        down_x = []

        for i, down in enumerate(self.down_layers):
            x = down(x)
            if i < self.dsa_start_level:
                feature = x
            else:
                patch_embedding = self.patch_embeddings[i-self.dsa_start_level]
                transformer_layer = self.transformer_layers[i-self.dsa_start_level]
                feature = patch_embedding(x)
                for blk in transformer_layer:
                    feature = blk(feature)
            down_x.append(feature)

        return feature, down_x

    def decode(self, x: torch.Tensor, down_x: list[torch.Tensor]) -> torch.Tensor:
        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            x = up(x) + down_x[i + 1]
            x = upl(x)

        if self.use_conv_final:
            x = self.conv_final(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, down_x = self.encode(x)
        down_x.reverse()

        x = self.decode(x, down_x)
        return x

class SegResNetVAE_DSA(SegResNet_DSA):
    def __init__(
        self,
        input_image_size: Sequence[int],
        vae_estimate_std: bool = False,
        vae_default_std: float = 0.3,
        vae_nz: int = 256,
        spatial_dims: int = 3,
        init_filters: int = 8,
        in_channels: int = 1,
        out_channels: int = 2,
        dropout_prob: float | None = None,
        act: tuple | str = ("RELU", {"inplace": True}),
        norm: tuple | str = ("GROUP", {"num_groups": 8}),
        norm_name: str = "",
        num_groups: int = 8,
        use_conv_final: bool = True,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        upsample_mode: UpsampleMode | str = UpsampleMode.PIXELSHUFFLE,
        interpolate_mode: InterpolateMode = InterpolateMode.LINEAR,
        dsa_img_size: Sequence[int] | int = 128,
        dsa_project_size: int = 64,
        dsa_num_heads: int = 4,
        dsa_pos_embed: bool = True,
        dsa_dropout_rate: float = 0.0,
        dsa_sa_type = 'parallel',
        dsa_bias: bool = False,
        dsa_num_layers: int = 3,
        dsa_start_level: int = 3,
    ):
        super().__init__(
            spatial_dims=spatial_dims,
            init_filters=init_filters,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_prob=dropout_prob,
            act=act,
            norm=norm,
            use_conv_final=use_conv_final,
            blocks_down=blocks_down,
            blocks_up=blocks_up,
            upsample_mode=upsample_mode,
            interpolate_mode=interpolate_mode,
            dsa_img_size=dsa_img_size,
            dsa_project_size=dsa_project_size,
            dsa_num_heads=dsa_num_heads,
            dsa_pos_embed=dsa_pos_embed,
            dsa_dropout_rate=dsa_dropout_rate,
            dsa_sa_type=dsa_sa_type,
            dsa_bias = dsa_bias,
            dsa_num_layers=dsa_num_layers,
            dsa_start_level=dsa_start_level
        )

        self.input_image_size = input_image_size
        self.smallest_filters = 16

        zoom = 2 ** (len(self.blocks_down) - 1)
        self.fc_insize = [s // (2 * zoom) for s in self.input_image_size]

        self.vae_estimate_std = vae_estimate_std
        self.vae_default_std = vae_default_std
        self.vae_nz = vae_nz
        self._prepare_vae_modules()
        self.vae_conv_final = self._make_final_conv(in_channels)

    def _prepare_vae_modules(self):
        zoom = 2 ** (len(self.blocks_down) - 1)
        v_filters = self.init_filters * zoom
        total_elements = int(self.smallest_filters * np.prod(self.fc_insize))

        self.vae_down = nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=v_filters),
            self.act_mod,
            get_conv_layer_segresnet(self.spatial_dims, v_filters, self.smallest_filters, stride=2, bias=True),
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.smallest_filters),
            self.act_mod,
        )
        self.vae_fc1 = nn.Linear(total_elements, self.vae_nz)
        self.vae_fc2 = nn.Linear(total_elements, self.vae_nz)
        self.vae_fc3 = nn.Linear(self.vae_nz, total_elements)

        self.vae_fc_up_sample = nn.Sequential(
            get_conv_layer_segresnet(self.spatial_dims, self.smallest_filters, v_filters, kernel_size=1),
            get_upsample_layer(self.spatial_dims, v_filters, upsample_mode=self.upsample_mode),
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=v_filters),
            self.act_mod,
        )

    def _get_vae_loss(self, net_input: torch.Tensor, vae_input: torch.Tensor):
        """
        Args:
            net_input: the original input of the network.
            vae_input: the input of VAE module, which is also the output of the network's encoder.
        """
        x_vae = self.vae_down(vae_input)
        x_vae = x_vae.view(-1, self.vae_fc1.in_features)
        z_mean = self.vae_fc1(x_vae)

        z_mean_rand = torch.randn_like(z_mean)
        z_mean_rand.requires_grad_(False)

        if self.vae_estimate_std:
            z_sigma = self.vae_fc2(x_vae)
            z_sigma = F.softplus(z_sigma)
            vae_reg_loss = 0.5 * torch.mean(z_mean**2 + z_sigma**2 - torch.log(1e-8 + z_sigma**2) - 1)

            x_vae = z_mean + z_sigma * z_mean_rand
        else:
            z_sigma = self.vae_default_std
            vae_reg_loss = torch.mean(z_mean**2)

            x_vae = z_mean + z_sigma * z_mean_rand

        x_vae = self.vae_fc3(x_vae)
        x_vae = self.act_mod(x_vae)
        x_vae = x_vae.view([-1, self.smallest_filters] + self.fc_insize)
        x_vae = self.vae_fc_up_sample(x_vae)

        for up, upl in zip(self.up_samples, self.up_layers):
            x_vae = up(x_vae)
            x_vae = upl(x_vae)

        x_vae = self.vae_conv_final(x_vae)
        vae_mse_loss = F.mse_loss(net_input, x_vae)
        vae_loss = vae_reg_loss + vae_mse_loss
        return vae_loss

    def forward(self, x):
        net_input = x
        x, down_x = self.encode(x)
        down_x.reverse()

        vae_input = x
        x = self.decode(x, down_x)

        if self.training:
            vae_loss = self._get_vae_loss(net_input, vae_input)
            return x, vae_loss

        return x, None

