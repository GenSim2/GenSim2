import numpy as np
import torch
from torch import nn
import torchvision
from torchvision import transforms
from functools import partial
from typing import Tuple, Union, Union, List
from termcolor import cprint
from einops import rearrange, repeat, reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

import IPython

from gensim2.paths import *
from gensim2.env.utils.pcd_utils import BOUND
from .clip.core.clip import build_model, load_clip, tokenize

IMGNET_TFM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        widths=[512],
        tokenize_dim=32,
        tanh_end=False,
        ln=True,
        num_of_copy=1,
        **kwargs,
    ):
        """vanilla MLP class with RELU activation and fixed width"""
        if not isinstance(input_dim, int):
            input_dim = np.prod(input_dim)

        super().__init__()
        modules = [nn.Linear(input_dim, widths[0]), nn.ReLU()]

        for i in range(len(widths) - 1):
            modules.extend([nn.Linear(widths[i], widths[i + 1])])
            if ln:
                modules.append(nn.LayerNorm(widths[i + 1]))
            modules.append(nn.ReLU())

        modules.append(nn.Linear(widths[-1], output_dim))
        if tanh_end:
            modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)
        self.num_of_copy = num_of_copy
        if self.num_of_copy > 1:
            self.net = nn.ModuleList(
                [nn.Sequential(*modules) for _ in range(num_of_copy)]
            )

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        if self.num_of_copy > 1:
            IPython.embed()
        else:
            y = self.net(x)
        y = y.reshape(y.shape[0], 1, y.shape[-1])  # 1 is the number of token
        return y


class ResNet(nn.Module):
    def __init__(
        self,
        output_dim=512,
        weights=None,  # "DEFAULT"->IMAGENET1K_V1,
        resnet_model="resnet18",
        num_of_copy=1,
        finetune=True,
        **kwargs,
    ):
        super().__init__()
        pretrained_model = getattr(torchvision.models, resnet_model)(weights=weights)
        # by default we use a separate image encoder for each view in downstream evaluation
        self.num_of_copy = num_of_copy
        self.net = nn.Sequential(*list(pretrained_model.children())[:-2])
        if not finetune:
            self.net.requires_grad_(False)  # Freeze the model

        if num_of_copy > 1:
            self.net = nn.ModuleList(
                [
                    nn.Sequential(*list(pretrained_model.children())[:-2])
                    for _ in range(num_of_copy)
                ]
            )

        self.output_dim = output_dim
        self.avgpool = nn.AvgPool2d(9, stride=1)
        self.proj = nn.Linear(7 * 12 * 512, output_dim)

    def forward(self, x):
        """
        x: dict of B x T x H x W x 4
        """
        x = torch.stack([*x.values()], dim=0)[..., :3]  # N x B x H x W x 4
        x = x.permute(1, 0, 4, 2, 3)
        # flatten first
        B, N, D, H, W = x.shape
        x = x.reshape(len(x), -1, 3, H, W)
        if self.num_of_copy > 1:
            # separate encoding for each view
            out = []
            iter_num = min(self.num_of_copy, x.shape[1])
            for idx in range(iter_num):
                input = x[:, idx]
                net = self.net[idx]
                out.append(net(input))
            feat = torch.stack(out, dim=1)
        else:
            x = x.reshape(-1, 3, H, W)
            feat = self.net(x)

        feat = self.avgpool(feat).contiguous()
        feat = self.proj(feat.view(feat.shape[0], -1))  # (B * Ncam) x 512
        feat = feat.view(B, -1, feat.shape[-1])  # concat all views

        return feat


class ViT(nn.Module):
    """Vision Transformer by DINOv2"""

    def __init__(
        self,
        model_name,
        patch_size=14,
        output_dim=512,
        num_of_copy=1,
        finetune=False,
        **kwargs,
    ):
        super(ViT, self).__init__(**kwargs)

        self.num_of_copy = num_of_copy
        backbone = torch.hub.load("facebookresearch/dinov2", model_name)

        self.net = backbone
        self.patch_size = patch_size

        if not finetune:
            self.net.requires_grad_(False)  # Freeze the model

        if num_of_copy > 1:
            self.net = nn.ModuleList(
                [nn.Sequential(*list(backbone.children())) for _ in range(num_of_copy)]
            )

        self.output_dim = output_dim
        self.proj = nn.Linear(384, output_dim)

    def forward(self, x):
        """
        x: dict of B x T x H x W x 4
        """
        x = torch.stack([*x.values()], dim=0)[..., :3]  # N x B x H x W x 4
        x = x.permute(1, 0, 4, 2, 3)
        # flatten first
        B, N, D, H, W = x.shape
        x = x.reshape(len(x), -1, 3, H, W)
        if self.num_of_copy > 1:
            # separate encoding for each view
            out = []
            iter_num = min(self.num_of_copy, x.shape[1])
            for idx in range(iter_num):
                input = x[:, idx]
                input = transforms.Pad(
                    (
                        (self.patch_size - W % self.patch_size) // 2,
                        (self.patch_size - H % self.patch_size) // 2,
                    )
                )(input)
                net = self.net[idx]
                out.append(net(input))
            feat = torch.stack(out, dim=1).contiguous()
        else:
            x = x.reshape(-1, 3, H, W)
            x = transforms.Pad(
                (
                    (self.patch_size - W % self.patch_size) // 2,
                    (self.patch_size - H % self.patch_size) // 2,
                )
            )(x)
            feat = self.net(x).contiguous()

        feat = self.proj(feat.view(feat.shape[0], -1))  # (B * Ncam) x 512
        feat = feat.view(B, -1, feat.shape[-1])  # concat all views

        return feat


class PointNet(nn.Module):
    """Simple Pointnet-Like Network"""

    def __init__(
        self,
        pcd_domain,
        cfg_name,
        point_dim=2,
        global_feat=None,
        finetune=False,
        pretrained_path=None,
        mlp_widths=None,
        color_extra=False,
        **kwargs,
    ):
        from openpoints.models import build_model_from_cfg
        from openpoints.utils import EasyConfig, load_checkpoint
        from gensim2.agent.utils.pcd_utils import update_openpoint_cfgs

        super(PointNet, self).__init__()

        self.domain = pcd_domain
        cfg = EasyConfig()
        cfg.load(
            f"{GENSIM_DIR}/agent/models/pointnet_cfg/{pcd_domain}/{cfg_name}.yaml",
            recursive=True,
        )
        update_openpoint_cfgs(cfg)
        self.in_channels = int(cfg.model.encoder_args.in_channels)
        self.pointnet = build_model_from_cfg(cfg.model)
        if pretrained_path is not None:  # load pre-train weights
            load_checkpoint(self.pointnet, pretrained_path=pretrained_path)
        if not finetune:
            self.pointnet.requires_grad_(False)  # Freeze the model

        self.point_dim = point_dim
        self.global_feat = global_feat

        self.mlp = None
        self.activation = None
        if mlp_widths is not None:
            self.activation = torch.nn.ReLU()
            self.mlp = torch.nn.Linear(mlp_widths[0], mlp_widths[1])

        self.color_pointnet = None
        if color_extra:
            cfg.model.encoder_args.in_channels = 3
            self.color_pointnet = build_model_from_cfg(cfg.model)

    def forward(self, x):
        """
        x: dict(['colors', 'pos', 'seg', 'heights', 'x']), B x N x D
        """
        # x["x"] = x["x"][..., : self.in_channels]
        if self.color_pointnet is not None:
            color_x = {"x": x["colors"], "pos": x["pos"].clone()}
            # color_x = {"x": torch.cat([x["colors"], x["x"][:,-1:,:]], axis= 1), "pos": x["pos"].clone()}
            color_x = self.color_pointnet.encoder.forward_cls_feat(color_x)

        if self.global_feat is not None:
            global_feats = []
            for preprocess in self.global_feat:
                if "max" in preprocess:
                    global_feats.append(
                        torch.max(x, dim=self.point_dim, keepdim=False)[0]
                    )
                elif preprocess in ["avg", "mean"]:
                    global_feats.append(
                        torch.mean(x, dim=self.point_dim, keepdim=False)
                    )
            x = torch.cat(global_feats, dim=1)
        x = self.pointnet.encoder.forward_cls_feat(x)
        if self.color_pointnet is not None:
            x = torch.cat([x, color_x], dim=1)
        if self.mlp is not None:
            x = self.mlp(self.activation(x))
        return x


class PerActVoxelEncoder(nn.Module):
    """PerAct's Voxel Encoder"""

    def __init__(
        self,
        batch_size: int,
        observation_horizon: int,
        pcd_channels: int,
        coordinate_bounds: list = [
            BOUND[0],
            BOUND[3],
            BOUND[1],
            BOUND[4],
            BOUND[2],
            BOUND[5],
        ],
        out_channels: int = 512,
        voxel_size: int = 100,
        voxel_patch_size: int = 9,
        voxel_patch_stride: int = 8,
        low_dim_size: int = 0,
        num_cameras: int = 3,
        image_resolution: list = [640, 480],
        input_axis=3,  # 3D tensors have 3 axes
        **kwargs,
    ):
        from gensim2.agent.models.peract import (
            Conv3DBlock,
            SpatialSoftmax3D,
            DenseBlock,
            VoxelGrid,
        )

        super().__init__()
        self.in_channels = pcd_channels + 4  # 3 index, 1 occupied
        self.out_channels = (
            out_channels if low_dim_size == 0 else out_channels // 2
        )  # if proprioception is used, we have to concat the proprioception feature
        self.low_dim_size = low_dim_size
        self.voxel_patch_size = voxel_patch_size
        self.voxel_patch_stride = voxel_patch_stride
        self.input_axis = input_axis

        self.voxelizer = VoxelGrid(
            coord_bounds=coordinate_bounds,
            voxel_size=voxel_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
            batch_size=batch_size * observation_horizon,
            feature_size=pcd_channels - 3,  # remove the xyz
            max_num_coords=np.prod(image_resolution) * num_cameras,
        )

        # voxel input preprocessing 1x1 conv encoder
        self.input_preprocess = Conv3DBlock(
            self.in_channels,
            self.out_channels,
            kernel_sizes=1,
            strides=1,
            norm=None,
            activation="relu",
        )

        # patchify conv
        self.patchify = Conv3DBlock(
            self.input_preprocess.out_channels,
            self.out_channels,
            kernel_sizes=self.voxel_patch_size,
            strides=self.voxel_patch_stride,
            norm=None,
            activation="relu",
        )

        # proprioception
        if self.low_dim_size > 0:
            self.proprio_preprocess = DenseBlock(
                self.low_dim_size,
                self.out_channels,
                norm=None,
                activation="relu",
            )

    def forward(
        self,
        pcd,
        proprio=None,  # state
    ):
        pcd = pcd["x"].swapaxes(1, 2)  # B, C, N
        pcd, coord_features = pcd[..., :3], pcd[..., 3:]  # [B, N, 3], [B, N, ?]
        # construct voxel grid
        voxel_grid = self.voxelizer.coords_to_bounding_voxel_grid(
            pcd, coord_features=coord_features, coord_bounds=None
        )

        # swap to channels fist
        voxel_grid = voxel_grid.permute(0, 4, 1, 2, 3).detach()

        # preprocess input
        d0 = self.input_preprocess(
            voxel_grid
        )  # [B,feature_size+7,100,100,100] -> [B,self.out_channel,100,100,100]

        # patchify input (5x5x5 patches)
        ins = self.patchify(
            d0
        )  # [B,self.out_channel,100,100,100] -> [B,self.out_channel,20,20,20]

        b, c, d, h, w, device = *ins.shape, ins.device
        axis = [d, h, w]
        assert (
            len(axis) == self.input_axis
        ), "input must have the same number of axis as input_axis"

        # concat proprio
        if self.low_dim_size > 0:
            p = self.proprio_preprocess(proprio)  # [B,4] -> [B,self.out_channel]
            p = p.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, d, h, w)
            ins = torch.cat([ins, p], dim=1)  # [B,self.out_channel,13,13,13]

        # channel last
        ins = rearrange(ins, "b d ... -> b ... d")  # [B,13,13,13,self.out_channel]

        return ins.reshape(b, -1, c)


class DP3PointNetEncoderXYZ(nn.Module):
    """DP3's Encoder for Pointcloud"""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 512,
        use_layernorm: bool = True,
        final_norm: str = "none",
        use_projection: bool = True,
        input_key="x",
        **kwargs,
    ):
        """_summary_

        Args:
            in_channels (int): feature size of input [3, 4, 5, 7, 8].
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        self.in_channels = in_channels
        self.input_key = input_key

        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), "cyan")
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), "cyan")

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )

        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels), nn.LayerNorm(out_channels)
            )
        elif final_norm == "none":
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")

        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)

    def forward(self, x):
        """
        x: dict(['colors', 'pos', 'seg', 'heights', 'x']), B x N x D
        """
        if self.input_key != "x":
            x = x["input_key"]
        else:
            if self.in_channels == 3:
                x = x["pos"]
            else:
                x = x["x"].swapaxes(1, 2)
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x


# class TextEncoder(nn.Module):
#     def __init__(self, pretrained_model_name_or_path, modality_embed_dim, finetune=False, **kwargs):
#         from sentence_transformers import SentenceTransformer

#         super(TextEncoder, self).__init__()
#         self.model = SentenceTransformer(pretrained_model_name_or_path)  # emb_dim: 384
#         if not finetune:
#             self.model.requires_grad_(False)  # Freeze the model

#         self.mlp = nn.Linear(384, modality_embed_dim)

#     def forward(self, x):
#         emb = self.model.encode(x, show_progress_bar=False, convert_to_tensor=True)
#         emb = self.mlp(emb)
#         return torch.unsqueeze(emb, 1)


class TextEncoder(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path,
        modality_embed_dim,
        finetune=False,
        **kwargs,
    ):

        super(TextEncoder, self).__init__()

        model, _ = load_clip("RN50", jit=False)
        self._clip_rn50 = build_model(model.state_dict())
        self._clip_rn50 = self._clip_rn50.float()
        self._clip_rn50.eval()
        del model

    def forward(self, x):
        tokens = tokenize(x).numpy()
        token_tensor = torch.from_numpy(tokens).to(next(self.parameters()).device)
        with torch.no_grad():
            text_feat, text_emb = self._clip_rn50.encode_text_with_embeddings(
                token_tensor
            )

        text_feat = text_feat.detach()
        text_emb = text_emb.detach()

        text_mask = torch.where(
            token_tensor == 0, token_tensor, 1
        )  # [1, max_token_len]

        return text_emb


class PolicyStem(nn.Module):
    """policy stem for encoders"""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return self.network(x)

    def freeze(self):
        for param in self.network.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.network.parameters():
            param.requires_grad = True

    def save(self, path):
        torch.save(self.state_dict(), path)

    def reset_latent(self):
        # TODO: figure out how to do sequential model
        pass

    @property
    def device(self):
        return next(self.parameters()).device
