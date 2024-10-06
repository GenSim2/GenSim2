# Code source from Jiayuan Gu: https://github.com/Jiayuan-Gu/torkit3d
import torch
import torch.nn as nn

from ..common.mlp import mlp1d_bn_relu, mlp_bn_relu, mlp_relu, mlp1d_relu
import logging

__all__ = [
    "PointNet",
    "PointNet_v1",
    "PointNet_v2",
    "PointNet_v3",
    "PointNet_v4",
    "PointNet_v22",
    "PointNet_v23",
    "PointNet_v5",
    "PointNet_v52",
    "PointNet_v53",
    "PointNet_v54",
]

# NOTE: add new versions of PointNet in the __all__ list above and common/torch_layers.py


class PointNet(nn.Module):
    """PointNet for classification.
    Notes:
        1. The original implementation includes dropout for global MLPs.
        2. The original implementation decays the BN momentum.
    """

    def __init__(
        self,
        in_channels=3,
        local_channels=(64, 64, 64, 128, 1024),
        global_channels=(512, 256),
        use_bn=True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = (local_channels + global_channels)[-1]
        self.use_bn = use_bn
        print("PointNet_original: use_bn:", use_bn)

        if use_bn:
            self.mlp_local = mlp1d_bn_relu(in_channels, local_channels)
            self.mlp_global = mlp_bn_relu(local_channels[-1], global_channels)
        else:
            self.mlp_local = mlp1d_relu(in_channels, local_channels)
            self.mlp_global = mlp_relu(local_channels[-1], global_channels)

        self.reset_parameters()

    def forward(self, points, points_feature=None, points_mask=None) -> dict:
        # points: [B, 3, N]; points_feature: [B, C, N], points_mask: [B, N]
        if points_feature is not None:
            input_feature = torch.cat([points, points_feature], dim=1)
        else:
            input_feature = points

        local_feature = self.mlp_local(input_feature)
        if points_mask is not None:
            local_feature = torch.where(
                points_mask.unsqueeze(1), local_feature, torch.zeros_like(local_feature)
            )
        global_feature, max_indices = torch.max(local_feature, 2)
        output_feature = self.mlp_global(global_feature)

        return {"feature": output_feature, "max_indices": max_indices}

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.momentum = 0.01


class PointNet_v1(nn.Module):
    """PointNet for classification.
    Notes:
        1. The original implementation includes dropout for global MLPs.
        2. The original implementation decays the BN momentum.
    """

    def __init__(
        self,
        in_channels=3,
        local_channels=(64, 64, 64, 128, 1024),
        global_channels=(512, 256),
        reduction_ratio=8,
        use_bn=True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = (local_channels + global_channels)[-1]
        self.use_bn = use_bn
        print("PointNet_v1: use_bn:", use_bn)
        if use_bn:
            self.mlp_local = mlp1d_bn_relu(in_channels, local_channels)
            self.mlp_global = mlp_bn_relu(local_channels[-1], global_channels)
        else:
            self.mlp_local = mlp1d_relu(in_channels, local_channels)
            self.mlp_global = mlp_relu(local_channels[-1], global_channels)

        self.excitation = nn.Sequential(
            nn.Linear(
                local_channels[-1], max(int(local_channels[-1] / reduction_ratio), 4)
            ),
            nn.ReLU(inplace=False),
            nn.Linear(
                max(int(local_channels[-1] / reduction_ratio), 4), local_channels[-1]
            ),
            nn.Sigmoid(),
        )

        self.reset_parameters()

    def forward(self, points, points_feature=None, points_mask=None) -> dict:
        # points: [B, 3+4, N]; points_feature: [B, C, N], points_mask: [B, N] (hardcode)
        group_num = 4

        group_id = torch.argmax(
            points[:, 3:, :], dim=1
        )  # e.g. tensor([[0, 1, 3, 2, 0], [0, 1, 3, 2, 0]]), [B, N]

        valid_group_ids = []
        for i in range(group_num):
            if (group_id == i).any():
                valid_group_ids.append(i)

        if points_feature is not None:
            input_feature = torch.cat([points, points_feature], dim=1)
        else:
            input_feature = points

        local_feature = self.mlp_local(input_feature)

        local_feature_u = torch.permute(local_feature, [0, 2, 1])  # [B, N, C] C=6
        # ic(local_feature_u)

        # v1: here we group points by 4 masks, and average the local features within each group. (group_id == i)
        # Squeeze
        squeezed_features_z = torch.zeros((group_num, local_feature_u.shape[2])).to(
            local_feature_u.device
        )  # [4, C]
        activations_s = torch.zeros((group_num, local_feature_u.shape[2])).to(
            local_feature_u.device
        )  # [4, C]
        local_feature_u1 = torch.zeros(local_feature_u.shape).to(
            local_feature_u.device
        )  # [4, N, C] dont do this inplace.
        for i in valid_group_ids:
            # Excitation
            squeezed_features_z[i] = torch.mean(
                local_feature_u[group_id == i], dim=0
            )  # 4 groups with feature [C]
            # TODO: should not mean among the batch here.
            # ic(squeezed_features_zi)
        for i in valid_group_ids:
            activations_s[i] = self.excitation(
                squeezed_features_z[i]
            )  # 4 groups with activation [C]
        # channel wise multiplication
        for i in valid_group_ids:
            local_feature_u1[group_id == i] = (
                local_feature_u[group_id == i] * activations_s[i]
            )

        local_feature_u2 = torch.permute(local_feature_u1, [0, 2, 1])  # [B, C, N]

        # ic(local_feature_u.shape)
        global_feature, max_indices = torch.max(local_feature_u2, 2)
        # ic(global_feature.shape)
        output_feature = self.mlp_global(global_feature)
        # ic(output_feature.shape)
        # ic(max_indices.shape)

        return {"feature": output_feature, "max_indices": max_indices}

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.momentum = 0.01


class PointNet_v2(nn.Module):
    """PointNet for classification.
    Notes:
        1. The original implementation includes dropout for global MLPs.
        2. The original implementation decays the BN momentum.
    """

    def __init__(
        self,
        in_channels=3,
        local_channels=(64, 64, 64, 128, 1024),
        global_channels=(512, 256),
        reduction_ratio=8,
        use_bn=True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = (local_channels + global_channels)[-1]
        self.use_bn = use_bn
        print("PointNet_v2: use_bn:", use_bn)
        if use_bn:
            self.mlp_local = mlp1d_bn_relu(in_channels, local_channels)
            self.mlp_global = mlp_bn_relu(local_channels[-1], global_channels)
        else:
            self.mlp_local = mlp1d_relu(in_channels, local_channels)
            self.mlp_global = mlp_relu(local_channels[-1], global_channels)

        self.excitation = nn.Sequential(  # [1, C] -> [1, C]
            nn.Linear(
                local_channels[-1], max(int(local_channels[-1] / reduction_ratio), 4)
            ),
            nn.ReLU(inplace=False),
            nn.Linear(
                max(int(local_channels[-1] / reduction_ratio), 4), local_channels[-1]
            ),
            nn.Softmax(),
        )

        self.group_excitation = nn.Sequential(  # [4,* C] => [4] # no B
            nn.Linear(local_channels[-1] * 4, 32),
            nn.ReLU(inplace=False),
            nn.Linear(32, 32),
            nn.ReLU(inplace=False),
            nn.Linear(32, 4),
            nn.Softmax(),
        )

        self.reset_parameters()

    def forward(self, points, points_feature=None) -> dict:
        # points: [B, 3+4, N]; points_feature: [B, C, N], points_mask: [B, N] (hardcode)
        group_num = 4

        group_id = torch.argmax(
            points_feature, dim=1
        )  # e.g. tensor([[0, 1, 3, 2, 0], [0, 1, 3, 2, 0]]), [B, N]

        valid_group_ids = []
        for i in range(group_num):
            if (group_id == i).any():
                valid_group_ids.append(i)

        if points_feature is not None:
            input_feature = torch.cat([points, points_feature], dim=1)
        else:
            input_feature = points

        local_feature = self.mlp_local(input_feature)

        local_feature_u = torch.permute(local_feature, [0, 2, 1])  # [B, N, C] C=6

        # v1: here we group points by 4 masks, and average the local features within each group. (group_id == i)
        # Squeeze
        squeezed_features_z = torch.zeros((group_num, local_feature_u.shape[2])).to(
            local_feature_u.device
        )  # [4, C]
        activations_s = torch.zeros((group_num, local_feature_u.shape[2])).to(
            local_feature_u.device
        )  # [4, C]
        local_feature_u1 = torch.zeros(local_feature_u.shape).to(
            local_feature_u.device
        )  # [4, N, C] dont do this inplace.
        for i in valid_group_ids:
            # Excitation
            squeezed_features_z[i] = torch.mean(
                local_feature_u[group_id == i], dim=0
            )  # 4 groups with feature [C]

        for i in valid_group_ids:
            activations_s[i] = self.excitation(
                squeezed_features_z[i]
            )  # 4 groups with activation [C]

        # group wise activation
        activations_g = self.group_excitation(
            squeezed_features_z.reshape(-1)
        )  # [4] (not [B, 4])

        # channel & group wise multiplication
        for i in valid_group_ids:
            local_feature_u1[group_id == i] = (
                local_feature_u[group_id == i] * activations_s[i] * activations_g[i]
            )

        local_feature_u2 = torch.permute(local_feature_u1, [0, 2, 1])  # [B, C, N]

        global_feature, max_indices = torch.max(local_feature_u2, 2)

        output_feature = self.mlp_global(global_feature)

        return {"feature": output_feature, "max_indices": max_indices}

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.momentum = 0.01
