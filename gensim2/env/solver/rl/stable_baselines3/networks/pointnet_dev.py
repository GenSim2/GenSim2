# Code source from Jiayuan Gu: https://github.com/Jiayuan-Gu/torkit3d

import torch
import torch.nn as nn

from common.mlp import mlp1d_bn_relu, mlp_bn_relu, mlp_relu, mlp1d_relu

from icecream import ic, install

install()
ic.configureOutput(includeContext=True, contextAbsPath=True, prefix="File ")

__all__ = ["PointNet", "PointNet_v1"]


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
        ic(global_feature)
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
            nn.Sigmoid(),
        )

        self.group_excitation = nn.Sequential(  # [4,* C] => [4] # no B
            nn.Linear(local_channels[-1] * 4, 32),
            nn.ReLU(inplace=False),
            nn.Linear(32, 32),
            nn.ReLU(inplace=False),
            nn.Linear(32, 4),
            nn.Sigmoid(),
        )

        self.reset_parameters()

    def forward(self, points, points_feature=None) -> dict:
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

        # channel&group wise multiplication
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


class PointNet_v3(nn.Module):
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
        print("PointNet_v3: use_bn:", use_bn)
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
            nn.Sigmoid(),
        )

        self.group_excitation = nn.Sequential(  # [4,* C] => [4] # no B
            nn.Linear(local_channels[-1] * 4, 32),
            nn.ReLU(inplace=False),
            nn.Linear(32, 32),
            nn.ReLU(inplace=False),
            nn.Linear(32, 4),
            nn.Sigmoid(),
        )

        self.reset_parameters()

    def forward(self, points, points_feature=None) -> dict:
        # points: [B, 3+4, N]; points_feature: [B, C, N], points_mask: [B, N] (hardcode)
        # handle_mask, faucet_mask, hand_mask, arm_mask
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

        # channel&group wise multiplication
        for i in valid_group_ids:
            if i in [1, 3]:  # v3: we manually set feature of faucet & arm to 0.
                continue
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


class PointNet_v4(nn.Module):
    """
    delete channel wise excitation, use max pooling within each group instead of avg pooling.
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

        assert global_channels[-1] % 2 == 0, "global_channels[-1] must be even"
        print("###################################")
        print("PointNet_v4: use_bn:", use_bn)
        print("global_channels:", global_channels)
        print("##################################")
        global_channels = list(global_channels)
        global_channels[-1] = global_channels[-1] // 2

        if use_bn:
            self.mlp_local = mlp1d_bn_relu(in_channels, local_channels)
            self.mlp_global = mlp_bn_relu(local_channels[-1], global_channels)
            self.mlp_group = mlp_bn_relu(local_channels[-1] * 4, global_channels)
        else:
            self.mlp_local = mlp1d_relu(in_channels, local_channels)
            self.mlp_global = mlp_relu(local_channels[-1], global_channels)
            self.mlp_group = mlp_relu(local_channels[-1] * 4, global_channels)

        self.reset_parameters()

    def forward(self, points, points_feature=None) -> dict:
        # points: [B, 3+4, N]; points_feature: [B, C, N], points_mask: [B, N] (hardcode)
        # handle_mask, faucet_mask, hand_mask, arm_mask
        group_num = 4

        group_id = torch.argmax(
            points[:, 3:, :], dim=1
        )  # e.g. tensor([[0, 1, 3, 2, 0], [0, 1, 3, 2, 0]]), [B, N]

        if points_feature is not None:
            input_feature = torch.cat([points, points_feature], dim=1)
        else:
            input_feature = points

        local_feature = self.mlp_local(input_feature)
        global_feature, max_indices = torch.max(local_feature, 2)  # [B, C, N]

        output_feature = self.mlp_global(global_feature)  # [B, C]

        # from v0 to v4: add group level feature extraction.
        local_feature_u = torch.permute(local_feature, [0, 2, 1])  # [B, N, C] C=6
        group_channel_feature = torch.zeros((group_num, local_feature_u.shape[2])).to(
            local_feature_u.device
        )  # [B, 4, C]
        for i in range(group_num):
            if (group_id == i).any():
                group_channel_feature[i] = torch.max(local_feature_u[group_id == i], 0)[
                    0
                ]
        # ic(group_channel_feature)  # TODO: can we make this [B, 4, C] & parallel?
        # ic(torch.max(local_feature, 2)[0])  # [B, C]
        ic(group_channel_feature.shape)
        group_output_feature = self.mlp_group(
            group_channel_feature.reshape(-1).repeat((local_feature.shape[0], 1))
        )  # [B, C]

        output = torch.cat((group_output_feature, output_feature), dim=1)  # [B, C]
        return {"feature": output, "max_indices": max_indices}

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.momentum = 0.01


import sys
import os

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../vision_dev")
)
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../vision_dev/models")
)

from data_utils import SemSegDataset
from models.pointnet_sem_seg import get_model


class PointNet_v5(nn.Module):
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
        in_channels = in_channels + 4  # add group id

        self.semsegnet = get_model(num_class=4, channel=3)
        ckpt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../vision_dev/checkpoints/model_693.pth",
        )
        self.semsegnet.load_state_dict(torch.load(ckpt_path)["model_state_dict"])
        self.semsegnet.eval()
        for p in self.parameters():
            p.requires_grad = False

        self.in_channels = in_channels
        self.out_channels = (local_channels + global_channels)[-1]
        self.use_bn = use_bn

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
            input = torch.cat([points, points_feature], dim=1)
        else:
            input = points

        ic(input.shape)
        input_seg, _ = self.semsegnet(input)
        input_seg = torch.transpose(input_seg, 1, 2)
        ic(input_seg.shape)  # [B, 4, N]
        ic(input.shape)
        input_feature = torch.cat([input, input_seg], dim=1)

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


if __name__ == "__main__":
    # import cProfile
    # import pstats
    # profiler = cProfile.Profile()
    # test PointNetSE
    net = PointNet_v5(in_channels=7, local_channels=(4, 8), global_channels=(512, 122))
    # net.reset_parameters()
    # # points: [B, 3+4, N]; points_feature: [B, C, N], points_mask: [B, N]
    points = torch.rand(10, 3, 512)
    # points = torch.tensor(
    #     [
    #         [
    #             [.2, .1, .3, 1., 0., 0., 0.],
    #             [-.2, .1, .3, 0., 1., 0., 0.],
    #             [.2, -.1, .3, 0., 0., 0., 1.],
    #             [-.2, .1, -.3, 0., 0., 1., 0.],
    #             [-.2, -.1, -.3, 1., 0., 0., 0.],
    #         ], [
    #             [.2, -.1, -.3, 1., 0., 0., 0.],
    #             [-.2, .1, .3, 0., 1., 0., 0.],
    #             [-.2, .1, .3, 0., 0., 0., 1.],
    #             [-.2, .1, -.3, 0., 0., 1., 0.],
    #             [-.2, -.1, -.3, 1., 0., 0., 0.],
    #         ],
    #     ]
    # )
    # ic(points.shape)
    # points = torch.permute(points, [0, 2, 1])
    # ic(points)
    # forward
    # profiler.enable()
    output = net(points)
    # profiler.disable()
    ic(output["feature"].shape)
    # profiler.print_stats(sort="time")
    # stats = pstats.Stats(profiler)
    # stats.dump_stats('profile.prof')

    # test loss backward
    torch.autograd.set_detect_anomaly(True)
    loss = torch.sum(output["feature"])
    loss.backward()

    # test process_faucet_pc_noise_seg

    # pc = torch.rand(10, 7, 512)
