import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open3d as o3d

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **kwargs,
        )

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        # permute to NCHW
        x = x.permute(0, 3, 1, 2)
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


def load_r3m(modelid):
    import os
    from os.path import expanduser
    import omegaconf
    import hydra
    import gdown
    import torch
    import copy

    VALID_ARGS = [
        "_target_",
        "device",
        "lr",
        "hidden_dim",
        "size",
        "l2weight",
        "l1weight",
        "langweight",
        "tcnweight",
        "l2dist",
        "bs",
    ]
    device = "cuda:0"

    def cleanup_config(cfg):
        config = copy.deepcopy(cfg)
        keys = config.agent.keys()
        for key in list(keys):
            if key not in VALID_ARGS:
                del config.agent[key]
        config.agent["_target_"] = "r3m.R3M"
        config["device"] = device

        ## Hardcodes to remove the language head
        ## Assumes downstream use is as visual representation
        config.agent["langweight"] = 0
        return config.agent

    def remove_language_head(state_dict):
        keys = state_dict.keys()
        ## Hardcodes to remove the language head
        ## Assumes downstream use is as visual representation
        for key in list(keys):
            if ("lang_enc" in key) or ("lang_rew" in key):
                del state_dict[key]
        return state_dict

    home = os.path.join(expanduser("~"), ".r3m")
    if modelid == "res50":
        foldername = "r3m_50"
        modelurl = "https://drive.google.com/uc?id=1Xu0ssuG0N1zjZS54wmWzJ7-nb0-7XzbA"
        configurl = "https://drive.google.com/uc?id=10jY2VxrrhfOdNPmsFdES568hjjIoBJx8"
    elif modelid == "res18":
        foldername = "r3m_18"
        modelurl = "https://drive.google.com/uc?id=1A1ic-p4KtYlKXdXHcV2QV0cUzI4kn0u-"
        configurl = "https://drive.google.com/uc?id=1nitbHQ-GRorxc7vMUiEHjHWP5N11Jvc6"
    else:
        raise NameError("Invalid Model ID")

    if not os.path.exists(os.path.join(home, foldername)):
        os.makedirs(os.path.join(home, foldername))
    modelpath = os.path.join(home, foldername, "model.pt")
    configpath = os.path.join(home, foldername, "config.yaml")
    if not os.path.exists(modelpath):
        gdown.download(modelurl, modelpath, quiet=False)
        gdown.download(configurl, configpath, quiet=False)

    modelcfg = omegaconf.OmegaConf.load(configpath)
    cleancfg = cleanup_config(modelcfg)
    rep = hydra.utils.instantiate(cleancfg)
    rep = torch.nn.DataParallel(rep)
    r3m_state_dict = remove_language_head(
        torch.load(modelpath, map_location=torch.device(device))["r3m"]
    )
    rep.load_state_dict(r3m_state_dict)
    return rep


class ResNet18(nn.Module):
    def __init__(self, output_channel, pretrain) -> None:
        super(ResNet18, self).__init__()
        from torchvision.models import resnet18, ResNet18_Weights

        if not pretrain:
            self.vision_extractor = resnet18()
            self.fc = nn.Linear(1000, output_channel)
        elif pretrain == "IMAGENET1K":
            self.vision_extractor = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.vision_extractor.eval()
            self.fc = nn.Linear(1000, output_channel)
        elif pretrain == "R3M":
            self.vision_extractor = load_r3m("res18")
            self.vision_extractor.eval()
            self.fc = nn.Linear(512, output_channel).to("cuda")
        else:
            raise NotImplementedError

    def forward(self, x):
        # x: B x 224 x 224 x 3
        x = torch.permute(x, (0, 3, 1, 2))  # x: B x 3 x 224 x 224
        out = self.vision_extractor(x)
        out = self.fc(out)
        return out


class ResNet50(nn.Module):
    def __init__(self, output_channel, pretrain=None) -> None:
        super(ResNet50, self).__init__()
        from torchvision.models import resnet50, ResNet50_Weights

        if not pretrain:
            self.vision_extractor = resnet50()
            self.fc = nn.Linear(1000, output_channel)
        elif pretrain == "IMAGENET1K":
            self.vision_extractor = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.vision_extractor.eval()
            self.fc = nn.Linear(1000, output_channel)
        elif pretrain == "R3M":
            self.vision_extractor = load_r3m("res50")
            self.vision_extractor.eval()
            self.fc = nn.Linear(2048, output_channel).to("cuda")
        else:
            raise NotImplementedError

    def forward(self, x):
        # x: B x 224 x 224 x 3
        x = torch.permute(x, (0, 3, 1, 2))  # x: B x 3 x 224 x 224
        out = self.vision_extractor(x)
        out = self.fc(out)
        return out


class PointNet(nn.Module):  # actually pointnet
    def __init__(self, point_channel=3, output_dim=256):
        # NOTE: we require the output dim to be 256, in order to match the pretrained weights
        super(PointNet, self).__init__()

        print(f"PointNetSmall")

        in_channel = point_channel
        mlp_out_dim = 256
        self.local_mlp = nn.Sequential(
            nn.Linear(in_channel, 64),
            nn.GELU(),
            nn.Linear(64, mlp_out_dim),
        )
        self.reset_parameters_()

    def reset_parameters_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: [B, N, 3]
        """
        # pc = x[0].cpu().detach().numpy()
        # Local
        x = self.local_mlp(x)
        # gloabal max pooling
        x = torch.max(x, dim=1)[0]
        # x, indices = torch.max(x, dim=1)
        # # save the point cloud and high light the max points
        # # ic(x.shape)
        # # pc = x[0].cpu().detach().numpy()
        # ic(pc.shape)
        # pc = pc.reshape(-1, 3)
        #
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pc)
        # # paint the key points red, others blue
        # colors = np.zeros((pc.shape[0], 3))
        # colors[indices[0].cpu().detach().numpy()] = [10, 10, 10]
        #
        # pcd.colors = o3d.utility.Vector3dVector(colors)
        #
        #
        # # view
        # # o3d.visualization.draw_geometries([pcd])
        # o3d.io.write_point_cloud("/home/helin/Code/output/test.ply", pcd)
        #
        # exit()
        return x


class PointNetMedium(nn.Module):  # actually pointnet
    def __init__(self, point_channel=3, output_dim=256):
        # NOTE: we require the output dim to be 256, in order to match the pretrained weights
        super(PointNetMedium, self).__init__()

        print(f"PointNetMedium")

        in_channel = point_channel
        mlp_out_dim = 256
        self.local_mlp = nn.Sequential(
            nn.Linear(in_channel, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, mlp_out_dim),
        )
        self.reset_parameters_()

    def reset_parameters_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: [B, N, 3]
        """
        # Local
        x = self.local_mlp(x)
        # gloabal max pooling
        x = torch.max(x, dim=1)[0]
        return x


class PointNetLarge(nn.Module):  # actually pointnet
    def __init__(self, point_channel=3, output_dim=256):
        # NOTE: we require the output dim to be 256, in order to match the pretrained weights
        super(PointNetLarge, self).__init__()

        print(f"PointNetLarge")

        in_channel = point_channel
        mlp_out_dim = 256
        self.local_mlp = nn.Sequential(
            nn.Linear(in_channel, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, mlp_out_dim),
        )

        self.reset_parameters_()

    def reset_parameters_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: [B, N, 3]
        """
        # Local
        x = self.local_mlp(x)
        # gloabal max pooling
        x = torch.max(x, dim=1)[0]
        return x


class PointNetLargeHalf(nn.Module):  # actually pointnet
    def __init__(self, point_channel=3, output_dim=256):
        # NOTE: we require the output dim to be 256, in order to match the pretrained weights
        super(PointNetLargeHalf, self).__init__()

        print(f"PointNetLargeHalf")

        in_channel = point_channel
        mlp_out_dim = 256
        self.local_mlp = nn.Sequential(
            nn.Linear(in_channel, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, mlp_out_dim),
            # nn.GELU(),
            # nn.Linear(128, 256),
            # nn.GELU(),
            # nn.Linear(256, mlp_out_dim),
        )

        self.reset_parameters_()

    def reset_parameters_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: [B, N, 3]
        """
        # Local
        x = self.local_mlp(x)
        # gloabal max pooling
        x = torch.max(x, dim=1)[0]
        return x


if __name__ == "__main__":
    b = 2
    img = torch.zeros(size=(b, 64, 64, 3))

    extractor = ResNet50(output_channel=256)

    out = extractor(img)
    print(out.shape)
