import torch
import torch.nn as nn
import torch.nn.functional as F

from .gpnet_utils import AttnDecoderBlock, AttnEncoderBlock, MlpMixerBlock

# from icecream import ic, install
# install()
# ic.configureOutput(includeContext=True, contextAbsPath=True, prefix='File ')


class GPNet_load2(nn.Module):
    def __init__(
        self,
        emb_dim=128,
        point_channel=3,
        group_dim=4,
        out_channel=256,
        pointnet_only=False,
    ):
        super(GPNet_load2, self).__init__()

        print(f"GPNet_load 2 pn only = {pointnet_only}")

        in_channel = point_channel
        mlp_out_dim = 256
        self.pointnet_only = pointnet_only
        self.local_mlp = nn.Sequential(
            nn.Linear(in_channel, 64),
            nn.GELU(),
            nn.Linear(64, mlp_out_dim),
        )

        # if not pointnet_only:
        self.encoder = AttnEncoderBlock(emb_dim=emb_dim, in_channel=mlp_out_dim)
        self.mixer = MlpMixerBlock(channel_dim=mlp_out_dim, group_dim=group_dim)
        self.decoder = AttnDecoderBlock(emb_dim=mlp_out_dim, in_channel=mlp_out_dim)
        self.group_tokens = nn.Parameter(torch.randn(1, group_dim, emb_dim))

        # self.norm1 = nn.LayerNorm(mlp_out_dim)
        self.norm2 = nn.LayerNorm(mlp_out_dim)

        self.reset_parameters_()

    def reset_parameters_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.norm2.weight)
        nn.init.zeros_(self.norm2.bias)

    def forward(self, x):
        """
        x: [B, 3, N]
        """
        x = x.permute(0, 2, 1)
        # Localckpionts/
        x = self.local_mlp(x)
        local_feats = x
        if not self.pointnet_only:
            # Encoder
            x = self.encoder(self.group_tokens, x, x)
            # Mixer
            x = self.mixer(x)
            # Decoder
            x = local_feats + self.norm2(
                self.decoder(local_feats, x, x)
            )  # 1011: add layer norms, larger local feats
        # Global
        # max pooling
        x = x.max(dim=1)[0]
        # x = self.global_mlp(x)  # 1015: remove global mlp, allow freezing the model and use the pretrained weights

        return x


class GPNet_load1(nn.Module):
    def __init__(
        self,
        emb_dim=128,
        point_channel=3,
        group_dim=4,
        out_channel=128,
        pointnet_only=False,
    ):
        super(GPNet_load1, self).__init__()

        print(f"GPNet_load 1 pn only = {pointnet_only}")

        in_channel = point_channel
        mlp_out_dim = 256
        self.pointnet_only = pointnet_only
        self.local_mlp = nn.Sequential(
            nn.Linear(in_channel, 64),
            nn.GELU(),
            nn.Linear(64, mlp_out_dim),
        )

        # if not pointnet_only:
        self.encoder = AttnEncoderBlock(emb_dim=emb_dim, in_channel=mlp_out_dim)
        self.mixer = MlpMixerBlock(channel_dim=mlp_out_dim, group_dim=group_dim)
        self.decoder = AttnDecoderBlock(emb_dim=mlp_out_dim, in_channel=mlp_out_dim)
        self.group_tokens = nn.Parameter(torch.randn(1, group_dim, emb_dim))

        # self.norm1 = nn.LayerNorm(mlp_out_dim)
        self.norm2 = nn.LayerNorm(mlp_out_dim)

        # self.output_mlp = nn.Sequential(
        #     nn.Linear(mlp_out_dim * 2, 128),
        #     nn.GELU(),
        #     nn.Linear(128, classes),
        # )

        self.global_mlp = nn.Sequential(
            nn.Linear(mlp_out_dim, 128),
            nn.GELU(),
            nn.Linear(128, out_channel),
        )
        self.reset_parameters_()

    def reset_parameters_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.norm2.weight)
        nn.init.zeros_(self.norm2.bias)

    def forward(self, x, feats):
        """
        x: [B, 3, N]
        """
        x = x.permute(0, 2, 1)
        # Localckpionts/
        x = self.local_mlp(x)
        local_feats = x
        if not self.pointnet_only:
            # Encoder
            x = self.encoder(self.group_tokens, x, x)
            # Mixer
            x = self.mixer(x)
            # Decoder
            x = local_feats + self.norm2(
                self.decoder(local_feats, x, x)
            )  # 1011: add layer norms, larger local feats
        # Global
        # max pooling
        x = x.max(dim=1)[0]
        x = self.global_mlp(x)

        return x


class GPNet(nn.Module):
    def __init__(
        self, emb_dim=128, group_dim=4, out_channel=128, pointnet_only=False, option=1
    ):
        super(GPNet, self).__init__()

        print(
            f"Using GPNet 1. PointNet only == {pointnet_only}, option = {option}. 1003"
        )

        self.option = option

        in_channel = 7
        mlp_out_dim = 256
        self.pointnet_only = pointnet_only
        if option == 1:
            self.local_mlp = nn.Sequential(
                nn.Linear(in_channel, 64),
                nn.GELU(),
                nn.Linear(64, mlp_out_dim),
            )
            self.global_mlp = nn.Sequential(
                nn.Linear(mlp_out_dim, 128),
                nn.GELU(),
                nn.Linear(128, out_channel),
            )
        elif option == 2:
            # compared to option 1, this one has more layers.
            self.local_mlp = nn.Sequential(
                nn.Linear(in_channel, 64),
                nn.GELU(),
                nn.Linear(64, 128),
                nn.GELU(),
                nn.Linear(128, mlp_out_dim),
            )
            self.global_mlp = nn.Sequential(
                nn.Linear(mlp_out_dim, 128),
                nn.GELU(),
                nn.Linear(128, 128),
                nn.GELU(),
                nn.Linear(128, out_channel),
            )
        if not pointnet_only:
            self.encoder = AttnEncoderBlock(emb_dim=emb_dim, in_channel=mlp_out_dim)
            self.mixer = MlpMixerBlock(channel_dim=mlp_out_dim, group_dim=group_dim)
            self.decoder = AttnDecoderBlock(emb_dim=mlp_out_dim, in_channel=mlp_out_dim)
            self.group_tokens = nn.Parameter(torch.randn(1, group_dim, emb_dim))
            if option == 2:
                # add another set
                self.encoder2 = AttnEncoderBlock(
                    emb_dim=emb_dim, in_channel=mlp_out_dim
                )
                self.mixer2 = MlpMixerBlock(
                    channel_dim=mlp_out_dim, group_dim=group_dim
                )
                self.decoder2 = AttnDecoderBlock(
                    emb_dim=mlp_out_dim, in_channel=mlp_out_dim
                )
                self.group_tokens2 = nn.Parameter(torch.randn(1, group_dim, emb_dim))

        self.reset_parameters_()

    def reset_parameters_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, x_label):
        """
        x: [B, 3, N]
        x_label: [B, 4, N]
        """
        # Local
        x = torch.cat([x, x_label], dim=1).transpose(1, 2)  # B * (N + sum(Ni)) * 7
        x = self.local_mlp(x)
        if not self.pointnet_only:
            local_feats = x  # TODO: is this really duplicated?
            # Encoder
            x = self.encoder(self.group_tokens, x, x)
            # Mixer
            x = self.mixer(x)
            # Decoder
            x = local_feats + self.decoder(local_feats, x, x)
            if self.option == 2:
                local_feats2 = x
                # add another set
                x = self.encoder2(self.group_tokens2 + self.group_tokens, x, x)
                x = self.mixer2(x)
                x = local_feats2 + self.decoder2(local_feats, x, x)
        # Global
        # max pooling
        x = x.max(dim=1)[0]
        x = self.global_mlp(x)
        return x


class GPNet_v1(nn.Module):
    """
    1006: GPNet_v1 adds layer normalization before residual connections.
    """

    def __init__(
        self, emb_dim=128, group_dim=4, out_channel=128, pointnet_only=False, option=1
    ):
        super(GPNet_v1, self).__init__()

        print(
            f"Using GPNet 1. PointNet only == {pointnet_only}, option = {option}. 1003"
        )

        self.option = option

        in_channel = 7
        mlp_out_dim = 256
        self.pointnet_only = pointnet_only
        if option == 1:
            self.local_mlp = nn.Sequential(
                nn.Linear(in_channel, 64),
                nn.GELU(),
                nn.Linear(64, mlp_out_dim),
            )
            self.global_mlp = nn.Sequential(
                nn.Linear(mlp_out_dim, 128),
                nn.GELU(),
                nn.Linear(128, out_channel),
            )
        elif option == 2:
            # compared to option 1, this one has more layers.
            self.local_mlp = nn.Sequential(
                nn.Linear(in_channel, 64),
                nn.GELU(),
                nn.Linear(64, 128),
                nn.GELU(),
                nn.Linear(128, mlp_out_dim),
            )
            self.global_mlp = nn.Sequential(
                nn.Linear(mlp_out_dim, 128),
                nn.GELU(),
                nn.Linear(128, 128),
                nn.GELU(),
                nn.Linear(128, out_channel),
            )
        if not pointnet_only:
            self.encoder = AttnEncoderBlock(emb_dim=emb_dim, in_channel=mlp_out_dim)
            self.mixer = MlpMixerBlock(channel_dim=mlp_out_dim, group_dim=group_dim)
            self.decoder = AttnDecoderBlock(emb_dim=mlp_out_dim, in_channel=mlp_out_dim)
            self.group_tokens = nn.Parameter(torch.randn(1, group_dim, emb_dim))
            if option == 2:
                # add another set
                self.encoder2 = AttnEncoderBlock(
                    emb_dim=emb_dim, in_channel=mlp_out_dim
                )
                self.mixer2 = MlpMixerBlock(
                    channel_dim=mlp_out_dim, group_dim=group_dim
                )
                self.decoder2 = AttnDecoderBlock(
                    emb_dim=mlp_out_dim, in_channel=mlp_out_dim
                )
                self.group_tokens2 = nn.Parameter(torch.randn(1, group_dim, emb_dim))

        self.norm1 = nn.LayerNorm(mlp_out_dim)
        self.norm2 = nn.LayerNorm(mlp_out_dim)
        self.reset_parameters_()

    def reset_parameters_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, x_label):
        """
        x: [B, 3, N]
        x_label: [B, 4, N]
        """
        # Local
        x = torch.cat([x, x_label], dim=1).transpose(1, 2)  # B * (N + sum(Ni)) * 7
        x = self.local_mlp(x)
        if not self.pointnet_only:
            local_feats = x  # TODO: is this really duplicated?
            # Encoder
            x = self.encoder(self.group_tokens, x, x)
            # Mixer
            x = self.mixer(x)
            # Decoder
            x = self.norm1(local_feats) + self.norm2(
                self.decoder(local_feats, x, x)
            )  # 1006: add layer norms
            if self.option == 2:
                local_feats2 = x
                # add another set
                x = self.encoder2(self.group_tokens2 + self.group_tokens, x, x)
                x = self.mixer2(x)
                x = local_feats2 + self.decoder2(local_feats, x, x)
        # Global
        # max pooling
        x = x.max(dim=1)[0]
        x = self.global_mlp(x)
        return x


class GPNet_v2(nn.Module):
    """
    1011: GPNet_v2 adds layer normalization only one. before residual connections.
    """

    def __init__(
        self, emb_dim=128, group_dim=4, out_channel=128, pointnet_only=False, option=1
    ):
        super(GPNet_v2, self).__init__()

        print(
            f"Using GPNet 2. PointNet only == {pointnet_only}, option = {option}. 1011"
        )

        self.option = option

        in_channel = 7
        mlp_out_dim = 256
        self.pointnet_only = pointnet_only
        if option == 1:
            self.local_mlp = nn.Sequential(
                nn.Linear(in_channel, 64),
                nn.GELU(),
                nn.Linear(64, mlp_out_dim),
            )
            self.global_mlp = nn.Sequential(
                nn.Linear(mlp_out_dim, 128),
                nn.GELU(),
                nn.Linear(128, out_channel),
            )
        elif option == 2:
            # compared to option 1, this one has more layers.
            self.local_mlp = nn.Sequential(
                nn.Linear(in_channel, 64),
                nn.GELU(),
                nn.Linear(64, 128),
                nn.GELU(),
                nn.Linear(128, mlp_out_dim),
            )
            self.global_mlp = nn.Sequential(
                nn.Linear(mlp_out_dim, 128),
                nn.GELU(),
                nn.Linear(128, 128),
                nn.GELU(),
                nn.Linear(128, out_channel),
            )
        if not pointnet_only:
            self.encoder = AttnEncoderBlock(emb_dim=emb_dim, in_channel=mlp_out_dim)
            self.mixer = MlpMixerBlock(channel_dim=mlp_out_dim, group_dim=group_dim)
            self.decoder = AttnDecoderBlock(emb_dim=mlp_out_dim, in_channel=mlp_out_dim)
            self.group_tokens = nn.Parameter(torch.randn(1, group_dim, emb_dim))
            if option == 2:
                # add another set
                self.encoder2 = AttnEncoderBlock(
                    emb_dim=emb_dim, in_channel=mlp_out_dim
                )
                self.mixer2 = MlpMixerBlock(
                    channel_dim=mlp_out_dim, group_dim=group_dim
                )
                self.decoder2 = AttnDecoderBlock(
                    emb_dim=mlp_out_dim, in_channel=mlp_out_dim
                )
                self.group_tokens2 = nn.Parameter(torch.randn(1, group_dim, emb_dim))

        # self.norm1 = nn.LayerNorm(mlp_out_dim)
        self.norm2 = nn.LayerNorm(mlp_out_dim)
        self.reset_parameters_()

    def reset_parameters_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, x_label):
        """
        x: [B, 3, N]
        x_label: [B, 4, N]
        """
        # Local
        x = torch.cat([x, x_label], dim=1).transpose(1, 2)  # B * (N + sum(Ni)) * 7
        x = self.local_mlp(x)
        if not self.pointnet_only:
            local_feats = x  # TODO: is this really duplicated?
            # Encoder
            x = self.encoder(self.group_tokens, x, x)
            # Mixer
            x = self.mixer(x)
            # Decoder
            x = local_feats + self.norm2(
                self.decoder(local_feats, x, x)
            )  # 1011: add layer norms
            if self.option == 2:
                local_feats2 = x
                # add another set
                x = self.encoder2(self.group_tokens2 + self.group_tokens, x, x)
                x = self.mixer2(x)
                x = local_feats2 + self.decoder2(local_feats, x, x)
        # Global
        # max pooling
        x = x.max(dim=1)[0]
        x = self.global_mlp(x)
        return x


class GPNet_v3(nn.Module):
    """
    1011: GPNet_v3 use larger local feats. compared to v2.
    """

    def __init__(
        self, emb_dim=128, group_dim=4, out_channel=128, pointnet_only=False, option=1
    ):
        super(GPNet_v3, self).__init__()

        print(
            f"Using GPNet 3. PointNet only == {pointnet_only}, option = {option}. 1011"
        )

        self.option = option

        in_channel = 7
        mlp_out_dim = 256
        self.pointnet_only = pointnet_only
        if option == 1:
            self.local_mlp = nn.Sequential(
                nn.Linear(in_channel, 64),
                nn.GELU(),
                nn.Linear(64, mlp_out_dim),
            )
            self.global_mlp = nn.Sequential(
                nn.Linear(mlp_out_dim, 128),
                nn.GELU(),
                nn.Linear(128, out_channel),
            )
        elif option == 2:
            # compared to option 1, this one has more layers.
            self.local_mlp = nn.Sequential(
                nn.Linear(in_channel, 64),
                nn.GELU(),
                nn.Linear(64, 128),
                nn.GELU(),
                nn.Linear(128, mlp_out_dim),
            )
            self.global_mlp = nn.Sequential(
                nn.Linear(mlp_out_dim, 128),
                nn.GELU(),
                nn.Linear(128, 128),
                nn.GELU(),
                nn.Linear(128, out_channel),
            )
        if not pointnet_only:
            self.encoder = AttnEncoderBlock(emb_dim=emb_dim, in_channel=mlp_out_dim)
            self.mixer = MlpMixerBlock(channel_dim=mlp_out_dim, group_dim=group_dim)
            self.decoder = AttnDecoderBlock(emb_dim=mlp_out_dim, in_channel=mlp_out_dim)
            self.group_tokens = nn.Parameter(torch.randn(1, group_dim, emb_dim))
            if option == 2:
                # add another set
                self.encoder2 = AttnEncoderBlock(
                    emb_dim=emb_dim, in_channel=mlp_out_dim
                )
                self.mixer2 = MlpMixerBlock(
                    channel_dim=mlp_out_dim, group_dim=group_dim
                )
                self.decoder2 = AttnDecoderBlock(
                    emb_dim=mlp_out_dim, in_channel=mlp_out_dim
                )
                self.group_tokens2 = nn.Parameter(torch.randn(1, group_dim, emb_dim))

        # self.norm1 = nn.LayerNorm(mlp_out_dim)
        self.norm2 = nn.LayerNorm(mlp_out_dim)
        self.reset_parameters_()

    def reset_parameters_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, x_label):
        """
        x: [B, 3, N]
        x_label: [B, 4, N]
        """
        # Local
        x = torch.cat([x, x_label], dim=1).transpose(1, 2)  # B * (N + sum(Ni)) * 7
        x = self.local_mlp(x)
        if not self.pointnet_only:
            local_feats = x  # TODO: is this really duplicated?
            # Encoder
            x = self.encoder(self.group_tokens, x, x)
            # Mixer
            x = self.mixer(x)
            # Decoder
            x = local_feats * 32 + self.norm2(
                self.decoder(local_feats, x, x)
            )  # 1011: add layer norms, larger local feats
            if self.option == 2:
                local_feats2 = x
                # add another set
                x = self.encoder2(self.group_tokens2 + self.group_tokens, x, x)
                x = self.mixer2(x)
                x = local_feats2 + self.decoder2(local_feats, x, x)
        # Global
        # max pooling
        x = x.max(dim=1)[0]
        x = self.global_mlp(x)
        return x


class GPNet_v4(nn.Module):
    """
    1011: use concat instead of add.
    """

    def __init__(
        self, emb_dim=128, group_dim=4, out_channel=128, pointnet_only=False, option=1
    ):
        super(GPNet_v4, self).__init__()

        assert not pointnet_only, "pointnet_only is not supported in v4."

        print(
            f"Using GPNet 4. PointNet only == {pointnet_only}, option = {option}. 1011"
        )

        self.option = option

        in_channel = 7
        mlp_out_dim = 256
        self.pointnet_only = pointnet_only
        if option == 1:
            self.local_mlp = nn.Sequential(
                nn.Linear(in_channel, 64),
                nn.GELU(),
                nn.Linear(64, mlp_out_dim),
            )
            self.global_mlp1 = nn.Sequential(
                nn.Linear(mlp_out_dim, 128),
                nn.GELU(),
                nn.Linear(128, int(out_channel / 2)),
            )
            self.global_mlp2 = nn.Sequential(
                nn.Linear(mlp_out_dim, 128),
                nn.GELU(),
                nn.Linear(128, int(out_channel / 2)),
            )
        if not pointnet_only:
            self.encoder = AttnEncoderBlock(emb_dim=emb_dim, in_channel=mlp_out_dim)
            self.mixer = MlpMixerBlock(channel_dim=mlp_out_dim, group_dim=group_dim)
            self.decoder = AttnDecoderBlock(emb_dim=mlp_out_dim, in_channel=mlp_out_dim)
            self.group_tokens = nn.Parameter(torch.randn(1, group_dim, emb_dim))

        # self.norm1 = nn.LayerNorm(mlp_out_dim)
        self.norm2 = nn.LayerNorm(mlp_out_dim)
        self.reset_parameters_()

    def reset_parameters_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, x_label):
        """
        x: [B, 3, N]
        x_label: [B, 4, N]
        """
        # Local
        x = torch.cat([x, x_label], dim=1).transpose(1, 2)  # B * (N + sum(Ni)) * 7
        x = self.local_mlp(x)
        out1 = self.global_mlp1(x.max(dim=1)[0])
        local_feats = x
        # Encoder
        x = self.encoder(self.group_tokens, x, x)
        # Mixer
        x = self.mixer(x)
        # Decoder
        x = self.norm2(self.decoder(local_feats, x, x))  # 1011: add layer norms
        # Global
        # max pooling
        x = x.max(dim=1)[0]
        out2 = self.global_mlp2(x)
        return torch.cat([out1, out2], dim=1)


class GPNet_v5(nn.Module):
    """
    1011: use concat instead of add.
    """

    def __init__(
        self, emb_dim=128, group_dim=4, out_channel=128, pointnet_only=False, option=1
    ):
        super(GPNet_v5, self).__init__()

        assert not pointnet_only, "pointnet_only is not supported in v5."

        print(
            f"Using GPNet 5. PointNet only == {pointnet_only}, option = {option}. 1011"
        )

        self.option = option

        in_channel = 7
        mlp_out_dim = 256
        self.pointnet_only = pointnet_only
        if option == 1:
            self.local_mlp = nn.Sequential(
                nn.Linear(in_channel, 64),
                nn.GELU(),
                nn.Linear(64, mlp_out_dim),
            )
            self.global_mlp = nn.Sequential(
                nn.Linear(mlp_out_dim * 2, 128),  # v5: similar to v4
                nn.GELU(),
                nn.Linear(128, out_channel),
            )
        if not pointnet_only:
            self.encoder = AttnEncoderBlock(emb_dim=emb_dim, in_channel=mlp_out_dim)
            self.mixer = MlpMixerBlock(channel_dim=mlp_out_dim, group_dim=group_dim)
            self.decoder = AttnDecoderBlock(emb_dim=mlp_out_dim, in_channel=mlp_out_dim)
            self.group_tokens = nn.Parameter(torch.randn(1, group_dim, emb_dim))

        # self.norm1 = nn.LayerNorm(mlp_out_dim)
        self.norm2 = nn.LayerNorm(mlp_out_dim)
        self.reset_parameters_()

    def reset_parameters_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, x_label):
        """
        x: [B, 3, N]
        x_label: [B, 4, N]
        """
        # Local
        x = torch.cat([x, x_label], dim=1).transpose(1, 2)  # B * (N + sum(Ni)) * 7
        x = self.local_mlp(x)
        x1 = x.max(dim=1)[0]
        local_feats = x
        # Encoder
        x = self.encoder(self.group_tokens, x, x)
        # Mixer
        x = self.mixer(x)
        # Decoder
        x = self.norm2(self.decoder(local_feats, x, x))  # 1011: add layer norms
        # Global
        # max pooling
        x2 = x.max(dim=1)[0]
        out = self.global_mlp(torch.cat([x1, x2], dim=1))
        return out


class SegGPNet(nn.Module):
    def __init__(
        self, emb_dim=128, group_dim=4, out_channel=128, pointnet_only=False, option=1
    ):
        super(SegGPNet, self).__init__()

        assert option == 1, "Only option 1 is supported for SegGPNet."
        assert group_dim == 4, "Only group_dim == 4 is supported for SegGPNet."

        print(f"Using SegGPNet 1. PointNet only == {pointnet_only}, option = {option}.")

        self.option = option

        mlp_out_dim = 256
        in_channel = 3
        self.pointnet_only = pointnet_only
        if option == 1:
            self.local_mlp0 = nn.Sequential(
                nn.Linear(in_channel, 64),
                nn.GELU(),
                nn.Linear(64, mlp_out_dim),
            )
            self.local_mlp1 = nn.Sequential(
                nn.Linear(in_channel, 64),
                nn.GELU(),
                nn.Linear(64, mlp_out_dim),
            )
            self.local_mlp2 = nn.Sequential(
                nn.Linear(in_channel, 64),
                nn.GELU(),
                nn.Linear(64, mlp_out_dim),
            )
            self.local_mlp3 = nn.Sequential(
                nn.Linear(in_channel, 64),
                nn.GELU(),
                nn.Linear(64, mlp_out_dim),
            )
            self.global_mlp = nn.Sequential(
                nn.Linear(mlp_out_dim, 128),
                nn.GELU(),
                nn.Linear(128, out_channel),
            )
        if not pointnet_only:
            # self.encoder = AttnEncoderBlock(emb_dim=emb_dim, in_channel=mlp_out_dim)
            self.mixer = MlpMixerBlock(channel_dim=mlp_out_dim, group_dim=group_dim)
            self.decoder = AttnDecoderBlock(emb_dim=mlp_out_dim, in_channel=mlp_out_dim)

        self.reset_parameters_()

    def reset_parameters_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, x_label):
        """
        x: [B, 3, N]
        x_label: [B, 4, N]
        """
        x = x.transpose(1, 2)  # B * N * 3
        x_label = x_label.transpose(1, 2)  # B * N * 4
        # Local
        # turn soft label into one hot
        x_label = F.one_hot(x_label.argmax(dim=-1), num_classes=4).float()

        label_id = torch.argmax(x_label, dim=2)

        x0 = self.local_mlp0(x)  # B, N, 256
        x0 = torch.einsum("bnc,bn->bnc", x0, x_label[:, :, 0])
        x0_ = x0.max(dim=1)[0]  # TODO: <0 cases?

        x1 = self.local_mlp1(x)  # B, N, 256
        x1 = torch.einsum("bnc,bn->bnc", x1, x_label[:, :, 1])
        x1_ = x1.max(dim=1)[0]  # TODO: <0 cases?

        x2 = self.local_mlp2(x)  # B, N, 256
        x2 = torch.einsum("bnc,bn->bnc", x2, x_label[:, :, 2])
        x2_ = x2.max(dim=1)[0]  # TODO: <0 cases?

        x3 = self.local_mlp3(x)  # B, N, 256
        x3 = torch.einsum("bnc,bn->bnc", x3, x_label[:, :, 3])
        x3_ = x3.max(dim=1)[0]  # TODO: <0 cases?

        local_feats = x0 + x1 + x2 + x3  # tested. B, N, 256

        emb = torch.stack([x0_, x1_, x2_, x3_], dim=1)  # B, 4, 256

        if not self.pointnet_only:
            # Mixer
            x = self.mixer(emb)
            # Decoder
            x = local_feats + self.decoder(local_feats, x, x)
            # Global
            # max pooling
            x = x.max(dim=1)[0]
            x = self.global_mlp(x)
        else:
            x = local_feats.max(dim=1)[0]
            x = self.global_mlp(x)

        return x


class SegGPNet_v1(nn.Module):
    """
    1006: add layer norms
    """

    def __init__(
        self, emb_dim=128, group_dim=4, out_channel=128, pointnet_only=False, option=1
    ):
        super(SegGPNet_v1, self).__init__()

        assert option == 1, "Only option 1 is supported for SegGPNet."
        assert group_dim == 4, "Only group_dim == 4 is supported for SegGPNet."

        print(f"Using SegGPNet 1. PointNet only == {pointnet_only}, option = {option}.")

        self.option = option

        mlp_out_dim = 256
        in_channel = 3
        self.pointnet_only = pointnet_only
        if option == 1:
            self.local_mlp0 = nn.Sequential(
                nn.Linear(in_channel, 64),
                nn.GELU(),
                nn.Linear(64, mlp_out_dim),
            )
            self.local_mlp1 = nn.Sequential(
                nn.Linear(in_channel, 64),
                nn.GELU(),
                nn.Linear(64, mlp_out_dim),
            )
            self.local_mlp2 = nn.Sequential(
                nn.Linear(in_channel, 64),
                nn.GELU(),
                nn.Linear(64, mlp_out_dim),
            )
            self.local_mlp3 = nn.Sequential(
                nn.Linear(in_channel, 64),
                nn.GELU(),
                nn.Linear(64, mlp_out_dim),
            )
            self.global_mlp = nn.Sequential(
                nn.Linear(mlp_out_dim, 128),
                nn.GELU(),
                nn.Linear(128, out_channel),
            )
        if not pointnet_only:
            # self.encoder = AttnEncoderBlock(emb_dim=emb_dim, in_channel=mlp_out_dim)
            self.mixer = MlpMixerBlock(channel_dim=mlp_out_dim, group_dim=group_dim)
            self.decoder = AttnDecoderBlock(emb_dim=mlp_out_dim, in_channel=mlp_out_dim)

        self.norm1 = nn.LayerNorm(mlp_out_dim)  # 1006: add layer norms
        self.norm2 = nn.LayerNorm(mlp_out_dim)
        self.reset_parameters_()

    def reset_parameters_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, x_label):
        """
        x: [B, 3, N]
        x_label: [B, 4, N]
        """
        x = x.transpose(1, 2)  # B * N * 3
        x_label = x_label.transpose(1, 2)  # B * N * 4
        # Local
        # turn soft label into one hot
        x_label = F.one_hot(x_label.argmax(dim=-1), num_classes=4).float()

        label_id = torch.argmax(x_label, dim=2)

        x0 = self.local_mlp0(x)  # B, N, 256
        x0 = torch.einsum("bnc,bn->bnc", x0, x_label[:, :, 0])
        x0_ = x0.max(dim=1)[0]  # TODO: <0 cases?

        x1 = self.local_mlp1(x)  # B, N, 256
        x1 = torch.einsum("bnc,bn->bnc", x1, x_label[:, :, 1])
        x1_ = x1.max(dim=1)[0]  # TODO: <0 cases?

        x2 = self.local_mlp2(x)  # B, N, 256
        x2 = torch.einsum("bnc,bn->bnc", x2, x_label[:, :, 2])
        x2_ = x2.max(dim=1)[0]  # TODO: <0 cases?

        x3 = self.local_mlp3(x)  # B, N, 256
        x3 = torch.einsum("bnc,bn->bnc", x3, x_label[:, :, 3])
        x3_ = x3.max(dim=1)[0]  # TODO: <0 cases?

        local_feats = x0 + x1 + x2 + x3  # tested. B, N, 256

        emb = torch.stack([x0_, x1_, x2_, x3_], dim=1)  # B, 4, 256

        if not self.pointnet_only:
            # Mixer
            x = self.mixer(emb)
            # Decoder
            x = self.norm1(local_feats) + self.norm2(
                self.decoder(local_feats, x, x)
            )  # 1006
            # Global
            # max pooling
            x = x.max(dim=1)[0]
            x = self.global_mlp(x)
        else:
            x = self.norm1(local_feats).max(dim=1)[0]  # 1006
            x = self.global_mlp(x)

        return x


class SegGPNet_v2(nn.Module):
    """
    1006: use only one local mlp head.
    """

    def __init__(
        self, emb_dim=128, group_dim=4, out_channel=128, pointnet_only=False, option=1
    ):
        super(SegGPNet_v2, self).__init__()

        assert option == 1, "Only option 1 is supported for SegGPNet."
        assert group_dim == 4, "Only group_dim == 4 is supported for SegGPNet."

        print(f"Using SegGPNet 1. PointNet only == {pointnet_only}, option = {option}.")

        self.option = option

        mlp_out_dim = 256
        in_channel = 3
        self.pointnet_only = pointnet_only
        if option == 1:
            self.local_mlp = nn.Sequential(
                nn.Linear(in_channel, 64),
                nn.GELU(),
                nn.Linear(64, mlp_out_dim),
            )
            self.global_mlp = nn.Sequential(
                nn.Linear(mlp_out_dim, 128),
                nn.GELU(),
                nn.Linear(128, out_channel),
            )
        if not pointnet_only:
            # self.encoder = AttnEncoderBlock(emb_dim=emb_dim, in_channel=mlp_out_dim)
            self.mixer = MlpMixerBlock(channel_dim=mlp_out_dim, group_dim=group_dim)
            self.decoder = AttnDecoderBlock(emb_dim=mlp_out_dim, in_channel=mlp_out_dim)

        self.reset_parameters_()

    def reset_parameters_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, x_label):
        """
        x: [B, 3, N]
        x_label: [B, 4, N]
        """
        x = x.transpose(1, 2)  # B * N * 3
        x_label = x_label.transpose(1, 2)  # B * N * 4
        # Local
        # turn soft label into one hot
        x_label = F.one_hot(x_label.argmax(dim=-1), num_classes=4).float()

        x = self.local_mlp(
            x
        )  # B, N, 256                         # 1006 v2: use only one local mlp
        x0 = torch.einsum("bnc,bn->bnc", x, x_label[:, :, 0])
        x0_ = x0.max(dim=1)[0]
        x1 = torch.einsum("bnc,bn->bnc", x, x_label[:, :, 1])
        x1_ = x1.max(dim=1)[0]
        x2 = torch.einsum("bnc,bn->bnc", x, x_label[:, :, 2])
        x2_ = x2.max(dim=1)[0]
        x3 = torch.einsum("bnc,bn->bnc", x, x_label[:, :, 3])
        x3_ = x3.max(dim=1)[0]

        local_feats = x0 + x1 + x2 + x3  # tested. B, N, 256

        ic(local_feats)

        emb = torch.stack([x0_, x1_, x2_, x3_], dim=1)  # B, 4, 256

        if not self.pointnet_only:
            # Mixer
            x = self.mixer(emb)
            # Decoder
            x = local_feats + self.decoder(local_feats, x, x)
            # Global
            # max pooling
            x = x.max(dim=1)[0]
            x = self.global_mlp(x)
        else:
            x = local_feats.max(dim=1)[0]
            x = self.global_mlp(x)

        return x


class SegGPNet_v3(nn.Module):
    """
    1006: use only one local mlp head, and add layer norm.
    """

    def __init__(
        self, emb_dim=128, group_dim=4, out_channel=128, pointnet_only=False, option=1
    ):
        super(SegGPNet_v3, self).__init__()

        assert option == 1, "Only option 1 is supported for SegGPNet."
        assert group_dim == 4, "Only group_dim == 4 is supported for SegGPNet."

        print(f"Using SegGPNet 1. PointNet only == {pointnet_only}, option = {option}.")

        self.option = option

        mlp_out_dim = 256
        in_channel = 3
        self.pointnet_only = pointnet_only
        if option == 1:
            self.local_mlp = nn.Sequential(
                nn.Linear(in_channel, 64),
                nn.GELU(),
                nn.Linear(64, mlp_out_dim),
            )
            self.global_mlp = nn.Sequential(
                nn.Linear(mlp_out_dim, 128),
                nn.GELU(),
                nn.Linear(128, out_channel),
            )
        if not pointnet_only:
            # self.encoder = AttnEncoderBlock(emb_dim=emb_dim, in_channel=mlp_out_dim)
            self.mixer = MlpMixerBlock(channel_dim=mlp_out_dim, group_dim=group_dim)
            self.decoder = AttnDecoderBlock(emb_dim=mlp_out_dim, in_channel=mlp_out_dim)

        self.norm1 = nn.LayerNorm(mlp_out_dim)
        self.norm2 = nn.LayerNorm(mlp_out_dim)
        self.reset_parameters_()

    def reset_parameters_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, x_label):
        """
        x: [B, 3, N]
        x_label: [B, 4, N]
        """
        x = x.transpose(1, 2)  # B * N * 3
        x_label = x_label.transpose(1, 2)  # B * N * 4
        # Local
        # turn soft label into one hot
        x_label = F.one_hot(x_label.argmax(dim=-1), num_classes=4).float()

        x = self.local_mlp(
            x
        )  # B, N, 256                         # v2: use only one local mlp
        x0 = torch.einsum("bnc,bn->bnc", x, x_label[:, :, 0])
        x0_ = x0.max(dim=1)[0]
        x1 = torch.einsum("bnc,bn->bnc", x, x_label[:, :, 1])
        x1_ = x1.max(dim=1)[0]
        x2 = torch.einsum("bnc,bn->bnc", x, x_label[:, :, 2])
        x2_ = x2.max(dim=1)[0]
        x3 = torch.einsum("bnc,bn->bnc", x, x_label[:, :, 3])
        x3_ = x3.max(dim=1)[0]

        local_feats = x0 + x1 + x2 + x3  # tested. B, N, 256

        emb = torch.stack([x0_, x1_, x2_, x3_], dim=1)  # B, 4, 256

        if not self.pointnet_only:
            # Mixer
            x = self.mixer(emb)
            # Decoder
            x = self.norm1(local_feats) + self.norm2(
                self.decoder(local_feats, x, x)
            )  # 1006 v3: add layer norm
            # Global
            # max pooling
            x = x.max(dim=1)[0]
            x = self.global_mlp(x)
        else:
            x = self.norm1(local_feats).max(dim=1)[0]  # 1006 v3: add layer norm
            x = self.global_mlp(x)

        return x


if __name__ == "__main__":
    x = torch.randn(2, 3, 101)
    x_label = torch.randn(2, 4, 101)
    model = SegGPNet_v3(
        pointnet_only=False, option=1, emb_dim=11, group_dim=4, out_channel=128
    )
    print(model)
    out = model(x, x_label)
    print(out.shape)

    # test backward
    loss = out.sum()
    loss.backward()

    x = torch.randn(2, 3, 101)
    x_label = torch.randn(2, 4, 101)
    model = GPNet_v1(
        pointnet_only=False, option=1, emb_dim=11, group_dim=4, out_channel=128
    )
    print(model)
    out = model(x, x_label)
    print(out.shape)

    # test backward
    loss = out.sum()
    loss.backward()
