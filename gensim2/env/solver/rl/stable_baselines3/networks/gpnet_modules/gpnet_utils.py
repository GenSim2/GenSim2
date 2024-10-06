import torch
import torch.nn as nn

# from icecream import ic, install
# install()
# ic.configureOutput(includeContext=True, contextAbsPath=True, prefix='File ')


class MlpMixerBlock(nn.Module):
    def __init__(self, channel_dim=512, group_dim=4):
        super(MlpMixerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(channel_dim)  # TODO: check this.
        self.norm2 = nn.LayerNorm(channel_dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(group_dim, group_dim),
            nn.GELU(),
            nn.Linear(group_dim, group_dim),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(channel_dim, channel_dim),
            nn.GELU(),
            nn.Linear(channel_dim, channel_dim),
        )

    def forward(self, x):
        x = x + self.mlp1(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        x = x + self.mlp2(self.norm2(x))
        return x


class AttnEncoderBlock(nn.Module):
    def __init__(self, emb_dim=128, in_channel=3):
        super(AttnEncoderBlock, self).__init__()
        self.k_proj_weight = nn.Parameter(torch.empty(in_channel, emb_dim))

        self.norm1 = nn.LayerNorm(in_channel)
        self.norm2 = nn.LayerNorm(emb_dim)

        self.reset_parameters_()

    def reset_parameters_(self):
        nn.init.trunc_normal_(self.k_proj_weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, query, key, value, require_attn=False):
        """
        query: gt, group tokens. [B, M=4, C=emb_dim]
        key: point_feats, [B, N=512, C0=in_channel]
        value: point_feats, [B, N=512, C0=in_channel]
        """
        # Norm
        key = self.norm1(key)
        query = self.norm2(query)
        key = torch.einsum(
            "bnc,ck->bkn", key, self.k_proj_weight
        )  # [B, N, C0] x [C0, C] -> [B, C, N]
        # [B, M, C0] x [B, C0, N] -> [B, M, N]
        attn = torch.einsum("bmc,bcn->bmn", query, key)
        attn = attn.softmax(dim=1)  # [B, M, N] [B, 4, 512]
        # [B, M, N] x [B, N, C0] -> [B, M, C0]
        out = torch.einsum("bmn,bnc->bmc", attn, value)
        if require_attn:
            return out, attn
        return out


class AttnDecoderBlock(nn.Module):
    def __init__(self, emb_dim=128, in_channel=3):
        super(AttnDecoderBlock, self).__init__()
        self.k_proj_weight = nn.Parameter(torch.empty(emb_dim, in_channel))
        self.v_proj_weight = nn.Parameter(torch.empty(emb_dim, in_channel))
        # self.q_proj_weight = nn.Parameter(torch.empty(in_channel, in_channel))  # Here we use I.

        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(in_channel)

        self.reset_parameters_()

    def reset_parameters_(self):
        nn.init.trunc_normal_(self.k_proj_weight, std=0.02)
        nn.init.trunc_normal_(self.v_proj_weight, std=0.02)
        # nn.init.trunc_normal_(self.q_proj_weight, std=.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, query, key, value):
        """
        query: point_feats, [B, N=512, C0=in_channel]
        key: gt, group tokens. [B, M=4, C=emb_dim]
        value: gt, group tokens. [B, M=4, C=emb_dim]
        """
        # Norm
        key = self.norm1(key)
        query = self.norm2(query)
        key = torch.einsum("bnc,ck->bkn", key, self.k_proj_weight)
        value = torch.einsum("bnc,ck->bnk", value, self.v_proj_weight)
        # query = torch.einsum('bnc,ck->bnk', query, self.q_proj_weight)
        attn = torch.einsum("bnc,bcm->bnm", query, key)
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bnm,bmc->bnc", attn, value)
        return out


if __name__ == "__main__":
    mlp_mixer_block = MlpMixerBlock(channel_dim=512, group_dim=4)
    x = torch.randn(1, 4, 512)
    print(mlp_mixer_block(x).shape)

    attn_encoder_block = AttnEncoderBlock(emb_dim=128, in_channel=3)
    query = torch.randn(1, 4, 128)
    key = torch.randn(1, 512, 3)
    value = torch.randn(1, 512, 3)
    print(attn_encoder_block(query, key, value).shape)

    attn_decoder_block = AttnDecoderBlock(emb_dim=128, in_channel=3)
    query = torch.randn(1, 512, 3)
    key = torch.randn(1, 4, 128)
    value = torch.randn(1, 4, 128)
    print(attn_decoder_block(query, key, value).shape)

    # test backward
    x = torch.randn(1, 4, 512, requires_grad=True)
    y = mlp_mixer_block(x)
    y.sum().backward()
    print(x.grad)

    # test backward
    query = torch.randn(1, 4, 128, requires_grad=True)
    key = torch.randn(1, 512, 3)
    value = torch.randn(1, 512, 3)
    y = attn_encoder_block(query, key, value)
    y.sum().backward()
    print(query.grad)

    # test backward
    query = torch.randn(1, 512, 3, requires_grad=True)
    key = torch.randn(1, 4, 128)
    value = torch.randn(1, 4, 128)
    y = attn_decoder_block(query, key, value)
    y.sum().backward()
    print(query.grad)
