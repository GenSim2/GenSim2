import torch.nn as nn
import torch
from torch import Tensor
from torch.nn import MultiheadAttention

from typing import Dict, Optional, Any
from collections import OrderedDict
import hydra

from .diffusion import DiffusionConditionalUnet1d
from gensim2.agent.utils.model_utils import *
from gensim2.agent.utils.utils import get_sinusoid_encoding_table
from gensim2.agent.models.peract import DenseBlock


class extract_tensor(nn.Module):
    def forward(self, x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        # Reshape shape (batch, hidden)
        return tensor


class QuickGELU(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: Tensor) -> Tensor:
        return input * torch.sigmoid(1.702 * input)


class residual_cross_attention(nn.Module):
    def __init__(self, output_dim, n_head=1, dropout=False):
        """single head attention"""
        super().__init__()
        self.attn = MultiheadAttention(output_dim, n_head, batch_first=True)
        self.ln_1 = nn.LayerNorm(output_dim)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(output_dim, output_dim)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(output_dim, output_dim)),
                ]
            )
        )
        self.ln_2 = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(p=0.2) if dropout else nn.Identity()

    def attention(self, x: torch.Tensor, y: torch.Tensor):
        """cross attention"""
        return self.attn(x, y, y)

    def forward(self, x, y, *args, **kwargs):
        attn, _ = self.attention(self.ln_1(x), self.ln_1(y))  # pre norm
        x = x + attn
        x = x + self.mlp(self.ln_2(x))
        x = self.dropout(x)
        return x


class residual_self_attention(nn.Module):
    def __init__(self, output_dim, n_head=1):
        """single head attention"""
        super().__init__()
        self.attn = MultiheadAttention(output_dim, n_head)
        self.ln_1 = nn.LayerNorm(output_dim)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(output_dim, output_dim)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(output_dim, output_dim)),
                ]
            )
        )
        self.ln_2 = nn.LayerNorm(output_dim)

    def attention(self, x: torch.Tensor):
        """self attention"""
        return self.attn(x, x, x)

    def forward(self, x, *args, **kwargs):
        # Output shape (batch, features, hidden)
        # Reshape shape (batch, hidden)
        attn = self.attention(self.ln_1(x))  # pre norm
        x = x + attn
        x = x + self.mlp(self.ln_2(x))  #
        return x


class MLP(nn.Module):
    def __init__(
        self,
        input_dim=10,
        output_dim=10,
        widths=[512],
        dropout=False,
        tanh_end=False,
        ln=True,
        **kwargs,
    ):
        """vanilla MLP head"""
        super().__init__()

        modules = [nn.Linear(input_dim, widths[0]), nn.ReLU()]

        for i in range(len(widths) - 1):
            modules.extend([nn.Linear(widths[i], widths[i + 1])])
            if dropout:
                modules.append(nn.Dropout(p=0.2))
            if ln:
                modules.append(nn.LayerNorm(widths[i + 1]))
            modules.append(nn.ReLU())

        modules.append(nn.Linear(widths[-1], output_dim))
        if tanh_end:
            modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)

    def forward(self, x, *args, **kwargs):
        y = self.net(x)
        return y


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim=10,
        output_dim=10,
        widths=[512],
        n_head=4,
        dropout=False,
        tanh_end=False,
        ln=False,
        **kwargs,
    ):
        """vanilla transformer head"""
        super().__init__()

        modules = [nn.Linear(input_dim, widths[0]), nn.ReLU()]

        for i in range(len(widths) - 1):
            # width must be fixed
            module = residual_self_attention(widths[i + 1], n_head=n_head)
            modules.extend([module])
            if dropout:
                modules.append(nn.Dropout(p=0.2))

        modules.append(nn.Linear(widths[-1], output_dim))
        if tanh_end:
            modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)

    def forward(self, x, *args, **kwargs):
        y = self.net(x)
        return y


class TransformerDecoder(nn.Module):
    """
    Transformer decoder, doing cross-attention of empty token with the input sequence
    """

    def __init__(
        self,
        token_dim,
        horizon,
        output_dim,
        num_layers=3,
        n_head=4,
        dropout=False,
        tanh_end=False,
        ln=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ):
        """vanilla transformer head"""
        super().__init__()
        self.token_dim = token_dim
        self.seq_len = horizon

        self.attns = []
        modules = []

        for i in range(num_layers):
            # width must be fixed
            attn = residual_cross_attention(token_dim, n_head=n_head, dropout=dropout)
            attn.to(device)
            self.attns.append(attn)

        modules.append(nn.Linear(token_dim, output_dim))
        if tanh_end:
            modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)

    def forward(self, y, *args, **kwargs):
        # x as positional encoding
        # x = torch.zeros(y.shape[0], self.seq_len, self.token_dim, device=y.device)
        positional_embedding = get_sinusoid_encoding_table(
            0, self.seq_len, self.token_dim
        )
        positional_embedding = positional_embedding.repeat((y.shape[0], 1, 1)).to(
            y.device
        )
        x = positional_embedding

        for attn in self.attns:
            y = attn(x, y)
        y = self.net(y)
        return y

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        for attn in self.attns:
            attn.to(*args, **kwargs)
        return self


class RNN(nn.Module):
    def __init__(
        self,
        input_dim=10,
        output_dim=10,
        widths=[512],
        dropout=False,
        tanh_end=False,
        ln=False,
        **kwargs,
    ):
        """vanilla RNN head"""
        super().__init__()

        modules = [nn.Linear(input_dim, widths[0]), nn.ReLU()]

        for i in range(len(widths) - 1):
            rnn = nn.RNN(
                widths[i],
                widths[i + 1],
                1,
                dropout=0.2 if dropout else 0,
                batch_first=True,
            )
            modules.extend([rnn, extract_tensor()])
            if ln:
                modules.append(nn.LayerNorm(widths[i + 1]))

        modules.append(nn.Linear(widths[-1], output_dim))
        if tanh_end:
            modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)

    def forward(self, x, *args, **kwargs):
        y = self.net(x)
        return y


class Diffusion(nn.Module):
    def __init__(
        self,
        input_dim,  # condition
        output_dim,  # action dim
        horizon,  # number of steps to predict
        noise_scheduler_type: str = "DDPM",
        num_train_timesteps: int = 100,
        beta_schedule: str = "squaredcos_cap_v2",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        prediction_type: str = "epsilon",
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        num_inference_steps=None,
        diffusion_step_embed_dim: int = 128,
        down_dims=(512, 1024, 2048),
        kernel_size: int = 5,
        n_groups: int = 8,
        use_film_scale_modulation: bool = True,
        do_mask_loss_for_padding: bool = True,
        **kwargs,
    ):
        from .diffusion import _make_noise_scheduler

        """Diffusion policy head"""
        super().__init__()

        self.prediction_type = prediction_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.do_mask_loss_for_padding = do_mask_loss_for_padding

        self.unet = DiffusionConditionalUnet1d(
            output_dim=output_dim,
            global_cond_dim=input_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            use_film_scale_modulation=use_film_scale_modulation,
        )

        self.noise_scheduler = _make_noise_scheduler(
            noise_scheduler_type,
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            clip_sample=clip_sample,
            clip_sample_range=clip_sample_range,
            prediction_type=prediction_type,
        )

        if num_inference_steps is None:
            self.num_inference_steps = self.noise_scheduler.num_train_timesteps
        else:
            self.num_inference_steps = num_inference_steps

    # ========= inference  ============
    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        device = next(iter(self.parameters())).device
        dtype = next(iter(self.parameters())).dtype

        # Sample prior.
        sample = torch.randn(
            size=(batch_size, self.horizon, self.output_dim),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            # Predict model output.
            model_output = self.unet(
                sample,
                torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
                global_cond=global_cond,
            )
            # Compute previous sample: x_t -> x_t-1
            sample = self.noise_scheduler.step(
                model_output, t, sample, generator=generator
            ).prev_sample
        return sample

    def generate_actions(self, x) -> Tensor:
        """
        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)
            "observation.image": (B, n_obs_steps, C, H, W)
        }
        """
        batch_size = x.shape[0]
        # run sampling
        samples = self.conditional_sample(batch_size, global_cond=x)

        # Currently all future aactions
        # # `horizon` steps worth of actions (from the first observation).
        # actions = sample[..., : self.output_shapes["action"][0]]
        # # Extract `n_action_steps` steps worth of actions (from the current observation).
        # start = n_obs_steps - 1
        # end = start + self.n_action_steps
        # actions = actions[:, start:end]

        return samples

    def compute_loss(self, x, target, **kwargs) -> Tensor:
        trajectory = target

        # Forward diffusion.
        # Sample noise to add to the trajectory.
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        # Sample a random noising timestep for each item in the batch.
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        # Add noise to the clean trajectories according to the noise magnitude at each timestep.
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)

        # Run the denoising network (that might denoise the trajectory, or attempt to predict the noise).
        pred = self.unet(noisy_trajectory, timesteps, global_cond=x)

        # Compute the loss.
        # The target is either the original trajectory, or the noise.
        if self.prediction_type == "epsilon":
            target = eps
        elif self.prediction_type == "sample":
            pass  # =target
        else:
            raise ValueError(f"Unsupported prediction type {self.prediction_type}")

        loss = F.mse_loss(pred, target, reduction="none")

        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if (
            self.do_mask_loss_for_padding
            and ("action_is_pad" in kwargs)
            and (kwargs["action_is_pad"] is not None)
        ):
            in_episode_bound = ~kwargs["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        return loss.mean()

    def forward(self, x: Tensor, target: Optional[Tensor] = None, **kwargs):
        if target is None:
            return self.generate_actions(x)

        """Run the batch through the model and compute the loss for training or validation."""
        loss = self.compute_loss(x, target)
        return {"loss": loss}


class QAttn(nn.Module):
    def __init__(
        self,
        feature_dim,
        voxel_size,
        num_rotation_classes=72,  # 5 degree increments (5*72=360) for each of the 3-axis,
        num_grip_classes=2,  # open or not open
        num_collision_classes=2,  # collisions allowed or not allowed
        final_dim=64,
    ):
        self.voxel_size = voxel_size
        self.num_rotation_classes = num_rotation_classes
        self.num_grip_classes = num_grip_classes
        self.num_collision_classes = num_collision_classes

        self.dense0 = DenseBlock(feature_dim, 256, None, "relu")
        self.dense1 = DenseBlock(256, final_dim, None, "relu")

        self.trans_ff = DenseBlock(
            final_dim, voxel_size * voxel_size * voxel_size, None, None
        )
        self.rot_grip_collision_ff = DenseBlock(
            final_dim,
            self.num_rotation_classes * 3
            + self.num_grip_classes
            + self.num_collision_classes,
            None,
            None,
        )

    def forward(self, x):
        # translation decoder
        trans = self.trans_ff(x)

        # rotation, gripper, and collision MLPs
        dense0 = self.dense0(x)
        dense1 = self.dense1(dense0)

        rot_and_grip_collision_out = self.rot_grip_collision_ff(dense1)
        rot_and_grip_out = rot_and_grip_collision_out[:, : -self.num_collision_classes]
        collision_out = rot_and_grip_collision_out[:, -self.num_collision_classes :]

        return (
            trans.resize(-1, self.voxel_size, self.voxel_size, self.voxel_size),
            rot_and_grip_out,
            collision_out,
        )


class PolicyHead(nn.Module):
    """policy head"""

    def __init__(self, model_spec, **kwargs):
        super().__init__()
        self.network = hydra.utils.instantiate(model_spec)

    def forward(self, x, *args, **kwargs):
        return self.network(x)

    def freeze(self):
        for param in self.network.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.network.parameters():
            param.requires_grad = True

    def save(self, path):
        torch.save(self.state_dict(), path)

    @property
    def device(self):
        return next(self.parameters()).device
