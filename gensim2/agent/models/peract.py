import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union
from functools import reduce
from operator import mul

MIN_DENOMINATOR = 1e-12
INCLUDE_PER_VOXEL_COORD = False

LRELU_SLOPE = 0.02


def act_layer(act):
    if act == "relu":
        return nn.ReLU()
    elif act == "lrelu":
        return nn.LeakyReLU(LRELU_SLOPE)
    elif act == "elu":
        return nn.ELU()
    elif act == "tanh":
        return nn.Tanh()
    elif act == "prelu":
        return nn.PReLU()
    else:
        raise ValueError("%s not recognized." % act)


def norm_layer2d(norm, channels):
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    elif norm == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    elif norm == "layer":
        return nn.GroupNorm(1, channels, affine=True)
    elif norm == "group":
        return nn.GroupNorm(4, channels, affine=True)
    else:
        raise ValueError("%s not recognized." % norm)


def norm_layer1d(norm, num_channels):
    if norm == "batch":
        return nn.BatchNorm1d(num_channels)
    elif norm == "instance":
        return nn.InstanceNorm1d(num_channels, affine=True)
    elif norm == "layer":
        return nn.LayerNorm(num_channels)
    else:
        raise ValueError("%s not recognized." % norm)


class Conv3DBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes: Union[int, list] = 3,
        strides=1,
        norm=None,
        activation=None,
        padding_mode="replicate",
        padding=None,
    ):
        super(Conv3DBlock, self).__init__()
        padding = kernel_sizes // 2 if padding is None else padding
        self.conv3d = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_sizes,
            strides,
            padding=padding,
            padding_mode=padding_mode,
        )

        if activation is None:
            nn.init.xavier_uniform_(
                self.conv3d.weight, gain=nn.init.calculate_gain("linear")
            )
            nn.init.zeros_(self.conv3d.bias)
        elif activation == "tanh":
            nn.init.xavier_uniform_(
                self.conv3d.weight, gain=nn.init.calculate_gain("tanh")
            )
            nn.init.zeros_(self.conv3d.bias)
        elif activation == "lrelu":
            nn.init.kaiming_uniform_(
                self.conv3d.weight, a=LRELU_SLOPE, nonlinearity="leaky_relu"
            )
            nn.init.zeros_(self.conv3d.bias)
        elif activation == "relu":
            nn.init.kaiming_uniform_(self.conv3d.weight, nonlinearity="relu")
            nn.init.zeros_(self.conv3d.bias)
        else:
            raise ValueError()

        self.activation = None
        self.norm = None
        if norm is not None:
            raise NotImplementedError("Norm not implemented.")
        if activation is not None:
            self.activation = act_layer(activation)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv3d(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.activation(x) if self.activation is not None else x
        return x


class SpatialSoftmax3D(torch.nn.Module):
    def __init__(self, depth, height, width, channel):
        super(SpatialSoftmax3D, self).__init__()
        self.depth = depth
        self.height = height
        self.width = width
        self.channel = channel
        self.temperature = 0.01
        pos_x, pos_y, pos_z = np.meshgrid(
            np.linspace(-1.0, 1.0, self.depth),
            np.linspace(-1.0, 1.0, self.height),
            np.linspace(-1.0, 1.0, self.width),
        )
        pos_x = torch.from_numpy(
            pos_x.reshape(self.depth * self.height * self.width)
        ).float()
        pos_y = torch.from_numpy(
            pos_y.reshape(self.depth * self.height * self.width)
        ).float()
        pos_z = torch.from_numpy(
            pos_z.reshape(self.depth * self.height * self.width)
        ).float()
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)
        self.register_buffer("pos_z", pos_z)

    def forward(self, feature):
        feature = feature.view(
            -1, self.height * self.width * self.depth
        )  # (B, c*d*h*w)
        softmax_attention = F.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x * softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * softmax_attention, dim=1, keepdim=True)
        expected_z = torch.sum(self.pos_z * softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y, expected_z], 1)
        feature_keypoints = expected_xy.view(-1, self.channel * 3)
        return feature_keypoints


class VoxelGrid(nn.Module):
    # Voxelizer modified from ARM for DDP training
    # Source: https://github.com/stepjam/ARM
    # License: https://github.com/stepjam/ARM/LICENSE

    def __init__(
        self,
        coord_bounds,
        voxel_size: int,
        device,
        batch_size,
        feature_size,  # e.g. rgb or image features
        max_num_coords: int,
    ):
        super(VoxelGrid, self).__init__()
        self._device = device
        self._voxel_size = voxel_size
        self._voxel_shape = [voxel_size] * 3
        self._voxel_d = float(self._voxel_shape[-1])
        self._voxel_feature_size = 4 + feature_size
        self._voxel_shape_spec = (
            torch.tensor(
                self._voxel_shape,
            ).unsqueeze(0)
            + 2
        )  # +2 because we crop the edges.
        self._coord_bounds = torch.tensor(
            coord_bounds,
            dtype=torch.float,
        ).unsqueeze(0)
        max_dims = self._voxel_shape_spec[0]
        self._total_dims_list = torch.cat(
            [
                torch.tensor(
                    [batch_size],
                ),
                max_dims,
                torch.tensor(
                    [4 + feature_size],
                ),
            ],
            -1,
        ).tolist()

        self.register_buffer(
            "_ones_max_coords", torch.ones((batch_size, max_num_coords, 1))
        )
        self._num_coords = max_num_coords

        shape = self._total_dims_list
        result_dim_sizes = torch.tensor(
            [reduce(mul, shape[i + 1 :], 1) for i in range(len(shape) - 1)] + [1],
        )
        self.register_buffer("_result_dim_sizes", result_dim_sizes)
        flat_result_size = reduce(mul, shape, 1)

        self._initial_val = torch.tensor(0, dtype=torch.float)
        flat_output = (
            torch.ones(flat_result_size, dtype=torch.float) * self._initial_val
        )
        self.register_buffer("_flat_output", flat_output)

        self.register_buffer("_arange_to_max_coords", torch.arange(4 + feature_size))
        self._flat_zeros = torch.zeros(flat_result_size, dtype=torch.float)

        self._const_1 = torch.tensor(
            1.0,
        )
        self._batch_size = batch_size

        # Coordinate Bounds:
        bb_mins = self._coord_bounds[..., 0:3]
        self.register_buffer("_bb_mins", bb_mins)
        bb_maxs = self._coord_bounds[..., 3:6]
        bb_ranges = bb_maxs - bb_mins
        # get voxel dimensions. 'DIMS' mode
        self._dims = dims = self._voxel_shape_spec.int()
        dims_orig = self._voxel_shape_spec.int() - 2
        self.register_buffer("_dims_orig", dims_orig)

        # self._dims_m_one = (dims - 1).int()
        dims_m_one = (dims - 1).int()
        self.register_buffer("_dims_m_one", dims_m_one)

        # BS x 1 x 3
        res = bb_ranges / (dims_orig.float() + MIN_DENOMINATOR)
        self._res_minis_2 = bb_ranges / (dims.float() - 2 + MIN_DENOMINATOR)
        self.register_buffer("_res", res)

        voxel_indicy_denmominator = res + MIN_DENOMINATOR
        self.register_buffer("_voxel_indicy_denmominator", voxel_indicy_denmominator)

        self.register_buffer("_dims_m_one_zeros", torch.zeros_like(dims_m_one))

        batch_indices = torch.arange(self._batch_size, dtype=torch.int).view(
            self._batch_size, 1, 1
        )
        self.register_buffer(
            "_tiled_batch_indices", batch_indices.repeat([1, self._num_coords, 1])
        )

        w = self._voxel_shape[0] + 2
        arange = torch.arange(
            0,
            w,
            dtype=torch.float,
        )
        index_grid = (
            torch.cat(
                [
                    arange.view(w, 1, 1, 1).repeat([1, w, w, 1]),
                    arange.view(1, w, 1, 1).repeat([w, 1, w, 1]),
                    arange.view(1, 1, w, 1).repeat([w, w, 1, 1]),
                ],
                dim=-1,
            )
            .unsqueeze(0)
            .repeat([self._batch_size, 1, 1, 1, 1])
        )
        self.register_buffer("_index_grid", index_grid)

    def _broadcast(self, src: torch.Tensor, other: torch.Tensor, dim: int):
        if dim < 0:
            dim = other.dim() + dim
        if src.dim() == 1:
            for _ in range(0, dim):
                src = src.unsqueeze(0)
        for _ in range(src.dim(), other.dim()):
            src = src.unsqueeze(-1)
        src = src.expand_as(other)
        return src

    def _scatter_mean(
        self, src: torch.Tensor, index: torch.Tensor, out: torch.Tensor, dim: int = -1
    ):
        out = out.scatter_add_(dim, index, src)

        index_dim = dim
        if index_dim < 0:
            index_dim = index_dim + src.dim()
        if index.dim() <= index_dim:
            index_dim = index.dim() - 1

        ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
        out_count = torch.zeros(out.size(), dtype=out.dtype, device=out.device)
        out_count = out_count.scatter_add_(index_dim, index, ones)
        out_count.clamp_(1)
        count = self._broadcast(out_count, out, dim)
        if torch.is_floating_point(out):
            out.true_divide_(count)
        else:
            out.floor_divide_(count)
        return out

    def _scatter_nd(self, indices, updates):
        indices_shape = indices.shape
        num_index_dims = indices_shape[-1]
        flat_updates = updates.view((-1,))
        indices_scales = self._result_dim_sizes[0:num_index_dims].view(
            [1] * (len(indices_shape) - 1) + [num_index_dims]
        )
        indices_for_flat_tiled = (
            ((indices * indices_scales).sum(dim=-1, keepdims=True))
            .view(-1, 1)
            .repeat(*[1, self._voxel_feature_size])
        )

        implicit_indices = (
            self._arange_to_max_coords[: self._voxel_feature_size]
            .unsqueeze(0)
            .repeat(*[indices_for_flat_tiled.shape[0], 1])
        )
        indices_for_flat = indices_for_flat_tiled + implicit_indices
        flat_indices_for_flat = indices_for_flat.view((-1,)).long()

        flat_scatter = self._scatter_mean(
            flat_updates, flat_indices_for_flat, out=torch.zeros_like(self._flat_output)
        )
        return flat_scatter.view(self._total_dims_list)

    def coords_to_bounding_voxel_grid(
        self, coords, coord_features=None, coord_bounds=None
    ):
        voxel_indicy_denmominator = self._voxel_indicy_denmominator
        res, bb_mins = self._res, self._bb_mins
        if coord_bounds is not None:
            bb_mins = coord_bounds[..., 0:3]
            bb_maxs = coord_bounds[..., 3:6]
            bb_ranges = bb_maxs - bb_mins
            res = bb_ranges / (self._dims_orig.float() + MIN_DENOMINATOR)
            voxel_indicy_denmominator = res + MIN_DENOMINATOR

        bb_mins_shifted = bb_mins - res  # shift back by one
        floor = torch.floor(
            (coords - bb_mins_shifted.unsqueeze(1))
            / voxel_indicy_denmominator.unsqueeze(1)
        ).int()
        voxel_indices = torch.min(floor, self._dims_m_one)
        voxel_indices = torch.max(voxel_indices, self._dims_m_one_zeros)

        # BS x NC x 3
        voxel_values = coords
        if coord_features is not None:
            voxel_values = torch.cat([voxel_values, coord_features], -1)

        _, num_coords, _ = voxel_indices.shape
        # BS x N x (num_batch_dims + 2)
        all_indices = torch.cat(
            [self._tiled_batch_indices[:, :num_coords], voxel_indices], -1
        )

        # BS x N x 4
        voxel_values_pruned_flat = torch.cat(
            [voxel_values, self._ones_max_coords[:, :num_coords]], -1
        )

        # import ipdb; ipdb.set_trace()

        # BS x x_max x y_max x z_max x 4
        scattered = self._scatter_nd(
            all_indices.view([-1, 1 + 3]),
            voxel_values_pruned_flat.view(-1, self._voxel_feature_size),
        )

        vox = scattered[:, 1:-1, 1:-1, 1:-1]
        if INCLUDE_PER_VOXEL_COORD:
            res_expanded = res.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            res_centre = (res_expanded * self._index_grid) + res_expanded / 2.0
            coord_positions = (
                res_centre + bb_mins_shifted.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            )[:, 1:-1, 1:-1, 1:-1]
            vox = torch.cat([vox[..., :-1], coord_positions, vox[..., -1:]], -1)

        occupied = (vox[..., -1:] > 0).float()
        vox = torch.cat([vox[..., :-1], occupied], -1)

        return torch.cat(
            [
                vox[..., :-1],
                self._index_grid[:, :-2, :-2, :-2] / self._voxel_d,
                vox[..., -1:],
            ],
            -1,
        )


class DenseBlock(nn.Module):
    def __init__(self, in_features, out_features, norm=None, activation=None):
        super(DenseBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

        if activation is None:
            nn.init.xavier_uniform_(
                self.linear.weight, gain=nn.init.calculate_gain("linear")
            )
            nn.init.zeros_(self.linear.bias)
        elif activation == "tanh":
            nn.init.xavier_uniform_(
                self.linear.weight, gain=nn.init.calculate_gain("tanh")
            )
            nn.init.zeros_(self.linear.bias)
        elif activation == "lrelu":
            nn.init.kaiming_uniform_(
                self.linear.weight, a=LRELU_SLOPE, nonlinearity="leaky_relu"
            )
            nn.init.zeros_(self.linear.bias)
        elif activation == "relu":
            nn.init.kaiming_uniform_(self.linear.weight, nonlinearity="relu")
            nn.init.zeros_(self.linear.bias)
        else:
            raise ValueError()

        self.activation = None
        self.norm = None
        if norm is not None:
            self.norm = norm_layer1d(norm, out_features)
        if activation is not None:
            self.activation = act_layer(activation)

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.activation(x) if self.activation is not None else x
        return x
