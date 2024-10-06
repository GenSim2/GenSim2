# Code source from Jiayuan Gu: https://github.com/Jiayuan-Gu/torkit3d
import torch
from torch import nn

from .conv import Conv1dBNReLU, Conv2dBNReLU
from .linear import LinearBNReLU

__all__ = ["mlp_bn_relu", "mlp1d_bn_relu", "mlp2d_bn_relu"]


def mlp_bn_relu(in_channels, out_channels_list):
    c_in = in_channels
    layers = []
    for c_out in out_channels_list:
        layers.append(LinearBNReLU(c_in, c_out, relu=True, bn=True))
        c_in = c_out
    return nn.Sequential(*layers)


def mlp_relu(in_channels, out_channels_list):
    c_in = in_channels
    layers = []
    for c_out in out_channels_list:
        layers.append(LinearBNReLU(c_in, c_out, relu=True, bn=False))
        c_in = c_out
    return nn.Sequential(*layers)


def mlp1d_bn_relu(in_channels, out_channels_list):
    c_in = in_channels
    layers = []
    for c_out in out_channels_list:
        layers.append(Conv1dBNReLU(c_in, c_out, 1, relu=True))
        c_in = c_out
    return nn.Sequential(*layers)


def mlp1d_relu(in_channels, out_channels_list):
    c_in = in_channels
    layers = []
    for c_out in out_channels_list:
        layers.append(Conv1dBNReLU(c_in, c_out, 1, relu=True, bn=False))
        c_in = c_out
    return nn.Sequential(*layers)


def mlp2d_bn_relu(in_channels, out_channels_list):
    c_in = in_channels
    layers = []
    for c_out in out_channels_list:
        layers.append(Conv2dBNReLU(c_in, c_out, 1, relu=True))
        c_in = c_out
    return nn.Sequential(*layers)


def mlp2d_relu(in_channels, out_channels_list):
    c_in = in_channels
    layers = []
    for c_out in out_channels_list:
        layers.append(Conv2dBNReLU(c_in, c_out, 1, relu=True, bn=False))
        c_in = c_out
    return nn.Sequential(*layers)
