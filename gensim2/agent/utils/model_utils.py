import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np


def module_max_param(module):
    def maybe_max(x):
        return float(torch.abs(x).max()) if x is not None else 0

    max_data = np.amax(
        [(maybe_max(param.data)) for name, param in module.named_parameters()]
    )
    return max_data


def module_mean_param(module):
    def maybe_mean(x):
        return float(torch.abs(x).mean()) if x is not None else 0

    max_data = np.mean(
        [(maybe_mean(param.data)) for name, param in module.named_parameters()]
    )
    return max_data


def module_max_gradient(module):
    def maybe_max(x):
        return torch.abs(x).max().item() if x is not None else 0

    max_grad = np.amax(
        [(maybe_max(param.grad)) for name, param in module.named_parameters()]
    )
    return max_grad
