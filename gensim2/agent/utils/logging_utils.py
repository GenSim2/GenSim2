import tabulate
from tqdm import tqdm

import numpy as np
import json

import torch
import gc
import csv
import os


def mkdir_if_missing(dst_dir):
    """make destination folder if it's missing"""
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)


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


def print_and_write(file_handle, text):
    print(text)

    if file_handle is not None:
        if type(file_handle) is list:
            for f in file_handle:
                f.write(text + "\n")
        else:
            file_handle.write(text + "\n")
    return text


def tabulate_print_state(result_dict):
    """print state dict"""
    result_dict = sorted(result_dict.items())
    headers = ["task", "success_rate"]
    data = [[kv[0], kv[1]] for kv in result_dict]
    data.append(["avg", np.mean([kv[1] for kv in result_dict])])
    str = tabulate.tabulate(data, headers, tablefmt="psql", floatfmt=".2f")
    print(str)
    return str


def log_results(cfg, total_rewards):
    tabulate_print_state(total_rewards)

    # log to offline csv
    mkdir_if_missing("outputs/sim_output_stats")
    result_paths = [
        os.path.join(cfg.output_dir, "eval_results.csv"),
        os.path.join(
            "outputs/sim_output_stats", cfg.output_dir.split("/")[-2] + "_results.csv"
        ),
    ]
    for path in result_paths:
        with open(path, "w") as f:
            writer = csv.writer(f)
            row_info_name = ["env_name", "success_rate"]
            writer.writerow(row_info_name)
            for k, v in total_rewards.items():
                try:
                    writer.writerow([k, v])
                except:
                    pass
            writer.writerow(
                ["avg", np.mean([float(kv[1]) for kv in sorted(total_rewards.items())])]
            )
