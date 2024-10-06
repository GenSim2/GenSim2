# -*- coding: UTF-8 -*-
"""
@Project ：stable-baselines3
@File    ：.py
@Author  ：Chen Bao
@Date    ：2022/12/21 下午1:00
"""

import argparse
import os
import random
from typing import List

import pandas as pd
import tqdm
import wandb
import icecream


# from dexart.utils.task_setting import TRAIN_CONFIG, IMG_CONFIG
# from main.eval.training_config import ExperimentSetting, TaskSetting, ExtractorSetting, lower_bounds
import torch.nn as nn
import numpy as np
from gensim2.env.solver.rl.stable_baselines3 import PPO
from gensim2.env.solver.rl.stable_baselines3.common.torch_layers import (
    PointNetImaginationExtractorGP,
)


def get_max_saved_model_index(run: wandb.apis.public.Run, **kwargs):
    """
    The function returns the max index of the checkpoint, given the constant eval_frequency.

    :param run: wandb.apis.public.Run
    :return: Max index of the checkpoint. If not found, return -eval_frequency.
    """
    import re

    files: wandb.apis.public.Files = run.files()
    max_index = -1
    for file in files:
        file_name = file.name
        match = re.search(r"model_([\d]+)\.zip", file_name)
        if match:
            current_index = int(match.group(1))
            if current_index > max_index:
                max_index = current_index
    return max_index


def filter_with_tags(
    runs: wandb.apis.public.Runs, exact_strs=None, blur_strs=None, blur_no_strs=None
):
    """
    The function filter the input wandb runs given three types of filter strings list.

    :param runs: wandb.apis.public.Runs
    :param exact_strs: List[str], the result runs must have the exactly same tag.
    :param blur_strs: List[str], the result runs must have the tag that contains the string.
    :param blur_no_strs: List[str], all the tag of the runs should not contain the string.
    :return: Filtered runs.
    """
    if blur_strs is None:
        blur_strs = []
    if exact_strs is None:
        exact_strs = []
    if blur_no_strs is None:
        blur_no_strs = []
    result_runs = list()
    for run in runs:
        tags = run.tags
        matched = True
        # print(f"tags = {tags}")
        for exact_str in exact_strs:
            if exact_str not in tags:
                matched = False
        for blur_str in blur_strs:
            blur_find = False
            for tag in tags:
                if blur_str in tag:
                    blur_find = True
            if not blur_find:
                matched = False
        for blur_no_str in blur_no_strs:
            blur_find = False
            for tag in tags:
                if blur_no_str in tag:
                    blur_find = True
            if blur_find:
                matched = False
        if matched:
            result_runs.append(run)
    return result_runs


def download_runs_reward(
    filtered_runs: List[wandb.apis.public.Run], training_setting: dict
):
    """

    :param filtered_runs:
    :param training_setting:
    :return:
    """
    print("=================================================================")
    print(f"{len(filtered_runs)} seeds detected!")
    print("=================================================================")

    task_name = training_setting["task_name"]
    index_name = training_setting["index_name"]
    eval_name = training_setting["eval_name"]
    extractor_name = training_setting["extractor_name"]
    save_path = training_setting["save_path"]
    lower_bound = training_setting["lower_bound"]
    freeze_type: int = training_setting["freeze_type"]
    freeze_type_str = "nofree" if freeze_type == 0 else f"free{freeze_type}"

    result_dict = dict()
    for i, run in enumerate(filtered_runs):  # enumerate random seed
        for j, row in run.history(samples=9999).iterrows():
            total_steps = row["time/total_timesteps"]
            reward = row["rollout/rollout_rew_mean"]
            if not result_dict.__contains__(total_steps):
                result_dict[total_steps] = list()
            result_dict[total_steps].append(reward)

    sorted_result = sorted(result_dict.items(), key=lambda x: x[0])

    i = 0
    results = list()
    for total_steps, reward_list in sorted_result:
        if i > 0:
            i -= 1
            continue
        if task_name == "faucet" and task_name == "toilet":
            i = 0
        elif task_name == "laptop":
            i = 1
        elif task_name == "bucket":
            i = 3
        current = (total_steps, 0, np.mean(reward_list), np.std(reward_list))
        results.append(current)

    df2 = pd.DataFrame(
        np.array(results), columns=["total_steps", "seed", "reward", "std"]
    )
    df2.to_csv(
        f"{save_path}/training_reward_{task_name}_{index_name}_{extractor_name}_{freeze_type_str}_{eval_name}_{lower_bound}.csv"
    )
    print(
        "save to ",
        f"{save_path}/training_reward_{task_name}_{index_name}_{extractor_name}_{freeze_type_str}_{eval_name}_{lower_bound}.csv",
    )


def download_models(
    filtered_runs: List[wandb.apis.public.Run],
    training_setting: dict,
    only_lowerbound=False,
):
    """

    :param only_lowerbound:
    :param filtered_runs:
    :param training_setting:
        [required field]
        lower_bound
        eval_frequency
        model_dir
    :param only_latest: if ture, only download the lower_bound one
    :return:
    """
    lower_bound: int = training_setting["lower_bound"]
    eval_frequency: int = training_setting["eval_frequency"]
    model_dir = training_setting["model_dir"]
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    first_index = 0 if not only_lowerbound else lower_bound
    for check_point_index in tqdm.tqdm(
        range(first_index, lower_bound + 1, eval_frequency)
    ):
        for seed_index, run in enumerate(filtered_runs):  # enumerate random seed
            checkpoint_name = f"model_{check_point_index}.zip"
            for trail in range(5):
                try:
                    print(f"downloading model {check_point_index}")
                    file = run.file(name=checkpoint_name).download(
                        root=model_dir, replace=True
                    )
                    os.rename(
                        os.path.join(model_dir, checkpoint_name),
                        os.path.join(model_dir, f"{seed_index}_{checkpoint_name}"),
                    )
                    break
                except Exception as e:
                    print(
                        f"Error in download model {check_point_index}, {4 - trail} times retry."
                    )
                    print(e)
