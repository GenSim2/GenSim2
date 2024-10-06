from typing import Tuple
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import hydra
from collections import OrderedDict

import numpy as np
import copy

import yaml
from collections import deque
import wandb
import logging

from gensim2.agent.utils.utils import dict_apply, batchify, sample_pcd_data, unbatchify

try:
    from gensim2.agent.utils.video import save_video
except:
    pass
from gensim2.agent.utils.model_utils import module_max_gradient, module_mean_param

info_key = [
    "loss",
    "max_action",
    "max_label_action",
    "max_gradient",
    "max_stem_gradient",
    "max_trunk_gradient",
    "max_head_gradient",
    "batch_time",
    "mean_param",
    "lr",
]

Loss = F.mse_loss


def log_stat(
    info_log,
    train_step,
    log_interval,
    log_name,
    domain,
    target,
    loss,
    output,
    model,
    optimizer,
    start_time,
    use_wandb=True,
):
    if domain + "_loss" not in info_log:
        info_log[domain + "_loss"] = deque([], maxlen=50)

    info_log[domain + "_loss"].append(loss.item())
    info_log["loss"].append(loss.item())
    info_log["max_label_action"].append(target.max().item())
    if output is not None:
        info_log["max_action"].append(output.max().item())

    info_log["max_gradient"].append(module_max_gradient(model))
    info_log["max_stem_gradient"].append(module_max_gradient(model.stems))
    info_log["max_trunk_gradient"].append(module_max_gradient(model.trunk))
    info_log["max_head_gradient"].append(module_max_gradient(model.heads))
    info_log["mean_param"].append(module_mean_param(model))
    info_log["batch_time"].append(time.time() - start_time)
    info_log["lr"].append(optimizer.param_groups[0]["lr"])
    if use_wandb and (train_step % log_interval == 0):
        wandb_metrics = {
            f"{log_name}/{k}": np.mean(v) for k, v in info_log.items() if len(v) > 0
        }
        wandb.log({"train_step": train_step, **wandb_metrics})


def train(
    log_interval,
    model,
    device,
    train_loader,
    optimizer,
    scheduler,
    epoch,
    pcd_npoints=8192,
    in_channels=4,
    log_name="train",
    debug=True,
):
    """training for one epoch"""
    info_log = {k: deque([], maxlen=20) for k in info_key}
    model.train()
    start_time = time.time()

    # combined_dataloader = train_loader  # WeightedDataLoader(train_loaders)
    epoch_size = len(train_loader)
    assert epoch_size > 0, "empty dataloader"
    pbar = tqdm(train_loader, position=1, leave=True)

    # randomly sample a dataloader with inverse probability square root to the number of data
    for batch_idx, batch in enumerate(pbar):
        batch["data"] = dict_apply(
            batch["data"], lambda x: x.to(device, non_blocking=True).float()
        )

        batch["data"] = batchify(batch["data"], exclude=["action"])

        if "pointcloud" in batch["data"]:
            sample_pcd_data(
                batch["data"]["pointcloud"],
                npoints=pcd_npoints,
                in_channels=in_channels,
            )

        data_time = time.time() - start_time
        start_time = time.time()
        output = model.forward_train(batch)
        target = batch["data"]["action"]
        if target.shape[1] == 0:
            print("empty target:", target.shape)
            continue

        if isinstance(output, (dict, OrderedDict)):
            domain_loss = output["loss"]
            output = None
        else:
            output = output.reshape(target.shape)
            domain_loss = Loss(output, target)

        # backward
        optimizer.zero_grad()
        domain_loss.backward()
        optimizer.step()
        scheduler.step()
        train_step = len(train_loader) * epoch + batch_idx

        log_stat(
            info_log,
            train_step,
            log_interval,
            log_name,
            batch["domain"][0],
            target,
            domain_loss,
            output,
            model,
            optimizer,
            start_time,
            use_wandb=not debug,
        )
        step_time = time.time() - start_time

        pbar.set_description(
            f"Epoch: {epoch} Step: {batch_idx}/{epoch_size} Time: {step_time:.3f} {data_time:.3f} Loss: {info_log[batch['domain'][0] + '_loss'][-1]:.3f} Grad: {info_log['max_gradient'][-1]:.3f}"
        )
        start_time = time.time()
        if batch_idx == epoch_size:
            break

    return {k: np.mean(v) for k, v in info_log.items() if len(v) > 1}


@torch.no_grad()
def test(
    model,
    device,
    test_loader,
    epoch,
    pcd_npoints=8192,
    in_channels=4,
    log_name="test",
    debug=True,
):
    """evaluate imitation losses on the test sets"""
    model.eval()
    test_loss, num_examples = 0, 0
    pbar = tqdm(test_loader, position=2, leave=False)

    for batch_idx, batch in enumerate(pbar):
        batch["data"] = dict_apply(
            batch["data"], lambda x: x.to(device, non_blocking=True).float()
        )

        batch["data"] = batchify(batch["data"], exclude=["action"])

        if "pointcloud" in batch["data"]:
            sample_pcd_data(
                batch["data"]["pointcloud"],
                npoints=pcd_npoints,
                in_channels=in_channels,
            )

        output = model.forward_train(batch)
        target = batch["data"]["action"].to(device)
        if target.shape[1] == 0:
            continue
        if isinstance(output, (dict, OrderedDict)):
            loss = output["loss"]
            output = None
        else:
            output = output.reshape(target.shape)
            loss = Loss(output, target)

        # logging
        test_loss += loss.item()
        num_examples += target.size(0)
        pbar.set_description(
            f"Test Epoch: {epoch} Step: {batch_idx} Domain: {batch['domain'][0]} Loss: {test_loss / (num_examples + 1):.3f}"
        )
    return test_loss / (num_examples + 1)


# sequential evaluation
def eval_policy(policy, cfg, env_name=None, seed=111, rollout_runner=None):
    # run the rollout function from each environment
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    policy.eval()
    if rollout_runner is None:
        rollout_runner = hydra.utils.instantiate(cfg.rollout_runner)
    success, reward, images = rollout_runner.run(
        policy=policy, env_name=env_name, seed=seed
    )
    if "RLBench" in cfg.rollout_runner._target_:
        tr = images
        tr.save(f"{cfg.output_dir}/{env_name}.mp4")
    else:
        if len(images) > 0:
            save_video(images, env_name, cfg.output_dir)

    return success, reward


def eval_policy_sequential(policy, cfg):
    env_names = [env_name.strip() for env_name in cfg.env_names]
    rollout_runner = None
    if "RLBench" in cfg.rollout_runner._target_:
        # from gensim2.env.utils.rlbench import ENV_DICT
        # env_names = [env_name for env_name in ENV_DICT.keys()]
        rollout_runner = hydra.utils.instantiate(cfg.rollout_runner)
    total_reward_list = []
    total_success_list = []

    for env_name in env_names:
        success, rewards = eval_policy(
            policy, cfg, env_name, rollout_runner=rollout_runner, seed=cfg.seed
        )
        total_success_list.append(success)

    total_success = {
        env_name: rew for env_name, rew in zip(env_names, total_success_list)
    }
    return total_success


def eval_policy_parallel(policy, cfg, seed=233):
    policy.eval()

    # run the rollout function from each environment
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    policy.eval()
    rollout_runner = hydra.utils.instantiate(cfg.rollout_runner)
    rollout_runner.initiate(policy=policy)
    rollout_runner.join()

    print("collecting results...")
    env_names, total_success_list, total_reward_list, images_list = (
        rollout_runner.collect_results()
    )

    if len(images_list) > 0:
        for i, images in enumerate(images_list):
            if len(images) > 0:
                save_video(images, env_names[i], cfg.output_dir)

    total_success = {
        env_name: rew for env_name, rew in zip(env_names, total_success_list)
    }
    return total_success
