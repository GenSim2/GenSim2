import os, sys
from typing import Union

import hydra
import torch
from torch.utils import data
from torchvision import transforms
from tqdm import trange

import csv
from torch.utils.data import DataLoader, RandomSampler
import wandb
from omegaconf import OmegaConf

from gensim2.agent.utils import utils, model_utils
from gensim2.agent.utils.warmup_lr_wrapper import WarmupLR
from gensim2.paths import *

sys.path.append(f"{GENSIM_DIR}/agent/third_party/")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from gensim2.agent import train_test


@hydra.main(
    config_path=f"{GENSIM_DIR}/agent/experiments/configs",
    config_name="config",
    version_base="1.2",
)
def run(cfg):
    """
    This script runs through the train / test / eval loop. Assumes single task for now.
    """
    is_eval = cfg.train.total_epochs == 0
    if not cfg.debug:
        run = wandb.init(
            project="mid-level",
            name=cfg.suffix,
            tags=[cfg.wb_tag],
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=False,
            resume="allow",
        )
        print("wandb url:", wandb.run.get_url())

    device = "cuda"
    domain_list = [d.strip() for d in cfg.domains.split(",")]
    domain = domain_list[0] if len(domain_list) == 1 else "_".join(domain_list)

    output_dir_full = cfg.output_dir.split("/")
    output_dir = "/".join(output_dir_full[:-2] + [domain, ""])
    if len(cfg.suffix):
        output_dir += f"{cfg.suffix}"
    else:
        output_dir += "-".join(output_dir_full[-2:])
    if is_eval:
        output_dir += "-eval"
    cfg.output_dir = output_dir
    utils.save_args_hydra(cfg.output_dir, cfg)

    print("cfg: ", cfg)
    print("output dir", cfg.output_dir)

    use_pcd = "pointcloud" in cfg.stem.modalities
    if use_pcd:
        cfg.dataset.use_pcd = use_pcd
        cfg.dataset.pcdnet_pretrain_domain = (
            cfg.rollout_runner.pcdnet_pretrain_domain
        ) = cfg.stem.pointcloud.pcd_domain
        cfg.rollout_runner.pcd_channels = cfg.dataset.pcd_channels
    cfg.dataset.horizon = (
        cfg.dataset.observation_horizon + cfg.dataset.action_horizon - 1
    )

    normalizer = None
    action_dim = 7  # 8 for rlbench, 7 for gensim2
    state_dim = 15  # 15 # 24 for rlbench, 15 for gensim2
    if not is_eval:
        dataset = hydra.utils.instantiate(
            cfg.dataset,
            dataset_name=domain if len(domain_list) == 1 else domain_list,
            **cfg.dataset,
        )
        normalizer = dataset.get_normalizer()

        val_dataset = dataset.get_validation_dataset()
        pcd_num_points = 1024
        if use_pcd:
            pcd_num_points = dataset.pcd_num_points
            assert pcd_num_points is not None

        train_loader = data.DataLoader(
            dataset, **cfg.dataloader, multiprocessing_context="fork"
        )
        test_loader = data.DataLoader(
            val_dataset, **cfg.val_dataloader, multiprocessing_context="fork"
        )

        print(f"Train size: {len(dataset)}. Test size: {len(val_dataset)}.")

        action_dim = dataset.action_dim
        state_dim = dataset.state_dim

    # initialize policy
    cfg.head["output_dim"] = cfg.network["action_dim"] = action_dim
    policy = hydra.utils.instantiate(cfg.network).to(device)
    cfg.stem.state["input_dim"] = state_dim
    policy.init_domain_stem(domain, cfg.stem)
    policy.init_domain_head(domain, cfg.head, normalizer=normalizer)

    # optimizer and scheduler
    policy.finalize_modules()
    print("cfg.train.pretrained_dir:", cfg.train.pretrained_dir)

    loaded_epoch = -1
    if len(cfg.train.pretrained_dir) > 0:
        if "pth" in cfg.train.pretrained_dir:
            assert os.path.exists(
                cfg.train.pretrained_dir
            ), "Pretrained model not found"
            print("load model from", cfg.train.pretrained_dir)
            policy.load_state_dict(torch.load(cfg.train.pretrained_dir))
            loaded_epoch = int(
                cfg.train.pretrained_dir.split("/")[-1].split(".")[0].split("_")[-1]
            )
        else:
            assert os.path.exists(
                os.path.join(cfg.train.pretrained_dir, f"model.pth")
            ), "Pretrained model not found"
            policy.load_state_dict(
                torch.load(os.path.join(cfg.train.pretrained_dir, f"model.pth"))
            )

        print("loaded trunk")
        # policy.load_trunk(os.path.join(cfg.train.pretrained_dir, f"model.pth"))
        if cfg.train.freeze_trunk:
            policy.freeze_trunk()
            print("trunk frozen")
    else:
        print("train from scratch")

    policy.to(device)
    opt = utils.get_optimizer(cfg.optimizer, policy)
    sch = utils.get_scheduler(cfg.lr_scheduler, optimizer=opt)

    sch = WarmupLR(
        sch,
        init_lr=cfg.warmup_lr.lr,
        num_warmup=cfg.warmup_lr.step,
        warmup_strategy="constant",
    )
    n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"number of params (M): {n_parameters / 1.0e6:.2f}")

    if not is_eval:
        # train / test loop
        pbar = trange(
            loaded_epoch + 1, loaded_epoch + 1 + cfg.train.total_epochs, position=0
        )
        for epoch in pbar:
            train_stats = train_test.train(
                cfg.log_interval,
                policy,
                device,
                train_loader,
                opt,
                sch,
                epoch,
                pcd_npoints=pcd_num_points,
                in_channels=dataset.pcd_channels,
                debug=cfg.debug,
            )
            test_loss = train_test.test(
                policy,
                device,
                test_loader,
                epoch,
                pcd_npoints=pcd_num_points,
                in_channels=dataset.pcd_channels,
                debug=cfg.debug,
            )
            train_steps = (epoch + 1) * len(train_loader)

            # Save the policy every epoch
            if epoch % cfg.save_interval == 0:
                policy_path = os.path.join(cfg.output_dir, f"model_{epoch}.pth")
            else:
                policy_path = os.path.join(cfg.output_dir, f"model.pth")
            policy.save(policy_path)
            if "loss" in train_stats:
                pbar.set_description(
                    f"Steps: {train_steps}. Train loss: {train_stats['loss']:.4f}. Test loss: {test_loss:.4f}"
                )

            if train_steps > cfg.train.total_iters:
                break

        policy.save(policy_path)
        pbar.close()

    # Evaluate jointly trained policy
    if cfg.parallel_eval:
        total_success = train_test.eval_policy_parallel(policy, cfg)
    else:
        total_success = train_test.eval_policy_sequential(policy, cfg)

    print("saved results to:", cfg.output_dir)
    # save the results
    utils.log_results(cfg, total_success)


if __name__ == "__main__":
    run()
