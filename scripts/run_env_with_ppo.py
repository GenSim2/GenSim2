import numpy as np
import hydra
import yaml
from pathlib import Path
import torch
import warnings
import ipdb
import uuid
import wandb
import torch.nn as nn
from pathlib import Path

from gensim2.env.task.origin_tasks import *
from gensim2.env.task.primitive_tasks import *
from gensim2.env.create_task import create_gensim
from gensim2.env.base.task_setting import EVAL_CAM_NAMES_CONFIG
from gensim2.env.solver.rl.timestep import *
from gensim2.env.solver.rl.stable_baselines3.common.vec_env.subproc_vec_env import (
    SubprocVecEnv,
)
from gensim2.env.solver.rl.stable_baselines3.ppo import PPO
from gensim2.env.solver.rl.wandb_callback import WorkspaceRL
from gensim2.env.solver.rl.eval_utils import get_max_saved_model_index

from hashlib import sha1, md5

warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.backends.cudnn.benchmark = True


def make_agent(obs_shape, action_shape, cfg):
    cfg.obs_shape = obs_shape
    cfg.action_shape = action_shape
    return hydra.utils.instantiate(cfg)


def setup_wandb(
    parser_config,
    exp_name,
    tags=None,
    project="debug",
    group="groupname",
    resume="False",
    run_id=None,
) -> wandb.sdk.wandb_run.Run:
    run: wandb.sdk.wandb_run.Run = wandb.init(
        project=project,
        name=exp_name,
        config=parser_config,
        monitor_gym=True,
        save_code=True,  # optional
        tags=tags,
        group=group,
        resume=resume,
        id=run_id,
        mode="online",
    )

    return run


def delete_file(dir_path, name):
    for f in dir_path.glob(name):
        f.unlink()


def run_ppo(
    env_name,
    task_code,
    iter=50000,
    horizon=50,
    num=100,
    bs=500,
    exp=21,
    lr=0.0001,
    seed=100,
    ep=10,
    use_wandb=True,
):
    # Run config parameters
    env_name = str(env_name)
    obs_mode = "state"
    sim = "Sapien"
    workers = 10
    eval_freq = 20
    eval_success_rate_freq = 20
    root_dir = Path.cwd()

    runid = str(uuid.uuid4())
    env_iter = iter * horizon * num

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        print("Using GPU, device:", device)
    else:
        device = torch.device("cpu")
        print("Using CPU")

    if use_wandb:
        wandb_tags = [
            env_name,
            "ppo",
            "statemlp",
            "batchsize" + str(bs),
            str(exp),
        ]  # task name

        config = {
            "n_env_horizon": num,
            "update_iteration": iter,
            "total_step": env_iter,
            "task_name": task_name,
            "seed": seed,
        }

        group_name = "-".join(wandb_tags)
        exp_name = f"{group_name}-lr + {str(lr)}-{str(seed)}-{runid}"
        wandb_run: wandb.sdk.wandb_run.Run = setup_wandb(
            config,
            exp_name=exp_name,
            tags=wandb_tags,
            project=env_name,
            group=group_name,
            resume="allow",
            run_id=runid,
        )
    else:
        raise NotImplementedError("wandb is required at this moment.")

    def create_env_fn():
        return create_gensim(
            task_name=env_name,
            sim_type=sim,
            task_code_input=task_code,
            use_gui=False,
            obs_mode=obs_mode,
            eval=False,
        )

    def create_env_eval_fn():
        return create_gensim(
            task_name=env_name,
            sim_type=sim,
            task_code_input=task_code,
            use_gui=False,
            obs_mode=obs_mode,
            headless=True,
            eval=True,
        )

    env = SubprocVecEnv([create_env_fn] * workers, "spawn")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=2,
        n_epochs=ep,
        n_steps=(num // workers) * horizon,
        learning_rate=lr,
        batch_size=bs,
        seed=seed,
        policy_kwargs={"activation_fn": nn.ReLU},
        min_lr=lr,
        max_lr=lr,
        adaptive_kl=0.02,
        target_kl=0.2,
        device=device,
    )

    # if "resume_exp" in cfg and cfg.resume_exp and wandb_run.resumed:
    #     api = wandb.Api(timeout=9)
    #     run: wandb.apis.public.Run = api.run(f"{cfg.project}/{runid}")
    #     saved_model_index = get_max_saved_model_index(run)
    #     if saved_model_index >= 0:
    #         file = run.file(name=f"model_{saved_model_index}.zip").download(root=root_dir, replace=True)
    #         last_checkpoint = root_dir / f"model_{saved_model_index}.zip"
    #         model = model.load(path=last_checkpoint,
    #                            env=env,
    #                            verbose=1,
    #                            n_epochs=cfg.ep,
    #                            n_steps=(cfg.num // cfg.workers) * cfg.horizon,
    #                            learning_rate=cfg.lr,
    #                            batch_size=cfg.bs,
    #                            seed=cfg.seed,
    #                            policy_kwargs={'activation_fn': nn.ReLU},
    #                            min_lr=cfg.lr,
    #                            max_lr=cfg.lr,
    #                            adaptive_kl=0.02,
    #                            target_kl=0.2, )
    #         model.num_timesteps -= cfg.horizon * cfg.num
    #         rollout = int(model.num_timesteps / (cfg.horizon * cfg.n))
    #         print(f"loaded model from {last_checkpoint}")
    #         print(f"restore from step {rollout}")
    #     else:
    #         print(f"model not saved")

    model.learn(
        total_timesteps=int(env_iter),
        callback=WorkspaceRL(
            model_save_freq=500,  # 50,
            eval_env_fn=create_env_eval_fn,
            train_env_fn=create_env_fn,
            eval_freq=eval_freq,
            eval_cam_names=["default_cam"],
            eval_success_rate_freq=eval_success_rate_freq,
            eval_success_rate_times=50,
            visualize_pc=False,
        ),
    )

    wandb_run.finish()
    print("Done")
    return True


@hydra.main(
    config_path="../gensim2/env/solver/rl/cfgs",
    config_name="close_laptop",
    version_base="1.1",
)
def main(cfg):
    root_dir = Path.cwd()

    runid = md5(repr(sorted(cfg.__dict__.items())).encode()).hexdigest()
    env_iter = cfg.iter * cfg.horizon * cfg.num

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        device = torch.device("cpu")

    if cfg.wandb:
        wandb_tags = [
            cfg.env,
            "ppo",
            "statemlp",
            "batchsize" + str(cfg.bs),
            str(cfg.exp),
            "debugging",
        ]  # task name

        config = {
            "n_env_horizon": cfg.num,
            "update_iteration": cfg.iter,
            "total_step": env_iter,
            "task_name": task_name,
            "seed": cfg.seed,
        }

        group_name = "-".join(wandb_tags)
        exp_name = f"{group_name}-lr + {str(cfg.lr)}-{str(cfg.seed)}-{runid}"
        wandb_run: wandb.sdk.wandb_run.Run = setup_wandb(
            config,
            exp_name=exp_name,
            tags=wandb_tags,
            project=cfg.env,
            group=group_name,
            resume="allow",
            run_id=runid,
        )
    else:
        raise NotImplementedError("wandb is required at this moment.")

    # if "resume_exp" in cfg and cfg.resume_exp:
    #     print(f"resuming exp")
    #     snapshot = ws.work_dir / "snapshot.pt"
    #     if snapshot.exists():
    #         print(f"load from snapshot: {snapshot}")
    #         ws.load_snapshot(snapshot)
    #     else:
    #         delete_file(ws.work_dir, "buffer/*.npz")
    #         delete_file(ws.work_dir, "tb/*")
    #         delete_file(ws.work_dir, "*.csv")
    # else:  # delete all buffer files
    #     delete_file(ws.work_dir, "buffer/*.npz")
    #     delete_file(ws.work_dir, "tb/*")
    #     delete_file(ws.work_dir, "*.csv")

    def create_env_fn():
        return create_gensim(
            task_name=cfg.env,
            sim_type=cfg.sim,
            use_gui=False,
            obs_mode=cfg.obs_mode,
            eval=False,
        )

    def create_env_eval_fn():
        return create_gensim(
            task_name=cfg.env,
            sim_type=cfg.sim,
            use_gui=False,
            obs_mode=cfg.obs_mode,
            headless=True,
            eval=True,
        )

    # if cfg.workers == 1:
    #     env = create_env_fn()
    # else:
    env = SubprocVecEnv([create_env_fn] * cfg.workers, "spawn")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=2,
        n_epochs=cfg.ep,
        n_steps=(cfg.num // cfg.workers) * cfg.horizon,
        learning_rate=cfg.lr,
        batch_size=cfg.bs,
        seed=cfg.seed,
        policy_kwargs={"activation_fn": nn.ReLU},
        min_lr=cfg.lr,
        max_lr=cfg.lr,
        adaptive_kl=0.02,
        target_kl=0.2,
        device=device,
    )

    if "resume_exp" in cfg and cfg.resume_exp and wandb_run.resumed:
        api = wandb.Api(timeout=9)
        run: wandb.apis.public.Run = api.run(f"{cfg.project}/{runid}")
        saved_model_index = get_max_saved_model_index(run)
        if saved_model_index >= 0:
            file = run.file(name=f"model_{saved_model_index}.zip").download(
                root=root_dir, replace=True
            )
            last_checkpoint = root_dir / f"model_{saved_model_index}.zip"
            model = model.load(
                path=last_checkpoint,
                env=env,
                verbose=1,
                n_epochs=cfg.ep,
                n_steps=(cfg.num // cfg.workers) * cfg.horizon,
                learning_rate=cfg.lr,
                batch_size=cfg.bs,
                seed=cfg.seed,
                policy_kwargs={"activation_fn": nn.ReLU},
                min_lr=cfg.lr,
                max_lr=cfg.lr,
                adaptive_kl=0.02,
                target_kl=0.2,
            )
            model.num_timesteps -= cfg.horizon * cfg.num
            rollout = int(model.num_timesteps / (cfg.horizon * cfg.n))
            print(f"loaded model from {last_checkpoint}")
            print(f"restore from step {rollout}")
        else:
            print(f"model not saved")

    model.learn(
        total_timesteps=int(env_iter),
        callback=WorkspaceRL(
            model_save_freq=500,  # 50,
            eval_env_fn=create_env_eval_fn,
            train_env_fn=create_env_fn,
            eval_freq=cfg.eval_freq,
            eval_cam_names=["default_cam"],
            eval_success_rate_freq=cfg.eval_success_rate_freq,
            eval_success_rate_times=25,
            visualize_pc=False,
        ),
    )

    wandb_run.finish()
    print("Done")


if __name__ == "__main__":
    main()
