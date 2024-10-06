import numpy as np
import icecream
import hydra
import yaml
from pathlib import Path
import torch
import warnings
import ipdb
import sys
import random
import wandb
import argparse
from run_env_with_ppo import setup_wandb

from gensim2.env.task.origin_tasks import *
from gensim2.env.task.primitive_tasks import *
from gensim2.env.create_task import create_gensim
import gensim2.env.solver.rl.utils as utils
from gensim2.env.solver.rl.logger import Logger
from gensim2.env.solver.rl.replay_buffer import (
    ReplayBufferStorage,
    make_replay_loader,
    make_expert_replay_loader,
)
from gensim2.env.solver.rl.video import VideoRecorder
from gensim2.env.solver.rl import specs
from gensim2.env.solver.rl.timestep import *
from gensim2.paths import GENSIM_DIR

warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.backends.cudnn.benchmark = True


def make_agent(obs_shape, action_shape, cfg):
    cfg.obs_shape = obs_shape
    cfg.action_shape = action_shape
    return hydra.utils.instantiate(cfg)


class WorkspaceRL:
    def __init__(self, cfg, wandb_run=None):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        self.wandb_run = wandb_run
        utils.set_seed_everywhere(2)
        self.device = torch.device("cuda")
        self.use_per = "use_per" in self.cfg and self.cfg.use_per

        self.setup()

        self.agent = make_agent(
            self.train_env.observation_space.shape,
            self.train_env.action_space.shape,
            cfg.agent,
        )

        if self.use_per:
            self.agent.use_per = True

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

        self.noise_zone = None
        self.noise_scale = 0.1
        self.noise_size = 3
        if "noise_zone" in self.cfg:
            self.noise_zone = self.cfg.noise_zone
            self.noise_scale = self.cfg.noise_scale
            self.noise_size = self.cfg.noise_size

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        self.train_env = create_gensim(
            task_name=self.cfg.env,
            sim_type=self.cfg.sim,
            use_gui=False,
            obs_mode=self.cfg.obs_mode,
            eval=False,
        )
        self.eval_env = create_gensim(
            task_name=self.cfg.env,
            sim_type=self.cfg.sim,
            use_gui=False,
            obs_mode=self.cfg.obs_mode,
            headless=True,
            eval=True,
        )

        # create replay buffer
        data_specs = [
            specs.Array(
                self.train_env.observation_space.shape, np.float32, "observation"
            ),  # Not support image obs for now
            specs.Array(self.train_env.action_space.shape, np.float32, "action"),
            specs.Array((1,), np.float32, "reward"),
            specs.Array((1,), np.float32, "discount"),
        ]

        self.replay_storage = ReplayBufferStorage(data_specs, self.work_dir / "buffer")

        self.replay_loader, self.replay_buf = make_replay_loader(
            self.work_dir / "buffer",
            self.cfg.replay_buffer_size,
            self.cfg.batch_size,
            self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot,
            self.cfg.nstep,
            self.cfg.discount,
            use_per=self.use_per,
        )

        self._replay_iter = None
        self.expert_replay_iter = None
        self.video_recorder = None
        self.train_video_recorder = None

        if self.cfg.record_video:
            self.video_recorder = VideoRecorder(
                self.work_dir if self.cfg.save_video else None
            )

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):

        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            done = False
            obs = self.eval_env.reset()
            if self.video_recorder:
                self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            for _ in range(self.cfg.max_episode_steps):
                if done:
                    break
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(obs, self.global_step, eval_mode=True)
                next_obs, reward, done, info = self.eval_env.step(action)
                # update render
                self.eval_env.update_render()
                # ipdb.set_trace()
                if self.video_recorder:
                    self.video_recorder.record(self.eval_env)
                total_reward += reward
                step += 1

            episode += 1
            if self.video_recorder and episode == 1:
                print("saving video")
                print("global episode", self.global_episode)
                path = self.video_recorder.save(f"{self.global_episode}")
                # upload video to wandb
                if self.wandb_run is not None:
                    self.wandb_run.log(
                        {"video": wandb.Video(str(path))}, step=self.global_frame
                    )

        with self.logger.log_and_dump_ctx(self.global_frame, ty="eval") as log:
            log("episode_reward", total_reward / episode)
            log("episode_length", step * self.cfg.action_repeat / episode)
            log("episode", self.global_episode)
            log("step", self.global_step)
            # create wandb logging with same key
            if self.wandb_run is not None:
                self.wandb_run.log(
                    {"episode_reward": total_reward / episode}, step=self.global_step
                )
                self.wandb_run.log(
                    {"episode_length": step * self.cfg.action_repeat / episode},
                    step=self.global_step,
                )
                self.wandb_run.log(
                    {"episode": self.global_episode}, step=self.global_step
                )
                self.wandb_run.log({"step": self.global_step}, step=self.global_step)

    def train(self):
        train_until_step = utils.Until(3100000, 2)
        seed_until_step = utils.Until(12000, 2)
        eval_every_step = utils.Every(20000, 2)

        if self.noise_zone is not None:
            print("\n Using Noisy Observation! \n")

        metrics = None

        while train_until_step(self.global_step):
            # ipdb.set_trace()
            episode_reward = 0
            done = False
            obs = self.train_env.reset()

            for episode_step in range(self.cfg.max_episode_steps):
                if done:
                    break

                # sample action
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(obs, self.global_step, eval_mode=False)

                # ipdb.set_trace()
                next_obs, reward, done, info = self.train_env.step(action)

                episode_reward += reward

                step_type = (
                    StepType.LAST
                    if done
                    else StepType.FIRST if episode_step == 0 else StepType.MID
                )

                timestep = ExtendedTimeStep(
                    step_type=step_type,
                    reward=reward,
                    discount=self.cfg.discount,
                    observation=obs,
                    action=action,
                )

                self.replay_storage.add(timestep)

                obs = next_obs

                # try to evaluate
                if eval_every_step(self.global_step):
                    self.logger.log(
                        "eval_total_time", self.timer.total_time(), self.global_frame
                    )
                    if self.wandb_run is not None:
                        self.wandb_run.log(
                            {"eval_total_time": self.timer.total_time()},
                            step=self.global_frame,
                        )
                    self.eval()

                # try to update the agent
                if not seed_until_step(self.global_step):
                    # Update
                    metrics = self.agent.update(self.replay_iter, self.global_step)
                    # ipdb.set_trace()
                    if self.use_per and len(metrics.keys()):
                        self.replay_buf.update(
                            metrics["tree_indices"], metrics["td_errors"]
                        )
                        metrics.pop("tree_indices", None)
                        metrics.pop("td_errors", None)
                    self.logger.log_metrics(metrics, self.global_frame, ty="train")

                self._global_step += 1

            self._global_episode += 1
            if metrics is not None:
                # log stats
                elapsed_time, total_time = self.timer.reset()

                with self.logger.log_and_dump_ctx(self.global_frame, ty="train") as log:
                    log("total_time", total_time)
                    log("episode_reward", episode_reward)
                    log("episode", self.global_episode)
                    log("buffer_size", len(self.replay_storage))
                    log("step", self.global_step)

                    # wandb logging
                    if self.wandb_run is not None:
                        self.wandb_run.log(
                            {"total time": total_time}, step=self.global_step
                        )
                        self.wandb_run.log(
                            {"episode reward": episode_reward}, step=self.global_step
                        )
                        self.wandb_run.log(
                            {"episode": self.global_episode}, step=self.global_step
                        )
                        self.wandb_run.log(
                            {"buffer_size": len(self.replay_storage)},
                            step=self.global_step,
                        )
                        self.wandb_run.log(
                            {"step": self.global_step}, step=self.global_step
                        )

                    # bytes = sys.getsizeof(self.replay_storage)
                    # gb = bytes / (1024 ** 3)
                    # print(f"replay storage size: {gb} GB")

                    # print(f"train env storage size: {sys.getsizeof(self.train_env) / (1024 ** 3)} GB")

    def collect_demo(self):
        pass


def delete_file(dir_path, name):
    for f in dir_path.glob(name):
        f.unlink()


@hydra.main(config_path="../gensim2/env/solver/rl/cfgs", config_name="close_laptop_gpt")
def main(cfg):
    root_dir = Path.cwd()

    if cfg.wandb:
        wandb_run: wandb.sdk.wandb_run.Run = setup_wandb(
            cfg, project=cfg.env, exp_name=cfg.experiment
        )
        ws = WorkspaceRL(cfg, wandb_run=wandb_run)
    else:
        ws = WorkspaceRL(cfg, wandb_run=None)

    if "resume_exp" in cfg and cfg.resume_exp:
        print(f"resuming exp")
        snapshot = ws.work_dir / "snapshot.pt"
        if snapshot.exists():
            print(f"load from snapshot: {snapshot}")
            ws.load_snapshot(snapshot)
        else:
            delete_file(ws.work_dir, "buffer/*.npz")
            delete_file(ws.work_dir, "tb/*")
            delete_file(ws.work_dir, "*.csv")
    else:  # delete all buffer files
        delete_file(ws.work_dir, "buffer/*.npz")
        delete_file(ws.work_dir, "tb/*")
        delete_file(ws.work_dir, "*.csv")

    ws.train()
    ws.collect_demo()
    # wandb_run.finish()
    print("Done")


if __name__ == "__main__":
    main()
