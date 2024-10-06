# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from typing import List, Optional
import numpy as np

from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
from collections import OrderedDict
from collections import deque
import transforms3d
import copy

from multiprocessing import Process, Queue
import multiprocessing as mp

try:
    mp.set_start_method("forkserver", force=True)
except RuntimeError:
    pass
from multiprocessing import Process, Queue

from gensim2.agent.utils.utils import dict_apply
from gensim2.agent.utils.pcd_utils import (
    visualize_point_cloud,
    randomly_drop_point,
    add_gaussian_noise,
)
from gensim2.env.task.origin_tasks import *
from gensim2.env.task.primitive_tasks import *
from gensim2.env.create_task import create_gensim
from gensim2.paths import GENSIM_DIR

try:
    from gensim2.env.utils.rlbench import *
    from rlbench.backend.exceptions import InvalidActionError
    from pyrep.objects.dummy import Dummy
    from pyrep.objects.vision_sensor import VisionSensor
    from pyrep.const import RenderMode
    from pyrep.errors import IKError, ConfigurationPathError
    from pyrep.objects import VisionSensor, Dummy
except Exception as e:
    print("RLBench not installed or error due to", e)

MAX_EP_STEPS = 100
MAX_RLBENCH_EP_STEPS = 100


def normalize_quaternion(q):
    norm = np.linalg.norm(q)
    if norm == 0:
        return q  # Handle the zero division case if needed
    return q / norm


def flat_pcd_sample(sample, pcd_channels=4):
    if "obs" in sample.keys():
        for key, val in sample.items():
            sample[key] = val
        del sample
    if "pointcloud" in sample.keys():
        if pcd_channels == 3:
            pass
            # sample['pointcloud']['pos'] = sample['pointcloud']['pos']
        elif pcd_channels == 5:
            sample["pointcloud"]["x"] = np.concatenate(
                [
                    sample["pointcloud"]["pos"],
                    sample["pointcloud"]["heights"],
                    sample["pointcloud"]["seg"][..., 1].unsqueeze(-1),
                ],
                axis=-1,
            )
        elif pcd_channels == 6:
            sample["pointcloud"]["x"] = np.concatenate(
                [sample["pointcloud"]["pos"], sample["pointcloud"]["colors"]], axis=-1
            )
        elif pcd_channels == 4:
            sample["pointcloud"]["x"] = np.concatenate(
                [sample["pointcloud"]["pos"], sample["pointcloud"]["heights"]], axis=-1
            )
        elif pcd_channels == 7:
            sample["pointcloud"]["x"] = np.concatenate(
                [
                    sample["pointcloud"]["pos"],
                    sample["pointcloud"]["colors"],
                    sample["pointcloud"]["heights"],
                ],
                axis=-1,
            )
        else:
            raise ValueError(f"Invalid pcd_channels: {pcd_channels}")
    return sample


def update_pcd_transform(pcdnet_pretrain_domain, pcd_setup_cfg=None):
    from openpoints.transforms import build_transforms_from_cfg

    if pcd_setup_cfg is None:
        from openpoints.utils import EasyConfig

        pcd_setup_cfg = EasyConfig()
        pcd_setup_cfg.load(
            f"{GENSIM_DIR}/agent/models/pointnet_cfg/{pcdnet_pretrain_domain}/pcd_setup.yaml",
            recursive=True,
        )

    # pcd_transform = build_transforms_from_cfg("train", pcd_setup_cfg.datatransforms)
    pcd_transform = build_transforms_from_cfg("val", pcd_setup_cfg.datatransforms)
    pcd_num_points = pcd_setup_cfg.num_points

    return pcd_transform, pcd_num_points


def preprocess_obs(sample, pcd_aug=None, pcd_transform=None, pcd_channels=4):
    if "pointcloud" in sample.keys():
        sample["pointcloud"]["x"] = sample["pointcloud"]["colors"]

        if pcd_aug is not None:
            pcd_aug = lambda x: randomly_drop_point(add_gaussian_noise(x))
            sample["pointcloud"]["pos"] = pcd_aug(sample["pointcloud"]["pos"])
        if pcd_transform is not None and "pointcloud" in sample.keys():
            sample["pointcloud"] = pcd_transform(sample["pointcloud"])

    sample = dict_apply(
        sample, lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else x
    )

    def unsqueeze(x, dim=0):
        if isinstance(x, np.ndarray):
            return np.expand_dims(x, dim)
        elif isinstance(x, torch.Tensor):
            return x.unsqueeze(dim)
        else:
            raise ValueError(f"Unsupported type: {type(x)}")

    sample = dict_apply(sample, unsqueeze)
    return flat_pcd_sample(sample, pcd_channels=pcd_channels)


def env_sample(env_names, ret_queue, **kwargs):
    for env_name in env_names:
        env = create_gensim(
            task_name=env_name,
            sim_type=kwargs["sim_type"],
            use_gui=kwargs["render"],
            eval=True,
            obs_mode=kwargs["obs_mode"],
            headless=False,
            asset_id="random",
        )


class RolloutRunner:
    """evaluate policy rollouts"""

    def __init__(
        self,
        env_names,
        episode_num=100,
        save_video=False,
        sim_type="sapien",
        render=False,
        obs_mode="pointcloud",
        pcd_channels=4,
        pcdnet_pretrain_domain="",
        random_reset=True,
        collision_pred=False,
    ):
        self.env_names = env_names
        self.save_video = save_video
        self.pcd_transform, self.pcd_num_points = update_pcd_transform(
            pcdnet_pretrain_domain
        )
        self.pcd_channels = pcd_channels
        self.pcd_aug = lambda x: randomly_drop_point(add_gaussian_noise(x))

        self.episode_num = episode_num
        self.envs = []
        self.env_name_dict = {}

        self.render = render
        self.random_reset = random_reset

        self.collision_pred = collision_pred

        for env_name in env_names:
            env = create_gensim(
                task_name=env_name,
                sim_type=sim_type,
                use_gui=self.render,
                eval=False,
                obs_mode=obs_mode,
                headless=False,
                asset_id="random",
            )
            self.envs.append(env)
            self.env_name_dict[env_name] = env

    @torch.no_grad()
    def run(self, policy, env_name, seed=233):
        episode_num = self.episode_num  # upper bound for number of trajectories
        imgs = OrderedDict()

        total_success = 0
        total_reward = 0
        env = self.env_name_dict[env_name]
        env.seed(seed)

        print(f"selected env name: {env_name}")
        pbar = tqdm(range(episode_num), position=1, leave=True)

        for i in pbar:
            eps_reward = 0
            traj_length = 0
            done = False
            policy.reset()
            obs = env.reset(random=self.random_reset)
            openloop_actions = deque()
            task_description = env.task.sub_task_descriptions[0]

            for task in env.sub_tasks:
                for t in range(MAX_EP_STEPS):
                    traj_length += 1

                    with torch.no_grad():
                        if len(openloop_actions) > 0:
                            action = openloop_actions.popleft()
                        else:
                            if "pointcloud" in obs.keys():
                                action = policy.get_action(
                                    preprocess_obs(
                                        obs,
                                        self.pcd_aug,
                                        self.pcd_transform,
                                        self.pcd_channels,
                                    ),
                                    pcd_npoints=self.pcd_num_points,
                                    in_channels=self.pcd_channels,
                                    task_description=task_description,
                                    t=t,
                                )
                            else:
                                action = policy.get_action(
                                    preprocess_obs(obs, None, None, 3),
                                    pcd_npoints=self.pcd_num_points,
                                    in_channels=3,
                                    task_description=task_description,
                                    t=t,
                                )
                            if len(action.shape) > 1:
                                for a in action[1:]:
                                    openloop_actions.append(a)
                                action = action[0]
                    action[-1] = 0.0 if action[-1] < 0.5 else 1.0
                    if self.collision_pred:
                        action[-2] = 0.0 if action[-2] < 0.5 else 1.0
                        ignore_collisions = bool(action[-1])
                        action = action[:-1]
                        next_obs, reward, done, info = env.step(
                            action, ignore_collisions=ignore_collisions
                        )
                    else:
                        next_obs, reward, done, info = env.step(action)
                    if self.render:
                        env.render()
                    if self.save_video:
                        for key, val in env.get_images().items():
                            if key not in imgs:
                                imgs[key] = []
                            imgs[key].append(val)

                    eps_reward += reward
                    obs = next_obs
                    task_description = info["next_task_description"]

                    if done:
                        break

                pbar.set_description(
                    f"{task} success: {info['sub_task_success']}, progress: {info['task_progress']}"
                )

                if not info["sub_task_success"]:
                    break

            total_reward += eps_reward
            total_success += info["success"]

        return total_success / episode_num, total_reward / episode_num, imgs


class RLBenchRolloutRunner:
    """evaluate policy rollouts"""

    def __init__(
        self,
        env_names,
        episode_num=100,
        save_video=False,
        render=False,
        obs_mode="pointcloud",
        pcd_channels=4,
        pcdnet_pretrain_domain="",
        random_reset=True,
        action_mode="joint_positions",
        collision_pred=False,
    ):
        self.env_names = env_names
        self.save_video = save_video

        self.pcd_transform, self.pcd_num_points = update_pcd_transform(
            pcdnet_pretrain_domain
        )
        self.pcd_channels = pcd_channels
        self.pcd_aug = lambda x: randomly_drop_point(add_gaussian_noise(x))

        self.episode_num = episode_num
        self.envs = []
        self.env_name_dict = {}

        self.render = render
        self.random_reset = random_reset
        self.action_mode = action_mode
        self.collision_pred = collision_pred

        self._error_type_counts = {
            "IKError": 0,
            "ConfigurationPathError": 0,
            "InvalidActionError": 0,
        }
        self.env, self.tr = create_rlbench_env(action_mode=action_mode)

    @torch.no_grad()
    def run(self, policy, env_name, seed=233):
        env = self.env.get_task(ENV_DICT[env_name])  # -> Task
        if self.save_video:
            self.tr._current_snaps = []
            self.tr._cam_motion.save_pose()

        episode_num = self.episode_num  # upper bound for number of trajectories
        imgs = OrderedDict()

        total_success = 0
        total_reward = 0
        np.random.seed(seed)

        print(f"selected env name: {env_name}")
        pbar = tqdm(range(episode_num), position=1, leave=True)

        for i in pbar:
            eps_reward = 0
            traj_length = 0
            done = False
            policy.reset()
            descriptions, obs = env.reset()
            task_description = np.random.choice(descriptions)
            openloop_actions = deque()

            if self.save_video:
                self.tr._cam_motion.restore_pose()

            for t in range(MAX_RLBENCH_EP_STEPS):
                if self.save_video:
                    self.tr.take_snap(obs)
                obs_data = OrderedDict()
                obs_data["state"] = obs.get_low_dim_data()
                obs_data["pointcloud"] = get_pcds(env, obs)

                traj_length += 1

                if done:
                    break
                with torch.no_grad():
                    if len(openloop_actions) > 0:
                        action = openloop_actions.popleft()
                    else:
                        action = policy.get_action(
                            preprocess_obs(
                                copy.deepcopy(obs_data),
                                None,
                                self.pcd_transform,
                                self.pcd_channels,
                            ),
                            # preprocess_obs(obs_data, self.pcd_aug, self.pcd_transform, self.pcd_channels),
                            pcd_npoints=self.pcd_num_points,
                            in_channels=self.pcd_channels,
                            task_description=task_description,
                            t=t,
                        )
                        # plot_pred(action, obs_data["pointcloud"]["pos"], obs_data["pointcloud"]["colors"], ".")
                        if len(action.shape) > 1:
                            if True:  # self.action_mode != "key_pose":
                                for a in action[1:]:
                                    openloop_actions.append(a)
                            action = action[0]
                action[-1] = 0.0 if action[-1] < 0.5 else 1.0
                if self.collision_pred:
                    action[-2] = 0.0 if action[-2] < 0.5 else 1.0  # -2 is gripper open
                    ignore_collisions = bool(action[-1])
                    action = action[:-1]  # remove ignore_collisions

                if self.action_mode == "gripper_pose" or self.action_mode == "key_pose":
                    # action[3:-1] = normalize_quaternion(action[3:-1])
                    rotation = transforms3d.euler.euler2quat(*action[3:-1])
                    action = np.concatenate(
                        [action[:3], rotation, np.array([action[-1]])]
                    )

                if self.collision_pred:
                    action = np.concatenate([action, np.array([ignore_collisions])])

                try:
                    next_obs, reward, done = env.step(action)
                    success = reward > 0  # done
                except (IKError, ConfigurationPathError, InvalidActionError) as e:
                    done = True
                    reward = 0.0
                    success = False

                    if isinstance(e, IKError):
                        self._error_type_counts["IKError"] += 1
                    elif isinstance(e, ConfigurationPathError):
                        self._error_type_counts["ConfigurationPathError"] += 1
                    elif isinstance(e, InvalidActionError):
                        self._error_type_counts["InvalidActionError"] += 1

                    print(e)
                eps_reward += reward
                if done:
                    self.tr.take_snap(obs)
                    break
                if self.render:
                    pass  # TODO
                    # env.render()
                obs = next_obs

            pbar.set_description(f"success: {eps_reward}, {done}, {success}")
            total_reward += eps_reward
            total_success += int(success)  # eps_reward

        if self.save_video:
            self.tr._snaps.extend(self.tr._current_snaps)

        return total_success / episode_num, total_reward / episode_num, self.tr


if __name__ == "__main__":
    # generate for all tasks
    runner = RolloutRunner(["all"], 200)
