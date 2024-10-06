import os
from collections import OrderedDict
from typing import Dict, Optional, Callable, List, Union, Tuple
import time
import gym
import icecream
import numpy as np
import open3d as o3d
import mplib
import transforms3d
from transforms3d.euler import euler2quat
from pathlib import Path
import json

from gensim2.env.utils.common import merge_dicts
from gensim2.env.utils.random import np_random
from gensim2.paths import *
from gensim2.env.utils.common import merge_dict_spaces


class GenSimBaseEnv(gym.Env):
    SUPPORTED_OBS_MODES = ("state", "none", "image", "pointcloud")

    def __init__(self, obs_mode="none", num_pcd=10240):

        # Simulation related attributes
        self.agent = None
        self.tool = None
        self.articulator = None
        self.rigid_body = None
        self.rigid_body_id = dict()
        self.articulator_id = dict()
        self.img_dict = OrderedDict()
        self.num_pcd = num_pcd

        # assert self.num_pcd == 4096

        # RL related attributes
        self.is_robot_free: Optional[bool] = None
        self.arm_dof: Optional[int] = None
        self.rl_step: Optional[Callable] = None
        self.robot_info = None
        self.velocity_limit: Optional[np.ndarray] = None
        self.kinematic_model = None
        self.eval = -1

        # Robot cache
        self.control_time_step = None
        self.ee_link_name = None
        self.ee_link = None
        self.cartesian_error = None

        # Observation mode
        if obs_mode is None:
            obs_mode = self.SUPPORTED_OBS_MODES[0]
        if obs_mode not in self.SUPPORTED_OBS_MODES:
            raise NotImplementedError("Unsupported obs mode: {}".format(obs_mode))
        self.obs_mode = obs_mode

        # Camera
        self.cameras = OrderedDict()
        self.resolutions = OrderedDict()
        self._buffer = {}
        # Cameras and rendering
        self.camera_pose_noise: Dict[str, Tuple[Optional[float], np.ndarray]] = (
            OrderedDict()
        )  # tuple for noise level and original camera pose
        self.time = 0
        self.current_step = 0
        self.np_random = None
        self.large_grip_force = False
        self.gripper_state = 1

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = np_random(seed)
        return [seed]

    def set_seed(self, seed=None):
        self.seed(seed)

    def get_observation(self):
        if self.obs_mode == "none":
            # Some cases do not need observations, e.g., MPC
            obs = OrderedDict()
        elif self.obs_mode == "state":
            obs = self.get_oracle_state()
        elif self.obs_mode == "image":
            obs = self.get_obs_images()
        elif self.obs_mode == "pointcloud":
            obs = self.get_obs_pcd()
        else:
            raise NotImplementedError(self.obs_mode)

        return obs

    def get_info(self):
        return {}

    def is_done(self):
        return self.current_step >= self.horizon

    def follow_path(self, result, update_image=-1):
        raise NotImplementedError

    @property
    def obs_dim(self):
        return len(self.get_oracle_state())

    @property
    def action_dim(self):
        return 8

    @property
    def horizon(self):
        return 50

    def get_contacts(self):
        raise NotImplementedError

    ############################# Added Env Info for Planner
    def get_object_axis(self):
        if self.articulator is not None:
            return self.articulator.axis
        else:
            return None

    def get_tool_keypoints(self):
        self.tool_keypoints = self.tool.get_keypoints()

        return self.tool_keypoints

    def get_object_keypoints(self):
        self.object_keypoints = {}
        if self.articulator is not None:
            if isinstance(self.articulator, list):
                for articulator in self.articulator:
                    self.object_keypoints.update(articulator.get_keypoints())
            else:
                self.object_keypoints.update(self.articulator.get_keypoints())
        if self.rigid_body is not None:
            if isinstance(self.rigid_body, list):
                for rigid in self.rigid_body:
                    self.object_keypoints.update(rigid.get_keypoints())
            else:
                self.object_keypoints.update(self.rigid_body.get_keypoints())

        return self.object_keypoints

    def get_object_pose(self):
        raise NotImplementedError

    def get_tool_pose(self):
        raise NotImplementedError

    def get_ee_pose(self):
        raise NotImplementedError

    def get_base_pose(self):
        raise NotImplementedError

    def get_joint_positions(self):
        raise NotImplementedError

    def compute_tool_keypoints_inhand(self):
        pass

    def add_noise_to_camera(self):
        pass

    #############################

    def reset_internal(self):
        self.current_step = 0
        self.time = 0
        # Reset camera pose
        self.add_noise_to_camera()

    def setup(self):
        self.init_root_pose = [-0.615, 0, 0]
        self.init_qpos = [
            0,
            -0.32,
            0.0,
            -2.617993877991494,
            0.0,
            2.23,
            0.7853981633974483,
            0,
            0,
        ]
        self.tcp_name = "fr3_hand"  # "panda_hand"
        self.ee_link_name = "fr3_hand"  # "panda_hand"

    def initialize_agent(self):
        if self.agent is not None:
            self.first_frame_robot_ee_pose = self.tcp.pose

    def open_gripper(self):
        raise NotImplementedError

    def init_gripper(self):
        raise NotImplementedError

    def close_gripper(self):
        raise NotImplementedError

    def move_to_pose(self, action, update_image=-1):
        pose = np.zeros(7)
        pose[:3] = action[:3]
        if action.shape[0] == 7:
            rot_euler = action[3:6]
            rot_quat = euler2quat(*rot_euler)
            pose[3:] = rot_quat
        else:
            pose[3:] = action[3:7]

        open_gripper = action[-1]
        self.set_gripper(open_gripper)

        self.plan_timestep = 0.1
        try:
            result = self.planner.plan_screw(
                pose, self.agent.get_qpos(), time_step=self.plan_timestep
            )
        except:
            return False

        if result["status"] != "Success":
            return False
        self.follow_path(result, update_image=update_image)
        return True

    def step_action(self, action, update_image=-1):
        # self.current_step += 1
        self.success_planned = False
        if not self.success_planned:
            self.success_planned = self.move_to_pose(action, update_image=update_image)

    def sim_step(self):
        raise NotImplementedError

    def step(self, action: np.ndarray, eval=-1, eval_cam_names=None):
        self.eval = eval

        if eval_cam_names is not None:
            if self.eval != -1:
                self.img_dict = {key: [] for key in self.cameras.keys()}
                # for cam_name, cam in self.cameras.items():
                #     self.img_dict[cam_name] = []

            # if self.eval != -1:
            #     self.img_dict = {key: [] for key in self.eval_cam_names}

        self.step_action(action, update_image=self.eval)
        self.sim_step()
        obs = self.get_observation()  # (36,)
        done = self.is_done()
        info = self.get_info()
        self.time = self.current_step * self.dt  # TODO: double-check

        if self.obs_mode == "image":
            info["image"] = obs["image"]
        else:
            if eval != -1:
                # Only get images during eval
                info["image"] = self.get_images()
        # Reference: https://github.com/openai/gym/blob/master/gym/wrappers/time_limit.py
        # Need to consider that is_done and timelimit can happen at the same time
        # if self.current_step >= self.horizon:
        #     done = True
        # if done:
        #     info["img_dict"] = self.img_dict # TODO: I do not know if this is useful

        return obs, done, info

    def update_render(self):
        raise NotImplementedError

    def get_images(self):
        raise NotImplementedError

    def get_oracle_state(self):
        oracle_state = self.get_robot_state()  # (15,)
        return oracle_state

    def get_obs_images(self):
        self.update_render()

        return OrderedDict(
            state=self.get_oracle_state(),
            image=self.get_images(),
        )

    def get_obs_pcd(self):
        self.update_render()

        return OrderedDict(
            state=self.get_oracle_state(),
            pointcloud=self.get_pcds(),
        )

    def get_robot_state(self):
        robot_qpos = self.agent.get_qpos()  # (9,)
        # robot_qvel = self.agent.get_qvel()  # (9,)
        # robot_qacc = self.agent.get_qacc()  # (9,)
        robot_ee_pose = self.tcp.pose  # Pose 7

        return np.concatenate(
            [
                # self.first_frame_robot_ee_pose.p,
                robot_ee_pose.p,  # - self.first_frame_robot_ee_pose.p,
                robot_ee_pose.q,
                robot_qpos[:-1],
                # [float(self.current_step) / float(self.horizon)],
            ],
            dtype=np.float32,  # 15
        )

    def get_camera_to_robot_pose(self, camera_name):
        raise NotImplementedError

    def get_camera_params(self, cam_uid):
        raise NotImplementedError

    @property
    def action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(self.action_dim,))

    @property
    def observation_space(self):
        if self.obs_mode == "none":
            return gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32
            )
        state_space = OrderedDict()
        if self.obs_mode == "state":
            high = np.inf * np.ones(self.obs_dim)
            low = -high
            # state_space["state"] = gym.spaces.Box(low=low, high=high)
            return gym.spaces.Box(low=low, high=high, dtype=np.float32)
        elif self.obs_mode in ["image", "pointcloud"]:
            low, high = 0, 255
            for name, cam in self.cameras.items():
                shape = self.resolutions[name]
                state_space["image"][name] = gym.spaces.Box(
                    low, high, shape=shape, dtype=np.uint8
                )
            self.update_pcd_observation_space(gym.spaces.Dict(state_space))
        elif self.obs_mode == "pointcloud":
            raise NotImplementedError
        state_space = gym.spaces.Dict(state_space)

        return state_space

    def update_pcd_observation_space(self, space):
        # Replace image observation spaces with point cloud ones
        image_space: space.Dict = space.spaces.pop("image")
        pcd_space = OrderedDict()

        for cam_uid in image_space:
            cam_image_space = image_space[cam_uid]
            cam_pcd_space = OrderedDict()

            h, w = self.cameras[cam_uid].height, self.cameras[cam_uid].width
            cam_pcd_space["pos"] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(h * w, 3), dtype=np.float32
            )
            cam_pcd_space["rgb"] = gym.spaces.Box(
                low=0, high=255, shape=(h * w, 3), dtype=np.uint8
            )
            cam_pcd_space["seg"] = gym.spaces.Box(
                low=0, high=(2**32 - 1), shape=(h * w, 4), dtype=np.uint32
            )

            pcd_space[cam_uid] = gym.spaces.Dict(cam_pcd_space)

        pcd_space = merge_dict_spaces(pcd_space.values())
        space.spaces["pointcloud"] = pcd_space

    def merge_pointclouds(self, pointcloud_obs):
        pointcloud_obs = merge_dicts(pointcloud_obs.values())
        for key, value in pointcloud_obs.items():
            buffer = self._buffer.get(key, None)
            pointcloud_obs[key] = np.concatenate(value, out=buffer)
            self._buffer[key] = pointcloud_obs[key]

        return pointcloud_obs

    ## Check functions

    def check_grasp(self, *args, **kwargs):
        raise NotImplementedError

    def check_contact_fingers(self, *args, **kwargs):
        raise NotImplementedError

    def check_contact(self, *args, **kwargs) -> bool:
        raise NotImplementedError

    def check_actor_pair_contact(self, *args, **kwargs) -> bool:
        raise NotImplementedError

    def check_actor_pair_contacts(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def check_actors_pair_contacts(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def check_actor_pair_contacts_in_distances(
        self,
        *args,
        **kwargs,
    ) -> np.ndarray:
        raise NotImplementedError

    def check_actors_pair_contacts_in_distance(
        self,
        *args,
        **kwargs,
    ) -> np.ndarray:
        raise NotImplementedError

    def create_camera(self):
        raise NotImplementedError

    def create_camera_from_pose(self):
        raise NotImplementedError

    def setup_camera_from_config(self, config: Dict[str, Dict], use_opencv_trans=True):
        for cam_name, cfg in config.items():
            if cam_name in self.cameras.keys():
                raise ValueError(f"Camera {cam_name} already exists in the environment")
            if "mount_actor_name" in cfg:
                self.create_camera(
                    cfg["pose"],
                    None,
                    None,
                    None,
                    name=cam_name,
                    resolution=cfg["resolution"],
                    fov=cfg["fov"],
                    mount_actor_name=cfg["mount_actor_name"],
                )
            else:
                if "position" in cfg:
                    self.create_camera(
                        None,
                        cfg["position"],
                        cfg["look_at_dir"],
                        cfg["right_dir"],
                        cam_name,
                        resolution=cfg["resolution"],
                        fov=cfg["fov"],
                    )
                elif "pose" in cfg:
                    self.create_camera_from_pose(
                        cfg["pose"],
                        cam_name,
                        resolution=cfg["resolution"],
                        fov=cfg["fov"],
                        use_opencv_trans=use_opencv_trans,
                    )
                else:
                    raise ValueError(f"Camera {cam_name} has no position or pose.")

            self.img_dict[cam_name] = []

            if "noise_scale" in cfg:
                self.camera_pose_noise[cam_name] = (cfg["noise_scale"], cfg["pose"])

    def load_hand_as_tool_info(self) -> Tuple[Dict, str]:
        asset_dir = ASSET_ROOT / "tools" / "hand" / "model1"
        keypoint_path = str(asset_dir / "model0_info.json")
        scale = json.load(open(keypoint_path))["scale"]
        scales = np.array([scale] * 3)
        hand_info = dict(
            scales=scales, collision_file=None, visual_file=None, name="hand"
        )

        return hand_info, keypoint_path

    def load_rigidbody(self, instance_cls):
        raise NotImplementedError

    def load_articulated_object(self, instance_cls):
        raise NotImplementedError


def compute_angle_between(x1, x2):
    """Compute angle (radian) between two vectors."""
    x1, x2 = normalize_vector(x1), normalize_vector(x2)
    dot_prod = np.clip(np.dot(x1, x2), -1, 1)
    return np.arccos(dot_prod).item()


def normalize_vector(x, eps=1e-6):
    x = np.asarray(x)
    assert x.ndim == 1, x.ndim
    norm = np.linalg.norm(x)
    return np.zeros_like(x) if norm < eps else (x / norm)
