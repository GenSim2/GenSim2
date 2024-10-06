from collections import OrderedDict
import numpy as np

import os
from typing import Type, List
from absl import app
from absl import flags

from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import (
    JointVelocity,
    EndEffectorPoseViaIK,
    JointPosition,
    EndEffectorPoseViaPlanning,
)
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *
from rlbench.backend.observation import Observation
from rlbench.backend.task import Task
from rlbench.demo import Demo

import logging
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation

from gensim2.env.utils.common import merge_dicts
from gensim2.env.utils.pcd_utils import (
    pcd_downsample,
    vis_pcd,
    pcd_filter_bound,
    pcd_filter_with_mask,
)

# ENV_DICT = { # PerAct Tasks
#     "OpenDrawer": OpenDrawer,
#     "SlideBlock": SlideBlockToTarget,
#     "SweepToDustpan": SweepToDustpan,
#     "MeatOffGrill": MeatOffGrill,
#     "TurnTap": TurnTap,
#     "PutInDrawer": PutItemInDrawer,
#     "CloseJar": CloseJar,
#     "DragStick": ReachAndDrag,
#     "StackBlocks": StackBlocks,
#     "ScrewBulb": LightBulbIn,
#     "PutInSafe": PutMoneyInSafe,
#     "PlaceWine": StackWine,
#     "PutInCupboard": PutGroceriesInCupboard,
#     "SortShape": PlaceShapeInShapeSorter,
#     "PushButtons": PushButtons,
#     "InsertPeg": InsertOntoSquarePeg,
#     "StackCups": StackCups,
#     "PlaceCups": PlaceCups,
# }

# ENV_DICT = {  # QAttn Tasks, these tasks do not need any collision mode check
#     "PhoneOnBase": PhoneOnBase,
#     "PickAndLift": PickAndLift,
#     "PickUpCup": PickUpCup,
#     "PutRubbishInBin": PutRubbishInBin,
#     "ReachTarget": ReachTarget,
#     "StackWine": StackWine,
#     "TakeLidOffSaucepan": TakeLidOffSaucepan,
#     "TakeUmbrellaOutOfUmbrellaStand": TakeUmbrellaOutOfUmbrellaStand,
# }

ENV_DICT = {  # DNAct Tasks
    # "OpenDrawer": OpenDrawer,
    "DragStick": ReachAndDrag,
    "SweepToDustpan": SweepToDustpan,
    "MeatOffGrill": MeatOffGrill,
    "TurnTap": TurnTap,
    "PutInDrawer": PutItemInDrawer,
    "PhoneOnBase": PhoneOnBase,
    "PutInSafe": PutMoneyInSafe,
    "PlaceWine": StackWine,
    "OpenFridge": OpenFridge,
    "SlideBlock": SlideBlockToTarget,
}

SCENE_BOUNDS = [-0.3, 0.7, -0.5, 0.5, 0.6, 1.6]


def create_rlbench_env(action_mode="gripper_pose", camera_resolution=[1280, 720]):
    obs_config = ObservationConfig(
        joint_forces=False,
        joint_positions=True,
        joint_velocities=True,
        task_low_dim_state=False,
        gripper_touch_forces=False,
        gripper_pose=True,
        gripper_open=True,
        gripper_matrix=True,
        gripper_joint_positions=True,
    )
    # obs_config.set_all(True)

    if action_mode == "gripper_pose":
        mode = MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaIK(), gripper_action_mode=Discrete()
        )
    elif action_mode == "joint_positions":
        mode = MoveArmThenGripper(
            arm_action_mode=JointPosition(), gripper_action_mode=Discrete()
        )
    elif action_mode == "key_pose":
        mode = MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(), gripper_action_mode=Discrete()
        )

    env = Environment(
        action_mode=mode,
        obs_config=obs_config,
        headless=False,
    )
    env.launch()

    # Add the camera to the scene
    cam_placeholder = Dummy("cam_cinematic_placeholder")
    cam = VisionSensor.create(camera_resolution)
    cam.set_pose(cam_placeholder.get_pose())
    cam.set_parent(cam_placeholder)

    cam_motion = CircleCameraMotion(cam, Dummy("cam_cinematic_base"), 0.005)
    tr = TaskRecorder(env, cam_motion, fps=30)

    return env, tr


def merge_pointclouds(pointcloud_obs):
    _buffer = {}
    pointcloud_obs = merge_dicts(pointcloud_obs.values())
    for key, value in pointcloud_obs.items():
        buffer = _buffer.get(key, None)
        pointcloud_obs[key] = np.concatenate(value, out=buffer)
        _buffer[key] = pointcloud_obs[key]

    return pointcloud_obs


def get_pcds(task, obs, num_pcd=10240):
    cams = {
        "front": task._scene._cam_front,
        "wrist": task._scene._cam_wrist,
        "shoulder_left": task._scene._cam_over_shoulder_left,
        "shoulder_right": task._scene._cam_over_shoulder_right,
        # "overhead": task._scene._cam_overhead
    }
    pcds = {
        "front": obs.front_point_cloud,
        "wrist": obs.wrist_point_cloud,
        "shoulder_left": obs.left_shoulder_point_cloud,
        "shoulder_right": obs.right_shoulder_point_cloud,
        # "overhead": obs.overhead_point_cloud,
    }
    colors = {
        "front": obs.front_rgb,
        "wrist": obs.wrist_rgb,
        "shoulder_left": obs.left_shoulder_rgb,
        "shoulder_right": obs.right_shoulder_rgb,
        # "overhead": obs.overhead_rgb,
    }
    masks = {
        "front": obs.front_mask,
        "wrist": obs.wrist_mask,
        "shoulder_left": obs.left_shoulder_mask,
        "shoulder_right": obs.right_shoulder_mask,
        # "overhead": obs.overhead_mask,
    }
    intrinsics = {
        "front": obs.misc["front_camera_intrinsics"],
        "wrist": obs.misc["wrist_camera_intrinsics"],
        "shoulder_left": obs.misc["left_shoulder_camera_intrinsics"],
        "shoulder_right": obs.misc["right_shoulder_camera_intrinsics"],
        # "overhead": obs.misc["overhead_intrinsics"],
    }
    extrinsics = {
        "front": obs.misc["front_camera_extrinsics"],
        "wrist": obs.misc["wrist_camera_extrinsics"],
        "shoulder_left": obs.misc["left_shoulder_camera_extrinsics"],
        "shoulder_right": obs.misc["right_shoulder_camera_extrinsics"],
        # "overhead": obs.misc["overhead_extrinsics"],
    }
    # TODO: pcd from cam space to world space and merge
    pointcloud_obs = OrderedDict()
    for cam in cams:
        cam_pcd = OrderedDict()
        pc = pcds[cam].reshape(-1, 3).astype(np.float32)
        rgb = colors[cam].reshape(-1, 3)
        mask = masks[cam].reshape(-1)

        # # transfer to world frame
        # cam_intrinsic = intrinsics[cam]
        # world2cam = extrinsics[cam]
        # base_pose_quat = task._scene.robot.arm.get_pose()
        # base_pose = np.eye(4)
        # base_pose[:3, :3] = transforms3d.quaternions.quat2mat(base_pose_quat[3:])
        # base_pose[:3, 3] = base_pose_quat[:3]
        # position_world = position.reshape(-1, 4) @ world2cam.T @ np.linalg.inv(base_pose).T
        # cam_pcd["pos"] = position_world[..., :3]

        cam_pcd["pos"] = pc[..., :3]
        cam_pcd["colors"] = rgb
        cam_pcd["seg"] = mask

        pointcloud_obs[cam] = cam_pcd
    pointcloud_obs = merge_pointclouds(pointcloud_obs)
    pointcloud_obs = pcd_downsample(
        pointcloud_obs, num=num_pcd, method="fps", bound_clip=True, bound=SCENE_BOUNDS
    )  # need crop, see peract

    return pointcloud_obs


class CameraMotion(object):
    def __init__(self, cam: VisionSensor):
        self.cam = cam

    def step(self):
        raise NotImplementedError()

    def save_pose(self):
        self._prev_pose = self.cam.get_pose()

    def restore_pose(self):
        self.cam.set_pose(self._prev_pose)


class CircleCameraMotion(CameraMotion):
    def __init__(self, cam: VisionSensor, origin: Dummy, speed: float):
        super().__init__(cam)
        self.origin = origin
        self.speed = speed  # in radians

    def step(self):
        self.origin.rotate([0, 0, self.speed])


class TaskRecorder(object):
    def __init__(self, env: Environment, cam_motion: CameraMotion, fps=30):
        self._env = env
        self._cam_motion = cam_motion
        self._fps = fps
        self._snaps = []
        self._current_snaps = []

    def take_snap(self, obs: Observation):
        self._cam_motion.step()
        self._current_snaps.append(
            (self._cam_motion.cam.capture_rgb() * 255.0).astype(np.uint8)
        )

    def record_task(self, task: Type[Task]):
        task = self._env.get_task(task)
        self._cam_motion.save_pose()
        while True:
            try:
                task.get_demos(
                    1,
                    live_demos=True,
                    callable_each_step=self.take_snap,
                    max_attempts=1,
                )
                break
            except RuntimeError:
                self._cam_motion.restore_pose()
                self._current_snaps = []
        self._snaps.extend(self._current_snaps)
        self._current_snaps = []
        return True

    def save(self, path):
        print("Converting to video ...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # OpenCV QT version can conflict with PyRep, so import here
        import cv2

        video = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc("m", "p", "4", "v"),
            self._fps,
            tuple(self._cam_motion.cam.get_resolution()),
        )
        for image in self._snaps:
            video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        video.release()
        self._snaps = []


def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = i < (len(demo) - 2) and (
        obs.gripper_open == demo[i + 1].gripper_open
        and obs.gripper_open == demo[i - 1].gripper_open
        and demo[i - 2].gripper_open == demo[i - 1].gripper_open
    )
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (
        stopped_buffer <= 0
        and small_delta
        and (not next_is_not_final)
        and gripper_state_no_change
    )
    return stopped


def keypoint_discovery(demo: Demo, stopping_delta=0.1, method="heuristic") -> List[int]:
    episode_keypoints = []
    if method == "heuristic":
        prev_gripper_open = demo[0].gripper_open
        stopped_buffer = 0
        for i, obs in enumerate(demo):
            stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
            stopped_buffer = 4 if stopped else stopped_buffer - 1
            # If change in gripper, or end of episode.
            last = i == (len(demo) - 1)
            if i != 0 and (obs.gripper_open != prev_gripper_open or last or stopped):
                episode_keypoints.append(i)
            prev_gripper_open = obs.gripper_open
        if (
            len(episode_keypoints) > 1
            and (episode_keypoints[-1] - 1) == episode_keypoints[-2]
        ):
            episode_keypoints.pop(-2)
        logging.debug("Found %d keypoints." % len(episode_keypoints), episode_keypoints)
        return episode_keypoints

    elif method == "random":
        # Randomly select keypoints.
        episode_keypoints = np.random.choice(range(len(demo)), size=20, replace=False)
        episode_keypoints.sort()
        return episode_keypoints

    elif method == "fixed_interval":
        # Fixed interval.
        episode_keypoints = []
        segment_length = len(demo) // 20
        for i in range(0, len(demo), segment_length):
            episode_keypoints.append(i)
        return episode_keypoints

    else:
        raise NotImplementedError


# find minimum difference between any two elements in list
def find_minimum_difference(lst):
    minimum = lst[-1]
    for i in range(1, len(lst)):
        if lst[i] - lst[i - 1] < minimum:
            minimum = lst[i] - lst[i - 1]
    return minimum


def plot_pred(predicted_trajs, pcds, rgbs, paths, gt_trajs=None):
    sampled_trajs = predicted_trajs
    tx, ty, tz = (
        sampled_trajs[:, 0],
        sampled_trajs[:, 1],
        sampled_trajs[:, 2],
    )

    if gt_trajs is not None:
        gt_traj = gt_trajs[0]
        gx, gy, gz = gt_traj[:, 0], gt_traj[:, 1], gt_traj[:, 2]

    # pcds = pcds.reshape(-1, 3)
    # rgbs = rgbs.reshape(-1, 3)

    # bound_min, bound_max = [SCENE_BOUNDS[0], SCENE_BOUNDS[2], SCENE_BOUNDS[4]], [SCENE_BOUNDS[1], SCENE_BOUNDS[3], SCENE_BOUNDS[5]]

    # bound = np.array([bound_min, bound_max], dtype=np.float32)

    # pcd_mask = (pcds > bound[0:1]) * (pcds < bound[1:2])
    # pcd_mask = np.all(pcd_mask, axis=1)
    # indices = np.where(pcd_mask)[0]

    # pcds = pcds[indices]
    # rgbs = rgbs[indices]
    # print(bound, pcds.shape, rgbs.shape)

    rgb_strings = [f"rgb{rgbs[i][0],rgbs[i][1],rgbs[i][2]}" for i in range(len(rgbs))]

    pcd_plots = [
        go.Scatter3d(
            x=pcds[:, 0],
            y=pcds[:, 1],
            z=pcds[:, 2],
            mode="markers",
            marker=dict(
                size=8,
                color=rgb_strings,
            ),
        )
    ]

    plot_data = [
        go.Scatter3d(
            x=tx,
            y=ty,
            z=tz,
            mode="markers",
            marker=dict(size=6, color="blue"),
        ),
    ] + pcd_plots

    if gt_trajs is not None:
        gt_plot = [
            go.Scatter3d(
                x=gx,
                y=gy,
                z=gz,
                mode="markers",
                marker=dict(size=10, color="red"),
            )
        ]
        plot_data += gt_plot

    fig = go.Figure(plot_data)
    path = f"./{paths}/plots"
    os.makedirs(path, exist_ok=True)
    existings = os.listdir(path)
    fig.write_html(os.path.join(path, f"vis_{len(existings)}.html"))


def normalize_quaternion(quat):
    return np.array(quat) / np.linalg.norm(quat, axis=-1, keepdims=True)


def quaternion_to_discrete_euler(quaternion, resolution):
    euler = Rotation.from_quat(quaternion).as_euler("xyz", degrees=True) + 180
    assert np.min(euler) >= 0 and np.max(euler) <= 360
    disc = np.around((euler / resolution)).astype(int)
    disc[disc == int(360 / resolution)] = 0
    return disc


def discrete_euler_to_quaternion(discrete_euler, resolution):
    euluer = (discrete_euler * resolution) - 180
    return Rotation.from_euler("xyz", euluer, degrees=True).as_quat()


def point_to_voxel_index(
    point: np.ndarray, voxel_size: np.ndarray, coord_bounds: np.ndarray
):
    bb_mins = np.array(coord_bounds[0:3])
    bb_maxs = np.array(coord_bounds[3:])
    dims_m_one = np.array([voxel_size] * 3) - 1
    bb_ranges = bb_maxs - bb_mins
    res = bb_ranges / (np.array([voxel_size] * 3) + 1e-12)
    voxel_indicy = np.minimum(
        np.floor((point - bb_mins) / (res + 1e-12)).astype(np.int32), dims_m_one
    )
    return voxel_indicy


def get_discrete_action(
    obs_tp1: Observation,
    obs_tm1: Observation,
    rlbench_scene_bounds: List[float],  # metric 3D bounds of the scene
    voxel_sizes: List[int],
    bounds_offset: List[float],
    rotation_resolution: int,
    crop_augmentation: bool,
):
    quat = normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    disc_rot = quaternion_to_discrete_euler(quat, rotation_resolution)

    attention_coordinate = obs_tp1.gripper_pose[:3]
    trans_indicies, attention_coordinates = [], []
    bounds = np.array(rlbench_scene_bounds)
    ignore_collisions = int(obs_tm1.ignore_collisions)
    for depth, vox_size in enumerate(
        voxel_sizes
    ):  # only single voxelization-level is used in PerAct
        if depth > 0:
            if crop_augmentation:
                shift = bounds_offset[depth - 1] * 0.75
                attention_coordinate += np.random.uniform(-shift, shift, size=(3,))
            bounds = np.concatenate(
                [
                    attention_coordinate - bounds_offset[depth - 1],
                    attention_coordinate + bounds_offset[depth - 1],
                ]
            )
        index = point_to_voxel_index(obs_tp1.gripper_pose[:3], vox_size, bounds)
        trans_indicies.extend(index.tolist())
        res = (bounds[3:] - bounds[:3]) / vox_size
        attention_coordinate = bounds[:3] + res * index
        attention_coordinates.append(attention_coordinate)

    rot_and_grip_indicies = disc_rot.tolist()
    grip = float(obs_tp1.gripper_open)
    rot_and_grip_indicies.extend([int(obs_tp1.gripper_open)])
    # return trans_indicies, rot_and_grip_indicies, ignore_collisions, np.concatenate(
    #     [obs_tp1.gripper_pose, np.array([grip])]), attention_coordinates
    return (
        trans_indicies,
        rot_and_grip_indicies,
        ignore_collisions,
        attention_coordinates,
    )
