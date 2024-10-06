from collections import OrderedDict

import open3d as o3d
import open3d.geometry as o3d_geom
import panda_py.libfranka
import pyrealsense2 as rs
import numpy as np
import sys, os
from collections import deque

import panda_py
from panda_py import libfranka
from panda_py import controllers

import transforms3d
import roboticstoolbox as rtb

import multiprocessing as mp

try:
    mp.set_start_method("spawn")
except:
    pass
from multiprocessing import Process, Queue, Event

import time
import threading
import concurrent.futures
from numba import njit

from gensim2.env.utils.pcd_utils import (
    uniform_sampling,
    fps_sampling,
    pcd_filter_bound,
    BOUND,
)
from gensim2.agent.utils.pcd_utils import *
from gensim2.agent.utils.calibration import *
from gensim2.env.task.origin_tasks import *
from gensim2.env.task.primitive_tasks import *
from gensim2.env.create_task import create_gensim
from gensim2.paths import GENSIM_DIR, ASSET_ROOT

sys.path.append(f"{GENSIM_DIR}/agent/third_party/")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# np.set_printoptions(threshold=sys.maxsize)

hostname = "172.16.0.2"
cameras = {
    "wrist_cam": "239722072125",
    "right_cam": "233522072900",
    "left_cam": "233622078546",
}  # wrist, left, right cam


def init_given_realsense(device):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(str(device))
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    profile = pipeline.start(config)

    # Skip 15 first frames to give the Auto-Exposure time to adjust
    for x in range(15):
        pipeline.wait_for_frames()

    return pipeline, config, profile


def get_depth_filter():
    # filter stuff
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, 4)
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 5)
    spatial.set_option(rs.option.filter_smooth_alpha, 1)
    spatial.set_option(rs.option.filter_smooth_delta, 50)
    spatial.set_option(rs.option.holes_fill, 3)
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter(mode=2)

    return (
        depth_to_disparity,
        disparity_to_depth,
        decimation,
        spatial,
        temporal,
        hole_filling,
    )


@njit
def filter_vectors(v, rgb):
    norms = np.sqrt((v**2).sum(axis=1))
    valid = norms > 0
    points = v[valid]
    colors = rgb[valid]
    return points, colors


@njit
def transform_points(points, transform, transform_T):
    return points @ transform_T + transform[:3, 3]


@njit
def concatenate_points_colors(points, colors):
    return np.concatenate(
        [points.reshape(-1, 3), colors.reshape(-1, 3) / 255.0], axis=-1
    )


class RealsenseProcess(Process):
    def __init__(self, name, device, process_event, end_event, pcd_queue) -> None:
        super(RealsenseProcess, self).__init__()

        self.process_event = process_event
        self.end_event = end_event
        self.pcd_queue = pcd_queue

        self.device = device
        self.name = name

    def init(self):
        self.temp_data = deque(maxlen=10)
        self.intrinsics_array = np.zeros((7,), dtype=np.float32)

        # Paras to compute point cloud
        # convert gl to cv
        gl2cv = transforms3d.euler.euler2mat(np.pi, 0, 0)
        self.gl2cv_homo = np.eye(4)
        self.gl2cv_homo[:3, :3] = gl2cv

        # self.right_T_base = left_T_base
        # self.left_T_base = right_T_base

        # self.left_T_base = left_T_base @ self.gl2cv_homo
        # self.left_T_base[:3, 3] += [0.03, 0.01, 0.0]
        # self.left_T_base[:3, 3] += [0.0, 0.0, 0.01]

        # self.right_T_base = right_T_base @ self.gl2cv_homo
        # self.right_T_base[:3, 3] += [0.0, 0.0, 0.01]

        self.pipeline, self.align, self.profile = init_given_realsense(self.device)
        (
            self.depth_to_disparity,
            self.disparity_to_depth,
            self.decimation,
            self.spatial,
            self.temporal,
            self.hole_filling,
        ) = get_depth_filter()
        self.decimation = None  # We do not use it

        align_to = rs.stream.color
        align = rs.align(align_to)

        # get
        color_stream = self.profile.get_stream(rs.stream.color)
        intr = color_stream.as_video_stream_profile().get_intrinsics()
        intr_mat = np.eye(3)
        intr_mat[0, 0] = intr.fx
        intr_mat[1, 1] = intr.fy
        intr_mat[0, 2] = intr.ppx
        intr_mat[1, 2] = intr.ppy

        self.intr_inv = np.linalg.inv(intr_mat)

        order = ["fx", "fy", "ppx", "ppy", "height", "width"]
        for i, name in enumerate(order):
            self.intrinsics_array[i] = getattr(intr, name)

        # if self.enable_depth:
        #     depth_sensor = self.profile.get_device().first_depth_sensor()
        #     self.depth_scale = depth_sensor.get_depth_scale()
        #     self.intrinsics_array[-1] = self.depth_scale

    def get_color_from_tex_coords(self, tex_coords, color_image):
        us = (tex_coords[:, 0] * 640).astype(int)
        vs = (tex_coords[:, 1] * 480).astype(int)

        us = np.clip(us, 0, 639)
        vs = np.clip(vs, 0, 479)

        colors = color_image[vs, us]

        return colors

    def capture_pcd(self):
        # print(f"Capturing point cloud from {self.name} ...")

        frames = self.pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # !!!!!TODO: compute point cloud by hand
        # Use intrinsics to get the point cloud
        # depth = depth * depth_scale
        # v, u = np.indices(depth.shape)
        # z = depth
        # uv1 = np.stack([u, v, np.ones_like(z)], axis=-1)
        # pc = uv1 @ intrinsics_inv.T * z[..., None]

        # # Camera to world
        # pc = pc @ extrinsics[:3, :3].T
        # pc += extrinsics[:3, 3]

        pc = rs.pointcloud()
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)  # x, y, z
        # print(f"Process {self.name}, Time taken to calculate point cloud: ", time.time() - start)

        v = points.get_vertices()
        v = np.ascontiguousarray(v).view(np.float32).reshape(-1, 3)
        tex_coords = (
            np.ascontiguousarray(points.get_texture_coordinates())
            .view(np.float32)
            .reshape(-1, 2)
        )
        rgb = np.ascontiguousarray(color_frame.get_data())

        colors = self.get_color_from_tex_coords(tex_coords, rgb)
        colors = np.ascontiguousarray(colors)

        points, colors = filter_vectors(v, colors)

        # print(f"Process {self.name}, Time taken to filter points: ", time.time() - start)

        # valid_bool = np.linalg.norm(points, axis=1) < 3.0
        # points = points[valid_bool]
        # colors = colors[valid_bool]

        if self.name == "wrist_cam":
            tcp_T_base = self.tcp_pose_queue.get().astype(np.float32)
            print(tcp_T_base)
            wristcam_T_base = tcp_T_base @ wrist_T_tcp
            # wristcam_T_base = tcp_T_base @ wrist_T_tcp
            # wristcam_T_base = wristcam_T_base.view(np.float32)
            wristcam_T_base_T = np.ascontiguousarray(wristcam_T_base.T[:3, :3])
            points = transform_points(points, wristcam_T_base, wristcam_T_base_T)
        elif self.name == "left_cam":
            points = transform_points(points, left_T_base, left_T_base_T)
        elif self.name == "right_cam":
            points = transform_points(points, right_T_base, right_T_base_T)

        cam_pcd = np.concatenate(
            [points.reshape(-1, 3), colors.reshape(-1, 3) / 255.0], axis=-1
        )

        self.temp_data.append(cam_pcd)

    def run(self):
        self.init()

        while not self.end_event.is_set():
            if self.process_event.is_set() and len(self.temp_data) > 0:
                pcd = self.temp_data.pop()
                self.pcd_queue.put(pcd)
                self.process_event.clear()

            self.capture_pcd()

    def terminate(self) -> None:
        # self.pipeline.stop()
        return super().terminate()


class RealRobot:
    def __init__(self):
        print("Intializing robot ...")

        # Initialize robot connection and libraries
        self.panda = panda_py.Panda(hostname)
        self.gripper = libfranka.Gripper(hostname)
        self.panda.enable_logging(int(1e2))

        # sim setup
        self.sim_env = create_gensim(
            task_name="TestArm",
            asset_id="",
            sim_type="Sapien",
            use_gui=False,
            use_ray_tracing=True,
            eval=False,
            obs_mode="pointcloud",
            headless=False,
            cam="default",
        )
        self.sim_env.reset()

        self.planner = self.sim_env.task.env.planner
        self.agent = self.sim_env.task.env.agent

        # camera params
        self.pipelines = {}

        self.init_robot()
        self.init_cameras()

        # other
        self.current_step = 0
        self.horizon = 50  # TODO
        self._buffer = {}

        print("Finished initializing robot.")

    def init_cameras(self):

        self.cam_processes = {}
        self.cam_event = {}
        self.cam_queue = {}
        self.env_event = Event()
        self.tcp_pose_queue = deque(maxlen=10)

        self.tcp_pose_queue.put(self.panda.get_pose())

        for name, id in cameras.items():
            self.cam_event[name] = Event()
            self.cam_queue[name] = Queue()
            self.cam_processes[name] = RealsenseProcess(
                name, id, self.cam_event[name], self.env_event, self.cam_queue[name]
            )
            self.cam_processes[name].daemon = True
            self.cam_processes[name].start()

    def init_robot(self):
        joint_pose = [
            0.00000000e00,
            -3.19999993e-01,
            0.00000000e00,
            -2.61799383e00,
            0.00000000e00,
            2.23000002e00,
            7.85398185e-01,
        ]

        self.panda.move_to_joint_position(joint_pose)
        self.gripper.move(width=0.08, speed=0.1)

        # replicate in sim
        action = np.zeros(self.sim_env.action_space.shape)
        action[:-1] = np.array(
            [
                0.00000000e00,
                -3.19999993e-01,
                0.00000000e00,
                -2.61799383e00,
                0.00000000e00,
                2.23000002e00,
                7.85398185e-01,
            ]
        )
        action[-1] = 1.0

        self.sim_env.step(action=action)

    def log_pose(self):
        while True:
            try:
                tmp = np.ascontiguousarray(self.panda.get_pose()).astype(np.float32)
                # log = self.panda.get_log().copy()
                # tmp = np.ascontiguousarray(log['O_T_EE'][-1]).reshape(4, 4).astype(np.float32)
                # tmp[3, :] = [0.0, 0.0, 0.0, 1.0]
                # # print(tmp)
                self.tcp_pose_queue.put(tmp)
                # time.sleep(0.001)
            except:
                print("FAILED")
                time.sleep(0.05)
                pass

    def vis_sim(self, obs):
        pcds = obs["pointcloud"]

        pcd_o3d = o3d.geometry.PointCloud()

        pcd = pcds["pos"]
        # pcds["pos"] = open3d_pcd_outlier_removal(pcd)  # should use in real
        # pcds["pos"] = add_crop_noise_to_points_data(pcd)
        # pcds["pos"] = add_pointoutlier_aug(pcd)
        # pcds["pos"] = add_gaussian_noise(pcd)  # Should use in sim
        # pcds["pos"] = randomly_drop_point(pcd)  # Should use in sim
        # pcds["pos"] = cutplane_pointcloud_aug(pcd)

        pcd_o3d.points = o3d.utility.Vector3dVector(pcds["pos"])
        # pcd_o3d.colors = o3d.utility.Vector3dVector(pcds["colors"]/255)

        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )

        # Initialize the visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Add the point cloud to the visualizer
        vis.add_geometry(pcd_o3d)
        vis.add_geometry(coordinate_frame)

        # Get the render options and set the point size
        render_option = vis.get_render_option()
        render_option.point_size = 5.0  # Set the point size to your desired value

        # Run the visualizer
        vis.run()
        vis.destroy_window()

    def vis_pcd(self, pcds):
        geom_list = []

        front_cam_pcd = o3d.geometry.PointCloud()
        front_cam_pcd.points = o3d.utility.Vector3dVector(pcds[..., :3])
        front_cam_pcd.colors = o3d.utility.Vector3dVector(pcds[..., 3:])
        # front_cam_pcd.points = o3d.utility.Vector3dVector(points)
        # front_cam_pcd.colors = o3d.utility.Vector3dVector(colors)

        # save the point cloud
        # o3d.io.write_point_cloud("combined_pcd.ply", front_cam_pcd)
        geom_list.append(front_cam_pcd)
        geom_list.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05))

        o3d.visualization.draw_geometries(geometry_list=geom_list)

    @property
    def tcp_pose(self):
        return np.ascontiguousarray(self.panda.get_pose()).astype(np.float32)

    def get_pcd(self, save=False, visualize=False):
        pcds = {}
        start_time = time.time()

        # thread the camera event to get the pcd

        for name in cameras.keys():
            self.cam_event[name].set()

        tmp_pcd = {
            "wrist_cam": {"points": None, "colors": None},
            "left_cam": {"points": None, "colors": None},
            "right_cam": {"points": None, "colors": None},
        }

        time.sleep(0.5)

        for name in cameras.keys():
            tmp_pcd[name]["points"], tmp_pcd[name]["colors"] = self.cam_queue[
                name
            ].get()

        for name in cameras.keys():
            points = tmp_pcd[name]["points"]
            colors = tmp_pcd[name]["colors"]

            if name == "wrist_cam":
                wristcam_T_base = self.tcp_pose @ wrist_T_tcp
                wristcam_T_base_T = np.ascontiguousarray(wristcam_T_base.T[:3, :3])
                points = transform_points(points, wristcam_T_base, wristcam_T_base_T)
            elif name == "left_cam":
                points = transform_points(points, left_T_base, left_T_base_T)
            elif name == "right_cam":
                points = transform_points(points, right_T_base, right_T_base_T)

            pcds[name] = np.concatenate(
                [points.reshape(-1, 3), colors.reshape(-1, 3) / 255.0], axis=-1
            )

        print(f"Time taken for getting pcd from queue: {time.time() - start_time}")
        pcds = np.concatenate([x for x in pcds.values()], axis=0)  # merge pcds

        # if visualize:
        #     self.vis_pcd(pcds)

        start_time = time.time()
        mask = pcd_filter_bound(pcds[..., :3])  # filter out points outside the bound
        pcds = pcds[mask]

        pcds = pcds[uniform_sampling(pcds, npoints=16384)]
        pcds = pcds[fps_sampling(pcds, npoints=8192)]

        points, colors = open3d_pcd_outlier_removal(
            pcds, radius_nb_num=300, radius=0.3, std_nb_num=20
        )

        points, colors = open3d_pcd_outlier_removal(
            pcds, radius_nb_num=20, radius=0.02, std_nb_num=20
        )

        pcds = np.concatenate([points, colors], axis=1)
        pcds = pcds[fps_sampling(pcds, npoints=1200)]

        if visualize:
            self.vis_pcd(pcds)

        # return {"pos": points, "colors": colors}
        return {"pos": pcds[..., :3], "colors": pcds[..., 3:]}

    def get_robot_state(self):
        """
        Get the real robot state.
        """

        # log = self.panda.get_log().copy()

        gripper_state = self.gripper.read_once()
        gripper_qpos = gripper_state.width

        robot_qpos = np.concatenate(
            [self.panda.get_log()["q"][-1], [gripper_qpos / 2.0]]
        )

        obs = np.concatenate(
            [self.panda.get_position(), self.panda.get_orientation(), robot_qpos],
            dtype=np.float32,  # 15
        )

        assert obs.shape == (15,), f"incorrect obs shape, {obs.shape}"

        return obs

    def get_obs(self, visualize=False):
        """
        Get the real robot observation.
        """
        start_time = time.time()
        pcd = self.get_pcd(visualize=visualize)
        # print(f"Time taken to get pcd: {time.time() - start_time}")
        start_time = time.time()
        state = self.get_robot_state()
        # print(f"Time taken to get state: {time.time() - start_time}")
        # obs = {"state": self.get_robot_state(), "pointcloud": self.get_pcd(visualize=visualize)}
        obs = {"state": state, "pointcloud": pcd}

        return obs

    def step(self, action, visualize=False):
        """
        Step robot in the real.
        """
        # not quite sure what the real robot action space is

        # Simple motion in cartesian space
        gripper = action[-1] * 0.08
        euler = action[3:-1]  # Euler angle
        quat = transforms3d.euler.euler2quat(*euler)

        pose = np.concatenate([action[:3], quat], axis=0)

        try:
            results = self.planner.plan_screw(
                pose, self.agent.get_qpos(), time_step=0.1
            )
            waypoints = results["position"][..., np.newaxis]

            self.panda.move_to_joint_position(waypoints=waypoints, speed_factor=0.2)
            self.gripper.move(width=gripper, speed=0.1)

            q_pose = np.concatenate(
                [self.panda.q, [gripper / 2.0], [gripper / 2.0]], axis=0
            )
            self.agent.set_qpos(q_pose)
        except:
            print("Failed to generate valid waypoints.")

        return self.get_obs(visualize=visualize)

    def end(self):
        self.env_event.set()
        for name in cameras.keys():
            self.cam_processes[name].join()
        self.panda.get_robot().stop()


if __name__ == "__main__":
    # test if each method works
    robot = RealRobot()

    robot.get_pcd(visualize=True)
