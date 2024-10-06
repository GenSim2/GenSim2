import panda_py
from panda_py import libfranka

import time
import threading
import transforms3d
import roboticstoolbox as rtb
from scipy.spatial.transform import Rotation as R

import multiprocessing as mp

from multiprocessing import Process, Queue, Event
from multiprocessing.managers import SharedMemoryManager

from gensim2.agent.utils.camera.multi_cam import MultiRealsense
from gensim2.agent.utils.shared_memory.shared_memory_queue import SharedMemoryQueue
from gensim2.agent.utils.calibration import *
from gensim2.agent.utils.robot.utils import *

from gensim2.env.utils.pcd_utils import (
    uniform_sampling,
    fps_sampling,
    pcd_filter_bound,
    BOUND,
)
from gensim2.agent.utils.shared_memory.shared_memory_ring_buffer import (
    SharedMemoryRingBuffer,
)
from gensim2.agent.utils.pcd_utils import *
from gensim2.agent.utils.calibration import *
from gensim2.env.task.origin_tasks import *
from gensim2.env.task.primitive_tasks import *
from gensim2.env.create_task import create_gensim
from gensim2.paths import GENSIM_DIR, ASSET_ROOT

# TODO fill in these values
hostname = ""
cameras = {
    "wrist_cam": "",
    "right_cam": "",
    "left_cam": "",
}  # wrist, left, right cam


class RealRobot:
    def __init__(self):
        print("Intializing robot ...")

        # Initialize robot connection and libraries
        self.panda = panda_py.Panda(hostname)
        self.gripper = libfranka.Gripper(hostname)
        self.panda.enable_logging(int(10))

        # sim setup
        self.sim_env = create_gensim(
            task_name="TestArm",
            asset_id="",
            sim_type="Sapien",
            use_gui=False,
            use_ray_tracing=True,
            eval=False,
            obs_mode="state",
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
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()

        # get initial pose
        init_pose = self.panda.get_pose()

        # same for time
        init_time = time.time()

        self.pose_buffer = SharedMemoryRingBuffer.create_from_examples(
            self.shm_manager,
            examples={
                "pose": init_pose,
                "timestamp": init_time,
            },
            get_time_budget=0.002,
            get_max_k=100,
        )

        for i in range(100):
            self.pose_buffer.put(
                {
                    "pose": init_pose,
                    "timestamp": init_time,
                },
                wait=True,
            )

        self.realsense = MultiRealsense(
            serial_numbers=cameras,
            shm_manager=self.shm_manager,
            resolution=(640, 480),
            capture_fps=30,
            put_fps=None,
            pose_buffer=self.pose_buffer,
        )

        self.realsense.start(wait=True)

        while not self.realsense.is_ready:
            time.sleep(0.1)

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
        self.gripper.move(width=0.0, speed=0.1)

        # replicate in sim
        action = np.zeros((9,))
        action[:-2] = joint_pose
        self.agent.set_qpos(action)

    def log_pose(self, verbose=False):
        while True:
            start_time = time.time()

            pose = np.ascontiguousarray(self.panda.get_pose()).astype(np.float32)
            init_time = time.time()

            data = {
                "pose": pose,
                "timestamp": init_time,
            }

            self.pose_buffer.put(data)

            elapsed_time = time.time() - start_time
            if elapsed_time < 0.001:
                time.sleep(0.001 - elapsed_time)

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

    @property
    def tcp_pose(self):
        return np.ascontiguousarray(self.panda.get_pose()).astype(np.float32)

    def get_robot_state(self):
        """
        Get the real robot state.
        """
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

        visualize: bool, whether to visualize the pointcloud.
        """
        pcds = self.realsense.get()
        pcds = pcds["pcds"]
        pcd = {"pos": pcds[..., :3], "colors": pcds[..., 3:]}

        if visualize:
            vis_pcd(pcds)

        state = self.get_robot_state()

        obs = {"state": state, "pointcloud": pcd}

        return obs

    def step(self, action, visualize=False):
        """
        Step robot in the real.
        """
        # Simple motion in cartesian space
        gripper = action[-1] * 0.08
        euler = action[3:-1]  # Euler angle
        quat = transforms3d.euler.euler2quat(*euler)

        pose = np.concatenate([action[:3], quat], axis=0)
        print(pose)

        try:
            results = self.planner.plan_screw(
                pose, self.agent.get_qpos(), time_step=0.1
            )
            waypoints = results["position"][..., np.newaxis]

            self.panda.move_to_joint_position(waypoints=waypoints, speed_factor=0.1)
            self.gripper.move(width=gripper, speed=0.3)

            q_pose = np.zeros((9,))
            q_pose[:-2] = self.panda.q
            q_pose[-2] = gripper / 2.0
            q_pose[-1] = gripper / 2.0

            self.agent.set_qpos(q_pose)
        except Exception as e:
            print(e)
            print("Failed to generate valid waypoints.")

        return self.get_obs(visualize=visualize)

    def test_sequence(self):
        """
        Test sequence of actions to test the robot.
        """
        for i in range(10):
            joint_pose = [
                0.00000000e00,
                -3.19999993e-01,
                0.00000000e00,
                -2.61799383e00,
                0.00000000e00,
                2.23000002e00,
                7.85398185e-01,
            ]

            self.panda.move_to_joint_position(joint_pose, speed_factor=0.1)

            self.get_obs(visualize=True)

            self.panda.move_to_start(speed_factor=0.1)

    def end(self):
        self.env_event.set()
        for name in cameras.keys():
            self.cam_processes[name].join()
        self.panda.get_robot().stop()


if __name__ == "__main__":
    data = RealRobot()

    t2 = threading.Thread(target=data.log_pose)
    t2.start()

    data.test_sequence()
    t2.join()
