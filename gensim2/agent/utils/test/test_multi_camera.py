import numpy as np
import time
import pyrealsense2 as rs

import panda_py
from panda_py import libfranka

import multiprocessing as mp

try:
    mp.set_start_method("spawn")
except:
    pass
from multiprocessing.managers import SharedMemoryManager
import logging

from gensim2.agent.utils.camera.multi_cam import MultiRealsense
from gensim2.agent.utils.shared_memory.shared_memory_queue import SharedMemoryQueue
from gensim2.agent.utils.calibration import *

import open3d as o3d

hostname = "172.16.0.2"
cameras = {
    "wrist_cam": "239722072125",
    "right_cam": "233522072900",
}  # wrist, left, right cam


def vis_pcd(pcds):
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


class RobotData:
    def __init__(self):
        self.panda = panda_py.Panda(hostname)
        self.robot = self.panda.get_robot()
        self.panda.move_to_start()

        self.state = self.panda.get_pose()

        logging.basicConfig(level=logging.INFO)
        self.panda.enable_logging(int(1e2))

    def __call__(self, state: libfranka.RobotState):
        self.state = state
        print(self.state)

    def move_panda(self):
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

            self.panda.move_to_joint_position(joint_pose, speed_factor=0.005)

            self.panda.move_to_start(speed_factor=0.005)

    def start_logging(self):
        while True:
            try:
                log = self.panda.get_log().copy()
                print(log["O_T_EE"][-1].reshape(4, 4).astype(np.float32))
                print(self.panda.get_pose().astype(np.float32))
            except:
                print("No log available")
        # self.robot.read(self.__call__)


# "left_cam": "023522070524",
if __name__ == "__main__":
    logger = mp.log_to_stderr()
    logger.setLevel(mp.SUBDEBUG)

    shm_manager = SharedMemoryManager()
    shm_manager.start()

    data = RobotData()

    pose_queue = SharedMemoryQueue.create_from_examples(
        shm_manager=shm_manager,
        examples={"pose": np.eye(4, dtype=np.float32), "timestamp": 0.0},
        buffer_size=5,
    )

    realsense = MultiRealsense(
        serial_numbers=cameras,
        shm_manager=shm_manager,
        resolution=(640, 480),
        capture_fps=30,
        put_fps=None,
        pose_queue=pose_queue,
    )

    realsense.start(wait=True)

    while not realsense.is_ready:
        time.sleep(0.1)

    vis_pcd(realsense.get()["pcds"])

    realsense.stop()
