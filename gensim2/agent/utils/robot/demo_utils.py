import panda_py
from panda_py import libfranka

import time
import threading
import transforms3d
import roboticstoolbox as rtb

import multiprocessing as mp

# mp.set_start_method("forkserver", force=True)
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
from gensim2.env.task.generated_tasks import *
from gensim2.env.create_task import create_gensim
from gensim2.paths import GENSIM_DIR, ASSET_ROOT

hostname = "172.16.0.2"
cameras = {
    "wrist_cam": "239722072125",
    "right_cam": "233522072900",
    "left_cam": "233622078546",
}


class Camera:
    def __init__(self) -> None:
        pass

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

    def get_obs(self, visualize=False):
        """
        Get the real robot observation.
        """
        start_time = time.time()
        pcds = self.realsense.get()
        pcds = pcds["pcds"]
        pcd = {"pos": pcds[..., :3], "colors": pcds[..., 3:]}
        # print(f"Time taken to get pcd: {time.time() - start_time}")
        if visualize:
            vis_pcd(pcds)
        start_time = time.time()
        state = self.get_robot_state()
        # print(f"Time taken to get state: {time.time() - start_time}")
        obs = {"state": state, "pointcloud": pcd}
