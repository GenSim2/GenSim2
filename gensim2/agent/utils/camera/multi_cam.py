import numpy as np
import time

import pyrealsense2 as rs
import cv2
from threadpoolctl import threadpool_limits
from typing import List, Optional, Union, Dict, Callable

from multiprocessing.managers import SharedMemoryManager
import multiprocessing as mp

# mp.set_start_method("forkserver", force=True)
import logging

from gensim2.agent.utils.camera.single_cam import SingleRealsense
from gensim2.agent.utils.shared_memory.shared_memory_queue import SharedMemoryQueue
from gensim2.agent.utils.shared_memory.shared_memory_ring_buffer import (
    SharedMemoryRingBuffer,
)
from gensim2.env.utils.pcd_utils import uniform_sampling, fps_sampling, pcd_filter_bound
from gensim2.agent.utils.pcd_utils import (
    open3d_pcd_outlier_removal,
    dbscan_outlier_removal_idx,
)
from gensim2.agent.utils.calibration import *

from numba import njit
import fpsample


@njit
def filter_vectors(v, rgb):
    norms = np.sqrt((v**2).sum(axis=1))
    valid = norms > 0
    points = v[valid]
    colors = rgb[valid]
    return points, colors


def get_color_from_tex_coords(tex_coords, color_image):
    us = (tex_coords[:, 0] * 640).astype(int)
    vs = (tex_coords[:, 1] * 480).astype(int)

    us = np.clip(us, 0, 639)
    vs = np.clip(vs, 0, 479)

    colors = color_image[vs, us]

    return colors


class MultiRealsense(mp.Process):
    def __init__(
        self,
        serial_numbers: Optional[Dict[str, str]] = None,
        shm_manager: Optional[SharedMemoryManager] = None,
        resolution=(1280, 720),
        capture_fps=30,
        put_fps=None,
        pose_buffer=None,
    ):
        super().__init__()
        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()

        n_cameras = len(serial_numbers.keys())

        cameras = dict()
        for cam_name in serial_numbers.keys():
            cameras[cam_name] = SingleRealsense(
                shm_manager=shm_manager,
                name=cam_name,
                serial_number=serial_numbers[cam_name],
                resolution=resolution,
                capture_fps=capture_fps,
                transform=calib[cam_name]["transform"],
                transform_T=calib[cam_name]["rotation"],
                put_fps=put_fps,
                get_max_k=10,
            )

        pcd_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples={
                "pcds": np.empty((16384, 6), dtype=np.float32),
            },
            get_time_budget=0.3,
            get_max_k=10,
        )

        # shared variables
        self.cameras = cameras
        self.pose_buffer = pose_buffer
        self.pcd_ring_buffer = pcd_ring_buffer
        self.pcd_process_ready_event = mp.Event()
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()

        # logging variables
        self.logger = mp.log_to_stderr()
        self.logger.setLevel(mp.SUBDEBUG)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @property
    def is_ready(self):
        is_ready = True
        for camera in self.cameras.values():
            if not camera.is_ready:
                is_ready = False

        is_ready = is_ready and self.pcd_process_ready_event.is_set()
        return is_ready

    def start(self, wait=True, put_start_time=None):
        if put_start_time is None:
            put_start_time = time.time()

        for camera in self.cameras.values():
            camera.start(wait=False, put_start_time=put_start_time)
            time.sleep(1)

        if wait:
            self.start_wait()

        # sleep for a bit to allow the cameras to start
        super().start()

    def stop(self, wait=True):
        for camera in self.cameras.values():
            camera.stop(wait=False)

        self.stop_event.set()
        if wait:
            self.stop_wait()

    def start_wait(self):
        for camera in self.cameras.values():
            camera.start_wait()

        # self.ready_event.wait()

    def stop_wait(self):
        for camera in self.cameras.values():
            camera.join()

    def end_wait(self):
        self.join()

    def get(self) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Return

        {
            'pcd': (N, 4096, 6),
        }

        Get the data from all cameras
        Select three frames that are synchronized

        Create a merged pointcloud from all cameras
        """
        return self.pcd_ring_buffer.get()

    def run(self):
        threadpool_limits(16)
        cv2.setNumThreads(16)
        iter_idx = 0
        while not self.stop_event.is_set():

            out = dict()
            # get camera data
            for i, camera in enumerate(self.cameras.values()):
                this_out = None
                if i in out:
                    this_out = out[i]

                this_out = camera.get(out=this_out)

                out[i] = this_out

            # process camera data
            tmp = dict()
            for i, camera in enumerate(self.cameras.values()):
                v = out[i]["vertices"]
                tex_coords = out[i]["tex"]
                color = out[i]["color"]

                colors = get_color_from_tex_coords(tex_coords, color)
                colors = np.ascontiguousarray(colors)

                points, colors = filter_vectors(v, colors)

                if camera.name == "wrist_cam":
                    poses = self.pose_buffer.get_last_k(k=50)

                    pose_times = np.abs(poses["timestamp"] - out[i]["timestamp"])
                    pose_idx = np.argmin(pose_times)

                    diff = poses["timestamp"][pose_idx] - out[i]["timestamp"]
                    tcp_T_base = poses["pose"][pose_idx]

                    wristcam_T_base = tcp_T_base @ camera.transform
                    wristcam_T_base_T = np.ascontiguousarray(wristcam_T_base.T[:3, :3])
                    points = points @ wristcam_T_base_T + wristcam_T_base[:3, 3]

                else:
                    points = points @ camera.transform_T + camera.transform[:3, 3]

                tmp[camera.name] = np.concatenate(
                    [points.reshape(-1, 3), colors.reshape(-1, 3) / 255.0], axis=-1
                )

            pcds = np.concatenate([x for x in tmp.values()], axis=0)  # merge pcds
            mask = pcd_filter_bound(pcds[..., :3])
            pcds = pcds[mask]

            pcds = pcds[uniform_sampling(pcds, npoints=16384)]

            # commented out for data collection. Uncomment for testing

            fps_sampling_idx = fpsample.fps_sampling(pcds[..., :3], 5000)

            pcds = pcds[fps_sampling_idx]

            pcds = pcds[
                dbscan_outlier_removal_idx(pcds[..., :3], eps=0.3, min_samples=300)
            ]
            pcds = pcds[
                dbscan_outlier_removal_idx(pcds[..., :3], eps=0.02, min_samples=5)
            ]
            pcds = pcds[uniform_sampling(pcds, npoints=4096)]

            pcds = {"pcds": pcds}

            self.pcd_ring_buffer.put(pcds)

            if iter_idx == 0:
                self.pcd_process_ready_event.set()

            iter_idx += 1
