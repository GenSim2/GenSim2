import numpy as np
import time
import enum
import json
import cv2

import pyrealsense2 as rs


from typing import Optional, Callable, Dict

import multiprocessing as mp

mp.set_start_method("fork", force=True)
import logging
from multiprocessing.managers import SharedMemoryManager
from threadpoolctl import threadpool_limits
from gensim2.agent.utils.shared_memory.shared_memory_ring_buffer import (
    SharedMemoryRingBuffer,
)
from gensim2.agent.utils.shared_memory.shared_ndarray import SharedNDArray
from gensim2.agent.utils.shared_memory.shared_memory_queue import (
    SharedMemoryQueue,
    Empty,
)
from gensim2.agent.utils.camera.timestamp_accumulator import (
    get_accumulate_timestamp_idxs,
)


class SingleRealsense(mp.Process):
    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        name,
        serial_number,
        resolution=(640, 480),
        transform: np.ndarray = None,
        transform_T: np.ndarray = None,
        capture_fps=30,
        put_fps=None,
        get_max_k=30,
        verbose=False,
    ):

        super().__init__()

        if put_fps is None:
            put_fps = capture_fps

        # create ring buffer
        resolution = tuple(resolution)
        shape = resolution[::-1]
        examples = dict()

        examples["color"] = np.empty(shape=shape + (3,), dtype=np.uint8)
        examples["vertices"] = np.empty(
            shape=(resolution[0] * resolution[1], 3), dtype=np.float32
        )
        examples["tex"] = np.empty(
            shape=(resolution[0] * resolution[1], 2), dtype=np.float32
        )

        examples["camera_capture_timestamp"] = 0.0
        examples["camera_receive_timestamp"] = 0.0
        examples["timestamp"] = 0.0
        examples["step_idx"] = 0

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=put_fps,
        )

        # copied variables
        self.name = name
        self.serial_number = serial_number
        self.resolution = resolution
        self.capture_fps = capture_fps
        self.put_fps = put_fps
        self.verbose = verbose
        self.put_start_time = None
        self.transform = transform
        self.transform_T = transform_T

        # shared variables
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.ring_buffer = ring_buffer

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= user API ===========
    def start(self, wait=True, put_start_time=None):
        self.put_start_time = put_start_time
        super().start()
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.end_wait()

    def start_wait(self):
        self.ready_event.wait()

    def end_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def get(self, k=None, out=None):
        """
        Get the latest k frames from the ring buffer if k is not None.
        """
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k, out=out)

    def run(self):
        """
        Main loop.
        """
        threadpool_limits(1)
        cv2.setNumThreads(1)

        w, h = self.resolution
        fps = self.capture_fps

        # configure realsense
        rs_config = rs.config()

        rs_config.enable_stream(rs.stream.color, w, h, rs.format.rgb8, fps)
        rs_config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)

        align_to = rs.stream.color
        rs.align(align_to)

        try:
            rs_config.enable_device(self.serial_number)

            # start pipeline
            pipeline = rs.pipeline()
            pipeline_profile = pipeline.start(rs_config)

            # report global time
            d = pipeline_profile.get_device().first_color_sensor()
            d.set_option(rs.option.global_time_enabled, 1)

            # put frequency regulation
            put_idx = None
            put_start_time = self.put_start_time
            if put_start_time is None:
                put_start_time = time.time()

            iter_idx = 0
            t_start = time.time()

            # iterate through the first few frames to get the camera ready
            for i in range(10):
                pipeline.wait_for_frames()

            while not self.stop_event.is_set():

                # wait for frames to come in
                frameset = pipeline.wait_for_frames()
                receive_time = time.time()

                # grab data
                data = dict()
                data["camera_receive_timestamp"] = receive_time
                # realsense report in ms
                data["camera_capture_timestamp"] = frameset.get_timestamp() / 1000.0

                color_frame = frameset.get_color_frame()
                data["color"] = np.asarray(color_frame.get_data())

                t = color_frame.get_timestamp() / 1000
                data["camera_capture_timestamp"] = t

                depth_frame = frameset.get_depth_frame()

                # working with frequency restriction first
                pc = rs.pointcloud()
                pc.map_to(color_frame)
                points = pc.calculate(depth_frame)

                data["vertices"] = (
                    np.ascontiguousarray(points.get_vertices())
                    .view(np.float32)
                    .reshape(-1, 3)
                )
                data["tex"] = (
                    np.ascontiguousarray(points.get_texture_coordinates())
                    .view(np.float32)
                    .reshape(-1, 2)
                )

                put_data = data

                local_idxs, global_idxs, put_idx = get_accumulate_timestamp_idxs(
                    timestamps=[receive_time],
                    start_time=put_start_time,
                    dt=1.0 / self.put_fps,
                    next_global_idx=put_idx,
                    allow_negative=True,
                )

                for step_idx in global_idxs:
                    put_data["step_idx"] = step_idx
                    put_data["timestamp"] = receive_time
                    self.ring_buffer.put(put_data, wait=True)

                # signal ready
                if iter_idx == 0:
                    self.ready_event.set()

                t_end = time.time()
                duration = t_end - t_start
                frequency = np.round(1 / duration, 1)
                t_start = t_end

                iter_idx += 1
                if self.verbose:
                    print(f"[SingleRealsense {self.serial_number}] FPS {frequency}")
        finally:
            rs_config.disable_all_streams()
            self.ready_event.set()

        if self.verbose:
            print(f"[SingleRealsense {self.serial_number}] Exiting worker process.")
