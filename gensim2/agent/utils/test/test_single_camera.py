import numpy as np
import time
import pyrealsense2 as rs

from gensim2.agent.utils.camera.single_cam import SingleRealsense
from multiprocessing.managers import SharedMemoryManager
from gensim2.agent.utils.calibration import *


if __name__ == "__main__":
    # Create a camera object
    serial_number = "239722073976"

    shm_manager = SharedMemoryManager()
    shm_manager.start()

    camera = SingleRealsense(
        shm_manager=shm_manager,
        name="right_cam",
        serial_number=serial_number,
        resolution=(640, 480),
        transform=right_T_base,
        transform_T=right_T_base_T,
        capture_fps=30,
        put_fps=None,
        get_max_k=10,
        verbose=False,
    )

    # Start the camera
    camera.start()

    time.sleep(0.5)

    print(camera.get(k=3))

    # Stop the camera
    camera.stop()
