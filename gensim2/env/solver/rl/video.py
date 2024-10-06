import cv2
import imageio
import numpy as np
import os
from collections import OrderedDict
import ipdb
import PIL.Image as Image
import time


class VideoRecorder:
    """
    A class to record and save videos of environment simulations by capturing frames from multiple camera sources
    in a simulated environment.

    Attributes:
    -----------
    save_dir : Path or None
        Directory where videos will be saved. Created as root_dir / 'eval_video' if root_dir is provided, otherwise None.
    render_size : int
        Size of the rendered video frames. Default is 256.
    fps : int
        Frames per second for the output video. Default is 20.
    frames : OrderedDict
        Stores video frames for each camera source from the environment.
    enabled : bool
        Flag indicating whether video recording is enabled.

    Methods:
    --------
    __init__(self, root_dir, render_size=256, fps=20)
        Initializes the video recorder, setting up the save directory and video parameters.

    init(self, env, enabled=True)
        Initializes the recording, clears previous frames, and enables video capture if a save directory is set.

    record(self, env)
        Captures frames from the environment and stores them in the frames dictionary for each camera source.

    save(self, file_name)
        Saves the recorded frames as `.mp4` files in the specified directory.
    """

    def __init__(self, root_dir, render_size=256, fps=20):
        if root_dir is not None:
            self.save_dir = root_dir / "eval_video"
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = OrderedDict()

    def init(self, env, enabled=True):
        self.frames = OrderedDict()

        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def record(self, env):
        if self.enabled:
            # import ipdb
            # ipdb.set_trace()
            images = env.get_images()
            # right_images = images['right_cam']
            # images = env.get_image()

            if images is None:
                return

            for name, img in images.items():
                if name not in self.frames:
                    self.frames[name] = np.expand_dims(
                        img, axis=0
                    )  # (480, 640, 4) -> (1, 480, 640, 4)
                else:
                    # self.frames[name].extend(img)
                    # import ipdb
                    # ipdb.set_trace()
                    tmp_img = np.expand_dims(
                        img, axis=0
                    )  # (480, 640, 4) -> (1, 480, 640, 4)
                    self.frames[name] = np.vstack(
                        (self.frames[name], tmp_img)
                    )  # (1, 480, 640, 4) -> (2, 480, 640, 4)

    def save(self, file_name):
        path = None
        if self.enabled:
            if not os.path.exists(self.save_dir / file_name):
                os.makedirs(self.save_dir / file_name)

            for key in self.frames:
                file_ending = key + ".mp4"
                path = self.save_dir / file_name / file_ending
                print("Saving video to", path)
                # convert self.frames[key] from float32 to uint8
                # self.frames[key] = np.array(self.frames[key], dtype=np.uint8)
                imageio.mimsave(str(path), self.frames[key], fps=self.fps)

            # flush the frames
            self.frames = OrderedDict()
        return path
