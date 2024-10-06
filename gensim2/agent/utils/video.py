import os
import cv2


def save_video(images, name, path, fps=10):
    """save video to path"""
    if isinstance(name, tuple):
        name = name[0]
    if not os.path.exists(path):
        os.makedirs(path)
    for key, val in images.items():
        video_path = os.path.join(path, f"{name}_{key}.mp4")
        writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (val[0].shape[1], val[0].shape[0]),
        )
        for img in val:
            writer.write(img[..., :3])
        writer.release()
