import json
import numpy as np
import sapien.core as sapien

from .rigid_body import RigidBody


class Tool(RigidBody):
    """
    load its corresponding keypoint json file
    """

    def __init__(self, info, keypoint_path):
        self.info = info
        self.keypoint_dict = json.load(open(keypoint_path))["keypoints"]
        keypoint_json = json.load(open(keypoint_path))
        self.scale = 1
        if "scale" in keypoint_json:
            self.scale = keypoint_json["scale"]
        self.instance = None

    def get_keypoints(self):
        keypoints_inworld = {}
        pose = self.instance.pose
        for name, kp in self.keypoint_dict.items():
            scaled_kp = np.array(kp) * self.scale
            kp_loc = sapien.Pose(scaled_kp)
            transformed_kp = pose.transform(kp_loc).p

            keypoints_inworld[name] = transformed_kp

        return keypoints_inworld

    def get_keypoint(self, name="head"):
        keypoint = self.keypoint_dict[name]

        # transform with the current pose
        pose = self.instance.pose
        keypoint_loc = sapien.Pose(keypoint)
        transformed_keypoint = pose.transform(keypoint_loc).p

        return transformed_keypoint

    def get_keypoint_vector(self, from_name, to_name):
        pose = self.instance.pose
        from_keypoint = self.get_keypoint(from_name)
        to_keypoint = self.get_keypoint(to_name)
        return np.array(to_keypoint) - np.array(from_keypoint)
