import numpy as np
import sapien.core as sapien
import transforms3d
from gensim2.env.base.base_task import GenSimBaseTask
from gensim2.env.utils.rewards import *
import gym
from gensim2.env.utils.pose import *
from gensim2.env.utils.rewards import (
    l2_distance_reward,
    alignment_reward,
    progress_reward,
    check_qpos,
)
import numpy as np
import transforms3d

from gensim2.env.base.base_task import GenSimBaseTask
from gensim2.env.utils.pose import *
from gensim2.env.utils.rewards import (
    l2_distance_reward,
    joint_fraction_reward,
    check_openness,
    progress_reward,
)


class StoreAppleInBucket(GenSimBaseTask):
    """Pick up an apple and place it into a bucket that is initially on its side, then set the bucket upright."""

    def __init__(self, env, asset_id=""):
        # Define the task description and the sequence of sub-tasks required to complete the long-horizon task.
        self.task_description = "Pick up an apple and place it into a bucket that is initially on its side, then set the bucket upright."
        self.sub_tasks = [
            "ReachRigidBody",
            "Grasp",
            "MoveAboveBucket",
            "UnGrasp",
            "LiftBucketUpright",
        ]
        self.sub_task_descriptions = [
            "Reach the apple with the gripper",
            "Grasp the apple with the gripper",
            "Move the gripper holding the apple above the bucket",
            "Release the apple into the bucket",
            "Lift the bucket into an upright position with the apple inside",
        ]
        self.success_criteria = [
            "distance_gripper_rigidbody",
            "distance_gripper_rigidbody",
            "distance_articulated_rigidbody",
            "distance_gripper_rigidbody",
            "articulated_open",
        ]

        # For articulated tasks, articulator should be specified, while others should be None.
        # The bucket_lift is the articulated object that will be manipulated to simulate storing the apple.
        # The apple is the rigid body object that will be grasped and placed into the bucket.
        super().__init__(
            env, articulator="bucket_lift", rigid_body="apple", asset_id=asset_id
        )

    def reset(self, random=False):
        # Reset the environment to the default state or a random state.
        super().reset()

        # Set the default openness and pose for the articulated object (bucket).
        # Set the default pose for the rigid body object (apple).
        # If random is True, set a random pose and openness for the articulated object and a random pose for the rigid body object.
        if not random:
            set_default_openness(self.articulator, self.articulator_name, self.asset_id)
            set_default_pose(self.articulator, self.articulator_name, self.asset_id)
            set_default_pose(self.rigid_body, self.rigid_body_name)
        else:
            set_random_pose(
                self.articulator, self.articulator_name, self.asset_id, self.task_type
            )
            set_random_openness(self.articulator, self.articulator_name, self.asset_id)
            set_random_pose(self.rigid_body, self.rigid_body_name, task=self.task_type)

        return self.env.get_observation()
