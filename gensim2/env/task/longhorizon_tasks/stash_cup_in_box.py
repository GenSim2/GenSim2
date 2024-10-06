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


class StashCupInBox(GenSimBaseTask):
    """Pick up a marker and place it inside an open drawer, then close the drawer."""

    def __init__(self, env, asset_id=""):
        # Define the task description
        self.task_description = "Pick up a marker and place it inside an open drawer, then close the drawer."
        # Define the sequence of sub-tasks required to complete the long-horizon task
        self.sub_tasks = [
            "OpenBox",
            "ReachRigidBodyHorizontal",
            "Grasp",
            "MoveInsideBox",
            "UnGrasp",
            "CloseBox",
        ]
        # Define the descriptions for each sub-task
        self.sub_task_descriptions = [
            "Open the drawer.",
            "Reach the marker and align the gripper above it.",
            "Grasp the marker with the gripper.",
            "Move the gripper with the marker to align above the open drawer.",
            "Release the marker inside the drawer.",
            "Close the drawer with the marker inside.",
        ]
        # Define the success criteria for each sub-task
        self.success_criteria = [
            "articulated_open",
            "distance_gripper_rigidbody",
            "",
            "distance_gripper_articulated",
            "",
            "articulated_closed",
        ]

        # Initialize the task with the drawer as the articulated object and the marker as the rigid body object.
        super().__init__(
            env, articulator="box_rotate", rigid_body="cup", asset_id=asset_id
        )

    def reset(self, random=False):
        # Reset the environment to the default state or a random state.
        super().reset()

        # Set the initial state of the drawer and marker
        if not random:
            set_default_openness(self.articulator, self.articulator_name, self.asset_id)
            set_default_pose(self.articulator, self.articulator_name, self.asset_id)
            set_default_pose(self.rigid_body, self.rigid_body_name)
        else:
            # Set a random state for the drawer and marker if the task is randomized
            set_random_pose(
                self.articulator, self.articulator_name, self.asset_id, self.task_type
            )
            set_random_openness(self.articulator, self.articulator_name, self.asset_id)
            set_random_pose(self.rigid_body, self.rigid_body_name, task=self.task_type)

        # Return the initial observation of the environment
        return self.env.get_observation()
