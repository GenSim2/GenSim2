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


class DropAppleIntoDrawer(GenSimBaseTask):
    """Open the drawer, pick up an apple, and drop it into the drawer."""

    def __init__(self, env, asset_id=""):
        # Define the task description and the sequence of sub-tasks required to complete the long-horizon task.
        self.task_description = (
            "Open the drawer, pick up an apple, and drop it into the drawer."
        )
        self.sub_tasks = [
            "OpenDrawer",
            "ReachRigidBody",
            "Grasp",
            "MoveToDrawer",
            "UnGrasp",
        ]
        self.sub_task_descriptions = [
            "Open a drawer",
            "Reach an apple with the gripper",
            "Close the gripper to grasp the apple",
            "Move the gripper with the apple above the drawer",
            "Open the gripper to drop the apple into the drawer",
        ]
        self.success_criteria = [
            "articulated_open",
            "distance_gripper_rigidbody",
            "",
            "distance_articulated_rigidbody",
            "",
        ]

        # Initialize the task with the drawer as the articulated object and the apple as the rigid body object.
        super().__init__(
            env, articulator="drawer", rigid_body="apple", asset_id=asset_id
        )

    def reset(self, random=False):
        # Reset the environment to the default state or a random state.
        super().reset()

        # Set the initial state of the drawer and the apple.
        if not random:
            set_default_openness(self.articulator, self.articulator_name, self.asset_id)
            set_default_pose(self.articulator, self.articulator_name, self.asset_id)
            set_default_pose(self.rigid_body, self.rigid_body_name)
        else:
            # For a random initial state, set random poses and openness for the drawer and the apple.
            set_random_pose(
                self.articulator, self.articulator_name, self.asset_id, self.task_type
            )
            set_random_openness(self.articulator, self.articulator_name, self.asset_id)
            set_random_pose(self.rigid_body, self.rigid_body_name, task=self.task_type)

        return self.env.get_observation()
