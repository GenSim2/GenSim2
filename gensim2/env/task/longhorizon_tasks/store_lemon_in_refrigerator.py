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


class StoreLemonInRefrigerator(GenSimBaseTask):
    """Open the refrigerator door, pick up a lemon, and place it inside the refrigerator."""

    def __init__(self, env, asset_id=""):
        # Define the task description and the sequence of sub-tasks required to complete the long-horizon task.
        self.task_description = "Open the refrigerator door, pick up a lemon, and place it inside the refrigerator."
        self.sub_tasks = [
            "OpenRefrigeratorDoor",
            "ReachRigidBody",
            "Grasp",
            "MoveToRefrigerator",
            "UnGrasp",
        ]
        self.sub_task_descriptions = [
            "Open the refrigerator door to access the interior.",
            "Reach the lemon to position the gripper around it.",
            "Close the gripper to grasp the lemon.",
            "Move the gripper with the grasped lemon into the refrigerator.",
            "Open the gripper to release the lemon inside the refrigerator.",
        ]
        self.success_criteria = [
            "articulated_open",
            "distance_gripper_rigidbody",
            "",
            "distance_articulated_rigidbody",
            "",
        ]

        # Initialize the task with the refrigerator as the articulated object and the lemon as the rigid body object.
        super().__init__(
            env, articulator="refrigerator", rigid_body="lemon", asset_id=asset_id
        )

    def reset(self, random=False):
        # Reset the environment to the default state or a random state.
        super().reset()

        if not random:
            # Set the refrigerator and lemon to their default positions and states.
            set_default_openness(self.articulator, self.articulator_name, self.asset_id)
            set_default_pose(self.articulator, self.articulator_name, self.asset_id)
            set_default_pose(self.rigid_body, self.rigid_body_name)
        else:
            # Set the refrigerator and lemon to random positions and states within the constraints of the task.
            set_random_pose(
                self.articulator, self.articulator_name, self.asset_id, self.task_type
            )
            set_random_openness(self.articulator, self.articulator_name, self.asset_id)
            set_random_pose(self.rigid_body, self.rigid_body_name, task=self.task_type)

        return self.env.get_observation()
