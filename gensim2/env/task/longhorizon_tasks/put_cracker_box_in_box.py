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


class PutCrackerBoxInBox(GenSimBaseTask):
    """Pick up the cracker box and place it inside the opened box, then close the box lid."""

    def __init__(self, env, asset_id=""):
        # Define the task description and the sequence of sub-tasks required to complete the main task.
        self.task_description = "Pick up the cracker box and place it inside the opened box, then close the box lid."
        self.sub_tasks = [
            "OpenBox",
            "ReachRigidBody",
            "Grasp",
            "MoveInsideBox",
            "UnGrasp",
            "CloseBox",
        ]
        self.sub_task_descriptions = [
            "Open a box",
            "Reach a cuboid-like object",
            "Close the gripper",
            "Move the gripper inside the box with the cracker box",
            "Open the gripper",
            "Close the box lid",
        ]
        self.success_criteria = [
            "articulated_open",
            "distance_gripper_rigidbody",
            "",
            "distance_articulated_rigidbody",
            "",
            "articulated_closed",
        ]

        # Initialize the task with the box as the articulated object and the cracker box as the rigid body object.
        super().__init__(
            env, articulator="box_rotate", rigid_body="cracker_box", asset_id=asset_id
        )

    def reset(self, random=False):
        # Reset the environment to the default state or a random state.
        super().reset()

        if not random:
            # Set the box and cracker box to their default positions and states.
            set_default_openness(self.articulator, self.articulator_name, self.asset_id)
            set_default_pose(self.articulator, self.articulator_name, self.asset_id)
            set_default_pose(self.rigid_body, self.rigid_body_name)
        else:
            # Set the box and cracker box to random positions and states.
            set_random_pose(
                self.articulator, self.articulator_name, self.asset_id, self.task_type
            )
            set_random_openness(self.articulator, self.articulator_name, self.asset_id)
            set_random_pose(self.rigid_body, self.rigid_body_name, task=self.task_type)

        return self.env.get_observation()
