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


class StoreBlockInBox(GenSimBaseTask):
    """Open the lid of the box, store a block into a slot inside the box, and close the lid."""

    def __init__(self, env, asset_id=""):
        # Task description is a brief summary of what the task is about.
        self.task_description = "Open the lid of the box, store a block into a slot inside the box, and close the lid."
        # Sub-tasks are the individual steps required to complete the long-horizon task.
        self.sub_tasks = [
            "OpenBox",
            "ReachXAxis",
            "Grasp",
            "MoveInsideBox",
            "UnGrasp",
            "CloseBox",
        ]
        # Sub-task descriptions provide more details on each sub-task.
        self.sub_task_descriptions = [
            "Rotate the box lid to an open position.",
            "Move the gripper to the position of the wood block to prepare for grasping.",
            "Close the gripper to grasp the block",
            "Move the gripper with the grasped block inside the box.",
            "Open the gripper to release the block ",
            "Rotate the box lid to a closed position to secure the block inside.",
        ]
        # Success criteria define how to evaluate if a sub-task has been successfully completed.
        self.success_criteria = [
            "articulated_open",
            "distance_gripper_rigidbody",
            "",
            "distance_articulated_rigidbody",
            "",
            "articulated_closed",
        ]

        # Initialize the task with the box as the articulated object and the gold coin as the rigid body object.
        # The 'articulator' and 'rigid_body' arguments are used to specify the objects involved in the task.
        super().__init__(
            env, articulator="box_rotate", rigid_body="wood_block", asset_id=asset_id
        )

    def reset(self, random=False):
        # Reset the environment to the default state or a random state.
        super().reset()

        # Set the initial state of the articulated object (box) and the rigid body object (coin).
        # If not random, set to default positions; otherwise, set to random positions within the task constraints.
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

        # Return the initial observation of the environment after reset.
        return self.env.get_observation()
