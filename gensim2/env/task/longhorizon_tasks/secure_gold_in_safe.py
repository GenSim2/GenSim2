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


class SecureGoldInSafe(GenSimBaseTask):
    """Place the gold bar inside the opened safe and then close the safe."""

    def __init__(self, env, asset_id=""):
        # Task description is a brief summary of what the task is about.
        self.task_description = (
            "Place the gold bar inside the opened safe and then close the safe."
        )
        # Sub-tasks are the individual steps required to complete the long-horizon task.
        self.sub_tasks = [
            "OpenSafe",
            "ReachRigidBody",
            "Grasp",
            "MoveIntoSafe",
            "UnGrasp",
            "CloseSafe",
        ]
        # Sub-task descriptions provide more details on each sub-task.
        self.sub_task_descriptions = [
            "Rotate the safe's door to an open position.",
            "Reach the gold bar and make it ready for grasping",
            "Close the gripper to grasp the gold bar",
            "Move the gripper holding the gold bar into the safe",
            "Open the gripper to release the gold bar",
            "Close the safe's door to secure the gold inside",
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

        # Initialize the task with the safe as the articulated object and the gold bar as the rigid body object.
        # The 'articulator' and 'rigid_body' arguments are used to specify the objects involved in the task.
        super().__init__(
            env, articulator="safe_rotate", rigid_body="gold", asset_id=asset_id
        )

    def reset(self, random=False):
        # Reset the environment to the default state or a random state.
        super().reset()

        # Set the initial state of the articulated object (safe) and the rigid body object (gold bar).
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
