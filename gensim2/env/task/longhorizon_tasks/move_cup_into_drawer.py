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


class MoveCupIntoDrawer(GenSimBaseTask):
    """Move a YCB object into a drawer."""

    def __init__(self, env, asset_id=""):
        # For articulated tasks, articulator should be specified, while others should be None.
        self.task_description = "Open the drawer, pick and place the cup into the drawer, and close the drawer."
        self.sub_tasks = [
            "OpenDrawer",
            "ReachRigidBodyEdge",
            "Grasp",
            "MoveToDrawer",
            "UnGrasp",
            "PushDrawerClose",
        ]
        self.sub_task_descriptions = [
            "Open the drawer",
            "Reach a regular rigid-body object",
            "Close the gripper",
            "Move the gripper into a drawer",
            "Open the gripper",
            "Push the drawer to a fully close position",
        ]
        self.success_criteria = [
            "articulated_open",
            "distance_gripper_rigidbody",
            "",
            "distance_articulated_rigidbody",
            "",
            "articulated_closed",
        ]

        super().__init__(env, articulator="drawer", rigid_body="cup", asset_id=asset_id)

    def reset(self, random=False):
        super().reset()

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
