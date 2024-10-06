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


class DropSpatulaInBucketAndLift(GenSimBaseTask):
    """Drop a spatula into an upright bucket and then lift the bucket."""

    def __init__(self, env, asset_id=""):
        self.task_description = (
            "Drop a spatula into an upright bucket and then lift the bucket."
        )
        self.sub_tasks = [
            "ReachThinObject",
            "Grasp",
            "MoveAboveBucket",
            "UnGrasp",
            "LiftBucketUpright",
        ]
        self.sub_task_descriptions = [
            "Reach the spatula with the gripper",
            "Grasp the spatula with the gripper",
            "Move the gripper above the bucket",
            "Release the spatula into the bucket",
            "Lift the bucket to a predefined height",
        ]
        self.success_criteria = [
            "distance_gripper_rigidbody",
            "",
            "distance_articulated_rigidbody",
            "",
            "articulated_open",
        ]

        # For articulated tasks, articulator should be specified, while others should be None.
        # Here, 'bucket_lift' is the articulated object that can be lifted, and 'spatula' is the rigid body object.
        super().__init__(
            env, articulator="bucket_lift", rigid_body="spatula", asset_id=asset_id
        )

    def reset(self, random=False):
        super().reset()

        # Set the bucket and spatula to their default positions.
        # The bucket is set upright and the spatula is placed on the table.
        if not random:
            set_default_openness(self.articulator, self.articulator_name, self.asset_id)
            set_default_pose(self.articulator, self.articulator_name, self.asset_id)
            set_default_pose(self.rigid_body, self.rigid_body_name)
        else:
            # For a random reset, place the bucket and spatula in random positions on the table.
            set_random_pose(
                self.articulator, self.articulator_name, self.asset_id, self.task_type
            )
            set_random_openness(self.articulator, self.articulator_name, self.asset_id)
            set_random_pose(self.rigid_body, self.rigid_body_name, task=self.task_type)

        return self.env.get_observation()
