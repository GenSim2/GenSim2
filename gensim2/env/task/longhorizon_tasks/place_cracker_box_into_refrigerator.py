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


class PlaceCrackerBoxIntoRefrigerator(GenSimBaseTask):
    def __init__(self, env, asset_id=""):
        self.task_description = "Open the refrigerator door, pick and place the cracker box into the refrigerator"
        self.sub_tasks = [
            "OpenRefrigeratorDoor",
            "ReachCuboid",
            "Grasp",
            "MoveToRefrigerator",
            "UnGrasp",
            "CloseRefrigeratorDoor",
        ]
        self.sub_task_descriptions = [
            "Open the refrigerator door",
            "Reach a cuboid object",
            "Close the gripper",
            "Move the gripper into the refrigerator",
            "Open the gripper",
            "Close the refrigerator door",
        ]
        self.success_criteria = [
            "articulated_open",
            "distance_gripper_rigidbody",
            "",
            "distance_articulated_rigidbody",
            "",
            "articulated_closed",
        ]

        # Initialize the task with the microwave as the articulated object and the cracker box as the rigid body object.
        super().__init__(
            env, articulator="refrigerator", rigid_body="cracker_box", asset_id=asset_id
        )

    def reset(self, random=False):
        # Reset the environment to the default state or a random state.
        super().reset()

        # Set the microwave and cracker box to their default or random positions and states.
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
