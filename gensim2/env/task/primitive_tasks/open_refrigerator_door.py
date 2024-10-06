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


class OpenRefrigeratorDoor(GenSimBaseTask):
    """Open the refrigerator door."""

    def __init__(self, env, asset_id=""):
        self.task_description = "Open the refrigerator door"
        self.sub_tasks = ["OpenRefrigeratorDoor"]
        self.sub_task_descriptions = ["Open the refrigerator door"]
        self.success_criteria = ["articulated_open"]
        # Initialize the task with the refrigerator as the articulated object.
        # The asset_id '10638' is specified for the refrigerator to ensure the correct instance is used.
        super().__init__(
            env, articulator="refrigerator", rigid_body=None, asset_id=asset_id
        )

    def get_progress_state(self):
        # Check if the refrigerator door is open enough to place the lemon inside.
        # The threshold can be adjusted based on the simulation environment's specifics.
        return check_openness(
            openness=self.articulator.get_openness()[0], fraction=1, threshold=0.1
        )

    def reset(self, random=False):
        # Reset the environment to the default state or a random state.
        super().reset()

        if not random:
            # Set the refrigerator door to its default closed position.
            set_default_openness(self.articulator, self.articulator_name, self.asset_id)
            # Set the refrigerator to its default pose.
            set_default_pose(self.articulator, self.articulator_name, self.asset_id)
        else:
            # Set the refrigerator door to a random openness.
            set_random_openness(self.articulator, self.articulator_name, self.asset_id)
            # Set the refrigerator to a random pose.
            set_random_pose(
                self.articulator, self.articulator_name, self.asset_id, self.task_type
            )

        return self.env.get_observation()
