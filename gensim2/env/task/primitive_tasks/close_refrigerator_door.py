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
from gensim2.env.utils.rewards import *


class CloseRefrigeratorDoor(GenSimBaseTask):
    """Close the refrigerator door."""

    def __init__(self, env, asset_id=""):
        # Define the task description and success criteria based on the task definition
        self.task_description = (
            "Close the refrigerator door to a fully closed position."
        )
        self.sub_tasks = ["CloseRefrigeratorDoor"]
        self.sub_task_descriptions = ["Close the refrigerator door"]
        self.success_criteria = ["articulated_closed"]

        # Initialize the base task with the specific articulated object 'refrigerator'
        # The articulator parameter is set to the asset name 'refrigerator' as defined in the task
        super().__init__(env, articulator="refrigerator", asset_id=asset_id)

    def reset(self, random=False):
        # Reset the environment and set the initial state of the refrigerator door
        super().reset()

        # If not random, set the refrigerator door to a default open position
        # If random, set a random pose and openness for the refrigerator door
        if not random:
            set_default_openness(self.articulator, self.articulator_name, self.asset_id)
            set_default_pose(self.articulator, self.articulator_name, self.asset_id)
        else:
            set_random_pose(
                self.articulator, self.articulator_name, self.asset_id, self.task_type
            )
            set_random_openness(self.articulator, self.articulator_name, self.asset_id)

        # Return the initial observation of the environment
        return self.env.get_observation()
