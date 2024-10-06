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


class CloseSuitcaseLid(GenSimBaseTask):
    """Close the lid of a suitcase."""

    def __init__(self, env, asset_id=""):
        # Define the task description and sub-tasks
        self.task_description = "Close the lid of a suitcase."
        self.sub_tasks = ["CloseSuitcaseLid"]
        self.sub_task_descriptions = ["Close the lid of a suitcase"]
        self.success_criteria = ["articulated_closed"]
        # For articulated tasks, articulator should be specified, while others should be None.
        # The articulator is the suitcase with a revolute joint for the lid.
        super().__init__(env, articulator="suitcase_rotate", asset_id=asset_id)

    def reset(self, random=False):
        # Reset the environment and set the initial state of the suitcase
        super().reset()

        # If not random, set the suitcase lid to a default open position
        # If random, set a random pose and openness for the suitcase lid
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
