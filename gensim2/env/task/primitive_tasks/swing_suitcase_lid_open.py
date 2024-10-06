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


class SwingSuitcaseLidOpen(GenSimBaseTask):
    """Swing the lid of a suitcase open."""

    def __init__(self, env, asset_id=""):
        # Define the task description and sub-tasks
        self.task_description = "Swing the suitcase lid open."
        self.sub_tasks = ["SwingSuitcaseLidOpen"]
        self.sub_task_descriptions = ["Swing the lid of a suitcase open"]
        self.success_criteria = ["articulated_open"]
        # For articulated tasks, the articulator should be specified, while others should be None.
        # The articulator here is 'suitcase_rotate', which is the asset we want to interact with.
        super().__init__(env, articulator="suitcase_rotate", asset_id=asset_id)

    def reset(self, random=False):
        # Reset the environment to the initial state before starting the task
        super().reset()

        # If not random, set the suitcase lid to its default closed position
        # Otherwise, set a random pose and openness for the suitcase lid
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
