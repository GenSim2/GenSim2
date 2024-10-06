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


class PushOvenClose(GenSimBaseTask):
    """Push the oven door to a closed position."""

    def __init__(self, env, asset_id=""):
        # Define the task description and success criteria
        self.task_description = "Push the oven door to a closed position."
        self.sub_tasks = ["PushOvenClose"]
        self.sub_task_descriptions = ["Push the oven door to a closed position"]
        self.success_criteria = ["articulated_closed"]
        # For articulated tasks, articulator should be specified, while others should be None.
        # Here, 'oven' is the name of the articulated object to be used.
        super().__init__(env, articulator="oven", asset_id=asset_id)

    def reset(self, random=False):
        # Reset the environment to the initial state
        super().reset()

        # Set the initial state of the oven door. If not random, set it to the default openness and pose.
        # Otherwise, set a random pose and openness within the allowed range for the task.
        if not random:
            # Set the oven door to a default open position
            set_default_openness(self.articulator, self.articulator_name, self.asset_id)
            # Set the oven in its default pose
            set_default_pose(self.articulator, self.articulator_name, self.asset_id)
        else:
            # Set a random pose for the oven within the task constraints
            set_random_pose(
                self.articulator, self.articulator_name, self.asset_id, self.task_type
            )
            # Set a random openness for the oven door
            set_random_openness(self.articulator, self.articulator_name, self.asset_id)

        # Return the initial observation of the environment
        return self.env.get_observation()
