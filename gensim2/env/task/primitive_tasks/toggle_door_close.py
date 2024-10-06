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


class ToggleDoorClose(GenSimBaseTask):
    """Close an open door."""

    def __init__(self, env, asset_id=""):
        # Define the task description and success criteria based on the task definition
        self.task_description = "Close an open door."
        self.sub_tasks = ["ToggleDoorClose"]
        self.sub_task_descriptions = ["Close an open door"]
        self.success_criteria = ["articulated_closed"]
        # For articulated tasks, articulator should be specified, while others should be None.
        # The articulator is the door, which is the asset we are going to manipulate.
        super().__init__(env, articulator="door", asset_id=asset_id)

    def reset(self, random=False):
        # Reset the environment to the initial state before starting the task
        super().reset()

        # If not random, set the door to its default open state
        # This ensures that the door is open at the start of the task
        if not random:
            set_default_openness(self.articulator, self.articulator_name, self.asset_id)
            set_default_pose(self.articulator, self.articulator_name, self.asset_id)
        else:
            # If random, set a random pose and openness for the door
            # This adds variability to the starting state of the door for the task
            set_random_pose(
                self.articulator, self.articulator_name, self.asset_id, self.task_type
            )
            set_random_openness(self.articulator, self.articulator_name, self.asset_id)

        # Return the current observation of the environment
        # This observation will be used by the robot to plan its actions
        return self.env.get_observation()
