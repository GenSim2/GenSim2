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
from gensim2.env.utils.pose import set_default_pose, set_random_pose
from gensim2.env.utils.rewards import l2_distance_reward


class PushToasterForward(GenSimBaseTask):
    """Push the toaster to a new location on the table."""

    def __init__(self, env, asset_id=""):
        # Define the task description and the success criteria
        self.task_description = "Push the toaster to a new location on the table."
        self.sub_tasks = ["PushToasterForward"]
        self.sub_task_descriptions = ["Push the toaster forward"]
        self.success_criteria = ["articulated_open"]

        # Initialize the base task with the articulated object 'toaster_move'
        # Since we are moving the toaster, we do not need to specify a tool or a tooled object
        super().__init__(env, articulator="toaster_move", asset_id=asset_id)

    def reset(self, random=False):
        # Reset the environment to its initial state
        super().reset()

        # Set the toaster to its default pose or a random pose depending on the 'random' flag
        if not random:
            # Set the toaster to its default pose
            set_default_pose(self.articulator, self.articulator_name, self.asset_id)
        else:
            # Set the toaster to a random pose on the table
            set_random_pose(
                self.articulator, self.articulator_name, self.asset_id, self.task_type
            )

        # Return the initial observation of the environment
        return self.env.get_observation()
