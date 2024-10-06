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


class RotateMicrowaveDoor(GenSimBaseTask):
    """Rotate the microwave door open."""

    def __init__(self, env, asset_id=""):
        # Define the task description and success criteria
        self.task_description = "Rotate the microwave door open."
        self.sub_tasks = ["RotateMicrowaveDoor"]
        self.sub_task_descriptions = ["Rotate the microwave door open"]
        self.success_criteria = ["articulated_open"]
        # Specify the articulated object to be used for this task
        super().__init__(env, articulator="microwave", asset_id=asset_id)

    def reset(self, random=False):
        # Reset the environment and set the initial state of the microwave door
        super().reset()

        # If not random, set the microwave door to its default closed position
        if not random:
            set_default_openness(self.articulator, self.articulator_name, self.asset_id)
            set_default_pose(self.articulator, self.articulator_name, self.asset_id)
        # If random, set a random pose and openness for the microwave door
        else:
            set_random_pose(
                self.articulator, self.articulator_name, self.asset_id, self.task_type
            )
            set_random_openness(self.articulator, self.articulator_name, self.asset_id)

        # Return the initial observation of the environment
        return self.env.get_observation()
