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


class RotateOvenKnob(GenSimBaseTask):
    """Rotate the oven knob to an open position."""

    def __init__(self, env, asset_id=""):
        # Define the task description and sub-tasks. In this case, there's only one sub-task.
        self.task_description = "Rotate the oven knob to an open position."
        self.sub_tasks = ["RotateOvenKnob"]
        self.sub_task_descriptions = ["Rotate the oven knob to an open position"]
        self.success_criteria = ["articulated_open"]
        # For articulated tasks, the articulator should be specified, while others should be None.
        # Here, 'oven' is the name of the asset as specified in the task description.
        super().__init__(env, articulator="oven", asset_id=asset_id)

    def reset(self, random=False):
        # Reset the environment to the initial state for the task.
        # This involves setting the pose and openness of the articulator (oven knob).
        super().reset()

        if not random:
            # If not random, set the default pose and openness for the oven knob.
            # This ensures the knob starts in a consistent position each time.
            set_default_openness(self.articulator, self.articulator_name, self.asset_id)
            set_default_pose(self.articulator, self.articulator_name, self.asset_id)
        else:
            # If random, set a random pose and openness for the oven knob.
            # This adds variability to the starting conditions of the task.
            set_random_pose(
                self.articulator, self.articulator_name, self.asset_id, self.task_type
            )
            set_random_openness(self.articulator, self.articulator_name, self.asset_id)

        # Return the initial observation of the environment after resetting.
        return self.env.get_observation()
