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
from gensim2.env.utils.rewards import l2_distance_reward, progress_reward


class SwayBagStrap(GenSimBaseTask):
    """Sway the strap of a bag to a new position."""

    def __init__(self, env, asset_id=""):
        # Define the task description and success criteria
        self.task_description = "Sway the strap of a bag to a new position."
        self.sub_tasks = ["SwayBagStrap"]
        self.sub_task_descriptions = ["Sway the strap of a bag"]
        self.success_criteria = ["distance_gripper_articulated"]

        # For articulated tasks, articulator should be specified, while others should be None.
        # Here, 'bag_swing' refers to the bag with a strap that can be swayed.
        super().__init__(env, articulator="bag_swing", asset_id=asset_id)

    def reset(self, random=False):
        # Reset the environment to the initial state
        super().reset()

        # Set the initial position and openness of the bag's strap
        # If not random, set to default pose and openness, otherwise set to a random pose and openness
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
