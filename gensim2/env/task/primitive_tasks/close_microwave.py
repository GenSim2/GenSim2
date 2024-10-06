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
from gensim2.env.utils.pose import (
    set_default_openness,
    set_default_pose,
    set_random_pose,
    set_random_openness,
)
from gensim2.env.utils.rewards import check_openness


class CloseMicrowave(GenSimBaseTask):
    """Close the microwave door."""

    def __init__(self, env, asset_id=""):
        # Define the task description and success criteria based on the task definition.
        self.task_description = "Close the microwave door."
        self.sub_tasks = ["CloseMicrowave"]
        self.sub_task_descriptions = ["Close the microwave door"]
        self.success_criteria = ["articulated_closed"]
        # For articulated tasks, the articulator should be specified, while others should be None.
        # The articulator is set to 'microwave' as per the task definition.
        super().__init__(env, articulator="microwave", asset_id=asset_id)

    def reset(self, random=False):
        # Reset the environment to the initial state for the task.
        super().reset()

        # If not random, set the microwave door to its default openness and pose.
        # This typically means the door is initially open and needs to be closed.
        if not random:
            set_default_openness(self.articulator, self.articulator_name, self.asset_id)
            set_default_pose(self.articulator, self.articulator_name, self.asset_id)
        else:
            # If random, set a random pose and openness for the microwave door.
            # This adds variability to the starting state of the task.
            set_random_pose(
                self.articulator, self.articulator_name, self.asset_id, self.task_type
            )
            set_random_openness(self.articulator, self.articulator_name, self.asset_id)

        # Return the initial observation of the environment after reset.
        return self.env.get_observation()

    def get_reward(self):
        reaching_reward_coef = 1
        openness_reward_coef = 1

        reaching_reward, reaching_done = l2_distance_reward(
            pos1=self.tcp.pos, pos2=self.articulator.pos
        )
        openness_reward, openness_done = joint_fraction_reward(
            openness=self.articulator.get_openness()[0], fraction=0.0
        )

        reward_list = [
            (reaching_reward_coef * reaching_reward, reaching_done),
            (openness_reward_coef * openness_reward, openness_done),
        ]

        reward = progress_reward(reward_list)

        return reward
