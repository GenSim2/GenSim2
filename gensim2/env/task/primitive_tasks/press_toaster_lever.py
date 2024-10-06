import numpy as np
import sapien.core as sapien
import transforms3d
from gensim2.env.base.base_task import GenSimBaseTask
from gensim2.env.utils.rewards import *
import gym
from gensim2.env.utils.pose import *
import numpy as np
import transforms3d


class PressToasterLever(GenSimBaseTask):
    """Press the lever of a toaster down"""

    def __init__(self, env, asset_id=""):
        self.task_description = "Press the lever of a toaster down"
        self.sub_tasks = ["PressToasterLever"]
        self.sub_task_descriptions = ["Press the lever of a toaster down"]
        self.success_criteria = ["articulated_open"]
        # For articulated tasks, articulator should be specified, while others should be None.
        # The asset_id is set to '103559' as specified in the task description.
        super().__init__(env, articulator="toaster_press", asset_id=asset_id)

    def reset(self, random=False):
        super().reset()

        if not random:
            set_default_openness(self.articulator, self.articulator_name, self.asset_id)
            set_default_pose(self.articulator, self.articulator_name, self.asset_id)
        else:
            set_random_pose(
                self.articulator, self.articulator_name, self.asset_id, self.task_type
            )
            set_random_openness(self.articulator, self.articulator_name, self.asset_id)

        return self.env.get_observation()

    def get_reward(self):
        # Define coefficients for the reaching and openness rewards.
        reaching_reward_coef = 1
        openness_reward_coef = 1

        # Calculate the reaching reward based on the distance between the TCP (Tool Center Point) and the toaster's lever.
        reaching_reward, reaching_done = l2_distance_reward(
            pos1=self.tcp.pos, pos2=self.articulator.pos
        )

        # Calculate the openness reward based on the current openness of the lever compared to the target (pressed down).
        openness_reward, openness_done = joint_fraction_reward(
            openness=self.articulator.get_openness()[0], fraction=1.0
        )

        # Combine the rewards into a list and calculate the overall progress reward.
        reward_list = [
            (reaching_reward_coef * reaching_reward, reaching_done),
            (openness_reward_coef * openness_reward, openness_done),
        ]

        # The progress reward is a weighted sum of the individual rewards.
        reward = progress_reward(reward_list)

        return reward
