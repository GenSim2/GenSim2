import numpy as np
import sapien.core as sapien
import transforms3d
from gensim2.env.base.base_task import GenSimBaseTask
from gensim2.env.utils.rewards import *
import gym
from gensim2.env.utils.pose import *
import numpy as np
import transforms3d


class PushDrawerClose(GenSimBaseTask):
    """Push the drawer to a fully close position"""

    def __init__(self, env, asset_id=""):
        self.task_description = "Push the drawer to a fully close position"
        self.sub_tasks = ["PushDrawerClose"]
        self.sub_task_descriptions = ["Push the drawer to a fully close position"]
        self.success_criteria = ["articulated_closed"]
        # Initialize the base task with the environment and specify the articulator as 'drawer'.
        super().__init__(env, articulator="drawer", asset_id=asset_id)

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
        # Define the coefficients for the reaching and openness rewards.
        reaching_reward_coef = 1
        openness_reward_coef = 1

        # Calculate the reaching reward based on the distance between the TCP (Tool Center Point) and the drawer's handle.
        reaching_reward, reaching_done = l2_distance_reward(
            pos1=self.tcp.pos, pos2=self.articulator.pos
        )

        # Calculate the openness reward based on the current openness of the drawer.
        # The goal is to have the drawer fully open, so fraction is set to 1.0.
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
