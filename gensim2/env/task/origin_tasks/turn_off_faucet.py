import numpy as np
import transforms3d

from gensim2.env.base.base_task import GenSimBaseTask
from gensim2.env.utils.rewards import (
    l2_distance_reward,
    joint_fraction_reward,
    check_openness,
    progress_reward,
)
from gensim2.env.utils.pose import *


class TurnOffFaucet(GenSimBaseTask):
    """Turn off a faucet."""

    def __init__(self, env, asset_id=""):
        self.task_description = "Turn off a faucet"
        self.sub_tasks = ["TurnOffFaucet"]
        self.sub_task_descriptions = ["Turn off a faucet"]
        self.success_criteria = ["articulated_closed"]
        # For articulated tasks, articulator should be specified, while others should be None.
        super().__init__(env, articulator="faucet", asset_id=asset_id)

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
