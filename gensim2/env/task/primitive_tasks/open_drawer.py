import numpy as np
import transforms3d

from gensim2.env.base.base_task import GenSimBaseTask
from gensim2.env.utils.pose import *
from gensim2.env.utils.rewards import (
    l2_distance_reward,
    joint_fraction_reward,
    check_openness,
    progress_reward,
)


class OpenDrawer(GenSimBaseTask):
    """Open the drawer."""

    def __init__(self, env, asset_id=""):
        self.task_description = "Open the drawer"
        self.sub_tasks = ["OpenDrawer"]
        self.sub_task_descriptions = ["Open the drawer"]
        self.success_criteria = ["articulated_open"]
        # For articulated tasks, articulator should be specified, while others should be None.
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
        reaching_reward_coef = 1
        openness_reward_coef = 1

        reaching_reward, reaching_done = l2_distance_reward(
            pos1=self.tcp.get_pos(), pos2=self.articulator.get_pos()
        )
        openness_reward, openness_done = joint_fraction_reward(
            openness=self.articulator.get_openness()[0], fraction=1.0
        )

        reward_list = [
            (reaching_reward_coef * reaching_reward, reaching_done),
            (openness_reward_coef * openness_reward, openness_done),
        ]

        reward = progress_reward(reward_list)

        return reward
