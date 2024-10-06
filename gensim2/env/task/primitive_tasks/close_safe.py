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


class CloseSafe(GenSimBaseTask):
    """Close a safe."""

    def __init__(self, env, asset_id=""):
        # Define the task description and success criteria
        self.task_description = "Close a safe."
        self.sub_tasks = ["CloseSafe"]
        self.sub_task_descriptions = ["Close a safe"]
        self.success_criteria = ["articulated_closed"]
        # For articulated tasks, articulator should be specified, while others should be None.
        # The articulator is the safe with a revolute joint that can be opened and closed.
        super().__init__(env, articulator="safe_rotate", asset_id=asset_id)

    def reset(self, random=False):
        # Reset the environment and set the initial state of the safe
        super().reset()

        # If not random, set the safe to a default open position and pose
        # Otherwise, set a random pose and openness for the safe
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

    def get_reward(self):
        reaching_reward_coef = 1
        openness_reward_coef = 1

        if not self.success_planned:
            return -0.1

        self.safe_T_tcp = self.articulator.get_keypoint_T_tcp("articulated_object_head")

        # reaching_reward, reaching_done = l2_distance_reward(pos1=self.tcp.pos, pos2=self.articulator.pos)
        reaching_reward, reaching_done = l2_norm_reward(self.safe_T_tcp)
        openness_reward, openness_done = joint_fraction_reward(
            openness=self.articulator.get_openness()[0], fraction=0.0
        )

        reward_list = [
            (reaching_reward_coef * reaching_reward, reaching_done),
            (openness_reward_coef * openness_reward, openness_done),
        ]

        reward = progress_reward(reward_list)

        print(reward)

        return reward
