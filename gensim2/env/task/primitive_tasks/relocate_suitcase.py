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


class RelocateSuitcase(GenSimBaseTask):
    """Move the suitcase to a new location on the table."""

    def __init__(self, env, asset_id=""):
        # Define the task description and name
        self.task_description = "Move the suitcase to a new location on the table."
        self.sub_tasks = ["RelocateSuitcase"]
        self.sub_task_descriptions = [
            "Move the suitcase to a new location on the table"
        ]
        self.success_criteria = ["articulated_open"]
        # For articulated tasks, articulator should be specified, while others should be None.
        # Here, the suitcase is the articulated object that we want to move.
        super().__init__(env, articulator="suitcase_move", asset_id=asset_id)

    def reset(self, random=False):
        # Reset the environment to the initial state
        super().reset()

        # Set the initial pose of the suitcase. If random is True, set a random pose,
        # otherwise set the default pose.
        if not random:
            set_default_pose(self.articulator, self.articulator_name, self.asset_id)
        else:
            set_random_pose(
                self.articulator, self.articulator_name, self.asset_id, self.task_type
            )

        # Return the initial observation of the environment
        return self.env.get_observation()

    def compute_reward(self, achieved_goal, desired_goal, info):
        # Compute the reward for the task. The reward is based on the L2 distance
        # between the gripper and the articulated object (suitcase) after the motion.
        # The closer the gripper is to the target location, the higher the reward.
        return l2_distance_reward(achieved_goal, desired_goal, info)
