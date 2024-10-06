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


class PushBox(GenSimBaseTask):
    """Push the box forward on the table."""

    def __init__(self, env, asset_id=""):
        # Define the task description and success criteria based on the task definition
        self.task_description = "Push the box forward on the table."
        self.sub_tasks = ["PushBox"]
        self.sub_task_descriptions = ["Push the box forward on the table"]
        self.success_criteria = ["articulated_open"]

        # Initialize the base task with the specific articulated object 'box_move'
        # No tool or tooled object is specified as the gripper itself is used to push the box
        super().__init__(env, articulator="box_move", asset_id=asset_id)

    def reset(self, random=False):
        # Reset the environment to the initial state before starting the task
        super().reset()

        # Set the box to its default pose or a random pose based on the 'random' flag
        if not random:
            set_default_pose(self.articulator, self.articulator_name, self.asset_id)
        else:
            set_random_pose(
                self.articulator, self.articulator_name, self.asset_id, self.task_type
            )

        # Return the initial observation to the environment
        return self.env.get_observation()

    def compute_reward(self, action, obs):
        # Compute the reward based on the distance between the box and the target area
        # The target position can be predefined in the environment and accessed here
        target_position = self.env.get_target_position()
        box_position = self.env.get_articulated_object_position(
            self.articulator_name, self.asset_id
        )

        # Use L2 distance for the reward calculation
        reward = l2_distance_reward(box_position, target_position)

        return reward
