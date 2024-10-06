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
from gensim2.env.utils.rewards import l2_distance_reward


class MoveBagForward(GenSimBaseTask):
    """Slide a bag forward on the table by a specified distance."""

    def __init__(self, env, asset_id=""):
        # Define the task description and the success criteria
        self.task_description = (
            "Slide a bag forward on the table by a specified distance."
        )
        self.sub_tasks = ["MoveBagForward"]
        self.sub_task_descriptions = ["Slide a bag forward on the table"]
        self.success_criteria = ["articulated_open"]

        # For articulated tasks, articulator should be specified, while others should be None.
        # Here, 'bag_move' is the articulated object that the robot will interact with.
        super().__init__(env, articulator="bag_move", asset_id=asset_id)

    def reset(self, random=False):
        # Reset the environment to the initial state
        super().reset()

        # Set the bag to its default position or a random position if specified
        if not random:
            set_default_pose(self.articulator, self.articulator_name, self.asset_id)
        else:
            set_random_pose(
                self.articulator, self.articulator_name, self.asset_id, self.task_type
            )

        # Return the initial observation of the environment
        return self.env.get_observation()

    def compute_reward(self, action, obs):
        # Compute the reward for the action taken by the robot
        # The reward is based on the distance between the gripper and the articulated object (bag)
        # after the action is performed. The closer the gripper is to the target position, the higher the reward.
        gripper_pose = self.env.get_gripper_pose()
        bag_pose = self.env.get_articulated_object_pose(
            self.articulator_name, self.asset_id
        )

        # Calculate the L2 distance between the gripper and the bag
        reward = l2_distance_reward(gripper_pose, bag_pose)

        return reward
