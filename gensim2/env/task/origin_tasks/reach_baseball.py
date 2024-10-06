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
from gensim2.env.utils.rewards import (
    l2_distance_reward,
    joint_fraction_reward,
    check_openness,
    progress_reward,
)


class ReachBall(GenSimBaseTask):

    def __init__(self, env):
        # Initialize the task with the microwave as the articulated object and the cracker box as the rigid body object.
        super().__init__(env, rigid_body="baseball")

    def get_progress_state(self):
        # Check if the microwave door is open enough to place the cracker box inside.
        if check_openness(
            openness=self.articulator.get_openness()[0], fraction=0.9, threshold=0.1
        ):
            # TODO: check if the cracker box is inside the microwave
            return 0.5
        return 0

    def reset(self, random=False):
        # Reset the environment to the default state or a random state.
        super().reset()

        if not random:
            set_default_pose(self.rigid_body, self.rigid_body_name)
        else:
            set_random_pose(self.rigid_body, self.rigid_body_name, task=self.task_type)

        return self.env.get_observation()

    def get_reward(self):
        # Define reward coefficients for different parts of the task.
        reaching_microwave_reward_coef = 1
        openness_microwave_reward_coef = 1

        reaching_cracker_box_reward_coef = 1
        placing_cracker_box_reward_coef = 1

        # Calculate the reward for reaching the microwave.
        reaching_microwave_reward, reaching_microwave_done = l2_distance_reward(
            pos1=self.tcp.get_pos(), pos2=self.articulator.get_pos()
        )
        # Calculate the reward for opening the microwave door.
        openness_microwave_reward, openness_microwave_done = joint_fraction_reward(
            openness=self.articulator.get_openness()[0], fraction=1.0
        )

        # Calculate the reward for reaching the cracker box.
        reaching_cracker_box_reward, reaching_cracker_box_done = l2_distance_reward(
            pos1=self.tcp.pos, pos2=self.rigid_body.pos
        )

        # Placeholder for the reward when the cracker box is placed inside the microwave.
        placing_cracker_box_reward, cracker_box_placed = 0.0, False
        # TODO: Implement the logic to check if the cracker box is placed inside the microwave and calculate the reward.

        # If the microwave door is not open, the robot cannot proceed to place the cracker box inside.
        if not openness_microwave_done:
            reaching_cracker_box_reward, placing_cracker_box_reward = 0, 0

        # Combine the rewards for each part of the task into a list.
        reward_list = [
            (
                reaching_microwave_reward_coef * reaching_microwave_reward,
                reaching_microwave_done,
            ),
            (
                openness_microwave_reward_coef * openness_microwave_reward,
                openness_microwave_done,
            ),
            (
                reaching_cracker_box_reward_coef * reaching_cracker_box_reward,
                reaching_cracker_box_done,
            ),
            (
                placing_cracker_box_reward_coef * placing_cracker_box_reward,
                cracker_box_placed,
            ),
        ]

        # Calculate the total reward based on the progress of each part of the task.
        reward = progress_reward(reward_list)

        return reward
