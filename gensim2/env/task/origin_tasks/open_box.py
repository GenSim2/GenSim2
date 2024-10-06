import numpy as np
import transforms3d

from gensim2.env.base.base_task import GenSimBaseTask
from gensim2.env.utils.pose import *
from gensim2.env.utils.rewards import (
    l2_distance_reward,
    joint_fraction_reward,
    check_openness,
    progress_reward,
    axis_alignment_reward,
)


class OpenBox(GenSimBaseTask):
    """Open a box."""

    def __init__(self, env, asset_id=""):
        self.task_description = "Open a box."
        self.sub_tasks = ["OpenBox"]
        self.sub_task_descriptions = ["Open a box"]
        self.success_criteria = ["articulated_open"]
        # For articulated tasks, articulator should be specified, while others should be None.
        super().__init__(env, articulator="box_rotate", asset_id=asset_id)

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

    # def get_reward(self):
    #     reaching_reward_coef = 1
    #     openness_reward_coef = 1

    #     reaching_reward, reaching_done = l2_distance_reward(pos1=self.tcp.pos, pos2=self.articulator.pos)
    #     openness_reward, openness_done = joint_fraction_reward(
    #         openness=self.articulator.get_openness()[0], fraction=1.0
    #     )

    #     reward_list = [
    #         (reaching_reward_coef * reaching_reward, reaching_done),
    #         (openness_reward_coef * openness_reward, openness_done),
    #     ]

    #     reward = progress_reward(reward_list)

    #     return reward

    def get_reward(self):
        # Initialize reward and done flag
        reward, done = 0.0, False

        if self.success_planned == -1:
            return -0.1

        # Check if the box is opened to the desired fraction
        box_openness_reward, box_openness_done = joint_fraction_reward(
            openness=self.articulator.get_openness()[0],
            fraction=1,  # Assuming the goal is to fully open the box
            threshold=0.05,
        )

        done = done or box_openness_done

        # Additional checks can include alignment of the tool with the box, ensuring the tool has correctly interacted with the box, etc.
        # For example, ensuring the tool's tail is near the articulated object's head (hammer head hits at the pin)

        # Make sure the robot gripper hits the box lid
        tool_tail_pos = self.tcp.pos @ self.tool.get_keypoints()["tool_tail"]
        box_head_pos = self.articulator.get_keypoints()["articulated_object_head"]
        box_contact_reward, box_contact_done = l2_distance_reward(
            tool_tail_pos, box_head_pos
        )

        # Assuming the presence of an incremental or progress-based reward system
        reward = progress_reward(
            [
                (box_openness_reward, box_openness_done),
                (box_contact_reward, box_contact_done),
            ]
        )

        return reward
