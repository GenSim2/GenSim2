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


class MoveInsideBox(GenSimBaseTask):
    """Move a YCB object into a box."""

    def __init__(self, env):
        # For articulated tasks, articulator should be specified, while others should be None.
        super().__init__(
            env, articulator="box_rotate", rigid_body=None, asset_id="100221"
        )

    def get_progress_state(self):
        if check_openness(
            openness=self.articulator.get_openness()[0], fraction=0.9, threshold=0.1
        ):
            return 0.5
        # TODO: check if the object is planced into the box

        return 0

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
        reaching_box_reward_coef = 1
        openness_box_reward_coef = 1

        reaching_ycb_reward_coef = 1
        reaching_ycb_goal_reward_coef = 1
        grasp_ycb_reward_coef = 1

        reaching_box_reward, reaching_box_done = l2_distance_reward(
            pos1=self.tcp.get_pos(), pos2=self.articulator.get_pos()
        )
        openness_box_reward, openness_box_done = joint_fraction_reward(
            openness=self.articulator.get_openness()[0], fraction=1.0
        )

        reaching_ycb_reward, reaching_ycb_done = l2_distance_reward(
            pos1=self.tcp.pos, pos2=self.rigid_body.pos
        )

        grasp_ycb_reward, is_grasped = self.grasp_reward(
            check_grasp_fun=self.env.check_grasp,
            actor=self.rigid_body.instance,
            max_angle=30,
        )

        reaching_ycb_goal_reward, reaching_ycb_goal_done = 0.0, False
        if is_grasped:
            reaching_ycb_goal_reward, reaching_ycb_goal_done = l2_distance_reward(
                pos1=self.goal_pos, pos2=self.rigid_body.p
            )

        if not openness_box_done:
            reaching_ycb_reward, grasp_ycb_reward, reaching_ycb_goal_reward = 0, 0, 0

        reward_list = [
            (reaching_box_reward_coef * reaching_box_reward, reaching_box_done),
            (openness_box_reward_coef * openness_box_reward, openness_box_done),
            (reaching_ycb_reward_coef * reaching_ycb_reward, reaching_ycb_done),
            (grasp_ycb_reward_coef * grasp_ycb_reward, is_grasped),
            (
                reaching_ycb_goal_reward_coef * reaching_ycb_goal_reward,
                reaching_ycb_goal_done,
            ),
        ]

        reward = progress_reward(reward_list)

        return reward
