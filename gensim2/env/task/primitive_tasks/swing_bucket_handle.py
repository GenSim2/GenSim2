import numpy as np
import sapien.core as sapien
import transforms3d
from gensim2.env.base.base_task import GenSimBaseTask
from gensim2.env.utils.rewards import *
import gym
from gensim2.env.utils.pose import *
import numpy as np
import transforms3d


class SwingBucketHandle(GenSimBaseTask):
    """Swing the bucket handle to a position perpendicular to its initial position."""

    def __init__(self, env, asset_id=""):
        self.task_description = "Swing the bucket handle to a position perpendicular to its initial position"
        self.sub_tasks = ["SwingBucketHandle"]
        self.sub_task_descriptions = [
            "Swing the bucket handle to a position perpendicular to its initial position"
        ]
        self.success_criteria = ["articulated_closed"]
        # Initialize the task by specifying the articulated object 'bucket' and its asset_id.
        super().__init__(env, articulator="bucket_swing", asset_id=asset_id)

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
        # Define the reward coefficients for reaching and angle completion.
        reaching_reward_coef = 1
        angle_reward_coef = 1

        # Calculate the reaching reward based on the distance between the TCP (Tool Center Point) and the bucket handle.
        reaching_reward, reaching_done = l2_distance_reward(
            pos1=self.tcp.pos, pos2=self.articulator.pos
        )

        # Calculate the angle reward based on how close the handle is to the target angle.
        current_angle = self.articulator.get_openness()[0]
        target_angle = np.pi / 2  # 90 degrees in radians
        angle_reward, angle_done = joint_fraction_reward(
            openness=current_angle, fraction=target_angle
        )

        # Combine the rewards into a list and calculate the total reward.
        reward_list = [
            (reaching_reward_coef * reaching_reward, reaching_done),
            (angle_reward_coef * angle_reward, angle_done),
        ]

        # Calculate the total reward based on the progress of each subtask.
        reward = progress_reward(reward_list)

        return reward
