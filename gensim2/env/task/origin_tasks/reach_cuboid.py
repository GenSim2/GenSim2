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


class ReachCuboid(GenSimBaseTask):
    """Open the microwave door, place the cracker box inside, and close the door."""

    def __init__(self, env):
        # Initialize the task with the microwave as the articulated object and the cracker box as the rigid body object.
        super().__init__(env, rigid_body="cracker_box")

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
        # super().reset()

        # Set the microwave and cracker box to their default or random positions and states.
        if not random:
            set_default_pose(self.rigid_body, self.rigid_body_name)
        else:
            set_random_pose(self.rigid_body, self.rigid_body_name, task=self.task_type)

        super().reset()

        return self.env.get_observation()

    def get_reward(self):
        return 0
