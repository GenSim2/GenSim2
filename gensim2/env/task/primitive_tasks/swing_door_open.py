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


class SwingDoorOpen(GenSimBaseTask):
    """Swing a door open."""

    def __init__(self, env, asset_id=""):
        # Define the task description and success criteria
        self.task_description = "Swing a door open."
        self.sub_tasks = ["SwingDoorOpen"]
        self.sub_task_descriptions = ["Swing a door open"]
        self.success_criteria = ["articulated_open"]
        # For articulated tasks, articulator should be specified, while others should be None.
        # The articulator in this task is the door with a revolute joint.
        super().__init__(env, articulator="door", asset_id=asset_id)

    def reset(self, random=False):
        # Reset the environment and set the initial state of the door
        super().reset()

        # If not random, set the door to a default closed position and pose
        # Otherwise, set a random pose and openness for the door
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
