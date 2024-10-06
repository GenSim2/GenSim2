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
    progress_reward,
    check_openness,
)


class LiftBucketUpright(GenSimBaseTask):
    """Lift a tipped-over bucket into an upright position."""

    def __init__(self, env, asset_id=""):
        # Define the task description and success criteria
        self.task_description = "Lift a tipped-over bucket into an upright position."
        self.sub_tasks = ["LiftBucketUpright"]
        self.sub_task_descriptions = [
            "Lift a tipped-over bucket into an upright position"
        ]
        self.success_criteria = ["articulated_open"]

        # For articulated tasks, articulator should be specified, while others should be None.
        # Here, 'bucket_lift' is the articulator that we will be using to refer to the bucket.
        super().__init__(env, articulator="bucket_lift", asset_id=asset_id)

    def reset(self, random=False):
        # Reset the environment to the initial state
        super().reset()

        # If not random, set the bucket to its default pose and openness
        # In this case, openness refers to the orientation of the bucket handle
        if not random:
            set_default_openness(self.articulator, self.articulator_name, self.asset_id)
            set_default_pose(self.articulator, self.articulator_name, self.asset_id)
        else:
            # If random, set a random pose and openness for the bucket
            # This adds variability to the starting state of the bucket
            set_random_pose(
                self.articulator, self.articulator_name, self.asset_id, self.task_type
            )
            set_random_openness(self.articulator, self.articulator_name, self.asset_id)

        # Return the initial observation of the environment
        return self.env.get_observation()
