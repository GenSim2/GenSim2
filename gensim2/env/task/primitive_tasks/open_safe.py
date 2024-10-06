import numpy as np
import transforms3d

from gensim2.env.base.base_task import GenSimBaseTask
from gensim2.env.utils.pose import (
    set_default_openness,
    set_default_pose,
    set_random_pose,
    set_random_openness,
)
from gensim2.env.utils.rewards import check_openness


class OpenSafe(GenSimBaseTask):
    """Rotate the safe's door to an open position."""

    def __init__(self, env, asset_id=""):
        # Define the task description and the sub-tasks involved.
        self.task_description = "Rotate the safe's door to an open position."
        self.sub_tasks = ["OpenSafe"]
        self.sub_task_descriptions = ["Rotate the safe's door to an open position"]
        self.success_criteria = ["articulated_open"]
        # For articulated tasks, the articulator should be specified, while others should be None.
        # Here, 'safe_rotate' is the name of the articulator to be used.
        super().__init__(env, articulator="safe_rotate", asset_id=asset_id)

    def reset(self, random=False):
        # Reset the environment to the initial state for the task.
        super().reset()

        # If not random, set the safe's door to its default position and openness.
        # Otherwise, set a random pose and openness for the safe's door.
        if not random:
            set_default_openness(self.articulator, self.articulator_name, self.asset_id)
            set_default_pose(self.articulator, self.articulator_name, self.asset_id)
        else:
            set_random_pose(
                self.articulator, self.articulator_name, self.asset_id, self.task_type
            )
            set_random_openness(self.articulator, self.articulator_name, self.asset_id)

        # Return the initial observation of the environment after reset.
        return self.env.get_observation()
