from PIL import Image
import numpy as np
from collections import OrderedDict
from typing import Dict
import copy

from gensim2.agent.utils.replay_buffer import ReplayBuffer
from gensim2.agent.utils.sampler import SequenceSampler, get_val_mask
from gensim2.agent.utils.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from gensim2.agent.utils.pcd_utils import (
    add_gaussian_noise,
    randomly_drop_point,
    voxelize_point_cloud,
)
from gensim2.agent.dataset.sim_traj_dataset import TrajDataset
from gensim2.paths import *

from transforms3d.euler import quat2euler

import os
import zarr
import pickle
import ipdb

ENV_NAMES = {
    "close_box": "CloseBox",
    "close_drawer": "PushDrawerClose",
    "close_laptop": "CloseLaptop",
    "close_safe": "CloseSafe",
    "lift_bucket": "LiftBucketUpright",
    "move_bag": "MoveBagForward",
    "open_box": "OpenBox",
    "open_drawer": "OpenDrawer",
    "open_laptop": "OpenLaptop",
    "open_safe": "OpenSafe",
    "swing_bucket": "SwingBucketHandle",
}


if __name__ == "__main__":

    dataset = TrajDataset(
        dataset_name="real_data_11task_10eps_quat_10240",
        from_empty=True,
        use_disk=True,
        load_from_cache=False,
    )

    # Load real world dataset
    dataset_path = os.path.join(
        GENSIM_DIR, "dataset_7task_10240"
    )  # /user/gensim_v2/gensim2/dataset

    # Iterate through each folder in dataset_path
    for folder in os.listdir(dataset_path):
        # Check that it is a folder
        if not os.path.isdir(os.path.join(dataset_path, folder)):
            continue

        print(folder)

        # Process each sub-folder
        eps = 0
        for subfolder in os.listdir(os.path.join(dataset_path, folder)):
            if eps == 10:
                break

            if not os.path.isdir(os.path.join(dataset_path, folder, subfolder)):
                continue

            # Load the data
            low_dim_obs_path = os.path.join(
                dataset_path, folder, subfolder, "low_dim_obs.pkl"
            )
            var_desc_path = os.path.join(
                dataset_path, folder, subfolder, "variation_description.pkl"
            )

            with open(low_dim_obs_path, "rb") as f:
                low_dim_obs = pickle.load(f)

            with open(var_desc_path, "rb") as f:
                task_description = pickle.load(f)

            steps = []

            action_broke = False
            for i in range(len(low_dim_obs)):
                # Iterate through each step
                v = i + 1

                if i == len(low_dim_obs) - 1:
                    # last element just copy action from previous
                    v = i

                p = low_dim_obs[v]["state"][0:3]
                q = low_dim_obs[v]["state"][3:7]

                euler = quat2euler(q)

                action = np.zeros(7)  # position (3), euler (3), gripper (1)
                action[0:3] = p
                action[3:6] = euler
                action[6] = low_dim_obs[v]["state"][-1]

                # Convert quaternion to euler angles
                low_dim_obs[i]["pointcloud"]["colors"] = np.zeros_like(
                    low_dim_obs[i]["pointcloud"]["colors"]
                ).astype(
                    np.uint8
                )  # no colors needed
                step = {"obs": low_dim_obs[i], "action": action}

                steps.append(step)

            env_name = ENV_NAMES[folder]
            dataset.append_episode(steps, task_description, env_name)

            eps += 1

            print("Processed episode", eps)
