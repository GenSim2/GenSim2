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
from gensim2.paths import *

from gensim2.env.utils.pcd_utils import BOUND as GENSIM_BOUNDS

try:
    from gensim2.env.utils.rlbench import SCENE_BOUNDS as RLBENCH_BOUNDS
except:
    print("RLBench is not installed. Skip.")

import os
import zarr
import ipdb


class TrajDataset:
    """
    Single Dataset class that converts simulation data into trajectory data.
    Explanations of parameters are in config
    """

    def __init__(
        self,
        dataset_name="",
        mode="train",
        episode_cnt=10,
        step_cnt=100,
        data_augmentation=False,
        data_ratio=1,
        use_disk=False,
        horizon=4,
        pad_before=0,
        pad_after=0,
        val_ratio=0.1,
        seed=233,
        action_horizon=1,
        observation_horizon=1,
        dataset_postfix="",
        dataset_encoder_postfix="",
        precompute_feat=False,
        image_encoder="resnet",
        env_rollout_fn=None,
        use_multiview=False,
        normalize_state=False,
        from_empty=True,
        use_pcd=False,
        pcdnet_pretrain_domain="",
        pcd_channels=None,
        load_from_cache=True,
        env_names=None,
        voxelization=False,
        voxel_size=0.01,
        **kwargs,
    ):
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.data_augmentation = data_augmentation
        self.episode_cnt = episode_cnt
        self.step_cnt = step_cnt
        self.action_horizon = action_horizon
        self.observation_horizon = observation_horizon
        self.precompute_feat = precompute_feat
        self.image_encoder = image_encoder
        self.data_ratio = data_ratio
        self.use_multiview = use_multiview
        self.normalize_state = normalize_state
        if use_pcd:
            assert pcd_channels is not None, "pcd_channels must be provided for pcd"
        if pcd_channels is not None:
            assert pcd_channels in [
                3,
                4,
                5,
                6,
                7,
            ], "pcd_channels must be one of [3, 4, 6, 7]"
        self.pcd_channels = pcd_channels
        self.mode = mode
        self.use_pcd = use_pcd
        self.pcd_transform = None
        self.pcdnet_pretrain_domain = pcdnet_pretrain_domain
        self.pcd_num_points = None
        self.env_names = env_names

        self.voxelization = voxelization
        self.voxel_size = voxel_size

        self.update_pcd_transform()

        self.dataset_path = [None]
        self.replay_buffer = [None]
        self.sample_ratio = [0.5, 0.5]
        if isinstance(dataset_name, list):
            self.replay_buffer = [None] * len(dataset_name)

        for idx, domain in enumerate(dataset_name):
            # if the dataset has not been downloaded, this line will download
            dataset_name = domain.strip()
            dataset_name_withpostfix = (
                domain + dataset_encoder_postfix + dataset_postfix
            )

            dataset_path = f"{GENSIM_DIR}/agent/data/" + dataset_name_withpostfix
            self.dataset_path[idx] = dataset_path
            load_from_cache = os.path.exists(dataset_path) and load_from_cache
            print(
                f"\n\n >>>dataset_path: {dataset_path} load_from_cache: {load_from_cache} \n\n"
            )

            if use_disk:
                # self.replay_buffer[idx] = ReplayBuffer.create_empty_zarr(storage=zarr.DirectoryStore(path=dataset_path))
                if load_from_cache:
                    self.replay_buffer[idx] = ReplayBuffer.create_from_path(
                        dataset_path, self.env_names
                    )
                else:
                    self.replay_buffer[idx] = ReplayBuffer.create_empty_zarr(
                        storage=zarr.DirectoryStore(path=dataset_path)
                    )
            else:
                self.replay_buffer[idx] = ReplayBuffer.create_empty_numpy()

        # loading datasets
        if not from_empty:
            if not (load_from_cache and use_disk):
                self.load_dataset()

            self.get_training_dataset(val_ratio, seed)
            self.get_sa_dim()

    def update_pcd_transform(self, pcd_setup_cfg=None):
        if not self.use_pcd:
            return
        assert (
            self.pcdnet_pretrain_domain != ""
        ), "pcdnet_domain must be provided for pcdnet"

        from openpoints.transforms import build_transforms_from_cfg

        if pcd_setup_cfg is None:
            from openpoints.utils import EasyConfig

            pcd_setup_cfg = EasyConfig()
            pcd_setup_cfg.load(
                f"{GENSIM_DIR}/agent/models/pointnet_cfg/{self.pcdnet_pretrain_domain}/pcd_setup.yaml",
                recursive=True,
            )

        # in case only val or test transforms are provided.
        if self.mode not in pcd_setup_cfg.keys() and self.mode in ["val", "test"]:
            trans_split = "val"
        else:
            trans_split = self.mode
        self.pcd_transform = build_transforms_from_cfg(
            trans_split, pcd_setup_cfg.datatransforms
        )
        self.pcd_num_points = pcd_setup_cfg.num_points

    def get_sa_dim(self):
        self.action_dim = self[0]["data"]["action"].shape[-1]  #  * self.action_horizon
        self.state_dim = self[0]["data"]["state"].shape[-1]

    def get_normalizer(self, mode="limits", **kwargs):
        """action normalizer"""
        data = self._sample_to_data(self.replay_buffer)
        self.normalizer = LinearNormalizer()
        self.normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        for k, v in self.normalizer.params_dict.items():
            print(f"normalizer {k} stats min: {v['input_stats'].min}")
            print(f"normalizer {k} stats max: {v['input_stats'].max}")
        return self.normalizer

    def append_episode(self, episode, description="", env_name=None):
        data = OrderedDict()

        def recursive_dict_update(d, u):
            for k, v in u.items():
                if isinstance(v, (dict, OrderedDict)):
                    d[k] = recursive_dict_update(d.get(k, {}), v)
                else:
                    if k not in d:
                        d[k] = []
                    d[k].append(v)
            return d

        def recursive_array(d):
            if isinstance(d, (dict, OrderedDict)):
                for k, v in d.items():
                    d[k] = recursive_array(v)
            elif isinstance(d, list):
                d = np.array(d)
            return d

        for dataset_step in episode:
            recursive_dict_update(data, dataset_step)
        for key, val in data.items():
            data[key] = recursive_array(data[key])

        self.replay_buffer.add_episode(data, description=description, env_name=env_name)

    def get_episode(self, idx):
        return self.replay_buffer.get_episode(idx)

    def _sample_to_data(self, sample):
        data = {"action": sample["action"]}
        if "state" in sample and self.normalize_state:
            data["state"] = sample["state"]  # 1 x N
        return data

    def get_training_dataset(self, val_ratio, seed):
        self.val_mask = [None] * len(self.replay_buffer)
        self.train_mask = [None] * len(self.replay_buffer)
        self.sampler = [None] * len(self.replay_buffer)
        for idx, replay_buffer in enumerate(self.replay_buffer):
            # split into train and test sets
            self.val_mask[idx] = get_val_mask(
                n_episodes=self.replay_buffer[idx].n_episodes,
                val_ratio=val_ratio,
                seed=seed,
            )
            self.train_mask[idx] = ~self.val_mask

            # considering hyperparameters and masking
            n_episodes = int(
                self.data_ratio
                * min(self.episode_cnt, self.replay_buffer[idx].n_episodes)
            )
            self.val_mask[idx][n_episodes:] = False
            self.train_mask[idx][n_episodes:] = False

            # normalize and create sampler
            self.sampler[idx] = SequenceSampler(
                replay_buffer=self.replay_buffer[idx],
                sequence_length=self.horizon,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                episode_mask=self.train_mask[idx],
            )
            print(
                f"{self.dataset_name[idx]} size: {len(self.sampler[idx])} episodes: {n_episodes} train: {self.train_mask[idx].sum()} eval: {self.val_mask[idx].sum()}"
            )

    def get_validation_dataset(self):
        val_set = [None] * len(self.replay_buffer)
        for idx, replay_buffer in enumerate(self.replay_buffer):
            val_set[idx] = copy.copy(self)
            val_set[idx].mode = "val"
            val_set[idx].update_pcd_transform()
            val_set[idx].sampler = SequenceSampler(
                replay_buffer=self.replay_buffer[idx],
                sequence_length=self.horizon,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                episode_mask=self.val_mask[idx],
            )
            val_set[idx].train_mask = self.val_mask[idx]
        return val_set

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int):
        """normalize observation and actions"""
        idx = np.random.choice(len(self.sampler), p=self.sample_ratio)
        sample = self.sampler[idx].sample_sequence(idx)

        # the full horizon is for the trajectory
        def recursive_horizon(data):
            for key, val in data.items():
                if isinstance(val, (dict, OrderedDict)):
                    recursive_horizon(val)
                else:
                    if (key != "action") and (key != "action_is_pad"):
                        if key == "language":
                            data[key] = val
                        else:
                            data[key] = val[: self.observation_horizon]
                    else:
                        if key == "action":
                            data["action"] = val[
                                self.observation_horizon
                                - 1 : self.action_horizon
                                + self.observation_horizon
                                - 1
                            ]
                        elif key == "action_is_pad":
                            data["action_is_pad"] = val[
                                self.observation_horizon
                                - 1 : self.action_horizon
                                + self.observation_horizon
                                - 1
                            ]

        if self.use_pcd:
            sample["obs"]["pointcloud"]["x"] = sample["obs"]["pointcloud"]["colors"]

        if self.data_augmentation:
            sample["obs"]["pointcloud"]["pos"] = self.pcd_aug(
                sample["obs"]["pointcloud"]["pos"]
            )

        if self.voxelization:
            sample["obs"]["pointcloud"] = np.array(
                [
                    voxelize_point_cloud(
                        sample["obs"]["pointcloud"]["pos"][i],
                        sample["obs"]["pointcloud"]["color"][i],
                        self.voxel_size,
                    )
                    for i in range(len(sample["obs"]["pointcloud"]["pos"]))
                ]
            )
        else:
            if self.pcd_transform is not None and "pointcloud" in sample["obs"].keys():
                seq_len, num_points = sample["obs"]["pointcloud"]["pos"].shape[:2]
                for key, val in sample["obs"][
                    "pointcloud"
                ].items():  # Reshape for tranform API
                    sample["obs"]["pointcloud"][key] = val.reshape(
                        seq_len * num_points, -1
                    )
                try:
                    sample["obs"]["pointcloud"] = self.pcd_transform(
                        sample["obs"]["pointcloud"]
                    )
                except Exception as e:
                    print(
                        f"Found error {e}, should not be a problem when initializing the dataset"
                    )
                for key, val in sample["obs"]["pointcloud"].items():
                    sample["obs"]["pointcloud"][key] = val.reshape(
                        seq_len, num_points, -1
                    )

            self.flat_sample(sample)

        # if "pointcloud" in sample.keys():
        #     assert sample["pointcloud"].shape[-1] == self.pcd_channels, f"pointcloud channel mismatch! expected {self.pcd_channels}, got {sample['pointcloud'].shape[-1]}"
        recursive_horizon(sample)

        return {"domain": self.dataset_name, "data": sample}

    def save_dataset(self):
        self.replay_buffer.save_to_path(self.dataset_path)

    def load_dataset(self):
        for idx, dataset_path in enumerate(self.dataset_path):
            self.replay_buffer[idx] = ReplayBuffer.copy_from_path(dataset_path)
            print("Replay buffer keys: ", self.replay_buffer[idx].keys())

    def flat_sample(self, sample):
        if "obs" in sample.keys():
            for key, val in sample["obs"].items():
                sample[key] = val
            del sample["obs"]
        if not self.use_pcd:
            del sample["pointcloud"]
        if "pointcloud" in sample.keys():
            if self.pcd_channels == 3:
                pass
                # sample['pointcloud']['pos'] = sample['pointcloud']['pos']
            elif self.pcd_channels == 6:
                sample["pointcloud"]["x"] = np.concatenate(
                    [sample["pointcloud"]["pos"], sample["pointcloud"]["colors"]],
                    axis=-1,
                )
            elif self.pcd_channels == 4:
                sample["pointcloud"]["x"] = np.concatenate(
                    [sample["pointcloud"]["pos"], sample["pointcloud"]["heights"]],
                    axis=-1,
                )
            elif self.pcd_channels == 5:
                sample["pointcloud"]["x"] = np.concatenate(
                    [
                        sample["pointcloud"]["pos"],
                        sample["pointcloud"]["heights"],
                        sample["pointcloud"]["seg"][..., 1].unsqueeze(-1),
                    ],
                    axis=-1,
                )
            elif self.pcd_channels == 7:
                sample["pointcloud"]["x"] = np.concatenate(
                    [
                        sample["pointcloud"]["pos"],
                        sample["pointcloud"]["colors"],
                        sample["pointcloud"]["heights"],
                    ],
                    axis=-1,
                )
            else:
                raise ValueError(f"Invalid pcd_channels: {self.pcd_channels}")

    def pcd_aug(self, pcd):
        pcd = add_gaussian_noise(pcd)
        pcd = randomly_drop_point(pcd)

        return pcd


def delete_indices(
    replay_buffer: ReplayBuffer,
    env_name: str,
) -> np.ndarray:
    episode_ends = replay_buffer.episode_ends[:]
    episode_desc = replay_buffer.episode_descriptions[:]
    env_names = replay_buffer.meta["env_names"]

    indices = list()
    for i in range(len(episode_ends)):
        if env_names[i] == env_name:
            start_idx = 0
            if i > 0:
                start_idx = episode_ends[i - 1]
            end_idx = episode_ends[i]
            eps_description = episode_desc[i]
            episode_length = end_idx - start_idx

            # TODO

            # remove i in replay.meta

            # remove start_idx:end_idx in replay.data


if __name__ == "__main__":
    import collections

    dataset = TrajDataset(
        dataset_name="rlbench_dnact_keypose_cleanpcd10240_100",  # "rlbench_dnact_keypose_cleanpcd10240_100",
        from_empty=False,
        use_disk=True,
        load_from_cache=True,
    )
    # dataset = TrajDataset(dataset_name="rlbench_peract_cleanpcd10240_100", from_empty=False, use_disk=True, load_from_cache=True)
    # dataset = TrajDataset(dataset_name="rlbench_peract_keypose_cleanpcd10240_100", from_empty=False, use_disk=True, load_from_cache=True)
    # dataset = TrajDataset(dataset_name="rlbench_peract_jointpos_cleanpcd10240_100", from_empty=False, use_disk=True, load_from_cache=True)
    # dataset.load_dataset()
    print(collections.Counter(dataset.replay_buffer.meta["episode_descriptions"]))
    if "env_names" in dataset.replay_buffer.meta.keys():
        print(collections.Counter(dataset.replay_buffer.meta["env_names"]))

    import ipdb

    ipdb.set_trace()

    from gensim2.env.utils.rlbench import plot_pred

    for i in range(15, 25):
        data = dataset.replay_buffer
        pcds = data["obs"]["pointcloud"]["pos"][i]
        rgbs = data["obs"]["pointcloud"]["colors"][i]
        action = data["action"][i]
        plot_pred(np.array([action]), pcds, rgbs, ".")
