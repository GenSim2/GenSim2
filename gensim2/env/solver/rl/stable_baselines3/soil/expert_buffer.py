from typing import Generator, Optional, Union

import h5py
import numpy as np
import torch as th
from gym import spaces

from gensim2.env.solver.rl.stable_baselines3.common.buffers import (
    RolloutBuffer,
    DictRolloutBuffer,
)
from gensim2.env.solver.rl.stable_baselines3.common.type_aliases import (
    RolloutBufferSamples,
    DictRolloutBufferSamples,
)
from gensim2.env.solver.rl.stable_baselines3.common.vec_env import VecNormalize


class ExpertRolloutBuffer(RolloutBuffer):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        dataset_path: str = "",
        load_in_memory=True,
    ):

        self.in_memory = load_in_memory
        assert load_in_memory

        if dataset_path.endswith(("pkl", "pickle")):
            if not load_in_memory:
                raise RuntimeError(f"Pickle dataset only support in memory load.")
            self.use_h5 = False

            self.original_data = np.load(dataset_path, allow_pickle=True)
            data_obs = []
            data_action = []
            for trajectory in self.original_data:
                data_obs.append(trajectory["observations"])
                data_action.append(trajectory["actions"])
            data_obs = np.concatenate(data_obs, axis=0)
            data_action = np.concatenate(data_action, axis=0)

            buffer_size = data_action.shape[0]
            if data_obs.shape[0] != buffer_size:
                raise RuntimeError(
                    "Demo Dataset Error: Obs num does not match Action num."
                )

        elif dataset_path.endswith("h5"):
            self.use_h5 = True

            self.original_data = h5py.File(dataset_path, "r")
            h5_data = self.original_data["data"]
            if "meta_data" in self.original_data.keys():
                self.h5_meta_data = self.original_data["meta_data"]

            h5_obs_dict = h5_data["observations"]
            if "oracle_state" not in h5_obs_dict.keys() or len(h5_obs_dict.keys()) > 1:
                raise RuntimeError(
                    f"For non dict observation space, only oracle_state is allowed."
                )
            h5_obs = h5_obs_dict["oracle_state"]
            h5_action = h5_data["actions"]
            buffer_size = h5_action.shape[0]
        else:
            raise RuntimeError(
                f"Unrecognized file type for dataset path: {dataset_path}"
            )

        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            gae_lambda,
            gamma,
            n_envs=1,
        )

        if not load_in_memory and self.use_h5:
            raise NotImplementedError
        elif self.use_h5:
            self.observations = h5_obs[:]
            self.actions = h5_action[:]
        else:
            self.observations = data_obs
            self.actions = data_action

        # Normalize the range of action
        self.actions = np.clip(self.actions, -1, 1)

        self.full = True

    def get(
        self, batch_size: Optional[int] = None, loop=True
    ) -> Generator[RolloutBufferSamples, None, None]:
        indices = np.random.permutation(self.buffer_size * self.n_envs)

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size

        start_idx = 0
        while start_idx < self.buffer_size:
            yield self._get_samples(
                indices[start_idx : min(start_idx + batch_size, self.buffer_size)]
            )
            start_idx += batch_size
            if start_idx >= self.buffer_size and loop:
                indices = np.random.permutation(self.buffer_size * self.n_envs)
                start_idx = 0

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
        )
        return RolloutBufferSamples(
            *tuple(map(self.to_torch, data)), None, None, None, None
        )


class DictExpertRolloutBuffer(DictRolloutBuffer):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        dataset_path: str = "",
        load_in_memory=True,
    ):
        self.in_memory = load_in_memory
        assert load_in_memory

        if dataset_path.endswith("h5"):
            self.original_data = h5py.File(dataset_path, "r")
            h5_data = self.original_data["data"]
            if "meta_data" in self.original_data.keys():
                self.h5_meta_data = self.original_data["meta_data"]

            h5_obs_dict = h5_data["observations"]
            h5_action = h5_data["actions"]
            buffer_size = h5_action.shape[0]

            super().__init__(
                buffer_size,
                observation_space,
                action_space,
                device,
                gae_lambda,
                gamma,
                n_envs=1,
            )

            self.actions = h5_action[:]
            for obs_name, obs_data in h5_obs_dict.items():
                if obs_name in observation_space.spaces:
                    self.observations[obs_name] = obs_data[:]

        else:
            raise RuntimeError(
                f"Unrecognized file type for dataset path: {dataset_path}"
            )
        # Normalize the range of action
        self.actions = np.clip(self.actions, -1, 1)

        self.full = True

    def get(
        self, batch_size: Optional[int] = None, loop=True
    ) -> Generator[RolloutBufferSamples, None, None]:
        indices = np.random.permutation(self.buffer_size * self.n_envs)

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size

        start_idx = 0
        while start_idx < self.buffer_size:
            yield self._get_samples(
                indices[start_idx : min(start_idx + batch_size, self.buffer_size)]
            )
            start_idx += batch_size
            if start_idx >= self.buffer_size and loop:
                indices = np.random.permutation(self.buffer_size * self.n_envs)
                start_idx = 0

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> DictRolloutBufferSamples:
        obs_dict = dict()
        for key, obs in self.observations.items():
            obs_dict[key] = self.to_torch(obs[batch_inds])
            if len(obs_dict[key].shape) == 4:  # 0808 Added for unknown bug
                obs_dict[key] = obs_dict[key].squeeze(1)
            # print(obs_dict[key].shape)
        return DictRolloutBufferSamples(
            observations=obs_dict,
            actions=self.to_torch(self.actions[batch_inds]),
            old_values=None,
            old_log_prob=None,
            advantages=None,
            returns=None,
        )
