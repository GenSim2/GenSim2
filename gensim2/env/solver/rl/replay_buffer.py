import datetime
import io
import random
import traceback
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
import pickle


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open("wb") as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open("rb") as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


class ReplayBufferStorage:
    def __init__(self, data_specs, replay_dir):
        self._data_specs = data_specs
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._preload()

    def __len__(self):
        return self._num_transitions

    def add(self, time_step):
        for spec in self._data_specs:
            value = time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype
            self._current_episode[spec.name].append(value)
        if time_step.last():
            episode = dict()
            for spec in self._data_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            self._current_episode = defaultdict(list)
            self._store_episode(episode)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob("*.npz"):
            _, _, eps_len = fn.stem.split("_")
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        eps_fn = f"{ts}_{eps_idx}_{eps_len}.npz"
        save_episode(episode, self._replay_dir / eps_fn)


class ReplayBuffer(IterableDataset):
    def __init__(
        self,
        replay_dir,
        max_size,
        num_workers,
        nstep,
        discount,
        fetch_every,
        save_snapshot,
    ):
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob("*.npz"), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split("_")[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()

        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
        obs = episode["observation"][idx - 1]
        action = episode["action"][idx]
        next_obs = episode["observation"][idx + self._nstep - 1]
        reward = np.zeros_like(episode["reward"][idx])
        discount = np.ones_like(episode["discount"][idx])
        for i in range(self._nstep):
            step_reward = episode["reward"][idx + i]
            reward += discount * step_reward
            discount *= episode["discount"][idx + i] * self._discount

        return (obs, action, reward, discount, next_obs)

    def __iter__(self):
        while True:
            yield self._sample()


class ExpertReplayBuffer(IterableDataset):
    def __init__(self, dataset_path, num_demos, obs_type):
        with open(dataset_path, "rb") as f:
            if obs_type == "pixels":
                obses, _, actions, _ = pickle.load(f)
            elif obs_type == "features":
                _, obses, actions, _ = pickle.load(f)

        self._episodes = []
        for i in range(num_demos):
            episode = dict(observation=np.array(obses[i]), action=np.array(actions[i]))
            self._episodes.append(episode)

    def _sample_episode(self):
        episode = random.choice(self._episodes)
        return episode

    def _sample(self):
        episode = self._sample_episode()
        # idx = np.random.randint(0, episode_len(episode)) + 1
        idx = np.random.randint(0, episode_len(episode))  # -1 for getting next obs
        obs = episode["observation"][idx]
        action = episode["action"][idx]
        next_obs = episode["observation"][idx + 1]

        return (obs, action, next_obs)

    def __iter__(self):
        while True:
            yield self._sample()


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(
    replay_dir,
    max_size,
    batch_size,
    num_workers,
    save_snapshot,
    nstep,
    discount,
    use_per,
):
    max_size_per_worker = max_size // max(1, num_workers)

    if use_per:
        iterable = PrioritizedReplayBuffer(
            replay_dir,
            max_size_per_worker,
            num_workers,
            nstep,
            discount,
            fetch_every=1000,
            save_snapshot=save_snapshot,
            batch_size=batch_size,
        )
    else:
        iterable = ReplayBuffer(
            replay_dir,
            max_size_per_worker,
            num_workers,
            nstep,
            discount,
            fetch_every=1000,
            save_snapshot=save_snapshot,
        )

    loader = torch.utils.data.DataLoader(
        iterable,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )
    return loader, iterable


def make_expert_replay_loader(replay_dir, batch_size, num_demos, obs_type):
    iterable = ExpertReplayBuffer(replay_dir, num_demos, obs_type)

    loader = torch.utils.data.DataLoader(
        iterable,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )
    return loader


class ReplayBufferStorage:
    def __init__(self, data_specs, replay_dir):
        self._data_specs = data_specs
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._preload()

    def __len__(self):
        return self._num_transitions

    def add(self, time_step):
        for spec in self._data_specs:
            value = time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype
            self._current_episode[spec.name].append(value)
        if time_step.last():
            episode = dict()
            for spec in self._data_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            self._current_episode = defaultdict(list)
            self._store_episode(episode)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob("*.npz"):
            _, _, eps_len = fn.stem.split("_")
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        eps_fn = f"{ts}_{eps_idx}_{eps_len}.npz"
        save_episode(episode, self._replay_dir / eps_fn)


class SumTree(object):
    def __init__(self, max_size):
        self._max_size = max_size
        self._tree = np.zeros(2 * max_size - 1)
        self._data = np.array([(None, 0)] * max_size)  # ep_fn, idx
        self._data_pointer = 0
        self._size = 0
        self.max = 1  # Initial max value to return (1 = 1^ω)

    def __len__(self):
        return self._size

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self._tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, score):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self._tree):
            return idx

        if score <= self._tree[left]:
            return self._retrieve(left, score)
        else:
            return self._retrieve(right, score - self._tree[left])

    @property
    def total(self):
        return self._tree[0]

    def add(self, data, priority):
        tree_idx = self._data_pointer + self._max_size - 1

        self._data[self._data_pointer] = data
        self.update(tree_idx, priority)

        self._data_pointer += 1
        if self._data_pointer >= self._max_size:
            self._data_pointer = 0

        if self._size < self._max_size:
            self._size += 1

    def update(self, idx, priority):
        change = priority - self._tree[idx]

        self._tree[idx] = priority
        self._propagate(idx, change)

        self.max = max(priority, self.max)

    def update_batch(self, indices, priorities):
        self._tree[indices] = priorities  # Set new values
        self._propagate_batch(indices)  # Propagate values
        current_max_value = np.max(priorities)
        self.max = max(current_max_value, self.max)

    def _propagate_batch(self, indices):
        parents = (indices - 1) // 2
        unique_parents = np.unique(parents)

        children_indices = unique_parents * 2 + np.expand_dims([1, 2], axis=1)
        self._tree[unique_parents] = np.sum(self._tree[children_indices], axis=0)

        if parents[0] != 0:
            self._propagate_batch(parents)

    def get(self, score):
        tree_idx = self._retrieve(0, score)
        data_idx = tree_idx - self._max_size + 1

        return (tree_idx, self._tree[tree_idx], self._data[data_idx])


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        replay_dir,
        max_size,
        num_workers,
        nstep,
        discount,
        fetch_every,
        save_snapshot,
        batch_size,
    ):
        super(PrioritizedReplayBuffer, self).__init__(
            replay_dir,
            max_size,
            num_workers,
            nstep,
            discount,
            fetch_every,
            save_snapshot,
        )
        self._sum_tree = SumTree(max_size)

        self._e = 0.01
        self._a = 0.6
        self._beta = 0.4
        self._beta_increment_per_sampling = 0.001

        self.batch_idx = 0
        self.batch_size = batch_size

    def _get_priority(self, error):
        return (np.abs(error) + self._e) ** self._a

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob("*.npz"), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split("_")[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = episode_len(episode)
        # To prevent the episode in sum tree is not existed in eps_fns
        # while eps_len + self._size > self._max_size:
        while self._size > self._max_size + 10000:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)

        for idx in range(len(episode["observation"])):
            if idx > eps_len - self._nstep:
                break
            priority = self._sum_tree.max
            self._sum_tree.add(
                (eps_fn, idx), priority
            )  # add eps_fn and idx as tree samples

        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len - self._nstep  # keep the same as the sum tree

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1

        eps_fn = None
        while eps_fn not in self._episodes:
            if eps_fn is not None:
                print("{} not found in sum tree!".format(eps_fn))
            segment = self._sum_tree.total / self.batch_size
            a = segment * self.batch_idx
            b = segment * (self.batch_idx + 1)
            random_priority = random.uniform(a, b)

            (tree_idx, priority, data) = self._sum_tree.get(random_priority)
            eps_fn, idx = data

        episode = self._episodes[eps_fn]

        if self.batch_idx == 0:
            self._beta = np.min([1.0, self._beta + self._beta_increment_per_sampling])
        obs = episode["observation"][idx - 1]
        action = episode["action"][idx]
        next_obs = episode["observation"][idx + self._nstep - 1]
        reward = np.zeros_like(episode["reward"][idx])
        discount = np.ones_like(episode["discount"][idx])
        for i in range(self._nstep):
            step_reward = episode["reward"][idx + i]
            reward += discount * step_reward
            discount *= episode["discount"][idx + i] * self._discount

        sampling_probability = priority / self._sum_tree.total
        is_weight = np.power(len(self._sum_tree) * sampling_probability, -self._beta)

        self.batch_idx += 1
        self.batch_idx %= self.batch_size

        return (obs, action, reward, discount, next_obs, tree_idx, is_weight)

    def update(self, tree_idx, error):
        p = self._get_priority(error)
        self._sum_tree.update_batch(tree_idx, p)
