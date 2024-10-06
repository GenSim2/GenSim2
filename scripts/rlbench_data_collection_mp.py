import numpy as np
import argparse
import collections
from collections import defaultdict, OrderedDict
import multiprocessing as mp
import time
import transforms3d

try:
    mp.set_start_method("forkserver", force=True)
    print("forkservered")
except RuntimeError:
    pass
from multiprocessing import Process, Queue


from gensim2.env.utils.rlbench import *
from gensim2.agent.dataset.sim_traj_dataset import TrajDataset

from rlbench_data_collection import parser


def env_sample(env_names, num_episodes, args, ret_queue):
    env, tr = create_rlbench_env("key_pose")
    # env, tr = create_rlbench_env("joint_positions")
    np.random.seed(int(time.time()))

    for env_name in env_names:

        task = env.get_task(ENV_DICT[env_name])  # -> Task

        num = 0
        while num < num_episodes:
            try:
                demo = task.get_demos(1, live_demos=True)[
                    0
                ]  # -> List[List[Observation]]
            except:
                pass
            num += 1
            steps = []
            descriptions, _ = task.reset()
            task_description = np.random.choice(descriptions)
            if args.action_mode == "key_pose":
                keypoint = keypoint_discovery(demo)
                cur_id = 0
                for next_id in keypoint:
                    for obs_idx in range(cur_id, next_id + 1):
                        keypoint = demo[next_id]
                        obs = demo[obs_idx]
                        obs_data = OrderedDict()
                        obs_data["state"] = obs.get_low_dim_data()
                        obs_data["pointcloud"] = get_pcds(task, obs)
                        # action = np.concatenate(
                        #     [keypoint.gripper_pose[:3], transforms3d.euler.quat2euler(keypoint.gripper_pose[3:])]
                        # )
                        action = keypoint.gripper_pose[:]
                        action = np.concatenate(
                            [action, np.array([keypoint.gripper_open])]
                        )
                        step = {}
                        # Peract-Style Action
                        before_keypoint = demo[max(0, next_id - 1)]
                        (
                            trans_indicies,
                            rot_grip_indicies,
                            ignore_collisions,
                            attention_coordinates,
                        ) = get_discrete_action(
                            keypoint,
                            before_keypoint,
                            SCENE_BOUNDS,
                            voxel_sizes=[100],
                            bounds_offset=[0.15],
                            rotation_resolution=5,
                            crop_augmentation=True,
                        )
                        action = np.concatenate([action, np.array([ignore_collisions])])
                        step = {
                            "attention_coordinates": attention_coordinates,
                            "trans_indicies": trans_indicies,
                            "rot_grip_indicies": rot_grip_indicies,
                            "ignore_collisions": ignore_collisions,
                        }

                        step.update({"obs": obs_data, "action": action})
                        steps.append(step)
                    cur_id = next_id
            else:
                for obs in demo:
                    obs_data = OrderedDict()
                    obs_data["state"] = obs.get_low_dim_data()
                    obs_data["pointcloud"] = get_pcds(task, obs)
                    if args.action_mode == "joint_positions":
                        action = obs.joint_positions
                    elif args.action_mode == "gripper_pose":
                        action = np.concatenate(
                            [
                                obs.gripper_pose[:3],
                                transforms3d.euler.quat2euler(obs.gripper_pose[3:]),
                            ]
                        )
                    else:
                        raise ValueError("Invalid action mode")
                    action = np.concatenate([action, np.array([obs.gripper_open])])
                    step = {"obs": obs_data, "action": action}
                    steps.append(step)
            if args.save:
                ret_queue.put((env_name, steps, task_description))


def split_process(envs, nprocs):
    envs = np.array_split(envs, nprocs)
    return envs


if __name__ == "__main__":
    parser.add_argument("--nprocs", type=int, default=20)
    args = parser.parse_args()
    dataset_name = "rlbench_dnact_keypose_quat_cleanpcd10240_50"
    # dataset_name = "rlbench_qattntasks_keypose_cleanpcd10240_100"
    if args.save:
        dataset = TrajDataset(
            dataset_name=dataset_name,
            from_empty=True,
            use_disk=True,
            load_from_cache=args.load_from_cache,
        )

    if args.env != "":
        envs = [args.env]
    else:
        envs = list(ENV_DICT.keys())

    ret_queue = Queue()
    processes = []
    if args.nprocs <= len(envs):
        processed_envs = split_process(envs, args.nprocs)
        num_episodes = [args.num_episodes] * args.nprocs
    else:
        total_nenv = len(envs)
        copy_env_num = args.nprocs // total_nenv
        processed_envs = envs * copy_env_num
        additional_env_num = args.nprocs % total_nenv

        if additional_env_num == 0:
            num_episodes = [args.num_episodes // copy_env_num] * (
                total_nenv * (copy_env_num - 1)
            )
            num_episodes += [
                args.num_episodes
                - args.num_episodes // copy_env_num * (copy_env_num - 1)
            ] * total_nenv

        else:
            num_episodes = [args.num_episodes // (copy_env_num + 1)] * (
                total_nenv * copy_env_num
            )
            processed_envs += envs[:additional_env_num]  # Add the remaining envs

            left_num_env = (
                args.num_episodes
                - args.num_episodes // (copy_env_num + 1) * copy_env_num
            )

            # modify the last num_episodes for envs without additional sampled processes
            num_episodes[-total_nenv + additional_env_num :] = [
                x + left_num_env
                for x in num_episodes[-total_nenv + additional_env_num :]
            ]

            num_episodes += [
                left_num_env
            ] * additional_env_num  # Add the remaining episodes for additional envs
        print(processed_envs)
        print(num_episodes)
        processed_envs = [[env] for env in processed_envs]

    for i, env_names in enumerate(processed_envs):
        p = Process(
            target=env_sample, args=(env_names, num_episodes[i], args, ret_queue)
        )
        p.daemon = True
        p.start()
        processes.append(p)

    while True:
        if all(not p.is_alive() for p in processes):
            break
        while not ret_queue.empty():
            env_name, steps, task_description = ret_queue.get()
            if args.save:
                dataset.append_episode(steps, task_description, env_name)
                print(f"{env_name}: Episode Succeed and Saved!")
                print(collections.Counter(dataset.replay_buffer.meta["env_names"]))
