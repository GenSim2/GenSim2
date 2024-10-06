import numpy as np
import argparse
import icecream
import yaml
import ipdb
import json
import collections
from collections import defaultdict
import multiprocessing as mp
import math

try:
    mp.set_start_method("forkserver", force=True)
    print("forkservered")
except RuntimeError:
    pass
from multiprocessing import Process, Queue

from gensim2.env.task.origin_tasks import *
from gensim2.env.task.primitive_tasks import *
from gensim2.env.solver.planner import KPAMPlanner
from gensim2.env.create_task import create_gensim
from gensim2.agent.dataset.sim_traj_dataset import TrajDataset

from kpam_data_collection import parser

parser.add_argument("--nprocs", type=int, default=20)
parser.add_argument("--seg", type=bool, default=False)

envs = [
    "OpenBox",
    "CloseBox",
    "OpenLaptop",
    "CloseLaptop",
    "OpenDrawer",
    "PushDrawerClose",
    "SwingBucketHandle",
    "LiftBucketUpright",
    "PressToasterLever",
    "PushToasterForward",
    "MoveBagForward",
    "OpenSafe",
    "CloseSafe",
    "RotateMicrowaveDoor",
    "CloseMicrowave",
    "CloseSuitcaseLid",
    "SwingSuitcaseLidOpen",
    "RelocateSuitcase",
    "TurnOnFaucet",
    "TurnOffFaucet",
    "SwingDoorOpen",
    "ToggleDoorClose",
    "CloseRefrigeratorDoor",
    "OpenRefrigeratorDoor",
]


def env_sample(env_names, num_episodes, args, ret_queue):
    for env_name in env_names:
        success_rate = {}
        env = create_gensim(
            task_name=env_name,
            asset_id=args.asset_id,
            use_ray_tracing=args.rt,
            use_gui=args.render,
            eval=False,
            obs_mode=args.obs_mode,
            headless=not args.render,
            cam=args.cam,
        )
        icecream.ic(env.horizon, env_name, env.task_description)

        eps = 0
        all_eps = 0
        while eps < num_episodes:
            all_eps += 1
            print(f"Collecting Episode {eps} for {env_name}")
            obs = env.reset(args.random)
            steps = []
            for task in env.sub_tasks:
                if task == "Grasp":
                    env.grasp()
                    continue
                elif task == "UnGrasp":
                    env.ungrasp()
                    continue
                sub_steps = []
                config_path = f"gensim2/env/solver/kpam/config/{task}.yaml"
                expert_planner = KPAMPlanner(
                    env, config_path, env.task.articulator.random_pose["rot"]
                )
                expert_planner.reset_expert()

                if env.viewer and env.viewer.closed:
                    break

                for i in range(args.max_steps):
                    action = expert_planner.get_action()
                    if args.save:
                        # remove segmentation mask from obs
                        if not args.seg:
                            del obs["pointcloud"]["seg"]

                        step = {
                            "obs": obs,
                            "action": action,
                        }
                        sub_steps.append(step)
                    obs, reward, done, info = env.step(action)
                    if args.render:
                        env.render()

                    if done:
                        steps.append((sub_steps, info["current_task_description"]))
                        break

                if not info["sub_task_success"]:
                    break

            print("Task Solved:", env.task.get_progress_state())
            # print("Openness:", env.task.articulator.get_openness())
            success = env.task.get_progress_state()

            if success >= 1.0:
                eps += 1
                if args.save:
                    for sub_steps, task_description in steps:
                        ret_queue.put([env_name, sub_steps, task_description])
                else:
                    print(f"{env_name}: Episode Succeed!")
            else:
                print(f"{env_name}: Episode Failed!")
        if all_eps != 0:
            success_rate[env_name] = eps / all_eps
            print("Success Rate:", success_rate)
            # with open(f"{env_name}_data_collection_result.json", "w") as f:
            #     json.dump(success_rate, f)
        if env.viewer:
            env.viewer.close()


def split_process(envs, nprocs):
    envs = np.array_split(envs, nprocs)
    return envs


if __name__ == "__main__":
    args = parser.parse_args()
    dataset_name = args.dataset
    print(dataset_name)
    if args.save:
        dataset = TrajDataset(
            dataset_name=dataset_name,
            from_empty=True,
            use_disk=True,
            load_from_cache=args.load_from_cache,
        )

    if args.env != "":
        envs = [args.env]

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
            env_name, step, task_description = ret_queue.get()
            if args.save:
                dataset.append_episode(step, task_description, env_name)
                print(f"{env_name}: Episode Succeed and Saved!")
                print(collections.Counter(dataset.replay_buffer.meta["env_names"]))
