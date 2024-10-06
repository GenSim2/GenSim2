import numpy as np
import icecream
import ipdb
import json
from collections import defaultdict
import os
import datetime

from gensim2.env.task.origin_tasks import *
from gensim2.env.task.primitive_tasks import *
from gensim2.env.solver.planner import KPAMPlanner
from gensim2.env.create_task import create_gensim
from gensim2.agent.dataset.sim_traj_dataset import TrajDataset

from common_parser import parser

parser.add_argument("--num_episodes", type=int, default=100)
parser.add_argument("--max_steps", type=int, default=500)
parser.add_argument("--load_from_cache", action="store_true")
parser.add_argument("--dataset", type=str, default="gensim2")

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

if __name__ == "__main__":
    args = parser.parse_args()
    dataset_name = args.dataset
    if args.save:
        dataset = TrajDataset(
            dataset_name=dataset_name,
            from_empty=True,
            use_disk=True,
            load_from_cache=args.load_from_cache,
        )

    success_rate = {}
    if args.env != "":
        envs = [args.env]

    for env_name in envs:
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
        while eps < args.num_episodes:
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
                load_file = open(config_path, mode="r")
                expert_planner = KPAMPlanner(
                    env, config_path, env.task.articulator.random_pose["rot"]
                )
                expert_planner.reset_expert()

                if env.viewer and env.viewer.closed:
                    break

                for i in range(args.max_steps):
                    action = expert_planner.get_action()
                    if args.save:
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
                        dataset.append_episode(sub_steps, task_description, env_name)
                    print("Episode Succeed and Saved!")
                else:
                    print("Episode Succeed!")
            else:
                print("Episode Failed!")
        success_rate[env_name] = eps / all_eps
        print("Success Rate:", success_rate)

        if env.viewer:
            env.viewer.close()
