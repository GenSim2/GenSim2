import numpy as np
import icecream
import ipdb
import json
import transforms3d

from gensim2.env.task.origin_tasks import *
from gensim2.env.task.primitive_tasks import *
from gensim2.env.solver.planner import KPAMPlanner
from gensim2.env.solver.planner_utils import build_plant
from gensim2.env.create_task import create_gensim
from gensim2.agent.dataset.sim_traj_dataset import TrajDataset

from common_parser import parser

parser.add_argument("--num_episodes", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=500)
parser.add_argument("--save", action="store_true")

envs = [
    "PlaceGolfBallIntoDrawer",
]

if __name__ == "__main__":
    args = parser.parse_args()
    dataset_name = "gensim_long_horizon_tasks"
    if args.save:
        dataset = TrajDataset(
            dataset_name=dataset_name,
            from_empty=True,
            use_disk=True,
            load_from_cache=False,
        )

    for env_name in envs:
        env = create_gensim(
            task_name=args.env,
            use_gui=args.render,
            eval=False,
            obs_mode=args.obs_mode,
            headless=not args.render,
        )

        icecream.ic(env.horizon, env_name, env.task_description)
        subtask_path = f"gensim2/env/solver/kpam/task_decomposition/{env_name}.json"
        subtasks = json.load(open(subtask_path, "r"))

        eps = 0
        while eps < args.num_episodes:
            if env.viewer and env.viewer.closed:
                break

            dataset_name = env_name + f"_{eps}"
            if args.save:
                dataset = TrajDataset(dataset_name=dataset_name, from_empty=True)

            steps = []
            print(f"Collecting Episode {eps} for {env_name}")
            obs = env.reset(args.random)

            for task in subtasks:
                if task == "Grasp":
                    curr_pose = env.get_ee_pose_in_base().copy()
                    action = np.zeros(8)
                    action[:3] = curr_pose[:3, 3]
                    action[3:-1] = transforms3d.quaternions.mat2quat(curr_pose[:3, :3])
                    action[-1] = 0.0

                    env.set_gripper_state(0)

                    if args.save:
                        step = {
                            "obs": obs,
                            "action": action,
                        }
                        steps.append(step)
                    obs, reward, done, info = env.step(action)
                    if args.render:
                        env.render()
                    env.reset_internal()
                    continue

                elif task == "UnGrasp":
                    curr_pose = env.get_ee_pose_in_base().copy()
                    action = np.zeros(8)
                    action[:3] = curr_pose[:3, 3]
                    action[3:-1] = transforms3d.quaternions.mat2quat(curr_pose[:3, :3])
                    action[-1] = 1.0

                    env.set_gripper_state(1)

                    if args.save:
                        step = {
                            "obs": obs,
                            "action": action,
                        }
                        steps.append(step)
                    obs, reward, done, info = env.step(action)
                    if args.render:
                        env.render()

                    for _ in range(10):

                        curr_pose = env.get_ee_pose_in_base().copy()
                        tf = np.eye(4)
                        tf[2, 3] += -0.015
                        next_pose = np.dot(curr_pose, tf)

                        action = np.zeros(8)
                        action[:3] = next_pose[:3, 3]
                        action[3:-1] = transforms3d.quaternions.mat2quat(
                            next_pose[:3, :3]
                        )
                        action[-1] = 1.0

                        if args.save:
                            step = {
                                "obs": obs,
                                "action": action,
                            }
                            steps.append(step)
                        obs, reward, done, info = env.step(action)
                        if args.render:
                            env.render()
                    env.reset_internal()
                    continue

                config_path = f"gensim2/env/solver/kpam/config/{task}.yaml"
                load_file = open(config_path, mode="r")
                expert_planner = KPAMPlanner(env, config_path)
                expert_planner.reset_expert()

                for i in range(args.max_steps):
                    action = expert_planner.get_action()
                    if args.save:
                        step = {
                            "obs": obs,
                            "action": action,
                        }
                        steps.append(step)
                    obs, reward, done, info = env.step(action)

                    env.render()
                    if done:
                        # print("Openness:", env.task.articulator.get_openness())
                        # print("Task Solved:", env.task.get_progress_state())
                        env.reset_internal()
                        break
            # success = env.task.get_progress_state()
            print(f"Episode length: {len(steps)}")
            success = True
            if success:
                eps += 1
                if args.save:
                    dataset.append_episode(steps, env.task_description)
                    print("Episode Succeed and Saved!")
                else:
                    print("Episode Succeed!")
            else:
                print("Episode Failed!")
