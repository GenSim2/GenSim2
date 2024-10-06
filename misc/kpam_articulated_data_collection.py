import numpy as np
import icecream

from gensim2.env.task.origin_tasks import *
from gensim2.env.task.primitive_tasks import *
from gensim2.env.solver.planner import KPAMPlanner
from gensim2.env.create_task import create_gensim
from gensim2.agent.dataset.sim_traj_dataset import TrajDataset

from common_parser import parser

parser.add_argument("--num_episodes", type=int, default=50)
parser.add_argument("--max_steps", type=int, default=500)
parser.add_argument("--save", action="store_true")

envs = [
    "OpenBox",
    "CloseBox",
    "OpenLaptop",
    "CloseLaptop",
    "TurnOnFaucet",
    "TurnOffFaucet",
    "OpenDrawer",
    "PushDrawerClose",
    "SwingBucketHandle",
    "PressToasterLever",
]

if __name__ == "__main__":
    args = parser.parse_args()
    dataset_name = "gensim_articulated_tasks"
    if args.save:
        dataset = TrajDataset(
            dataset_name=dataset_name,
            from_empty=True,
            use_disk=True,
            load_from_cache=False,
        )

    success_rate = {}
    if args.env != "":
        envs = [args.env]

    for env_name in envs:

        env = create_gensim(
            task_name=env_name,
            asset_id=args.asset_id,
            use_gui=args.render,
            eval=False,
            obs_mode=args.obs_mode,
            headless=not args.render,
        )

        icecream.ic(env.horizon, env_name, env.task_description)
        print("======================================")
        config_path = f"gensim2/env/solver/kpam/config/{env_name}.yaml"
        load_file = open(config_path, mode="r")
        expert_planner = KPAMPlanner(env, config_path)

        eps = 0
        all_eps = 0
        while eps < args.num_episodes:
            all_eps += 1
            steps = []
            print(f"Collecting Episode {eps} for {env_name}")
            obs = env.reset(args.random)
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
                    steps.append(step)
                obs, reward, done, info = env.step(action)

                if args.render:
                    env.render()
                if done:
                    if env.task.articulator is not None:
                        print("Openness:", env.task.articulator.get_openness())
                    print("Task Solved:", env.task.get_progress_state())
                    success = env.task.get_progress_state()
                    break
            if success:
                eps += 1
                if args.save:
                    dataset.append_episode(steps, env.task_description)
                    print("Episode Succeed and Saved!")
                else:
                    print("Episode Succeed!")
            else:
                print("Episode Failed!")

        success_rate[env_name] = eps / all_eps

    print("Success Rate:", success_rate)
