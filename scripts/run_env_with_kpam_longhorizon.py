import numpy as np
import argparse
import icecream
import yaml
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

parser.add_argument("--num_episodes", type=int, default=5)
parser.add_argument("--max_steps", type=int, default=500)

if __name__ == "__main__":
    args = parser.parse_args()
    env = create_gensim(
        task_name=args.env, use_gui=args.render, eval=False, obs_mode=args.obs_mode
    )

    icecream.ic(env.horizon)

    if args.render:
        env.render()
    for eps in range(args.num_episodes):
        steps = []
        obs = env.reset(args.random)

        for task in env.sub_tasks:
            if task == "Grasp":
                env.grasp()
                continue
            elif task == "UnGrasp":
                env.ungrasp()
                continue

            config_path = f"gensim2/env/solver/kpam/config/{task}.yaml"
            load_file = open(config_path, mode="r")
            expert_planner = KPAMPlanner(env, config_path)
            expert_planner.reset_expert()

            if env.viewer.closed:
                break

            for i in range(args.max_steps):
                action = expert_planner.get_action()
                obs, reward, done, info = env.step(action)

                if args.render:
                    env.render()
                if done:
                    break

        print("Task Solved:", env.task.get_progress_state())
