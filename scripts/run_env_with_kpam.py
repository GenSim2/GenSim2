import numpy as np
import argparse
import icecream
import hydra
import yaml
import ipdb

from gensim2.env.task.origin_tasks import *
from gensim2.env.task.primitive_tasks import *
from gensim2.env.solver.planner import KPAMPlanner
from gensim2.env.create_task import create_gensim
from gensim2.agent.dataset.sim_traj_dataset import TrajDataset

from common_parser import parser

from transforms3d.quaternions import quat2mat
from transforms3d.euler import mat2euler, euler2mat

parser.add_argument("--num_episodes", type=int, default=5)
parser.add_argument("--max_steps", type=int, default=500)
parser.add_argument("--video", action="store_true")
parser.add_argument("--early_stop", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()
    eps = 0
    env = create_gensim(
        task_name=args.env,
        asset_id=args.asset_id,
        use_gui=args.render,
        use_ray_tracing=args.rt,
        eval=False,
        obs_mode=args.obs_mode,
        headless=not args.render,
        cam=args.cam,
    )
    if args.video:
        import datetime

        id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        video_dir = f"videos/{id}"
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        frames = {}
        for cam in env.cameras.keys():
            frames[cam] = []

    while eps < args.num_episodes:
        # icecream.ic(env.horizon, args.env, env.task_description)
        print("======================================")
        config_path = f"gensim2/env/solver/kpam/config/{args.env}.yaml"
        load_file = open(config_path, mode="r")
        expert_planner = KPAMPlanner(
            env, config_path, env.task.articulator.random_pose["rot"]
        )

        print(f"Running Episode {eps} for {args.env}")
        obs = env.reset(args.random)

        if args.video:
            for cam in env.cameras.keys():
                frames[cam].append(obs["image"][cam])

        expert_planner.reset_expert()

        actions = []

        if env.viewer and env.viewer.closed:
            break
        import time

        t = time.time()
        traj = []
        qpos = env.get_joint_positions()
        openness = env.articulator.get_openness()[0]

        traj.append({"qpos": qpos, "openness": openness})

        for i in range(args.max_steps):
            action = expert_planner.get_action()
            obs, reward, done, info = env.step(action, whole_eps=not args.early_stop)

            if args.video:
                for cam in env.cameras.keys():
                    frames[cam].append(obs["image"][cam])

            if args.render:
                env.render()

            if done:
                if env.task.articulator is not None:
                    print("Openness:", env.task.articulator.get_openness())
                print("Task Solved:", env.task.get_progress_state())
                print("Time:", print(time.time() - t))
                success = env.task.get_progress_state()

                if args.video:
                    for cam in env.cameras.keys():
                        # save frames to folder
                        from PIL import Image

                        for i, frame in enumerate(frames[cam]):
                            img = Image.fromarray(frame)
                            image_dir = f"{video_dir}/images"
                            if not os.path.exists(image_dir):
                                os.makedirs(image_dir)
                            img.save(f"{image_dir}/{args.env}_{eps}_{cam}_{i}.png")

                        import imageio

                        imageio.mimsave(
                            f"{video_dir}/{args.env}_{eps}_{cam}.mp4",
                            frames[cam],
                            fps=30,
                        )

                break
        if success:
            eps += 1
            print("Episode Succeed!")
        else:
            print("Episode Failed!")
