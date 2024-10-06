import numpy as np
import argparse
from collections import defaultdict, OrderedDict
import transforms3d
import time

from gensim2.agent.dataset.sim_traj_dataset import TrajDataset
from gensim2.env.utils.rlbench import *

from common_parser import parser

parser.add_argument("--num_episodes", type=int, default=100)
parser.add_argument("--max_steps", type=int, default=500)
parser.add_argument("--save", action="store_true")
parser.add_argument("--load_from_cache", action="store_true")
parser.add_argument(
    "--action_mode", type=str, default="key_pose"
)  # "joint_positions") # joint_positions, gripper_pose, key_pose

if __name__ == "__main__":
    parser.add_argument("--nprocs", type=int, default=20)
    args = parser.parse_args()
    dataset_name = "rlbench_test_keypose"
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

    env, tr = create_rlbench_env("joint_positions")
    np.random.seed(int(time.time()))

    for i, env_name in enumerate(envs):

        task = env.get_task(ENV_DICT[env_name])  # -> Task

        num = 0
        while num < args.num_episodes:
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
                        action = np.concatenate(
                            [
                                keypoint.gripper_pose[:3],
                                transforms3d.euler.quat2euler(
                                    keypoint.gripper_pose[3:]
                                ),
                            ]
                        )
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
                dataset.append_episode(steps, task_description, env_name)
                print(f"{env_name}: {i} Episode Succeed and Saved!")
            del demo

    env.shutdown()
