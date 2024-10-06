import numpy as np
from typing import Optional


class GenSimTaskWrapper:
    def __init__(self, task):
        self.task = task
        self.dt = task.env.dt
        self.horizon = task.env.horizon
        self.cameras = task.env.cameras

    @property
    def gripper_state(self):
        return self.task.env.gripper_state

    @property
    def task_description(self):
        return self.task.task_description

    # TODO: Need to remove these two properties and change the scripts that use them
    @property
    def time(self):
        return self.task.env.time

    @property
    def viewer(self):
        return self.task.env.viewer

    @property
    def renderer(self):
        return self.task.env.renderer

    @property
    def scene(self):
        return self.task.env.sim

    @property
    def observation_space(self):
        return self.task.env.observation_space

    @property
    def action_space(self):
        return self.task.env.action_space

    @property
    def progress(self):
        return self.task.progress

    @property
    def sub_tasks(self):
        return self.task.sub_tasks

    @property
    def articulator(self):
        return self.task.env.articulator

    def setup(self):
        raise NotImplementedError

    def get_observation(self):
        return self.task.env.get_observation()

    def get_images(self):
        return self.task.env.get_images()

    def step(self, action, eval=-1, eval_cam_names=None, whole_eps=False):
        obs, done, info = self.task.env.step(action, eval, eval_cam_names)

        try:
            reward = self.task.get_reward()
        except:
            reward = 0

        try:
            info["success"] = self.task.get_progress_state() >= 1
        except:
            info["success"] = False

        if info["success"] and not whole_eps:
            done = True

        info["current_task"] = self.task.sub_tasks[self.task.task_step]
        info["current_task_description"] = self.task.sub_task_descriptions[
            self.task.task_step
        ]

        all_done = False
        sub_task_success = True
        if done:
            if self.progress[self.task.task_step] < 1:
                sub_task_success = False
            self.reset_internal()
            if self.task.task_step >= len(self.task.success_criteria):
                all_done = True

        if all_done:
            info["next_task"] = "Task done"
            info["next_task_description"] = "Task done"
        else:
            info["next_task"] = self.task.sub_tasks[self.task.task_step]
            info["next_task_description"] = self.task.sub_task_descriptions[
                self.task.task_step
            ]

        info["task_step"] = self.task.task_step
        info["task_progress"] = self.task.progress
        info["all_done"] = all_done
        info["sub_task_success"] = sub_task_success

        return obs, reward, done, info

    def reset(self, random=False):
        return self.task.reset(random)

    def reset_internal(self):
        return self.task.reset_internal()

    def reset_info(self):
        return self.task.reset_info()

    def update_render(self):
        return self.task.env.update_render()

    def render(self):
        return self.task.env.render()

    def close(self):
        return self.task.close()

    def seed(self, seed: Optional[int] = None):
        return self.task.env.seed(seed)

    # Get things from env specific codes
    def get_ee_pose(self):
        return self.task.env.get_ee_pose()

    def get_base_pose(self):
        return self.task.env.get_base_pose()

    def get_ee_pose_in_base(self):
        base_pose = self.get_base_pose()
        ee_pose = self.get_ee_pose()
        return np.dot(np.linalg.inv(base_pose), ee_pose)

    def get_tool_pose(self):
        return self.task.env.get_tool_pose()

    def get_object_pose(self, obj_type):
        return self.task.env.get_object_pose(obj_type)

    def get_joint_positions(self):
        qpos = self.task.env.get_joint_positions()
        if qpos.shape[0] == 9:
            return qpos
        elif qpos.shape[0] == 10:
            return qpos[1:]

    def get_tool_keypoints(self):
        return self.task.env.get_tool_keypoints()

    def get_object_poses(self):
        return self.task.env.get_object_poses()

    def get_object_keypoints(self):
        return self.task.env.get_object_keypoints()

    def get_object_axis(self):
        return self.task.env.get_object_axis()

    def set_joint_positions(self, qpos):
        self.task.env.set_joint_positions(qpos)

    def grasp(self):
        self.task.env.grasp()
        self.task.progress[self.task.task_step] = 1
        self.task.reset_internal()

    def ungrasp(self):
        self.task.env.ungrasp()
        self.task.progress[self.task.task_step] = 1
        self.task.reset_internal()

    def test(self):
        print("Test from wrapper")

    def set_gripper_state(self, state):
        self.task.env.gripper_state = state
