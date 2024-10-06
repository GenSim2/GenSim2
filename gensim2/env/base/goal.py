from typing import Union
import numpy as np


class Goal(object):
    def __init__(self, template, object_list, target=0):
        assert template in ["distance", "joint"]
        self.template = template
        self.object_list = object_list
        self.target = target

    def get_reward(self):
        if self.template == "distance":
            try:
                obj1_pos = self.object_list[0].pose.p
            except:
                obj1_pos = self.object_list[0]
            try:
                obj2_pos = self.object_list[1].pose.p
            except:
                obj2_pos = self.object_list[1]
            distance = np.linalg.norm(obj1_pos - obj2_pos)
            reward = 1 - np.tanh(3.0 * distance)  # encourage palm be close to handle
            # print("reward = ", reward)
            # if self.object_list[0].name == 'laptop':
            #     distance_threshold = 0.05
            # elif self.object_list[0].name == 'box':
            #     distance_threshold = 0.05
            # else:
            distance_threshold = 0.05
            if distance > distance_threshold:
                condition_fulfilled = False
            else:
                condition_fulfilled = True

        elif self.template == "joint":
            instance = self.object_list[0]
            openness = instance.get_openness()[0]
            reward = np.tanh(3.0 * abs(self.target - openness))

            if abs(self.target - openness) > 0.05:
                condition_fulfilled = False
            else:
                condition_fulfilled = True
        elif self.template == "grasp":
            instance1 = self.object_list[0]
            instance2 = self.object_list[1]
        else:
            raise NotImplementedError
        return reward, condition_fulfilled


def translate_point_rews(keypoints, target, cost_type="point_point_dist"):
    rew = -np.linalg.norm(np.array(keypoints) - np.array(target))
    print("keypoint distance reward:", rew)
    # the larger and close to 0, the better.
    return rew


def translate_vec_rews(keypoint_vec, target_vec, target_value):
    # inner product
    keypoint_vec = keypoint_vec / np.linalg.norm(keypoint_vec)
    target_vec = target_vec / np.linalg.norm(target_vec)
    ang_diff = (keypoint_vec * target_vec).sum()
    rew = -np.abs(ang_diff - target_value)
    print("keypoint vec rew:", rew)
    # the difference between of the inner product and the absolute value. the larger and close to 0, the better.
    return rew


class KeypointGoal(object):
    """
    More complex goal reward template for tool-use tasks.
    Compared to the distance costs that apply to single point,
    keypoint supports multiple points and relations between the points.
    There are three keypoints on the tool: head, tail, side.
    The function tries to compute a costs based on point distance, vector
    alignment and orthogonality. The distance and the angle can be thought of as
    generalizations of the single point.
    """

    def __init__(
        self,
        point_curr_list,
        point_target_list,
        vec_curr_list,
        vec_target_list,
        vec_target_value_list,
    ):
        self.point_curr_list = point_curr_list
        self.point_target_list = point_target_list
        self.vec_curr_list = vec_curr_list
        self.vec_target_list = vec_target_list
        self.vec_target_value_list = vec_target_value_list

    def get_reward(self):
        total_rewards = 0
        finish_task = True

        if len(self.point_curr_list) > 0:
            keypoint_rew = translate_point_rews(
                self.point_curr_list, self.point_target_list
            )
            finish_task = finish_task & (keypoint_rew > -0.05)
            total_rewards += keypoint_rew

        if len(self.vec_curr_list) > 0:
            for keypoint_vec, target_vec, target_value in zip(
                self.vec_curr_list, self.vec_target_list, self.vec_target_value_list
            ):
                keypoint_rew += translate_vec_rews(
                    keypoint_vec, target_vec, target_value
                )
                finish_task = finish_task & (keypoint_rew > -0.05)
                total_rewards += keypoint_rew

        return total_rewards, finish_task


class GoalChecker(object):
    def __init__(self):
        self.goals = list()
        self.state = 0

    def append(self, goal: Union[Goal, KeypointGoal]):
        self.goals.append(goal)

    def get_reward(self):
        total_reward = 0
        self.state = 0
        for goal in self.goals:
            cur_reward, condition_fulfilled = goal.get_reward()
            total_reward = total_reward + cur_reward  # state specific reward
            if not condition_fulfilled:
                break
            total_reward += 1  # state progressive reward
            self.state += 1
        return total_reward

    def get_state(self):
        return self.state
