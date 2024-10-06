import numpy as np
from gensim2.env.solver.planner import *


def check_ontop(top_obj, bottom_obj):
    top_obj_height = top_obj.pos[2]
    bottom_obj_height = bottom_obj.pos[2]

    return (np.linalg.norm(top_obj.pos[2] - bottom_obj.pos[2]) < 0.1) and (
        0.04 <= top_obj_height - bottom_obj_height <= 0.07
    )


def check_objs_distance(obj1, obj2, threshold=0.05):
    pos1 = list(obj1.get_keypoints().values())[-1][:2]
    pos2 = list(obj2.get_keypoints().values())[-1][:2]

    return np.linalg.norm(pos1 - pos2) < threshold


def check_gripper_obj_distance(gripper, obj, threshold=0.05):
    pos1_list = [pos[:2] for pos in list(gripper.get_keypoints().values())]
    pos2 = list(obj.get_keypoints().values())[-1][:2]

    for pos1 in pos1_list:
        if np.linalg.norm(pos1 - pos2) < threshold:
            return True

    return False


def check_distance(distance, threshold=0.05):
    return distance < threshold


def check_openness(openness, fraction, threshold=0.05):
    return abs(openness - fraction) < threshold


def check_qpos(qpos, lower=None, upper=None):
    if lower is None:
        return qpos < upper
    elif upper is None:
        return qpos > lower
    else:
        return (qpos > lower) and (qpos < upper)


def check_height(obj, height, threshold=0.005):
    if isinstance(obj, list):
        return [check_height(ob) for ob in obj]
    else:
        obj_height = obj.pos[2]
        return abs(obj_height - height) < threshold


def task_finished_reward(task_finished):
    reward = 1.0 if task_finished else 0.0
    return reward, task_finished


def l2_norm_reward(pos, threshold=0.05):
    distance = np.linalg.norm(pos)
    reward = 1.0 - np.tanh(3.0 * distance)
    done = check_distance(distance, threshold)
    return reward, done


def l2_distance_reward(pos1, pos2, threshold=0.05):
    distance = np.linalg.norm(pos1 - pos2)
    reward = 1.0 - np.tanh(3.0 * distance)
    done = check_distance(distance, threshold)
    return reward, done


def joint_fraction_reward(openness, fraction, threshold=0.05):
    # reward = np.tanh(3.0 * abs(fraction - openness))
    reward = np.tanh(3.0 * (1 - np.abs(fraction - openness)))
    done = check_openness(openness, fraction, threshold)
    reward += float(done) * 10
    return reward, done


def height_reward(obj_height, height, threshold=0.005):
    reward = min(obj_height - height, 0.5)
    done = check_height(obj_height, height, threshold)
    return reward, done


def grasp_reward(check_grasp_func, max_angle, **kwargs):
    is_grasped = check_grasp_func(max_angle=max_angle, **kwargs)
    reward = 3.0 if is_grasped else 0.0
    return reward, is_grasped


def keypoint_reward(
    keypoints,
    keypoint_targets,
    vectors,
    vector_targets,
    vector_target_values,
    threshold=-0.05,
):
    reward, done = 0.0, True

    for point, target in zip(keypoints, keypoint_targets):
        cur_reward, cur_done = l2_distance_reward(point, target)
        reward += cur_reward
        done = done if cur_done else cur_done

    for vector, target, value in zip(vectors, vector_targets, vector_target_values):
        vector = vector / np.linalg.norm(vector)
        target = target / np.linalg.norm(target)
        ang_diff = (vector * target).sum()
        cur_reward = -np.abs(ang_diff - value)
        cur_done = cur_reward > threshold
        done = done if cur_done else cur_done
        # print(f'value: {value}, ang_diff: {ang_diff}, vec_rew: {cur_reward}, cur_done: {cur_done}, done: {done}')

    return reward, done


def alignment_reward(vector, target, target_value, threshold=-0.05):
    vector = vector / np.linalg.norm(vector)
    target = target / np.linalg.norm(target)
    ang_diff = (vector * target).sum()
    reward = -np.abs(ang_diff - target_value)
    done = reward > threshold
    return reward, done


def axis_alignment_reward(from_pos, to_pos, target, threshold=0.01):
    vector = to_pos - from_pos
    vector = vector / np.linalg.norm(vector)
    target = target / np.linalg.norm(target)
    diff = np.dot(vector, target)
    reward = -np.abs(diff - target).sum()

    done = reward < (-1) * threshold
    return reward, done


def axis_parallel_reward(
    from_pos1, to_pos1, from_pos2, to_pos2, target=-1, threshold=0.01
):
    vec1 = to_pos1 - from_pos1
    vec2 = to_pos2 - from_pos2

    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)

    # check if parallel
    val = np.cross(vec1, vec2).sum()
    reward = -np.abs(val - target)
    done = reward < -1 * threshold

    return reward, done


def progress_reward(reward_list):
    reward = 0.0
    for rew, done in reward_list:
        reward += rew + int(done)
        if not done:
            break
    return reward


def height_reward(obj):
    # check height from table
    obj_height = obj.pos[2]
    target_height = 0.1
    reward = 0.0
    done = False
    if obj_height > target_height:
        reward = 0.5
        done = True
    return reward, done
