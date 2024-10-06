import numpy as np
import gensim2.env.solver.kpam.SE3_utils as SE3_utils
from gensim2.env.solver.kpam.optimization_problem import OptimizationProblemkPAM


def keypoint_l2_cost(problem, from_keypoints, goal_keypoints, weight, xyzrpy):
    # type: (OptimizationProblemkPAM, np.ndarray, np.ndarray, float, np.ndarray) -> np.ndarray
    """
    :param from_keypoints: (N, 3) np.ndarray with float
    :param goal_keypoints: (N, 3) np.ndarray with float
    :param weight: (N, 3) np.ndarray with float
    :param xyzrpy: np.ndarray with potentially symbolic variable
    :return: The cost value
    """
    cost = 0.0
    T_eps = SE3_utils.xyzrpy_to_matrix(xyzrpy)
    T = np.dot(problem.T_init, T_eps)

    # The iteration on keypoints
    num_keypoints = from_keypoints.shape[0]
    for i in range(num_keypoints):
        q = SE3_utils.transform_point(T, from_keypoints[i, :])
        q_goal = goal_keypoints[i, :]
        delta = q - q_goal
        cost += np.dot(delta, delta)

    return weight * cost


def point2plane_cost(
    problem, from_keypoints, goal_keypoints, weight, plane_normal, xyzrpy
):
    cost = 0.0
    T_eps = SE3_utils.xyzrpy_to_matrix(xyzrpy)
    T = np.dot(problem.T_init, T_eps)

    # The iteration on keypoints
    num_keypoints = from_keypoints.shape[0]
    for i in range(num_keypoints):
        q = SE3_utils.transform_point(T, from_keypoints[i, :])
        q_goal = goal_keypoints[i, :]
        delta = q - q_goal
        delta_normal = np.dot(delta, plane_normal)
        cost += np.dot(delta_normal, delta_normal)

    return weight * cost


def vector_dot_symbolic(problem, from_axis, to_axis, xyzrpy):
    # type: (OptimizationProblemkPAM, np.ndarray, np.ndarray, np.ndarray) -> np.ndarray
    """
    dot(T.dot(from_axis), to_axis)
    :param problem:
    :param from_axis:
    :param to_axis:
    :param xyzrpy:
    :return:
    """
    T_eps = SE3_utils.xyzrpy_to_matrix_symbolic(xyzrpy)
    T = np.dot(problem.T_init, T_eps)
    R = T[0:3, 0:3]
    transformed_from = np.dot(R, from_axis)
    return np.dot(transformed_from, to_axis)


def transform_point_symbolic(problem, from_keypoint, xyzrpy):
    # type: (OptimizationProblemkPAM, np.ndarray, np.ndarray) -> np.ndarray
    """
    Transform the point using symbolic expression
    :param problem:
    :param from_keypoint: np.ndarray with float
    :param xyzrpy: np.ndarray with symbolic variable
    :return:
    """
    T_eps = SE3_utils.xyzrpy_to_matrix_symbolic(xyzrpy)
    T = np.dot(problem.T_init, T_eps)
    return SE3_utils.transform_point(T, from_keypoint)
