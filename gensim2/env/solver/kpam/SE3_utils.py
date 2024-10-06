import numpy as np


def transform_point_cloud(T_homo, pc):
    # type: (np.ndarray, np.ndarray) -> np.ndarray
    """
    Transform a point cloud using homogeous transform
    :param T_homo: 4x4 transformation
    :param pc: (N, 3) point cloud
    :return: (N, 3) point cloud
    """
    transformed_pc = T_homo[0:3, 0:3].dot(pc.T)
    transformed_pc = transformed_pc.T
    transformed_pc[:, 0] += T_homo[0, 3]
    transformed_pc[:, 1] += T_homo[1, 3]
    transformed_pc[:, 2] += T_homo[2, 3]
    return transformed_pc


def xyzrpy_to_matrix(xyzrpy):
    """
    Create 4x4 homogeneous transform matrix from pos and rpy
    """
    xyz = xyzrpy[0:3]
    rpy = xyzrpy[3:6]

    T = np.zeros([4, 4], dtype=xyzrpy.dtype)
    T[0:3, 0:3] = rpy_to_rotation_matrix(rpy)
    T[3, 3] = 1
    T[0:3, 3] = xyz
    return T


def rpy_to_rotation_matrix(rpy):
    """
    Creates 3x3 rotation matrix from rpy
    See http://danceswithcode.net/engineeringnotes/rotations_in_3d/rotations_in_3d_part1.html
    """
    u = rpy[0]
    v = rpy[1]
    w = rpy[2]

    R = np.zeros([3, 3], dtype=rpy.dtype)

    # first row
    R[0, 0] = np.cos(v) * np.cos(w)
    R[0, 1] = np.sin(u) * np.sin(v) * np.cos(w) - np.cos(u) * np.sin(w)
    R[0, 2] = np.sin(u) * np.sin(w) + np.cos(u) * np.sin(v) * np.cos(w)

    # second row
    R[1, 0] = np.cos(v) * np.sin(w)
    R[1, 1] = np.cos(u) * np.cos(w) + np.sin(u) * np.sin(v) * np.sin(w)
    R[1, 2] = np.cos(u) * np.sin(v) * np.sin(w) - np.sin(u) * np.cos(w)

    # third row
    R[2, 0] = -np.sin(v)
    R[2, 1] = np.sin(u) * np.cos(v)
    R[2, 2] = np.cos(u) * np.cos(v)

    return R


def transform_point(T, p):
    """
    Transform a point via a homogeneous transform matrix T

    :param: T 4x4 numpy array
    """

    p_homog = np.concatenate((p, np.array([1])))
    q = np.dot(T, p_homog)
    q = q.squeeze()
    q = q[0:3]
    return q


def transform_vec(T, v):
    """
    Transform a vector via a homogeneous transform matrix T

    :param: T 4x4 numpy array
    """
    v = np.array(v)
    R = T[:3, :3]
    u = np.dot(R, v)
    return u


def xyzrpy_to_matrix_symbolic(xyzrpy):
    """
    Create 4x4 homogeneous transform matrix from pos and rpy
    """
    xyz = xyzrpy[0:3]
    rpy = xyzrpy[3:6]

    T = np.zeros([4, 4], dtype=xyzrpy.dtype)
    T[0:3, 0:3] = rpy_to_rotation_matrix_symbolic(rpy)
    T[3, 3] = 1
    T[0:3, 3] = xyz
    return T


def rpy_to_rotation_matrix_symbolic(rpy):
    """
    Creates 3x3 rotation matrix from rpy
    See http://danceswithcode.net/engineeringnotes/rotations_in_3d/rotations_in_3d_part1.html
    """
    from pydrake.math import sin, cos

    u = rpy[0]
    v = rpy[1]
    w = rpy[2]

    R = np.zeros([3, 3], dtype=rpy.dtype)

    # first row
    R[0, 0] = cos(v) * cos(w)
    R[0, 1] = sin(u) * sin(v) * cos(w) - cos(u) * sin(w)
    R[0, 2] = sin(u) * sin(w) + cos(u) * sin(v) * cos(w)

    # second row
    R[1, 0] = cos(v) * sin(w)
    R[1, 1] = cos(u) * cos(w) + sin(u) * sin(v) * sin(w)
    R[1, 2] = cos(u) * sin(v) * sin(w) - sin(u) * cos(w)

    # third row
    R[2, 0] = -sin(v)
    R[2, 1] = sin(u) * cos(v)
    R[2, 2] = cos(u) * cos(v)

    return R
