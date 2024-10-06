import math
import numpy as np
from numpy.linalg import norm


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The reuslt is for euler angles (ZYX) radians
def rotationMatrixToEulerAngles(R):

    assert isRotationMatrix(R)

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


# Calculates Rotation Matrix given euler angles (ZYX).
def eulerAnglesToRotationMatrix(theta):

    R_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(theta[0]), -math.sin(theta[0])],
            [0, math.sin(theta[0]), math.cos(theta[0])],
        ]
    )
    R_y = np.array(
        [
            [math.cos(theta[1]), 0, math.sin(theta[1])],
            [0, 1, 0],
            [-math.sin(theta[1]), 0, math.cos(theta[1])],
        ]
    )
    R_z = np.array(
        [
            [math.cos(theta[2]), -math.sin(theta[2]), 0],
            [math.sin(theta[2]), math.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )
    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


# Calculates rotation matrix given axis-angle (radian).
def axisAngleToRotationMatrix(axis, theta):
    # Normlaize the axis
    rot_axis = np.array(axis)
    rot_axis = rot_axis / norm(rot_axis)
    R = np.eye(4)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    kx = rot_axis[0]
    ky = rot_axis[1]
    kz = rot_axis[2]
    R[:3, :3] = np.array(
        [
            [
                cos_theta + kx**2 * (1 - cos_theta),
                -sin_theta * kz + (1 - cos_theta) * kx * ky,
                sin_theta * ky + (1 - cos_theta) * kx * kz,
            ],
            [
                sin_theta * kz + (1 - cos_theta) * kx * ky,
                cos_theta + ky**2 * (1 - cos_theta),
                -sin_theta * kx + (1 - cos_theta) * ky * kz,
            ],
            [
                -sin_theta * ky + (1 - cos_theta) * kx * kz,
                sin_theta * kx + (1 - cos_theta) * ky * kz,
                cos_theta + kz**2 * (1 - cos_theta),
            ],
        ]
    )

    return R
