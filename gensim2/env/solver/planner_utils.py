import json

from colored import fg
import IPython
from pydrake.all import *
import cv2
import numpy as np
import IPython
from transforms3d.quaternions import *
from transforms3d.euler import *
from transforms3d.axangles import *
import open3d as o3d
import ipdb

robot_plant = None

franka_gripper_points = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.053, -0.0, 0.075],
        [-0.053, 0.0, 0.075],
        [0.053, -0.0, 0.105],
        [-0.053, 0.0, 0.105],
    ]
)

anchor_seeds = np.array(
    [
        [0.0, -1.285, 0, -2.356, 0.0, 1.571, 0.785, 0, 0],
        [2.5, 0.23, -2.89, -1.69, 0.056, 1.46, -1.27, 0, 0],
        [2.8, 0.23, -2.89, -1.69, 0.056, 1.46, -1.27, 0, 0],
        [2, 0.23, -2.89, -1.69, 0.056, 1.46, -1.27, 0, 0],
        [2.5, 0.83, -2.89, -1.69, 0.056, 1.46, -1.27, 0, 0],
        [0.049, 1.22, -1.87, -0.67, 2.12, 0.99, -0.85, 0, 0],
        [-2.28, -0.43, 2.47, -1.35, 0.62, 2.28, -0.27, 0, 0],
        [-2.02, -1.29, 2.20, -0.83, 0.22, 1.18, 0.74, 0, 0],
        [-2.2, 0.03, -2.89, -1.69, 0.056, 1.46, -1.27, 0, 0],
        [-2.5, -0.71, -2.73, -0.82, -0.7, 0.62, -0.56, 0, 0],
        [-2, -0.71, -2.73, -0.82, -0.7, 0.62, -0.56, 0, 0],
        [-2.66, -0.55, 2.06, -1.77, 0.96, 1.77, -1.35, 0, 0],
        [1.51, -1.48, -1.12, -1.55, -1.57, 1.15, 0.24, 0, 0],
        [-2.61, -0.98, 2.26, -0.85, 0.61, 1.64, 0.23, 0, 0],
    ]
)


def add_table_collision_free_constraint(
    ik, plant, frame, bb_size=[0.12, 0.08, 0.08], table_height=0.1
):
    # apprxoimate a link with a bounding box and add as position constraints
    min_height = -0.01 + table_height
    max_num = 100
    y_bound = 1

    ik.AddPositionConstraint(
        frame,
        [0, 0, 0],
        plant.world_frame(),
        [0, -y_bound, min_height],
        [max_num, y_bound, max_num],
    )
    ik.AddPositionConstraint(
        frame,
        [0, 0, -bb_size[2]],
        plant.world_frame(),
        [-0.3, -y_bound, min_height],
        [max_num, y_bound, max_num],
    )
    ik.AddPositionConstraint(
        frame,
        [0, 0, bb_size[2]],
        plant.world_frame(),
        [-0.3, -y_bound, min_height],
        [max_num, y_bound, max_num],
    )
    ik.AddPositionConstraint(
        frame,
        [0, -bb_size[1], 0],
        plant.world_frame(),
        [-0.3, -y_bound, min_height],
        [max_num, y_bound, max_num],
    )
    ik.AddPositionConstraint(
        frame,
        [0, bb_size[1], 0],
        plant.world_frame(),
        [-0.3, -y_bound, min_height],
        [max_num, y_bound, max_num],
    )
    ik.AddPositionConstraint(
        frame,
        [bb_size[0], 0, 0],
        plant.world_frame(),
        [-0.3, -y_bound, min_height],
        [max_num, y_bound, max_num],
    )
    ik.AddPositionConstraint(
        frame,
        [-bb_size[0], 0, 0],
        plant.world_frame(),
        [-0.3, -y_bound, min_height],
        [max_num, y_bound, max_num],
    )


def make_gripper_pts(points, color=(1, 0, 0)):
    # o3d.visualization.RenderOption.line_width = 8.0
    line_index = [[0, 1], [1, 2], [1, 3], [3, 5], [2, 4]]

    cur_gripper_pts = points.copy()
    cur_gripper_pts[1] = (cur_gripper_pts[2] + cur_gripper_pts[3]) / 2.0
    line_set = o3d.geometry.LineSet()

    line_set.points = o3d.utility.Vector3dVector(cur_gripper_pts)
    line_set.lines = o3d.utility.Vector2iVector(line_index)
    line_set.colors = o3d.utility.Vector3dVector(
        [color for i in range(len(line_index))]
    )
    return line_set


def get_interp_time(curr_time, finish_time, ratio):
    """get interpolated time between curr and finish"""
    return (finish_time - curr_time) * ratio + curr_time


def se3_inverse(RT):
    RT = RT.reshape(4, 4)
    R = RT[:3, :3]
    T = RT[:3, 3].reshape((3, 1))
    RT_new = np.eye(4, dtype=np.float32)
    RT_new[:3, :3] = R.transpose()
    RT_new[:3, 3] = -1 * np.dot(R.transpose(), T).reshape((3))
    return RT_new


def rotAxis(axis, angle):
    if axis == "x":
        return rotX(angle)
    elif axis == "y":
        return rotY(angle)
    elif axis == "z":
        return rotZ(angle)
    else:
        Rot = np.eye(4)
        Rot[:3, :3] = axangle2mat(axis, angle)
        return Rot


def rotZ(rotz):
    RotZ = np.array(
        [
            [np.cos(rotz), -np.sin(rotz), 0, 0],
            [np.sin(rotz), np.cos(rotz), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    return RotZ


def rotY(roty):
    RotY = np.array(
        [
            [np.cos(roty), 0, np.sin(roty), 0],
            [0, 1, 0, 0],
            [-np.sin(roty), 0, np.cos(roty), 0],
            [0, 0, 0, 1],
        ]
    )
    return RotY


def rotX(rotx):
    RotX = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(rotx), -np.sin(rotx), 0],
            [0, np.sin(rotx), np.cos(rotx), 0],
            [0, 0, 0, 1],
        ]
    )
    return RotX


def mat2axangle_(mat):
    axangle = mat2axangle(mat[:3, :3])
    return axangle[0] * axangle[1]


def pack_pose(pose, rot_type="quat"):  # for action
    rot_index = 4 if rot_type == "quat" else 3
    if rot_type == "quat":
        rot_func = mat2quat
    elif rot_type == "euler":
        rot_func = mat2euler
    elif rot_type == "axangle":
        rot_func = mat2axangle_
    packed = np.zeros(3 + rot_index)
    packed[3 : 3 + rot_index] = rot_func(pose[:3, :3])
    packed[:3] = pose[:3, 3]
    return packed


def dense_sample_traj_times(sample_times, task_completion_time, start_idx=1):
    """densify the waypoints for IK trajectory"""
    ik_times = list(sample_times)
    for i in range(start_idx, len(sample_times)):
        ratio = 1 if sample_times[i] < task_completion_time else 2
        N = int((sample_times[i] - sample_times[i - 1]) * ratio)  #
        for j in range(N):
            ik_times.insert(
                i, get_interp_time(sample_times[i - 1], sample_times[i], j / N)
            )
    ik_times = sorted(np.unique(ik_times))
    return ik_times


def compose_circular_key_frames(
    center_pose,
    start_time=0,
    r=0.02,
    discrete_frame_num=5,
    total_degrees=np.pi * 2,
    total_time=3,
):
    """generating the circular motion for whisk for instance"""
    poses = []
    times = [
        start_time + (i + 1) / discrete_frame_num * total_time
        for i in range(discrete_frame_num)
    ]
    for i in range(discrete_frame_num):
        pose = center_pose.copy()
        pose[:2, 3] += (
            rotZ(i / discrete_frame_num * total_degrees)[:3, :3] @ np.array([r, 0, 0])
        )[:2]
        poses.append(pose)
    return times, poses


def compose_rotate_y_key_frames(
    center_pose,
    start_time=0,
    r=0.02,
    discrete_frame_num=5,
    total_degrees=np.pi * 2,
    total_time=3,
):
    """generating the circular motion for whisk for instance"""
    poses = []
    times = [
        start_time + (i + 1) / discrete_frame_num * total_time
        for i in range(discrete_frame_num)
    ]
    for i in range(discrete_frame_num):
        pose = center_pose.copy()
        pose[:2, 3] += (
            rotZ(i / discrete_frame_num * total_degrees)[:3, :3] @ np.array([r, 0, 0])
        )[:2]
        poses.append(pose)
    return times, poses


def compose_rotating_key_frames(
    ef_pose,
    center_point,
    start_time=0,
    discrete_frame_num=5,
    total_degrees=np.pi * 2,
    total_time=3,
):
    """generating the circular motion with facing center for wrench for instance"""
    poses = []
    times = [
        start_time + (i + 1) / discrete_frame_num * total_time
        for i in range(discrete_frame_num)
    ]
    rel_pose = ef_pose.copy()
    rel_pose[:3, 3] -= center_point

    for i in range(discrete_frame_num):  # clockwise
        pose = rotZ(-(i + 1) / discrete_frame_num * total_degrees) @ rel_pose.copy()
        pose[:3, 3] += center_point
        poses.append(pose)
    return times, poses


def solve_ik_kpam(
    constraint_dicts,
    gripper_frame,
    keypoints_in_hand,
    keypoints_object_in_world,
    p0,
    q0,
    centering_joint,
    rot_tol=np.pi / 10,
    add_table_col=True,
    add_gripper_faceup=False,
    timeout=False,
    consider_collision=False,
    table_height=0.15,
):
    """
    the simple case for the kpam problem
    always assume the head tool keyponint match the head object keypoint
    the head and tail orthogonal to the 0,0,1
    and the tail and side orthogonal to 0,0,1
    minimize cost and both joint space and the pose space
    """
    ik_context = robot_plant.CreateDefaultContext()
    ik = InverseKinematics(robot_plant, ik_context)

    # separate out cost and constraint
    # cost is more or less address by the position and orientation constraint

    # separate out axis
    for constraint in constraint_dicts:
        if "target_axis" in constraint:
            from_name = constraint["axis_from_keypoint_name"]
            to_name = constraint["axis_to_keypoint_name"]
            target_axis = constraint["target_axis"]
            vec = keypoints_in_hand[to_name] - keypoints_in_hand[from_name]
            tol = constraint["tolerance"]
            tgt = np.arccos(constraint["target_inner_product"])
            lower_bound = max(tgt - tol, 0)
            upper_bound = min(tgt + tol, np.pi)

            ik.AddAngleBetweenVectorsConstraint(
                gripper_frame,
                vec,
                robot_plant.world_frame(),
                target_axis,
                lower_bound,
                upper_bound,
            )

        elif "target_axis_from_keypoint_name" in constraint:
            from_name = constraint["axis_from_keypoint_name"]
            to_name = constraint["axis_to_keypoint_name"]
            target_from_name = constraint["target_axis_from_keypoint_name"]
            target_to_name = constraint["target_axis_to_keypoint_name"]
            tool_vec = keypoints_in_hand[to_name] - keypoints_in_hand[from_name]
            object_vec = (
                keypoints_object_in_world[target_to_name]
                - keypoints_object_in_world[target_from_name]
            )
            tol = constraint["tolerance"]
            tgt = np.arccos(constraint["target_inner_product"])
            lower_bound = max(tgt - tol, 0)
            upper_bound = min(tgt + tol, np.pi)

            ik.AddAngleBetweenVectorsConstraint(
                gripper_frame,
                tool_vec,
                robot_plant.world_frame(),
                object_vec,
                lower_bound,
                upper_bound,
            )
        else:
            name = constraint["keypoint_name"]
            target_name = constraint["target_keypoint_name"]
            tool_point = keypoints_in_hand[name]
            object_point = keypoints_object_in_world[target_name]
            tol = constraint["tolerance"]

            ik.AddPositionConstraint(
                gripper_frame,
                tool_point,
                robot_plant.world_frame(),
                object_point - tol,
                object_point + tol,
            )

    """solving IK to match tool head keypoint and the object keypoint"""
    # maybe add slackness

    # make sure the arm does not go backward
    ik.AddPositionConstraint(
        gripper_frame, [0, 0, 0], robot_plant.world_frame(), [0.05, -1, 0], [1, 1, 1]
    )

    # no rotation constraint
    # ik.AddOrientationConstraint(gripper_frame, RotationMatrix(), plant.world_frame(), pose.rotation(), rot_tol)

    if add_gripper_faceup:
        ik.AddAngleBetweenVectorsConstraint(
            gripper_frame,
            [1, 0, 0],
            robot_plant.world_frame(),
            [0, 0, -1],
            np.pi / 12,
            np.pi,
        )

    # not touching table constraints add elbow
    # if add_table_col:
    #     add_table_collision_free_constraint(ik, robot_plant, gripper_frame, [0.03, 0.04, 0.08])
    #     add_table_collision_free_constraint(
    #         ik,
    #         robot_plant,
    #         robot_plant.GetFrameByName("panda_link6"),
    #         [0.03, 0.03, 0.03],
    #     )
    #     add_table_collision_free_constraint(
    #         ik,
    #         robot_plant,
    #         robot_plant.GetFrameByName("panda_link7"),
    #         [0.03, 0.03, 0.03],
    #     )

    if consider_collision:
        ik.AddMinimumDistanceConstraint(0.01)  # 0.03

    prog = ik.get_mutable_prog()
    q = ik.q()
    solver = SnoptSolver()
    # if timeout:
    #    solver.SetSolverOption(solver.id(), "Major Iterations Limit", 1000)

    # as well as pose space costs experiments
    # print("added quadratic loss")
    joint_cost_mat = np.identity(len(q))
    joint_cost_mat[0, 0] = 10  # 000
    prog.AddQuadraticErrorCost(joint_cost_mat, centering_joint, q)
    ik.AddPositionCost(
        gripper_frame, [0, 0, 0], robot_plant.world_frame(), p0.translation(), np.eye(3)
    )
    ik.AddOrientationCost(
        gripper_frame, RotationMatrix(), robot_plant.world_frame(), p0.rotation(), 1
    )

    prog.SetInitialGuess(q, q0)
    result = solver.Solve(ik.prog())  #
    if result.is_success():
        return result
    else:
        return None


def solve_ik_traj_with_standoff(
    endpoint_pose,
    endpoint_joints,
    q_traj,
    waypoint_times,
    keytimes,
    keyposes,
    rot_tol=0.03,
):
    """run a small trajopt on the trajectory with the solved IK from end-effector traj"""
    # make sure the beginning and the end do not get updated

    waypoint_num = len(waypoint_times)
    prog = MathematicalProgram()
    q = prog.NewContinuousVariables(9, waypoint_num)
    gripper_frame = robot_plant.GetFrameByName("panda_hand")
    plant_contexts = [robot_plant.CreateDefaultContext() for i in range(waypoint_num)]
    q0 = np.array([q_traj.value(t) for t in waypoint_times])

    for idx, time in enumerate(waypoint_times):
        if time in keytimes:  # standoff
            keypoint_idx = keytimes.index(time)
            pose = RigidTransform(keyposes[keypoint_idx])
            prog.AddConstraint(
                OrientationConstraint(
                    robot_plant,
                    gripper_frame,
                    RotationMatrix(),
                    robot_plant.world_frame(),
                    pose.rotation(),
                    rot_tol,
                    plant_contexts[idx],
                ),
                q[:, idx],
            )

            prog.AddConstraint(
                PositionConstraint(
                    robot_plant,
                    robot_plant.world_frame(),
                    pose.translation(),
                    pose.translation(),
                    gripper_frame,
                    [0, 0, 0],
                    plant_contexts[idx],
                ),
                q[:, idx],
            )

        # # table constraint
        # prog.AddConstraint(
        #     PositionConstraint(
        #         robot_plant,
        #         robot_plant.world_frame(),
        #         [0.05, -0.7, 0.06],
        #         [1, 0.7, 1],
        #         gripper_frame,
        #         [0, 0, 0],
        #         plant_contexts[idx],
        #     ),
        #     q[:, idx],
        # )

    # add some other constraints
    prog.AddConstraint(np.sum((q[:, 0] - endpoint_joints[:, 0]) ** 2) == 0)

    # Add smoothness cost
    weight = np.ones((9, 1))
    weight[0] = 10.0
    weight[-1] = 10.0
    prog.AddQuadraticCost(np.sum(weight * (q[:, 1:] - q[:, :-1]) ** 2))
    prog.SetInitialGuess(q, q0.squeeze().T)

    # add linear constraint
    try:
        solver = SnoptSolver()
        result = solver.Solve(prog)
    except:
        solver = IpoptSolver()
        result = solver.Solve(prog)

    if result.is_success():
        return result
    else:
        return None


def build_plant(time_step=0.004, discrete_contact_solver="sap", init_qpos=None):

    global robot_plant
    multibody_plant_config = MultibodyPlantConfig(
        time_step=time_step,
        discrete_contact_solver=discrete_contact_solver,
    )
    builder = DiagramBuilder()
    robot_plant, _ = AddMultibodyPlant(multibody_plant_config, builder)
    franka = AddFranka(robot_plant, init_qpos)
    robot_plant.Finalize()

    diagram = builder.Build()
    fk_context = diagram.CreateDefaultContext()
    fk_plant_context = robot_plant.GetMyMutableContextFromRoot(fk_context)

    return robot_plant, fk_plant_context


def AddFranka(plant, init_qpos):
    franka_combined_path = "assets/robot/panda_drake/panda_arm_hand.urdf"
    parser = Parser(plant)
    franka = parser.AddModelFromFile(franka_combined_path)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("panda_link0"))

    # Set default positions:
    if init_qpos is None:
        init_qpos = [
            0,
            -0.32,
            0.0,
            -2.617993877991494,
            0.0,
            2.23,
            0.7853981633974483,
            0.04,
            0.04,
        ]

    index = 0
    for joint_index in plant.GetJointIndices(franka):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, RevoluteJoint):
            joint.set_default_angle(init_qpos[index])
            index += 1
    return franka
