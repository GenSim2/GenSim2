import json
import gensim2.env.solver.kpam.SE3_utils as SE3_utils

# The specification of optimization problem
import gensim2.env.solver.kpam.term_spec as term_spec
import gensim2.env.solver.kpam.mp_terms as mp_terms
from gensim2.env.solver.kpam.optimization_problem import (
    OptimizationProblemkPAM,
    solve_kpam,
)
from gensim2.env.solver.kpam.optimization_spec import OptimizationProblemSpecification
from gensim2.env.solver.kpam.mp_builder import OptimizationBuilderkPAM

# import gensim2.env.solver.planner_utils as plannerutils
from gensim2.env.solver.planner_utils import *
from colored import fg
import IPython
import numpy as np
import time
from pydrake.all import *
import ipdb
import yaml

MOTION_DICT = {
    "move-up": ["translate_z", 0.06],
    "move-down": ["translate_z", -0.06],
    "move-left": ["translate_y", 0.08],
    "move-right": ["translate_y", -0.08],
    "move-forward": ["translate_x", 0.1],
    "move-backward": ["translate_x", -0.1],
}


class KPAMPlanner:
    """
    A general class of keypoint-based trajectory optimization methods to solve
    robotic tasks. Includes task specific motion.
    https://github.com/liruiw/Fleet-Tools/blob/master/core/expert/base_expert.py
    Mostly works with simple and kinematic tasks.
    """

    def __init__(self, env, cfg_path, obj_rot=0):
        self.env = env
        self.cfg_path = cfg_path
        self.plan_time = 0
        self.goal_joint = None
        self.kpam_success = False
        self.joint_plan_success = False
        self.obj_rot = obj_rot

        if type(cfg_path) is dict:
            self.cfg = cfg_path
        else:
            self.cfg = yaml.load(open(cfg_path), Loader=yaml.SafeLoader)

        if "actuation_time" in self.cfg:
            self.actuation_time = self.cfg["actuation_time"] + self.env.time
            self.pre_actuation_times = [
                t + self.env.time for t in self.cfg["pre_actuation_times"]
            ]
            self.post_actuation_times = [
                t + self.env.time for t in self.cfg["post_actuation_times"]
            ]
            # self.pre_actuation_poses_intool = self.cfg["pre_actuation_poses_intool"]
            # self.post_actuation_poses_intool = self.cfg["post_actuation_poses_intool"]

            self.pre_actuation_motions = self.cfg["pre_actuation_motions"]
            self.post_actuation_motions = self.cfg["post_actuation_motions"]

        elif "post_actuation_motions" in self.cfg:
            self.actuation_time = 24
            self.pre_actuation_motions = self.cfg["pre_actuation_motions"]
            self.post_actuation_motions = self.cfg["post_actuation_motions"]
            self.pre_actuation_times = [
                24 - 4 * i for i in range(len(self.pre_actuation_motions), 0, -1)
            ]
            self.post_actuation_times = [
                24 + 4 * (i + 1) for i in range(len(self.post_actuation_motions))
            ]

        elif "pre_actuation_motions" in self.cfg:
            self.actuation_time = 24
            self.pre_actuation_motions = self.cfg["pre_actuation_motions"]
            self.post_actuation_motions = []
            self.pre_actuation_times = [
                24 - 4 * i for i in range(len(self.pre_actuation_motions), 0, -1)
            ]
            self.post_actuation_times = []

        elif "post_actuation_motions" in self.cfg:
            self.actuation_time = 24 + self.env.time
            self.pre_actuation_motions = self.cfg["pre_actuation_motions"]
            self.post_actuation_motions = self.cfg["post_actuation_motions"]
            self.pre_actuation_times = [
                24 + self.env.time - 4 * i
                for i in range(len(self.pre_actuation_motions), 0, -1)
            ]
            self.post_actuation_times = [
                24 + self.env.time + 4 * (i + 1)
                for i in range(len(self.post_actuation_motions))
            ]

        elif "pre_actuation_motions" in self.cfg:
            self.actuation_time = 24 + self.env.time
            self.pre_actuation_motions = self.cfg["pre_actuation_motions"]
            self.post_actuation_motions = []
            self.pre_actuation_times = [
                24 - 4 * i + self.env.time
                for i in range(len(self.pre_actuation_motions), 0, -1)
            ]
            self.post_actuation_times = []

        self.plant, self.fk_context = build_plant(
            init_qpos=self.env.get_joint_positions()
        )

        self.reset_expert()

    def setup(self):
        # load keypoints
        self.solved_ik_times = []
        self.joint_traj_waypoints = []

    def reset_expert(self):
        """reinitialize expert state"""
        self.joint_space_traj = None
        self.task_space_traj = None
        self.plan_succeeded = False
        self.plan_time = 0.0
        self.setup()

    def check_plan_empty(self):
        """check if already have a plan"""
        return self.joint_space_traj is None

    def get_pose_from_translation(self, translation, pre_pose):
        """get the pose from translation"""
        pose = np.eye(4)
        translation = np.array(translation)
        pose[:3, 3] = translation
        actuation_pose = pre_pose @ pose
        return actuation_pose

    def get_pose_from_translation_inworld(self, translation, pre_pose):
        """get the pose from translation"""
        translation = np.array(translation)
        actuation_pose = pre_pose.copy()
        actuation_pose[:3, 3] += translation
        return actuation_pose

    def get_pose_from_rotation(self, rotation, pre_pose):
        """get the pose from rotation"""
        axis = self.env.get_object_axis()
        Rot = rotAxis(angle=rotation, axis=axis)
        actuation_pose = (
            self.object_pose @ Rot @ se3_inverse(self.object_pose) @ pre_pose
        )
        return actuation_pose

    def generate_actuation_poses(self):
        """build the post-activation trajectory specified in the config
        (1) reach above the screw
        (2) hammer it by moving downward
        """
        self.pre_actuation_poses = []
        self.post_actuation_poses = []

        curr_pose = self.task_goal_hand_pose
        for motion in self.pre_actuation_motions:
            mode = motion[0]
            value = motion[1]

            assert mode in ["translate_x", "translate_y", "translate_z", "rotate"]
            assert type(value) == float

            if mode == "rotate":
                curr_pose = self.get_pose_from_rotation(value, curr_pose)
                self.pre_actuation_poses.append(curr_pose)
            else:
                value_vec = [0, 0, 0]
                if mode == "translate_x":
                    value_vec[0] = value
                elif mode == "translate_y":
                    value_vec[1] = value
                elif mode == "translate_z":
                    value_vec[2] = value
                curr_pose = self.get_pose_from_translation(value_vec, curr_pose)
                self.pre_actuation_poses.append(curr_pose)

        self.pre_actuation_poses.reverse()

        curr_pose = self.task_goal_hand_pose
        for motion in self.post_actuation_motions:
            if type(motion) == list:
                mode = motion[0]
                value = motion[1]

                if mode == "rotate":
                    curr_pose = self.get_pose_from_rotation(value, curr_pose)
                    self.post_actuation_poses.append(curr_pose)
                else:
                    value_vec = [0, 0, 0]
                    if mode == "translate_x":
                        value_vec[0] = value * np.cos(self.obj_rot)
                        value_vec[1] = value * np.sin(self.obj_rot)
                    elif mode == "translate_y":
                        value_vec[0] = -value * np.sin(self.obj_rot)
                        value_vec[1] = value * np.cos(self.obj_rot)
                    elif mode == "translate_z":
                        value_vec[2] = value
                    curr_pose = self.get_pose_from_translation(value_vec, curr_pose)
                    self.post_actuation_poses.append(curr_pose)

            elif type(motion) == str:
                mode = MOTION_DICT[motion][0]
                value = MOTION_DICT[motion][1]

                value_vec = [0, 0, 0]
                if mode == "translate_x":
                    value_vec[0] = value
                elif mode == "translate_y":
                    value_vec[1] = value
                elif mode == "translate_z":
                    value_vec[2] = value
                curr_pose = self.get_pose_from_translation_inworld(value_vec, curr_pose)
                self.post_actuation_poses.append(curr_pose)

        self.sample_times = (
            [self.env.time]
            + self.pre_actuation_times
            + [self.actuation_time]
            + self.post_actuation_times
        )

        self.traj_keyframes = (
            [self.ee_pose.reshape(4, 4)]
            + self.pre_actuation_poses
            + [self.task_goal_hand_pose]
            + self.post_actuation_poses
        )

    def get_env_info(self):
        # get current end effector pose, joint angles, object poses, and keypoints from the environment
        self.tool_keypoint_in_world = self.env.get_tool_keypoints()
        self.object_keypoint_in_world = self.env.get_object_keypoints()
        self.dt = self.env.dt  # simulator dt
        self.time = self.env.time  # current time
        self.base_pose = self.env.get_base_pose()
        self.joint_positions = self.env.get_joint_positions()

        self.curr_tool_keypoints = self.compute_tool_keypoints_inbase()
        self.curr_object_keypoints = self.compute_object_keypoints_inbase()
        self.ee_pose = self.compute_hand_pose_inbase()
        self.tool_pose = self.compute_tool_pose_inbase()
        self.object_pose = self.compute_object_pose_inbase()
        self.tool_keypoints_in_hand = self.compute_tool_keypoints_inhand()
        self.tool_rel_pose = self.compute_tool_inhand()

    def compute_hand_pose_inbase(self):
        ee_pose = self.env.get_ee_pose()
        inv_base_pose = se3_inverse(self.base_pose)
        hand_pose_inbase = inv_base_pose.dot(ee_pose)
        hand_pose_inbase = hand_pose_inbase.reshape(4, 4)

        return hand_pose_inbase

    def compute_tool_pose_inbase(self):
        tool_pose = self.env.get_tool_pose()
        inv_base_pose = se3_inverse(self.base_pose)
        tool_pose_inbase = inv_base_pose.dot(tool_pose)
        tool_pose_inbase = tool_pose_inbase.reshape(4, 4)

        return tool_pose_inbase

    def compute_object_pose_inbase(self):
        object_pose = self.env.get_object_pose(obj_type=self.cfg["category_name"])
        inv_base_pose = se3_inverse(self.base_pose)
        object_pose_inbase = inv_base_pose.dot(object_pose)
        object_pose_inbase = object_pose_inbase.reshape(4, 4)

        return object_pose_inbase

    def compute_tool_keypoints_inbase(self):
        inv_base_pose = se3_inverse(self.base_pose)
        tool_keypoints_inbase = {}
        for name, loc in self.tool_keypoint_in_world.items():
            tool_keypoints_inbase[name] = inv_base_pose.dot(
                np.array([loc[0], loc[1], loc[2], 1])
            )[:3]
        # tool_keypoints_inbase = np.array(
        #     [inv_base_pose.dot(np.array([pt[0], pt[1], pt[2], 1])) for pt in self.tool_keypoint_in_world]
        # )
        # tool_keypoints_inbase = tool_keypoints_inbase[:, :3]

        return tool_keypoints_inbase

    def compute_object_keypoints_inbase(self):
        inv_base_pose = se3_inverse(self.base_pose)
        object_keypoints_inbase = {}
        for name, loc in self.object_keypoint_in_world.items():
            object_keypoints_inbase[name] = inv_base_pose.dot(
                np.array([loc[0], loc[1], loc[2], 1])
            )[:3]
        # object_keypoints_inbase = np.array(
        #     [inv_base_pose.dot(np.array([pt[0], pt[1], pt[2], 1])) for pt in self.object_keypoint_in_world]
        # )
        # object_keypoints_inbase = object_keypoints_inbase[:, :3]

        return object_keypoints_inbase

    def compute_tool_keypoints_inhand(self):
        inv_ee_pose = se3_inverse(self.ee_pose)
        tool_keypoints_inhand = {}
        for name, loc in self.curr_tool_keypoints.items():
            tool_keypoints_inhand[name] = inv_ee_pose.dot(
                np.array([loc[0], loc[1], loc[2], 1])
            )[:3]
        # tool_keypoints_inhand = np.array(
        #     [inv_ee_pose.dot(np.array([pt[0], pt[1], pt[2], 1])) for pt in self.curr_tool_keypoints]
        # )
        # tool_keypoints_inhand = tool_keypoints_inhand[:, :3]

        return tool_keypoints_inhand

    def compute_tool_inhand(self):
        inv_ee_pose = se3_inverse(self.ee_pose)
        tool_rel_pose = inv_ee_pose.dot(self.tool_pose)

        return tool_rel_pose

    def create_opt_problem(self, optimization_spec):
        """create a keypoint optimization problem from the current keypoint state"""
        optimization_spec.load_from_config(self.cfg)
        # print(f"load tool keypoint file from {self.cfg_path}")
        # # self.curr_tool_keypoint_head, self.curr_tool_keypoint_tail
        # # match the table
        # constraint_update_keypoint_target = {"tool_head": self.curr_object_keypoints[0]}

        # # minimize movement
        # cost_update_keypoint_target = {
        #     "tool_head": self.curr_tool_keypoints[0],
        #     "tool_tail": self.curr_tool_keypoints[1],
        #     "tool_side": self.curr_tool_keypoints[2],
        # }

        # for term in optimization_spec._cost_list:
        #     if hasattr(term, "keypoint_name") and term.keypoint_name in cost_update_keypoint_target.keys():
        #         term.target_position = cost_update_keypoint_target[term.keypoint_name]

        for term in optimization_spec._constraint_list:
            if (
                hasattr(term, "target_axis_frame")
                and term.target_axis_frame == "object"
            ):
                axis_inobject = term.target_axis.copy()
                term.target_axis = SE3_utils.transform_vec(
                    self.object_pose, axis_inobject
                ).tolist()

        return optimization_spec

    # tool use related
    def solve_actuation_joint(self, generate_traj=True):
        """solve the formulated kpam problem and get goal joint"""

        optimization_spec = OptimizationProblemSpecification()
        optimization_spec = self.create_opt_problem(optimization_spec)

        constraint_dicts = [c.to_dict() for c in optimization_spec._constraint_list]

        # need to parse the kpam config file and create a kpam problem
        indexes = np.random.randint(len(anchor_seeds), size=(8,))
        random_seeds = [self.joint_positions.copy()] + [
            anchor_seeds[idx] for idx in indexes
        ]
        solutions = []

        for seed in random_seeds:
            res = solve_ik_kpam(
                constraint_dicts,
                self.plant.GetFrameByName("panda_hand"),
                self.tool_keypoints_in_hand,
                self.curr_object_keypoints,
                RigidTransform(self.ee_pose.reshape(4, 4)),
                seed.reshape(-1, 1),
                self.joint_positions.copy(),
                rot_tol=0.01,
                timeout=True,
                consider_collision=False,
            )

            if res is not None:
                solutions.append(res.get_x_val()[:9])

        if len(solutions) == 0:
            print("empty solution in kpam")
            self.goal_joint = self.joint_positions[:9].copy()
            self.kpam_success = False
        else:
            self.kpam_success = True
            solutions = np.array(solutions)
            joint_positions = self.joint_positions[:9]
            dist_to_init_joints = np.linalg.norm(
                solutions - joint_positions.copy(), axis=-1
            )
            res = solutions[np.argmin(dist_to_init_joints)]
            self.goal_joint = res

            self.plant.SetPositions(self.fk_context, res)

        # self.task_goal_hand_pose = self.differential_ik.ForwardKinematics(diff_ik_context)
        self.task_goal_hand_pose = self.plant.EvalBodyPoseInWorld(
            self.fk_context, self.plant.GetBodyByName("panda_hand")
        )
        self.task_goal_hand_pose = np.array(self.task_goal_hand_pose.GetAsMatrix4())
        self.task_goal_tool_pose = self.task_goal_hand_pose @ self.tool_rel_pose

        # # Transform the keypoint
        # self.curr_solution_tool_keypoint_head = SE3_utils.transform_point(
        #     self.task_goal_hand_pose, tool_keypoint_loc_inhand[0, :]
        # )
        # self.curr_solution_tool_keypoint_tail = SE3_utils.transform_point(
        #     self.task_goal_hand_pose, tool_keypoint_loc_inhand[1, :]
        # )
        # self.curr_solution_tool_keypoint_side = SE3_utils.transform_point(
        #     self.task_goal_hand_pose, tool_keypoint_loc_inhand[2, :]
        # )

        self.plan_time = self.env.time

        # self.goal_keypoint = np.stack(
        #     (
        #         self.curr_solution_tool_keypoint_head,
        #         self.curr_solution_tool_keypoint_tail,
        #         self.curr_solution_tool_keypoint_side,
        #     ),
        #     axis=0,
        # )

    def get_task_traj_from_joint_traj(self):
        """forward kinematics the joint trajectory to get the task trajectory"""
        self.pose_traj = []
        ik_times = dense_sample_traj_times(self.sample_times, self.actuation_time)
        self.dense_ik_times = ik_times
        for traj_time in ik_times:
            # diff_ik_context = self.differential_ik.GetMyMutableContextFromRoot(self.context)
            set_joints = self.joint_space_traj.value(traj_time)
            self.plant.SetPositions(self.fk_context, set_joints)
            pose = self.plant.EvalBodyPoseInWorld(
                self.fk_context, self.plant.GetBodyByName("panda_hand")
            )
            self.pose_traj.append(pose.GetAsMatrix4())

        self.task_space_traj = PiecewisePose.MakeLinear(
            ik_times, [RigidTransform(p) for p in self.pose_traj]
        )

    def solve_postactuation_traj(self):
        """
        generate the full task trajectory with a FirstOrderHold
        """
        self.generate_actuation_poses()

    def save_traj(self, traj):
        """save the trajectory to a txt file"""
        traj = np.array(traj)
        np.savetxt("traj.txt", traj, delimiter=",")

    def solve_joint_traj(self, densify=True):
        """
        solve for the IKs for each individual waypoint as an initial guess, and then
        solve for the whole trajectory with smoothness cost
        """
        keyposes = self.traj_keyframes
        keytimes = self.sample_times

        self.joint_traj_waypoints = [self.joint_positions.copy()]
        self.joint_space_traj = PiecewisePolynomial.FirstOrderHold(
            [self.env.time, self.actuation_time],
            np.array([self.joint_positions.copy(), self.goal_joint]).T,
        )

        if densify:
            self.dense_traj_times = dense_sample_traj_times(
                self.sample_times, self.actuation_time
            )
        else:
            self.dense_traj_times = self.sample_times

        print("solve traj endpoint")

        # interpolated joint
        res = solve_ik_traj_with_standoff(
            [self.ee_pose.reshape(4, 4), self.task_goal_hand_pose],
            np.array([self.joint_positions.copy(), self.goal_joint]).T,
            q_traj=self.joint_space_traj,
            waypoint_times=self.dense_traj_times,
            keyposes=keyposes,
            keytimes=keytimes,
        )

        # solve the standoff and the remaining pose use the goal as seed.
        # stitch the trajectory
        if res is not None:
            # use the joint trajectory to build task trajectory for panda
            self.joint_plan_success = True
            self.joint_traj_waypoints = res.get_x_val().reshape(-1, 9)
            self.joint_traj_waypoints = list(self.joint_traj_waypoints)
            self.joint_space_traj = PiecewisePolynomial.CubicShapePreserving(
                self.dense_traj_times, np.array(self.joint_traj_waypoints).T
            )
            if densify:
                self.get_task_traj_from_joint_traj()
        else:
            print("endpoint trajectory not solved! environment termination")
            self.joint_plan_success = False
            self.env.need_termination = True
            if densify:
                self.get_task_traj_from_joint_traj()

    def get_joint_action(self):
        """get the joint space action"""
        if self.check_plan_empty():
            print("no joint trajectory")
            return self.env.reset()

        # lookahead
        return self.joint_space_traj.value(self.env.time + self.env.env_dt).reshape(-1)

    def get_pose_action(self, traj_eff_pose):
        """get the task space action"""
        # traj_eff_pose_inworld = self.base_pose @ traj_eff_pose
        action = pack_pose(traj_eff_pose, rot_type="euler")
        action = np.concatenate([action, [self.env.gripper_state]])
        return action

    def get_actuation_joint(self):
        if self.goal_joint is not None:
            return self.goal_joint
        if self.check_plan_empty():
            self.solve_actuation_joint()
            return self.goal_joint
        raise ValueError("no actuation joint")

    def get_action(self, mode="pose"):
        self.get_env_info()

        if self.check_plan_empty():
            s = time.time()
            self.solve_actuation_joint()
            self.solve_postactuation_traj()
            self.solve_joint_traj()
            print(
                "env time: {:.3f} plan generation time: {:.3f}".format(
                    self.env.time, time.time() - s
                )
            )

        """get the task-space action from the kpam expert joint trajectory"""

        traj_eff_pose = self.task_space_traj.value(self.time + self.dt)
        pose_action = self.get_pose_action(traj_eff_pose)

        return pose_action

    def get_actuation_qpos(self):
        """get the actuation qpos"""
        self.get_env_info()
        self.solve_actuation_joint()

        return self.goal_joint, self.kpam_success

    def get_sparse_traj_qpos(self):
        """get the sparse trajectory qpos"""
        self.get_env_info()
        self.solve_actuation_joint()
        self.solve_postactuation_traj()
        self.solve_joint_traj(densify=False)

        return self.joint_traj_waypoints, self.joint_plan_success
