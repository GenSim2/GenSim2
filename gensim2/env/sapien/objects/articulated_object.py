import numpy as np
import json
import sapien.core as sapien
from .rigid_body import RigidBody
import ipdb


class ArticulatedObject(RigidBody):
    def __init__(
        self,
        instance: sapien.Articulation,
        name,
        tcp_link,
        keypoint_path=None,
        open_qpos=1,
        close_qpos=0,
    ):
        super().__init__(instance, name, tcp_link)

        self.open_qpos = open_qpos
        self.close_qpos = close_qpos
        if keypoint_path is not None:
            keypoint_json = json.load(open(keypoint_path))
            self.keypoint_dict = keypoint_json["keypoints"]
            self.scale = 1
            self.axis = None
            if "scale" in keypoint_json:
                self.scale = keypoint_json["scale"]
            if "axis" in keypoint_json:
                self.axis = keypoint_json["axis"]

        # TODO: please keep these codes in case we need
        # if name == "laptop":
        #     self.handle_link = self.instance.get_active_joints()[0].get_child_link()
        #     self.handle2link_relative_pose = self.update_laptop_relative_pose()
        # elif name == "box":
        #     self.handle_link = self.instance.get_active_joints()[0].get_child_link()
        #     self.handle2link_relative_pose = self.update_box_relative_pose()
        # elif name == "faucet":
        #     revolute_joint = self.instance.get_active_joints()[0]
        #     self.handle_link = revolute_joint.get_child_link()

        self.handle_link = self.instance.get_active_joints()[0].get_child_link()
        self.handle2link_relative_pose = self.handle_link.get_collision_shapes()[
            0
        ].get_local_pose()

    def set_openness(self, openness):
        assert 0 <= openness <= 1
        self.instance.set_qpos(
            self.close_qpos + openness * (self.open_qpos - self.close_qpos)
        )

    def get_openness(self):
        joint = self.instance.get_qpos()[0]
        openness = (joint - self.close_qpos) / (self.open_qpos - self.close_qpos)
        openness = np.clip(openness, 0, 1)
        return [openness]

    def get_pose(self) -> sapien.Pose:
        return self.handle_link.pose.transform(self.handle2link_relative_pose)

    def get_keypoints(self):
        """return head, tail, or side keypoint based on the given name"""
        keypoints_inworld = {}
        pose = self.get_pose()
        base_pose = self.instance.get_pose()
        for name, kp in self.keypoint_dict.items():
            scaled_kp = np.array(kp) * self.scale
            kp_loc = sapien.Pose(scaled_kp)
            if "base" in name:
                transformed_kp = base_pose.transform(kp_loc).p
            else:
                transformed_kp = pose.transform(kp_loc).p
            keypoints_inworld[name] = transformed_kp

        return keypoints_inworld

    def get_keypoint_T_tcp(self, keypoint_name):
        return super().get_keypoint_T_tcp(keypoint_name)

    def update_laptop_relative_pose(self):

        vertices_relative_pose_list = list()
        vertices_global_pose_list = list()
        # get all the collision mesh of laptop upper face
        for collision_mesh in self.handle_link.get_collision_shapes():
            vertices = collision_mesh.geometry.vertices
            for vertex in vertices:
                vertex_relative_pose = sapien.Pose(
                    vertex * collision_mesh.geometry.scale
                ).transform(collision_mesh.get_local_pose())
                vertices_relative_pose_list.append(vertex_relative_pose)
                vertices_global_pose_list.append(
                    self.handle_link.pose.transform(vertex_relative_pose)
                )

        z_max = 1e9
        max_z_index = 0
        sum_pos = np.zeros(3)
        for i, vertex_global_pose in enumerate(vertices_global_pose_list):
            sum_pos += vertex_global_pose.p
            z = vertex_global_pose.p[2]
            if z < z_max:
                z_max = z
                max_z_index = i
        mean_pos = sum_pos / len(vertices_global_pose_list)

        # for x and z, we use the corresponding value of the highest vertex
        # for y, we use the mean of all the vertices
        y = vertices_global_pose_list[max_z_index].p[1]
        z = z_max
        x = mean_pos[0]

        handle_global_pose = sapien.Pose(np.array([x, y, z]))
        link_global_pose = self.handle_link.pose

        relative_pose = link_global_pose.inv().transform(handle_global_pose)
        return relative_pose

    def update_box_relative_pose(self):

        vertices_relative_pose_list = list()
        vertices_global_pose_list = list()
        # get all the collision mesh of laptop upper face
        for collision_mesh in self.handle_link.get_collision_shapes():
            vertices = collision_mesh.geometry.vertices
            for vertex in vertices:
                vertex_relative_pose = sapien.Pose(
                    vertex * collision_mesh.geometry.scale
                ).transform(collision_mesh.get_local_pose())
                vertices_relative_pose_list.append(vertex_relative_pose)
                vertices_global_pose_list.append(
                    self.handle_link.pose.transform(vertex_relative_pose)
                )

        z_max = -1e9
        max_z_index = 0
        sum_pos = np.zeros(3)
        for i, vertex_global_pose in enumerate(vertices_global_pose_list):
            sum_pos += vertex_global_pose.p
            z = vertex_global_pose.p[2]
            if z > z_max:
                z_max = z
                max_z_index = i
        mean_pos = sum_pos / len(vertices_global_pose_list)

        # for x and z, we use the corresponding value of the highest vertex
        # for y, we use the mean of all the vertices
        y = vertices_global_pose_list[max_z_index].p[1]
        z = z_max
        x = mean_pos[0]

        handle_global_pose = sapien.Pose(np.array([x, y, z]))
        link_global_pose = self.handle_link.pose

        relative_pose = link_global_pose.inv().transform(handle_global_pose)
        return relative_pose
