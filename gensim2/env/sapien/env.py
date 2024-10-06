from pathlib import Path
from typing import Dict, Optional, Sequence, List, Union, Tuple
from collections import OrderedDict
import gym
import numpy as np
import time
import open3d as o3d
import transforms3d
from transforms3d.euler import euler2quat
import sapien.core as sapien
from sapien.sensor import StereoDepthSensor, StereoDepthSensorConfig
import json
from PIL import Image, ImageColor

from gensim2.env.utils.pcd_utils import (
    pcd_downsample,
    add_gaussian_noise,
    add_perlin_noise,
)
from gensim2.env.sapien.sim import SapienSim
from gensim2.env.sapien.objects.tool import Tool
from gensim2.env.sapien.objects.articulated_object import ArticulatedObject
from gensim2.env.sapien.objects.rigid_body import RigidBody
from gensim2.env.utils.pose import set_random_pose
from gensim2.env.base.base_env import GenSimBaseEnv, compute_angle_between
from gensim2.paths import *
import ipdb

DENSE_ASSET_LIST = ["faucet", "bucket_swing", "bag_swing"]
SPARSE_ASSET_LIST = ["toaster_press"]
LARGE_FRICTION_ASSET_LIST = ["toaster_press"]
SMALL_FRICTION_ASSET_LIST = ["bucket_swing"]

LARGE_GRIP_FORCE_ASSET_LIST = ["mug"]


class SapienEnv(SapienSim, GenSimBaseEnv):
    """
    Sapien Env contains the concrete functions for calling when running pipeline codes.
    """

    def __init__(
        self,
        obs_mode="state",
        use_gui=True,
        use_ray_tracing=False,
        headless=False,
        frame_skip=5,
        cam="default",
        **renderer_kwargs,
    ):
        GenSimBaseEnv.__init__(self, obs_mode=obs_mode)
        SapienSim.__init__(
            self,
            use_gui=use_gui,
            use_ray_tracing=use_ray_tracing,
            frame_skip=frame_skip,
            use_visual_obs=(obs_mode in ["image", "pointcloud"]),
            headless=headless,
            cam=cam,
            **renderer_kwargs,
        )
        self.cam_type = cam
        if self.cam_type == "real":
            self.sensor_config = StereoDepthSensorConfig()
            self.opencv_tran_inv = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])

        sapien2opencv = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        self.sapien2opencv_quat = transforms3d.quaternions.mat2quat(sapien2opencv)

    def initialize_agent(self):
        if self.agent is not None:
            self.agent.set_root_pose(sapien.Pose([-0.615, 0, 0]))
            self.agent.set_qpos(self.init_qpos)

        super().initialize_agent()

    def get_contacts(self):
        return self.sim.get_contacts()

    def take_picture(self):
        """Take pictures from all cameras (non-blocking)."""
        for name, cam in self.cameras.items():
            if name == "default_cam" and not self.use_visual_obs:
                continue
            cam.take_picture()
            if self.cam_type == "real":
                cam.compute_depth()

    def get_images(self):
        self.update_render()
        self.take_picture()
        images = OrderedDict()
        for name, cam in self.cameras.items():
            if name == "default_cam":
                continue

            if self.cam_type == "default":
                # old cam
                rgba = cam.get_float_texture("Color")  # [H, W, 4]
                rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
                images[name] = rgba_img
            elif self.cam_type == "real":
                rgb = cam.get_rgb()  # [H, W, 3]
                rgb_img = (rgb * 255).clip(0, 255).astype("uint8")
                images[name] = rgb_img

        return images

    def get_pcds_from_cameras(self):
        pointcloud_obs = OrderedDict()

        def vis_pcd(pcds):
            import open3d as o3d

            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(pcds["pos"])
            # pcd_o3d.colors = o3d.utility.Vector3dVector(pcds["colors"])

            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=1, origin=[0, 0, 0]
            )
            # Initialize the visualizer
            vis = o3d.visualization.Visualizer()
            vis.create_window()

            # Add the point cloud to the visualizer
            vis.add_geometry(pcd_o3d)
            vis.add_geometry(coordinate_frame)

            # Get the render options and set the point size
            render_option = vis.get_render_option()
            render_option.point_size = 1.0  # Set the point size to your desired value

            # Run the visualizer
            vis.run()
            vis.destroy_window()

        def vis_img(sensor):
            import matplotlib.pyplot as plt

            if self.cam_type == "real":
                rgb = sensor.get_rgb()
                ir_l, ir_r = sensor.get_ir()
                depth = sensor.get_depth()

                plt.subplot(221)
                plt.title("RGB Image")
                plt.imshow((rgb * 255).astype(np.uint8))
                plt.subplot(222)
                plt.title("Left Infrared Image")
                plt.imshow((ir_l * 255).astype(np.uint8), cmap="gray")
                plt.subplot(223)
                plt.title("Right Infrared Image")
                plt.imshow((ir_r * 255).astype(np.uint8), cmap="gray")
                plt.subplot(224)
                plt.title("Depth Map")
                plt.imshow(depth)
                plt.show()
            elif self.cam_type == "default":
                rgba = sensor.get_float_texture("Color")  # [H, W, 4]
                rgb = (rgba[..., :3] * 255).clip(0, 255).astype("uint8")
                plt.imshow(rgb)
                plt.show()

        def vis_seg(sensor):
            seg_labels = sensor.get_uint32_texture("Segmentation")  # [H, W, 4]
            colormap = sorted(set(ImageColor.colormap.values()))
            color_palette = np.array(
                [ImageColor.getrgb(color) for color in colormap], dtype=np.uint8
            )

            label_image = seg_labels[..., 1].astype(np.uint8)  # actor-level
            mask0 = np.isin(label_image, self.articulator_id["drawer"])
            mask1 = np.isin(label_image, self.rigid_body_id["golf_ball"])
            label0_image = label_image * mask0
            label1_image = label_image * mask1
            label1_pil = Image.fromarray(color_palette[label1_image])
            label1_pil.save("label1.png")
            label0_pil = Image.fromarray(color_palette[label0_image])
            label0_pil.save("label0.png")

        for cam_uid, cam in self.cameras.items():
            if cam_uid == "default_cam":
                continue
            cam_pcd = {}
            if self.cam_type == "default":
                rgba = cam.get_float_texture("Color")
                position = cam.get_float_texture("Position")
                seg = cam.get_uint32_texture("Segmentation").astype(np.uint8)

                # Each pixel is (x, y, z, z_buffer_depth) in OpenGL camera space
                position[..., 3] = position[..., 2] < 0  # (512,512,4)

                # Convert to world space
                cam2world = self.get_camera_params(cam_uid)["cam2world_gl"]
                base_pose = self.get_base_pose()
                xyzw = (
                    position.reshape(-1, 4) @ cam2world.T @ np.linalg.inv(base_pose).T
                )
                cam_pcd["pos"] = xyzw[..., :3]

                # Extra keys
                rgb = rgba[..., :3]
                rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
                cam_pcd["colors"] = rgb.reshape(-1, 3)
                cam_pcd["seg"] = seg.reshape(-1, 4)

            elif self.cam_type == "real":
                pc = cam.get_pointcloud(with_rgb=True)

                position = pc[..., :3]
                position = np.concatenate(
                    [position, np.ones((position.shape[0], 1))], axis=-1
                )
                world2cam = cam.get_pose().to_transformation_matrix()
                world2cam[:3, :3] = world2cam[:3, :3] @ self.opencv_tran_inv
                base_pose = self.get_base_pose()
                position_world = (
                    position.reshape(-1, 4) @ world2cam.T @ np.linalg.inv(base_pose).T
                )
                cam_pcd["pos"] = position_world[..., :3]

                rgb = pc[..., 3:]
                rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
                cam_pcd["colors"] = rgb
                cam_pcd["seg"] = np.zeros_like(rgb)

            pointcloud_obs[cam_uid] = cam_pcd

        return pointcloud_obs

    def get_pcds(self):
        self.update_render()
        self.take_picture()

        pointcloud_obs = self.get_pcds_from_cameras()
        pointcloud_obs = self.merge_pointclouds(pointcloud_obs)
        pointcloud_obs = pcd_downsample(
            pointcloud_obs, num=self.num_pcd, method="fps", bound_clip=True
        )

        return pointcloud_obs

    def create_camera(
        self,
        pose: np.ndarray,
        position: np.ndarray,
        look_at_dir: np.ndarray,
        right_dir: np.ndarray,
        name: str,
        resolution: Sequence[Union[float, int]],
        fov: Union[list, float],
        mount_actor_name: str = None,
    ):
        if not len(resolution) == 2:
            raise ValueError(
                f"Resolution should be a 2d array, but now {len(resolution)} is given."
            )
        if isinstance(fov, float):
            fov = [fov, fov]

        if mount_actor_name is not None:
            mount = [
                actor
                for actor in self.sim.get_all_actors()
                if actor.get_name() == mount_actor_name
            ]
            mount.extend(
                [
                    link
                    for link in self.agent.get_links()
                    if link.get_name() == mount_actor_name
                ]
            )

            if len(mount) == 0:
                raise ValueError(
                    f"Camera mount {mount_actor_name} not found in the env."
                )
            if len(mount) > 1:
                raise ValueError(
                    f"Camera mount {mount_actor_name} name duplicates! To mount an camera on an actor,"
                    f" give the mount a unique name."
                )
            mount = mount[0]
            sapien_pose = sapien.Pose(p=pose[:3], q=pose[3:])

            if self.cam_type == "default":
                cam = self.sim.add_mounted_camera(
                    name,
                    mount,
                    # May need to check this function
                    sapien_pose * sapien.Pose(q=self.sapien2opencv_quat),
                    width=resolution[0],
                    height=resolution[1],
                    fovy=fov[0],
                    fovx=fov[1],
                    near=0.1,
                    far=10,
                )
            elif self.cam_type == "real":
                cam = StereoDepthSensor(name, self.sim, self.sensor_config, mount=mount)
                cam.set_local_pose(sapien_pose * sapien.Pose(q=self.sapien2opencv_quat))
        else:
            # Construct camera pose
            look_at_dir = look_at_dir / np.linalg.norm(look_at_dir)
            right_dir = (
                right_dir
                - np.sum(right_dir * look_at_dir).astype(np.float64) * look_at_dir
            )
            right_dir = right_dir / np.linalg.norm(right_dir)
            up_dir = np.cross(look_at_dir, -right_dir)
            rot_mat_homo = np.stack([look_at_dir, -right_dir, up_dir, position], axis=1)
            pose_mat = np.concatenate([rot_mat_homo, np.array([[0, 0, 0, 1]])])
            pose_cam = sapien.Pose.from_transformation_matrix(pose_mat)
            # print("camera_name = ", name, 'Pose = ', pose_cam)
            if self.cam_type == "default":
                cam = self.sim.add_camera(
                    name,
                    width=resolution[0],
                    height=resolution[1],
                    fovy=fov[0],
                    near=0.1,
                    far=10,
                )
            elif self.cam_type == "real":
                cam = StereoDepthSensor(name, self.sim, self.sensor_config)
            cam.set_local_pose(pose_cam)

        self.cameras.update({name: cam})

    def create_camera_from_pose(
        self,
        pose: np.ndarray,
        name: str,
        resolution: Sequence[Union[float, int]],
        fov: float,
        use_opencv_trans: bool,
    ):
        if isinstance(fov, list):
            fov = fov[0]
        pose = sapien.Pose(pose[:3], pose[3:])
        if not len(resolution) == 2:
            raise ValueError(
                f"Resolution should be a 2d array, but now {len(resolution)} is given."
            )
        if use_opencv_trans:
            pose_cam = pose * sapien.Pose(q=self.sapien2opencv_quat)
        else:
            pose_cam = pose

        if self.cam_type == "default":
            cam = self.sim.add_camera(
                name,
                width=resolution[0],
                height=resolution[1],
                fovy=fov,
                near=0.1,
                far=10,
            )
        elif self.cam_type == "real":
            cam = StereoDepthSensor(name, self.sim, self.sensor_config)

        cam.set_local_pose(pose_cam)
        self.cameras.update({name: cam})
        self.resolutions.update({name: resolution})

    def get_camera_params(self, cam_uid):
        """Get camera parameters."""
        return dict(
            extrinsic_cv=self.cameras[cam_uid].get_extrinsic_matrix(),
            cam2world_gl=self.cameras[cam_uid].get_model_matrix(),
            intrinsic_cv=self.cameras[cam_uid].get_intrinsic_matrix(),
        )

    def add_noise_to_camera(self):
        for cam_name, (noise_level, pose) in self.camera_pose_noise.items():
            if noise_level is None:
                continue
            if cam_name in self.cameras:
                pos_noise = self.np_random.randn(3) * noise_level * 0.03
                rot_noise = self.np_random.randn(3) * noise_level * 0.1
                quat_noise = transforms3d.euler.euler2quat(*rot_noise)
                perturb_pose = sapien.Pose(p=pos_noise, q=quat_noise)
                original_pose = sapien.Pose(p=pose[:3], q=pose[3:])
                self.cameras[cam_name].set_local_pose(
                    original_pose
                    * perturb_pose
                    * sapien.Pose(q=self.sapien2opencv_quat)
                )

    def open_gripper(self):
        curr_qpos = self.agent.get_qpos()
        for i, joint in enumerate(self.active_joints[:-2]):
            joint.set_drive_target(curr_qpos[i])
        for joint in self.active_joints[-2:]:
            joint.set_drive_target(0.04)
        for i in range(100):
            if (i % 4 == 0) and (self.use_gui):
                self.render()
            self.sim.step()

    def init_gripper(self):
        for joint in self.active_joints[-2:]:
            joint.set_drive_target(0)

    def close_gripper(self):
        curr_qpos = self.agent.get_qpos()
        for i, joint in enumerate(self.active_joints[:-2]):
            joint.set_drive_target(curr_qpos[i])
        for joint in self.active_joints[-2:]:
            if self.large_grip_force:
                joint.set_drive_property(stiffness=200, damping=0)
            else:
                joint.set_drive_property(stiffness=20, damping=0)
            joint.set_drive_target(0)

        for i in range(100):
            if i % 4 == 0:
                self.render()
            self.sim.step()

    def set_gripper(self, openness):
        curr_qpos = self.agent.get_qpos()
        for i, joint in enumerate(self.active_joints[:-2]):
            joint.set_drive_target(curr_qpos[i])
        for joint in self.active_joints[-2:]:
            if self.large_grip_force:
                joint.set_drive_property(stiffness=200, damping=0)
            else:

                joint.set_drive_property(stiffness=20, damping=0)
            joint.set_drive_target(int(openness) * 0.04)

    def grasp(self):
        curr_pose = self.get_ee_pose_in_base().copy()
        action = np.zeros(7)
        action[:3] = curr_pose[:3, 3]
        action[3:-1] = transforms3d.euler.mat2euler(curr_pose[:3, :3])
        action[-1] = 0.0

        self.gripper_state = 0

        obs, done, info = self.step(action)

        if self.use_gui:
            self.render()

    def ungrasp(self):
        curr_pose = self.get_ee_pose_in_base().copy()
        action = np.zeros(7)
        action[:3] = curr_pose[:3, 3]
        action[3:-1] = transforms3d.euler.mat2euler(curr_pose[:3, :3])
        action[-1] = 1.0

        self.gripper_state = 1

        obs, done, info = self.step(action)

        if self.use_gui:
            self.render()

        for _ in range(10):
            curr_pose = self.get_ee_pose_in_base().copy()
            tf = np.eye(4)
            tf[2, 3] += -0.008
            next_pose = np.dot(curr_pose, tf)
            action = np.zeros(7)
            action[:3] = next_pose[:3, 3]
            action[3:-1] = transforms3d.euler.mat2euler(next_pose[:3, :3])
            action[-1] = 1.0
            obs, done, info = self.step(action)

            if self.use_gui:
                self.render()

    # def reset_internal(self):
    #     # if self.init_state is not None:
    #     #     self.sim.unpack(self.init_state)
    #     # self.reset_sim()
    #     # if self.init_state is None:
    #     #     self.init_state = self.sim.pack()

    #     super().reset_internal()

    def get_object_pose(self, obj_type):
        if self.articulator is not None and obj_type == "Articulated":
            return self.articulator.get_pose().to_transformation_matrix()
        elif self.rigid_body is not None and obj_type == "RigidBody":
            return self.rigid_body.pose.to_transformation_matrix()

    def get_tool_pose(self):
        return self.tool.pose.to_transformation_matrix()

    def get_ee_pose(self):
        return self.ee_link.pose.to_transformation_matrix()

    def get_base_pose(self):
        return self.base_link.pose.to_transformation_matrix()

    def get_ee_pose_in_base(self):
        base_pose = self.get_base_pose()
        ee_pose = self.get_ee_pose()
        return np.dot(np.linalg.inv(base_pose), ee_pose)

    def get_joint_positions(self):
        return self.agent.get_qpos()

    def set_joint_positions(self, qpos):
        self.agent.set_qpos(qpos)

    def check_grasp(self, actor: sapien.ActorBase, min_impulse=1e-6, max_angle=85):
        assert isinstance(actor, sapien.ActorBase), type(actor)
        contacts = self.sim.get_contacts()

        limpulse = get_pairwise_contact_impulse(contacts, self.finger1_link, actor)
        rimpulse = get_pairwise_contact_impulse(contacts, self.finger2_link, actor)

        # direction to open the gripper
        ldirection = self.finger1_link.pose.to_transformation_matrix()[:3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[:3, 1]

        # angle between impulse and open direction
        langle = compute_angle_between(ldirection, limpulse)
        rangle = compute_angle_between(rdirection, rimpulse)

        lflag = (
            np.linalg.norm(limpulse) >= min_impulse and np.rad2deg(langle) <= max_angle
        )
        rflag = (
            np.linalg.norm(rimpulse) >= min_impulse and np.rad2deg(rangle) <= max_angle
        )

        return all([lflag, rflag])

    def check_contact_fingers(self, actor: sapien.ActorBase, min_impulse=1e-6):
        assert isinstance(actor, sapien.ActorBase), type(actor)
        contacts = self.sim.get_contacts()

        limpulse = get_pairwise_contact_impulse(contacts, self.finger1_link, actor)
        rimpulse = get_pairwise_contact_impulse(contacts, self.finger2_link, actor)

        return (
            np.linalg.norm(limpulse) >= min_impulse,
            np.linalg.norm(rimpulse) >= min_impulse,
        )

    def check_contact(
        self,
        actors1: List[sapien.Actor],
        actors2: List[sapien.Actor],
        impulse_threshold=1e-2,
    ) -> bool:
        actor_set1 = set(actors1)
        actor_set2 = set(actors2)
        for contact in self.sim.get_contacts():
            contact_actors = {contact.actor0, contact.actor1}
            if (
                len(actor_set1 & contact_actors) > 0
                and len(actor_set2 & contact_actors) > 0
            ):
                impulse = [point.impulse for point in contact.points]
                if np.sum(np.abs(impulse)) < impulse_threshold:
                    continue
                return True
        return False

    def check_actor_pair_contact(
        self, actor1: sapien.Actor, actor2: sapien.Actor, impulse_threshold=1e-2
    ) -> bool:
        actor_pair = {actor1, actor2}
        for contact in self.sim.get_contacts():
            contact_actors = {contact.actor0, contact.actor1}
            if contact_actors == actor_pair:
                impulse = [point.impulse for point in contact.points]
                if np.sum(np.abs(impulse)) < impulse_threshold:
                    continue
                return True
        return False

    def check_actor_pair_contacts(
        self, actors1: List[sapien.Actor], actor2: sapien.Actor, impulse_threshold=1e-2
    ) -> np.ndarray:
        actor_set1 = set(actors1)
        contact_buffer = np.zeros(len(actors1))
        for contact in self.sim.get_contacts():
            contact_actors = {contact.actor0, contact.actor1}
            # print("contact_acotrs = ", contact_actors)
            if len(actor_set1 & contact_actors) > 0 and actor2 in contact_actors:
                impulse = [point.impulse for point in contact.points]
                if np.sum(np.abs(impulse)) < impulse_threshold:
                    continue
                contact_actors.remove(actor2)
                actor_index = actors1.index(contact_actors.pop())
                contact_buffer[actor_index] = 1
        return contact_buffer

    def check_actors_pair_contacts(
        self,
        actors1: List[sapien.Actor],
        actors2: List[sapien.Actor],
        impulse_threshold=1e-2,
    ) -> np.ndarray:
        contact_buffer = np.zeros(len(actors1))
        for actor2 in actors2:
            contact_buffer_local = self.check_actor_pair_contacts(
                actors1, actor2, impulse_threshold
            )
            contact_buffer += contact_buffer_local
        contact_buffer = np.clip(contact_buffer, 0, 1)
        return contact_buffer

    def check_actor_pair_contacts_in_distances(
        self,
        actors1: List[sapien.Actor],
        actor2: sapien.Actor,
        centers: List[np.ndarray],
        radii: List[float],
        impulse_threshold=1e-2,
        reverse=False,
    ) -> np.ndarray:
        actor_set1 = set(actors1)
        contact_buffer = np.zeros(len(actors1))
        for contact in self.sim.get_contacts():
            contact_actors = {contact.actor0, contact.actor1}
            # print("contact_acotrs = ", contact_actors)
            if len(actor_set1 & contact_actors) > 0 and actor2 in contact_actors:
                impulse = [point.impulse for point in contact.points]

                distance = np.ones(len(contact.points))
                for center, radius in zip(centers, radii):
                    if reverse:
                        distance = np.logical_and(
                            distance,
                            np.array(
                                [
                                    np.linalg.norm(point.position - center) < radius
                                    for point in contact.points
                                ]
                            ),
                        )
                    else:
                        distance = np.logical_and(
                            distance,
                            np.array(
                                [
                                    np.linalg.norm(point.position - center) > radius
                                    for point in contact.points
                                ]
                            ),
                        )
                if (np.sum(np.abs(impulse)) < impulse_threshold) or np.sum(
                    np.array(distance)
                ) > 0:
                    continue
                contact_actors.remove(actor2)
                actor_index = actors1.index(contact_actors.pop())
                contact_buffer[actor_index] = 1
        return contact_buffer

    def check_actors_pair_contacts_in_distance(
        self,
        actors1: List[sapien.Actor],
        actors2: List[sapien.Actor],
        centers: List[np.ndarray],
        radii: List[float],
        impulse_threshold=1e-2,
        reverse=False,
    ) -> np.ndarray:
        contact_buffer = np.zeros(len(actors1))
        for actor2 in actors2:
            contact_buffer_local = self.check_actor_pair_contacts_in_distances(
                actors1, actor2, centers, radii, impulse_threshold, reverse=reverse
            )
            contact_buffer += contact_buffer_local
        contact_buffer = np.clip(contact_buffer, 0, 1)
        return contact_buffer

    def load_hand_as_tool(self):
        hand_info, hand_keypoint_path = self.load_hand_as_tool_info()
        tool = Tool(hand_info, hand_keypoint_path)
        return tool

    def load_rigidbody(self, instance_cls):
        if instance_cls in LARGE_GRIP_FORCE_ASSET_LIST:
            self.large_grip_force = True
        else:
            self.large_grip_force = False
        visual_file = RIGIDBODY_OBJECTS_ROOT / instance_cls / "textured.obj"
        collision_file = RIGIDBODY_OBJECTS_ROOT / instance_cls / "collision.obj"
        keypoint_path = RIGIDBODY_OBJECTS_ROOT / instance_cls / "info.json"
        info = json.load(open(keypoint_path))
        scale = 1 if "scale" not in info else info["scale"]
        builder = self.sim.create_actor_builder()
        scales = np.array([scale] * 3)
        density = 100
        builder.add_multiple_collisions_from_file(
            str(collision_file), scale=scales, density=density
        )
        builder.add_visual_from_file(str(visual_file), scale=scales)
        instance = builder.build(name=instance_cls)

        obj = RigidBody(
            instance=instance,
            name=instance_cls,
            tcp_link=self.tcp,
            keypoint_path=keypoint_path,
        )
        self.rigid_body_id[instance_cls] = [instance.id]
        self.obs_register_list.append(obj.get_pos_wrt_tcp)
        return obj

    def load_articulated_object(self, instance_cls, instance_id):

        loader: sapien.URDFLoader = self.sim.create_urdf_loader()
        loader.load_multiple_collisions_from_file = True
        urdf_path = str(
            ARTICULATED_OBJECTS_ROOT / instance_cls / instance_id / "mobility.urdf"
        )
        keypoint_path = str(
            ARTICULATED_OBJECTS_ROOT / instance_cls / instance_id / "info.json"
        )
        info = json.load(open(keypoint_path))
        loader.scale = (
            1
            if "scale" not in json.load(open(keypoint_path))
            else json.load(open(keypoint_path))["scale"]
        )
        if instance_cls in DENSE_ASSET_LIST:
            instance: sapien.Articulation = loader.load(
                urdf_path, config={"density": 7800}
            )
        elif instance_cls in SPARSE_ASSET_LIST:
            instance: sapien.Articulation = loader.load(
                urdf_path, config={"density": 10}
            )
        else:
            instance: sapien.Articulation = loader.load(
                urdf_path, config={"density": 100}
            )

        for joint in instance.get_joints():
            if instance_cls in LARGE_FRICTION_ASSET_LIST:
                joint.set_friction(1e8)
            elif instance_cls in SMALL_FRICTION_ASSET_LIST:
                joint.set_friction(0.05)
            elif "box" in instance_cls or "suitcase" in instance_cls:
                joint.set_friction(0.25)
            elif "laptop" in instance_cls or "dishwasher" in instance_cls:
                joint.set_friction(0.4)
            elif "faucet" in instance_cls:
                joint.set_friction(0.01)
            else:
                joint.set_friction(0.1)

        joint = instance.get_active_joints()[0]
        if "joint_position_range" in info:
            joint_position_range = np.array(
                info["joint_position_range"], dtype=np.float32
            )
            joint.set_limits(np.expand_dims(joint_position_range, axis=0))
        else:
            joint_position_range = joint.get_limits()[0]

        obj = ArticulatedObject(
            instance=instance,
            name=instance_cls,
            keypoint_path=keypoint_path,
            open_qpos=joint_position_range[1],
            close_qpos=joint_position_range[0],
            tcp_link=self.tcp,
        )
        self.articulator = obj
        self.articulator_id[instance_cls] = [link.id for link in instance.get_links()]
        self.obs_register_list.append(obj.get_pos_wrt_tcp)
        self.obs_register_list.append(obj.get_openness)
        return obj


def get_pairwise_contact_impulse(
    contacts: List[sapien.Contact], actor0: sapien.ActorBase, actor1: sapien.ActorBase
):
    pairwise_contacts = get_pairwise_contacts(contacts, actor0, actor1)
    total_impulse = compute_total_impulse(pairwise_contacts)
    return total_impulse


def get_pairwise_contacts(
    contacts: List[sapien.Contact], actor0: sapien.ActorBase, actor1: sapien.ActorBase
) -> List[Tuple[sapien.Contact, bool]]:
    pairwise_contacts = []
    for contact in contacts:
        if contact.actor0 == actor0 and contact.actor1 == actor1:
            pairwise_contacts.append((contact, True))
        elif contact.actor0 == actor1 and contact.actor1 == actor0:
            pairwise_contacts.append((contact, False))
    return pairwise_contacts


def compute_total_impulse(contact_infos: List[Tuple[sapien.Contact, bool]]):
    total_impulse = np.zeros(3)
    for contact, flag in contact_infos:
        contact_impulse = np.sum([point.impulse for point in contact.points], axis=0)
        # Impulse is applied on the first actor
        total_impulse += contact_impulse * (1 if flag else -1)
    return total_impulse
