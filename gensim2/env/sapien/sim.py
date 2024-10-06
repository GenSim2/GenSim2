from functools import cached_property
from pathlib import Path
from typing import Optional, Dict, Sequence, Union
from collections import OrderedDict

import icecream
import numpy as np
import sapien.core as sapien
import transforms3d.quaternions
from transforms3d.euler import euler2quat
from sapien.core import Pose
from sapien.utils import Viewer
import mplib

from gensim2.env.sapien.objects.link import Link
from gensim2.env.sapien.utils import get_entity_by_name
from gensim2.paths import *
from .constructor import get_engine_and_renderer, add_default_scene_light

try:
    from sim_web_visualizer import (
        create_sapien_visualizer,
        bind_visualizer_to_sapien_scene,
    )
except ImportError:
    print("Cannot import sim_web_visualizer. Please install it first.")

# In Sapien, Articulation is a set of joints and links (behave like a rigid body), i.e., the robot and objects. Actor is an alias of rigid body.


class SapienSim(object):
    """
    SapienSim contains the env initialization codes that will not be used when running pipeline codes.
    """

    SUPPORTED_RENDER_MODES = ("human", "rgb_array", "cameras")

    def __init__(
        self,
        use_gui=True,
        use_ray_tracing=False,
        frame_skip=5,
        use_visual_obs=False,
        no_rgb=False,
        headless=False,
        web_viewer=False,
        cam="default",
        **renderer_kwargs,
    ):
        headless = headless or (not use_visual_obs)
        icecream.ic(use_gui, use_visual_obs, no_rgb, headless)

        if web_viewer:
            create_sapien_visualizer(
                port=6000, host="localhost", keep_default_viewer=False
            )

        engine, renderer = get_engine_and_renderer(
            use_gui=use_gui,
            use_ray_tracing=use_ray_tracing,
            no_rgb=no_rgb,
            **renderer_kwargs,
        )
        self.use_gui = use_gui
        self.engine = engine
        self.renderer = renderer
        self.frame_skip = frame_skip
        self.cam_type = cam

        self.viewer: Optional[Viewer] = None
        self.sim: Optional[sapien.Scene] = None
        self.init_state: Optional[Dict] = None
        self.robot: Optional[sapien.Articulation] = None
        self.robot_name = ""

        self.use_visual_obs = use_visual_obs
        self.headless = headless
        self.no_rgb = no_rgb and (not use_gui)

        self.current_step = 0
        self._sim_freq = 100
        self._sim_steps_per_control = 80

        self.dt = self._sim_steps_per_control / self._sim_freq

        # Construct scene
        self.construct_scene()

        # Dummy camera creation to initial geometry object
        if self.renderer and not self.no_rgb:
            cam = self.sim.add_camera(
                "init_not_used", width=10, height=10, fovy=1, near=0.1, far=1
            )
            self.sim.remove_camera(cam)

        # load table
        self.table = self.create_table(
            table_height=0.6, table_half_size=[0.65, 0.65, 0.025]
        )
        # self.create_room()

        self.obs_register_list = list()

    def construct_scene(self):
        scene_config = sapien.SceneConfig()
        self.sim = self.engine.create_scene(config=scene_config)
        self.sim.set_timestep(1.0 / self._sim_freq)
        try:
            self.sim = bind_visualizer_to_sapien_scene(
                self.sim, self.engine, self.renderer
            )
        except Exception as e:
            print("Bind web visualizer error!", e)

        # If headless is enabled, setup scene lighting for rendering
        if (not self.headless) and (self.cam_type != "real"):
            print("updating lights")
            if len(self.sim.get_all_lights()) <= 1:
                add_default_scene_light(self.sim, self.renderer)
                self.sim.update_render()
        if self.headless and (self.cam_type != "real"):
            print("updating lights")
            if len(self.sim.get_all_lights()) <= 1:
                add_default_scene_light(self.sim, self.renderer)
                self.sim.update_render()

    def setup_agent(self, tool=None, robot="panda"):
        # panda_path = f"{ASSET_ROOT}/robot/panda/panda.urdf"
        if robot == "panda":
            panda_path = f"{ASSET_ROOT}/robot/panda/panda_newgripper.urdf"
            self.tcp_name = "panda_hand"
            self.ee_link_name = "panda_hand"
            self.base_link_name = "panda_link0"
        elif robot == "fr3":
            panda_path = (
                f"{ASSET_ROOT}/robot/fr3/fr3_nogripper.urdf"  # panda_newgripper.urdf"
            )
            self.tcp_name = "fr3_hand"
            self.ee_link_name = "fr3_hand"
            self.base_link_name = "fr3_link0"

        loader: sapien.URDFLoader = self.sim.create_urdf_loader()
        loader.fix_root_link = True

        self.init_root_pose = sapien.Pose([-0.615, 0, 0])
        self.init_qpos = [
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

        self.agent_builder: sapien.ArticulationBuilder = (
            loader.load_file_as_articulation_builder(panda_path)
        )

        self.agent = self.agent_builder.build(fix_root_link=True)

        self.agent.set_root_pose(self.init_root_pose)
        self.agent.set_qpos(self.init_qpos)

        self.active_joints = self.agent.get_active_joints()
        for joint in self.active_joints:
            joint.set_drive_property(stiffness=1000, damping=200)

        # setup planner
        link_names = [link.get_name() for link in self.agent.get_links()]
        joint_names = [joint.get_name() for joint in self.agent.get_active_joints()]

        self.planner = mplib.Planner(
            urdf=panda_path,
            srdf=panda_path.replace("urdf", "srdf"),
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group=self.ee_link_name,
            joint_vel_limits=np.ones(7),
            joint_acc_limits=np.ones(7),
        )

        self.tcp = Link(get_entity_by_name(self.agent.get_links(), self.tcp_name))
        self.ee_link = Link(
            get_entity_by_name(self.agent.get_links(), self.ee_link_name)
        )
        self.base_link = Link(
            get_entity_by_name(self.agent.get_links(), self.base_link_name)
        )
        self.finger1_link = Link(
            get_entity_by_name(self.agent.get_links(), "panda_leftfinger")
        )
        self.finger2_link = Link(
            get_entity_by_name(self.agent.get_links(), "panda_rightfinger")
        )
        self.first_frame_robot_ee_pose = self.tcp.pose

        tool.instance = self.ee_link
        self.tool = tool

    def follow_path(self, result, update_image=-1):
        n_step = result["position"].shape[0]
        # assert n_step > 0, "no path" # We just skip

        for i in range(n_step):
            qf = self.agent.compute_passive_force(
                external=False, gravity=True, coriolis_and_centrifugal=True
            )
            self.agent.set_qf(qf)
            for j in range(7):
                self.active_joints[j].set_drive_target(result["position"][i][j])
                self.active_joints[j].set_drive_velocity_target(
                    result["velocity"][i][j]
                )
            self.sim.step()
            if update_image > -1:
                self.get_images()

    def sim_step(self):
        self.pre_step()
        for i in range(self._sim_steps_per_control):
            self.sim.step()
        self.post_step()
        self.current_step += 1

    def pre_step(self):
        pass

    def post_step(self):
        pass

    def reset_sim(self):
        # self.robot.set_pose(self.robot_init_pose)
        pass

    def __del__(self):
        self.sim = None

    def update_render(self):
        self.sim.update_render()

    def render(self, mode="human"):
        assert self.use_gui
        if mode == "human":
            self.update_render()
            if self.viewer is None:
                self.create_viewer()
            if len(self.sim.get_all_lights()) <= 1:
                add_default_scene_light(self.sim, self.renderer)
            self.viewer.render()
            return self.viewer
        else:
            raise NotImplementedError

    def create_viewer(self):
        viewer = Viewer(renderer=self.renderer)
        try:
            # Use the web renderer
            viewer.set_scene(self.sim._scene)
            viewer.scene = self.sim
        except Exception as e:
            print(
                "Not using web render, set scene error, try to use the original sapien render. Error:",
                e,
            )
            viewer.set_scene(self.sim)

        # The rotation of the free camera is represented as [roll(x), pitch(-y), yaw(-z)]
        # The camera now looks at the origin
        # viewer.set_camera_xyz(x=-4, y=0, z=2)
        # viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 4), y=0)
        viewer.set_camera_xyz(x=0, y=-0.7, z=0.5)
        viewer.set_camera_rpy(r=0, p=-np.pi / 7.0, y=-np.pi / 2.0)
        viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

        viewer.toggle_axes(False)
        viewer.toggle_camera_lines(False)

        self.viewer = viewer

    def create_table(self, table_height=1.0, table_half_size=(0.8, 0.8, 0.025)):
        builder = self.sim.create_actor_builder()
        # Top
        top_pose = sapien.Pose([0, 0, -table_half_size[2]])
        top_material = self.sim.create_physical_material(1, 0.2, 0.01)
        builder.add_box_collision(
            pose=top_pose, half_size=table_half_size, material=top_material
        )
        if (self.renderer is not None) and (not self.no_rgb):
            asset_dir = ASSET_ROOT / "misc"
            table_map_path = asset_dir / "table_map.jpg"
            # Leg
            table_cube_path = asset_dir / "cube.obj"
            table_visual_material = self.renderer.create_material()
            table_visual_material.set_metallic(0.0)
            table_visual_material.set_specular(0.3)
            table_visual_material.set_diffuse_texture_from_file(str(table_map_path))
            # table_visual_material.set_base_color(np.array([1, 1, 1, 1]))
            table_visual_material.set_roughness(0.3)
            leg_size = np.array([0.025, 0.025, (table_height / 2 - table_half_size[2])])
            leg_height = -table_height / 2 - table_half_size[2]
            x = table_half_size[0] - 0.1
            y = table_half_size[1] - 0.1
            if self.use_gui or self.use_visual_obs:
                builder.add_visual_from_file(
                    str(table_cube_path),
                    pose=top_pose,
                    material=table_visual_material,
                    scale=table_half_size,
                    name="surface",
                )
                builder.add_box_visual(
                    pose=sapien.Pose([x, y, leg_height]),
                    half_size=leg_size,
                    material=table_visual_material,
                    name="leg0",
                )
                builder.add_box_visual(
                    pose=sapien.Pose([x, -y, leg_height]),
                    half_size=leg_size,
                    material=table_visual_material,
                    name="leg1",
                )
                builder.add_box_visual(
                    pose=sapien.Pose([-x, y, leg_height]),
                    half_size=leg_size,
                    material=table_visual_material,
                    name="leg2",
                )
                builder.add_box_visual(
                    pose=sapien.Pose([-x, -y, leg_height]),
                    half_size=leg_size,
                    material=table_visual_material,
                    name="leg3",
                )
        return builder.build_static("table")

    def create_box(
        self,
        pose: np.ndarray,
        half_size,
        color=None,
        name="",
    ) -> sapien.Actor:
        """Create a box.

        Args:
            pose: 6D pose of the box.
            half_size: [3], half size along x, y, z axes.
            color: [3] or [4], rgb or rgba
            name: name of the actor.

        Returns:
            sapien.Actor
        """
        half_size = np.array(half_size)
        builder: sapien.ActorBuilder = self.sim.create_actor_builder()
        builder.add_box_collision(half_size=half_size)  # Add collision shape
        builder.add_box_visual(half_size=half_size, color=color)  # Add visual shape
        box: sapien.Actor = builder.build(name=name)
        # Or you can set_name after building the actor
        # box.set_name(name)
        box.set_pose(pose)
        return box

    def create_room(self, length=3):
        room_color = [20.0] * 3
        box = self.create_box(
            sapien.Pose(p=np.array([-length, 0.0, 0])),
            half_size=np.array([1, 100, 100]),
            color=room_color,
            name="box1",
        )
        box.lock_motion()
        box = self.create_box(
            sapien.Pose(p=np.array([length, 0.0, 0])),
            half_size=np.array([1, 100, 100]),
            color=room_color,
            name="box1",
        )
        box.lock_motion()
        box = self.create_box(
            sapien.Pose(p=np.array([0, length, 0])),
            half_size=np.array([100, 1, 100]),
            color=room_color,
            name="box1",
        )
        box.lock_motion()
        box = self.create_box(
            sapien.Pose(p=np.array([0, -length, 0])),
            half_size=np.array([100, 1, 100]),
            color=room_color,
            name="box1",
        )
        box.lock_motion()

        box = self.create_box(
            sapien.Pose(p=np.array([0, 0, 9])),
            half_size=np.array([100, 100, 1]),
            color=room_color,
            name="box1",
        )
        box.lock_motion()

    def setup_lighting(self):
        self.sim.set_ambient_light([0.5, 0.5, 0.5])
        self.sim.add_directional_light(
            np.array([-1, -1, -1]), np.array([0.5, 0.5, 0.5]), shadow=True
        )
        self.sim.add_directional_light([0, 0, -1], [0.9, 0.8, 0.8], shadow=False)

        intensity = 20
        self.sim.add_spot_light(
            np.array([0, 0, 3]),
            direction=np.array([0, 0, -1]),
            inner_fov=0.3,
            outer_fov=1.0,
            color=np.array([0.5, 0.5, 0.5]) * intensity,
            shadow=True,
        )
        self.sim.add_spot_light(
            np.array([1, 0, 2]),
            direction=np.array([-1, 0, -1]),
            inner_fov=0.3,
            outer_fov=1.0,
            color=np.array([0.5, 0.5, 0.5]) * intensity,
            shadow=True,
        )
        # self.sim.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        # self.sim.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
        # self.sim.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
        # self.sim.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)

        visual_material = self.renderer.create_material()
        visual_material.set_base_color(np.array([0.5, 0.5, 0.5, 1]))
        visual_material.set_roughness(0.7)
        visual_material.set_metallic(1)
        visual_material.set_specular(0.04)

        # self.sim.add_ground(-5)
