import numpy as np
import yaml

from gensim2.env.base.task_setting import *
from gensim2.env.sapien.objects.tool import Tool
from gensim2.env.utils.pose import set_random_pose
from gensim2.env.utils.rewards import *
from gensim2.paths import *


class GenSimBaseTask(object):
    def __init__(
        self,
        env,
        articulator=None,
        rigid_body=None,
        tool=None,
        asset_id="",
        is_eval=False,
        number_of_rigid=1,
    ):
        self.env = env
        self.dt = env.dt
        self.time = env.time
        self.horizon = env.horizon

        assert (
            (articulator is None)
            or isinstance(articulator, str)
            or isinstance(articulator, list)
        ), "articulator must be a string"
        assert (
            (rigid_body is None)
            or isinstance(rigid_body, str)
            or isinstance(rigid_body, list)
        ), "rigid_body must be a string or a list"
        assert (tool is None) or isinstance(tool, str), "tool must be a string"

        self.articulator_name = articulator
        self.rigid_body_name = rigid_body
        self.tool_name = tool

        self.number_of_rigid = number_of_rigid

        self.articulator = self.rigid_body = None
        self.tool = self.env.load_hand_as_tool()

        self.asset_id_mode = asset_id

        self.camera_names = ["wrist", "left", "right", "default"]

        self.env.setup_agent(tool=self.tool)
        for cam in self.camera_names:
            self.env.setup_camera_from_config(CAMERA_CONFIG[cam])

        self.reset()

    @property
    def tcp(self):
        return self.env.tcp

    @property
    def success_planned(self):
        return self.env.success_planned

    @property
    def task_type(self):
        if self.articulator_name is not None:
            if self.rigid_body_name is not None:
                return "longhorizon"
            else:
                return "articulated"
        else:
            if self.rigid_body_name is not None:
                return "rigidbody"

    def reset(self, *args, **kwargs):
        # Reset the internal state of the environment and the agent
        self.reset_objects()
        self.env.reset_internal()
        self.env.initialize_agent()
        self.env.gripper_state = 1
        self.task_step = 0
        self.progress = np.zeros(len(self.success_criteria))

        return self.env.get_observation()

    def reset_objects(self):
        # Load articulator and rigid body if there are any
        if self.asset_id_mode == "":
            self.asset_id = get_asset_id(
                self.articulator_name, random=False, task_type=self.task_type
            )
        elif self.asset_id_mode == "random":
            self.asset_id = get_asset_id(
                self.articulator_name, random=True, task_type=self.task_type
            )
        elif self.asset_id_mode == "train":
            self.asset_id = get_train_asset_id(
                self.articulator_name, task_type=self.task_type
            )
        elif self.asset_id_mode == "test":
            self.asset_id = get_test_asset_id(
                self.articulator_name, task_type=self.task_type
            )
        else:
            self.asset_id = self.asset_id_mode
        print("Asset ID: ", self.asset_id)

        if self.articulator_name is not None:
            if self.articulator is not None:
                if isinstance(self.articulator, list):
                    for articulator in self.articulator:
                        self.env.sim.remove_articulation(articulator.instance)
                else:
                    self.env.sim.remove_articulation(self.articulator.instance)
                self.articulator = None

            if isinstance(self.articulator_name, list):
                self.articulator = []
                for articulator, id in zip(self.articulator_name, self.asset_id):
                    self.articulator.append(
                        self.env.load_articulated_object(
                            instance_cls=articulator, instance_id=id
                        )
                    )
            else:
                self.articulator = self.env.load_articulated_object(
                    instance_cls=self.articulator_name, instance_id=self.asset_id
                )
            self.env.articulator = self.articulator

        if self.rigid_body_name is not None:
            if self.rigid_body is not None:
                if isinstance(self.rigid_body, list):
                    for rigid in self.rigid_body:
                        self.env.sim.remove_actor(rigid.instance)
                else:
                    self.env.sim.remove_actor(self.rigid_body.instance)
                self.rigid_body = None
            # TODO: to be fixed
            # if self.number_of_rigid > 1:
            #     self.rigid_body = []
            #     for i in range(self.number_of_rigid):
            #         self.rigid_body.append(self.env.load_rigidbody(instance_cls=self.rigid_body_name))
            if isinstance(self.rigid_body_name, list):
                self.rigid_body = []
                for rigid in self.rigid_body_name:
                    self.rigid_body.append(self.env.load_rigidbody(instance_cls=rigid))
            else:
                self.rigid_body = self.env.load_rigidbody(
                    instance_cls=self.rigid_body_name
                )
            self.env.rigid_body = self.rigid_body

    def reset_internal(self):
        self.env.reset_internal()
        self.task_step += 1

    def reset_info(self):
        info = self.env.get_info()
        info["image"] = self.env.get_images()
        return info

    def get_reward(self):
        raise NotImplementedError

    def update_progress(self):
        criterion = self.success_criteria[self.task_step]

        if criterion == "distance_articulated_rigidbody":
            new_progress = check_objs_distance(self.articulator, self.rigid_body, 0.05)
        elif criterion == "distance_gripper_rigidbody":
            new_progress = check_gripper_obj_distance(self.tool, self.rigid_body, 0.05)
        elif criterion == "distance_gripper_articulated":
            new_progress = check_gripper_obj_distance(self.tool, self.articulator, 0.05)
        elif criterion == "articulated_open":
            new_progress = check_openness(self.articulator.get_openness()[0], 1, 0.2)
        elif criterion == "articulated_closed":
            new_progress = check_openness(self.articulator.get_openness()[0], 0, 0.2)

        self.progress[self.task_step] = new_progress

    def get_progress_state(self):
        try:
            self.update_progress()
        except:
            pass

        return np.average(self.progress)
