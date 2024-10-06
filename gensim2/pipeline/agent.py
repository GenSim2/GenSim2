import numpy as np
import os
from collections import OrderedDict
import IPython
import random
import json
import traceback
import ipdb

import gensim2.pipeline.utils.utils as g_utils
from gensim2.paths import *


class Agent(object):
    """
    class that design new origin_tasks and codes for simulation environments
    """

    def __init__(self, cfg, memory):
        self.cfg = cfg
        self.model_output_dir = cfg["model_output_dir"]
        self.prompt_folder = f"prompts/{cfg['prompt_folder']}"
        self.data_folder = f"prompts/{cfg['prompt_data_folder']}"
        self.memory = memory
        self.chat_log = memory.chat_log
        self.use_template = cfg["use_template"]

    def propose_assets(self, tasks):
        """propose asset urdf"""
        g_utils.add_to_txt(
            self.chat_log, "================= Asset Design!", with_print=True
        )
        self.task_assets = []
        for asset_name in tasks["assets-used"]:
            self.task_assets.append(self.memory.online_asset_buffer[asset_name])

        if os.path.exists(f"{self.prompt_folder}/prompt_asset.txt"):
            print("os.path exists")
            g_utils.add_to_txt(
                self.chat_log, "================= Asset Generation!", with_print=True
            )
            asset_prompt_text = open(f"{self.prompt_folder}/prompt_asset.txt").read()

            if self.use_template:
                asset_prompt_text = asset_prompt_text.replace(
                    "TASK_NAME_TEMPLATE", self.new_task["task-name"]
                )
                asset_prompt_text = asset_prompt_text.replace(
                    "ASSET_STRING_TEMPLATE", str(self.new_task["assets-used"])
                )
                print("Template Asset PROMPT: ", asset_prompt_text)

            res = g_utils.generate_feedback(
                asset_prompt_text, temperature=0, interaction_txt=self.chat_log
            )
            print(
                "Save asset to:",
                self.model_output_dir,
                self.new_task["task-name"] + "_asset_output",
            )
            g_utils.save_text(
                self.model_output_dir, f'{self.new_task["task-name"]}_asset_output', res
            )
            asset_list = g_utils.extract_assets(res)
            IPython.embed()
            # save_urdf(asset_list)
        else:
            print("os.path does not exist")
            print(self.prompt_folder)
            asset_list = OrderedDict()
        return asset_list

    def propose_label_keypoints(self, task, assets):
        """prompt for keypoint selection."""
        g_utils.add_to_txt(
            self.chat_log, "================= Keypoint Label Design!", with_print=True
        )

        # Within the same class of objects there's different types of objects.
        # Such as different styles of hammers or microwaves.
        # We randomly select an object to work with.
        asset = random.choice(assets)
        print("asset:", asset)

        asset_urdf = asset["asset_path"]
        try:
            asset_obj = asset["obj_path"]
        except:
            asset_obj = None
        asset_keypoint_json = asset["gpt_keypoints_path"]
        if not os.path.exists(asset_keypoint_json):
            print(f"{asset_keypoint_json} keypoint absent!")
            return []

        asset_keypoints = json.load(open(asset_keypoint_json))["keypoints"]
        imgs = g_utils.render_urdf_sapien(asset_urdf, asset_obj, asset_keypoints)

        try:
            if os.path.exists(f"{self.prompt_folder}/prompt_keypoints.txt"):
                keypoint_prompt_text = open(
                    f"{self.prompt_folder}/prompt_keypoints.txt"
                ).read()
                # [(img)]
                res = g_utils.generate_feedback_visual(
                    keypoint_prompt_text,
                    imgs,
                    temperature=0,
                    interaction_txt=self.chat_log,
                )
                g_utils.save_text(
                    self.model_output_dir,
                    f'{self.new_task["task-name"]}_keypoint_output',
                    res,
                )
                keypoint_name_list = g_utils.extract_keypoints(
                    res, "keypoint_name_list"
                )
                # print(f"{keypoint_name_list=}")
        except:
            print("Keypoint Prompt Failure")
            keypoint_name_list = ["red", "green", "blue"]

        # index to get keypoint spatial locations
        keypoint_list = g_utils.index_keypoints(asset_keypoints, keypoint_name_list)
        return keypoint_list

    def save_chatlog(self):
        chat_logs = ""
        for i in range(len(self.chat_log)):
            chat_logs += self.chat_log[i] + "\n"
        g_utils.save_text(
            self.model_output_dir, f'{self.new_task["task-name"]}_chat_log', chat_logs
        )

    def propose_task(self, proposed_task_names):
        """generate language descriptions for the task"""
        g_utils.add_to_txt(
            self.chat_log, "================= Task Design!", with_print=True
        )

        if self.use_template:
            task_prompt_text = open(f"{self.prompt_folder}/prompt_task.txt").read()
            task_asset_replacement_str = g_utils.format_dict_prompt(
                self.memory.online_asset_buffer, self.cfg["task_asset_candidate_num"]
            )
            task_prompt_text = task_prompt_text.replace(
                "TASK_ASSET_PROMPT", task_asset_replacement_str
            )

            task_desc_replacement_str = g_utils.format_dict_prompt(
                self.memory.online_task_buffer,
                self.cfg["task_description_candidate_num"],
            )
            print("prompt task description candidates:")
            print(task_desc_replacement_str)
            task_prompt_text = task_prompt_text.replace(
                "TASK_DESCRIPTION_PROMPT", task_desc_replacement_str
            )

            if len(self.cfg["target_task_name"]) > 0:
                task_prompt_text = task_prompt_text.replace(
                    "PLACEHOLDER", "named " + self.cfg["target_task_name"]
                )
            elif len(self.cfg["target_object_name"]) > 0:
                task_prompt_text = task_prompt_text.replace(
                    "PLACEHOLDER", "with object " + self.cfg["target_object_name"]
                )
            else:
                task_prompt_text = task_prompt_text.replace("PLACEHOLDER", "")
        else:
            task_prompt_text = open(f"{self.prompt_folder}/prompt_task.txt").read()

        # maximum number
        print("online_task_buffer size:", len(self.memory.online_task_buffer))
        total_tasks = self.memory.online_task_buffer

        MAX_NUM = 20
        if len(total_tasks) > MAX_NUM:
            total_tasks = dict(random.sample(total_tasks.items(), MAX_NUM))

        res = g_utils.generate_feedback(
            task_prompt_text,
            temperature=self.cfg["gpt_temperature"],
            interaction_txt=self.chat_log,
        )
        print("Res")
        print(res)
        # Extract dictionary for task name, descriptions, and assets
        task_def = g_utils.extract_dict(res, prefix="new_task")
        print("Task def")
        print(task_def)

        try:
            exec(task_def, globals())
            self.new_task = new_task

            self.save_chatlog()
            return self.new_task
        except:
            self.new_task = ""
            print(str(traceback.format_exc()))
            return None

    def propose_task_bottomup(self, proposed_task_names):
        """generate language descriptions for the task"""
        g_utils.add_to_txt(
            self.chat_log, "================= Task Design!", with_print=True
        )

        if self.use_template:
            task_prompt_text = open(f"{self.prompt_folder}/prompt_task.txt").read()
            task_lib_prompt_text = open(f"{self.data_folder}/task_lib.json").read()
            task_asset_replacement_str = g_utils.format_dict_prompt(
                self.memory.online_asset_buffer, self.cfg["task_asset_candidate_num"]
            )
            task_prompt_text = task_prompt_text.replace(
                "TASK_ASSET_PROMPT", task_asset_replacement_str
            )

            task_prompt_text = task_prompt_text.replace(
                "TASK_LIB_PROMPT", task_lib_prompt_text
            )

            task_desc_replacement_str = g_utils.format_dict_prompt(
                self.memory.online_task_buffer,
                self.cfg["task_description_candidate_num"],
            )
            print("prompt task description candidates:")
            print(task_desc_replacement_str)
            task_prompt_text = task_prompt_text.replace(
                "TASK_DESCRIPTION_PROMPT", task_desc_replacement_str
            )

            if len(self.cfg["target_task_name"]) > 0:
                task_prompt_text = task_prompt_text.replace(
                    "PLACEHOLDER", "named " + self.cfg["target_task_name"]
                )
            elif len(self.cfg["target_object_name"]) > 0:
                task_prompt_text = task_prompt_text.replace(
                    "PLACEHOLDER", "with object " + self.cfg["target_object_name"]
                )
            else:
                task_prompt_text = task_prompt_text.replace("PLACEHOLDER", "")

        else:
            task_prompt_text = open(f"{self.prompt_folder}/prompt_task.txt").read()

        # maximum number
        print("online_task_buffer size:", len(self.memory.online_task_buffer))
        total_tasks = self.memory.online_task_buffer

        MAX_NUM = 20
        if len(total_tasks) > MAX_NUM:
            total_tasks = dict(random.sample(total_tasks.items(), MAX_NUM))

        res = g_utils.generate_feedback(
            task_prompt_text,
            temperature=self.cfg["gpt_temperature"],
            interaction_txt=self.chat_log,
        )
        print("Res")
        print(res)
        # Extract dictionary for task name, descriptions, and assets
        tasks = g_utils.extract_multiple_tasks(res)
        self.new_task = tasks[0]
        sub_tasks = tasks[1:]

        self.save_chatlog()

        return self.new_task, sub_tasks

        try:
            exec(task_def, globals())
            self.new_task = new_task
            self.save_chatlog()
            return self.new_task
        except:
            self.new_task = ""
            print(str(traceback.format_exc()))
            return None

    def decompose_task(self):
        """decompose task into task name, description, and solution"""
        g_utils.add_to_txt(
            self.chat_log, "================= Task Decomposition!", with_print=True
        )

        if os.path.exists(f"{self.prompt_folder}/prompt_task_decomposition.txt"):
            task_decomposition_prompt = open(
                f"{self.prompt_folder}/prompt_task_decomposition.txt"
            ).read()
            task_decomposition_prompt = task_decomposition_prompt.replace(
                "TASK_NAME_TEMPLATE", self.new_task["task-name"]
            )
            task_decomposition_prompt = task_decomposition_prompt.replace(
                "TASK_DESCRIPTION_TEMPLATE", str(self.new_task)
            )

            res = g_utils.generate_feedback(
                task_decomposition_prompt,
                temperature=0.0,
                interaction_txt=self.chat_log,
            )
            task_decomposition = g_utils.extract_multiple_tasks(res)
            print(task_decomposition)

            return task_decomposition

    def template_reference_prompt(self):
        """select which code reference to reference"""
        if os.path.exists(
            f"{self.prompt_folder}/prompt_code_reference_selection_template.txt"
        ):
            self.chat_log = g_utils.add_to_txt(
                self.chat_log, "================= Code Reference!", with_print=True
            )
            code_reference_question = open(
                f"{self.prompt_folder}/prompt_code_reference_selection_template.txt"
            ).read()
            code_reference_question = code_reference_question.replace(
                "TASK_NAME_TEMPLATE", self.new_task["task-name"]
            )
            code_reference_question = code_reference_question.replace(
                "TASK_CODE_LIST_TEMPLATE",
                str(list(self.memory.online_code_buffer.keys())),
            )

            code_reference_question = code_reference_question.replace(
                "TASK_STRING_TEMPLATE", str(self.new_task)
            )
            res = g_utils.generate_feedback(
                code_reference_question, temperature=0.0, interaction_txt=self.chat_log
            )
            code_reference_cmd = g_utils.extract_list(res, prefix="code_reference")
            exec(code_reference_cmd, globals())
            task_code_reference_replace_prompt = ""
            for key in code_reference:
                if key in self.memory.online_code_buffer:
                    task_code_reference_replace_prompt += (
                        f"```\n{self.memory.online_code_buffer[key]}\n```\n\n"
                    )
                else:
                    print("missing task reference code:", key)
        else:
            task_code_reference_replace_prompt = g_utils.sample_list_reference(
                self.memory.base_task_codes,
                sample_num=self.cfg["task_code_candidate_num"],
            )
        return task_code_reference_replace_prompt

    def reward_template_reference_prompt(self):
        """select which reward reference to reference"""
        if os.path.exists(
            f"{self.prompt_folder}/prompt_reward_reference_selection_template.txt"
        ):
            self.chat_log = g_utils.add_to_txt(
                self.chat_log, "================= Reward Reference!", with_print=True
            )
            reward_reference_question = open(
                f"{self.prompt_folder}/prompt_reward_reference_selection_template.txt"
            ).read()
            reward_reference_question = reward_reference_question.replace(
                "TASK_NAME_TEMPLATE", self.new_task["task-name"]
            )
            reward_reference_question = reward_reference_question.replace(
                "TASK_CODE_LIST_TEMPLATE",
                str(list(self.memory.online_code_buffer.keys())),
            )

            reward_reference_question = reward_reference_question.replace(
                "TASK_STRING_TEMPLATE", str(self.new_task)
            )
            res = g_utils.generate_feedback(
                reward_reference_question,
                temperature=0.0,
                interaction_txt=self.chat_log,
            )
            reward_reference_cmd = g_utils.extract_list(res, prefix="reward_reference")
            exec(reward_reference_cmd, globals())
            reward_code_reference_replace_prompt = ""
            for key in reward_reference:
                if key in self.memory.online_code_buffer:
                    reward_code_reference_replace_prompt += (
                        f"```\n{self.memory.online_code_buffer[key]}\n```\n\n"
                    )
                else:
                    print("missing reward reference code:", key)
        else:
            reward_code_reference_replace_prompt = g_utils.sample_list_reference(
                self.memory.base_task_codes,
                sample_num=self.cfg["task_code_candidate_num"],
            )
        return reward_code_reference_replace_prompt

    def implement_task(self, task=None):
        if task is None:
            task = self.new_task
        """generate Code for the task"""
        code_prompt_text = open(
            f"{self.prompt_folder}/prompt_code_split_template.txt"
        ).read()
        code_prompt_text = code_prompt_text.replace(
            "TASK_NAME_TEMPLATE", task["task-name"]
        )

        if self.use_template and os.path.exists(
            f"{self.prompt_folder}/prompt_code_reference_selection_template.txt"
        ):
            task_code_reference_replace_prompt = self.template_reference_prompt()
            code_prompt_text = code_prompt_text.replace(
                "TASK_CODE_REFERENCE_TEMPLATE", task_code_reference_replace_prompt
            )

        if os.path.exists(f"{self.prompt_folder}/prompt_reward_lib.txt"):
            reward_code_reference_replace_prompt = open(
                f"{self.prompt_folder}/prompt_reward_lib.txt"
            ).read()
            code_prompt_text = code_prompt_text.replace(
                "REWARD_CODE_REFERENCE_TEMPLATE", reward_code_reference_replace_prompt
            )

        if os.path.exists(f"{self.prompt_folder}/prompt_code_split_template.txt"):
            self.chat_log = g_utils.add_to_txt(
                self.chat_log, "================= Code Generation!", with_print=True
            )
            code_prompt_text = code_prompt_text.replace(
                "TASK_STRING_TEMPLATE", str(task)
            )

        res = g_utils.generate_feedback(
            code_prompt_text, temperature=0, interaction_txt=self.chat_log
        )
        code, task_name = g_utils.extract_code(res)

        if len(task_name) == 0:
            print("empty task name:", task_name)
            return None

        return code, task_name

    def implement_reward(self, env_code, n=1, task=None):
        if task is None:
            task = self.new_task
        """implement reward function"""
        self.chat_log = g_utils.add_to_txt(
            self.chat_log, "================= Reward Generation!", with_print=True
        )

        if os.path.exists(f"{self.prompt_folder}/prompt_reward.txt") and os.path.exists(
            f"{self.prompt_folder}/prompt_reward_lib.txt"
        ):
            # Load and format keypoint information
            asset_cls = task["assets-used"][0]
            asset_id = get_asset_id(asset_cls)
            info_json = json.load(open(self.get_asset_info(asset_cls, asset_id)))
            object_keypoint_desc = g_utils.format_keypoint_prompt(
                info_json["keypoint_descriptions"]
            )
            model0_info = json.load(open("assets/tools/hand/model1/model0_info.json"))[
                "keypoint_descriptions"
            ]
            tool_keypoint_desc = g_utils.format_keypoint_prompt(model0_info)

            # Structure reward prompting
            reward_prompt_text = open(f"{self.prompt_folder}/prompt_reward.txt").read()
            reward_lib_prompt_text = open(
                f"{self.prompt_folder}/prompt_reward_lib.txt"
            ).read()
            reward_prompt_text = reward_prompt_text.replace(
                "ENVIRONMENT_CODE", env_code
            )
            reward_prompt_text = reward_prompt_text.replace(
                "REWARD_LIBRARY", reward_lib_prompt_text
            )
            reward_prompt_text = reward_prompt_text.replace(
                "TOOL_KEYPOINT_DESC", tool_keypoint_desc
            )
            reward_prompt_text = reward_prompt_text.replace(
                "OBJECT_KEYPOINT_DESC", object_keypoint_desc
            )
            reward_prompt_text = reward_prompt_text.replace(
                "TASK_NAME_TEMPLATE", task["task-name"]
            )
            reward_prompt_text = reward_prompt_text.replace(
                "TASK_PLAN_TEMPLATE", task["task-description"]
            )

            res = g_utils.generate_feedback(
                reward_prompt_text, temperature=0, interaction_txt=self.chat_log, n=n
            )
            code = g_utils.extract_reward(res)
            return code
        else:
            return ""

    def propose_task_solution(self):
        # old version: coarse plan
        g_utils.add_to_txt(
            self.chat_log, "================= Task Solution Design!", with_print=True
        )
        plan_prompt_text = open(f"{self.prompt_folder}/prompt_plan.txt").read()
        plan_prompt_text = plan_prompt_text.replace(
            "TASK_NAME_TEMPLATE", self.new_task["task-name"]
        )
        plan_prompt_text = plan_prompt_text.replace(
            "TASK_DESCRIPTION_TEMPLATE", self.new_task["task-description"]
        )
        res = g_utils.generate_feedback(
            plan_prompt_text, temperature=0.2, interaction_txt=self.chat_log
        )
        self.new_task["task-plan"] = res.lstrip("Plan:\n")

    def propose_task_solution_new(self, n=1):
        # save self.chat_log to a file
        self.save_chatlog()

        try:
            """prompt to write optimization config."""
            g_utils.add_to_txt(
                self.chat_log,
                "================= Task Solution Design New!",
                with_print=True,
            )
            plan_prompt_text = open(
                f"{self.prompt_folder}/prompt_task_config.txt"
            ).read()
            plan_prompt_text = plan_prompt_text.replace(
                "TASK_NAME_TEMPLATE", self.new_task["task-name"]
            )
            plan_prompt_text = plan_prompt_text.replace(
                "TASK_DESCRIPTION_TEMPLATE", self.new_task["task-description"]
            )
            res = g_utils.generate_feedback(
                plan_prompt_text, temperature=0.0, interaction_txt=self.chat_log, n=n
            )
            self.new_task["task-solution"] = res  # keep the string form for reward
            task_config = g_utils.extract_opt_config(
                res
            )  # keep the yaml form for planner
            print(task_config)
            return task_config
        except Exception as e:
            print(str(e))
            print("task config creation failure")
            return ""

    def propose_task_solution_visual(self, images, n=1):
        self.save_chatlog()

        try:
            """prompt to write optimization config."""
            g_utils.add_to_txt(
                self.chat_log,
                "================= Task Solution Design Visual!",
                with_print=True,
            )
            asset_info_path = self.memory.online_asset_buffer[
                self.new_task["assets-used"][0]
            ]["keypoints_path"]
            tool_info_path = "assets/tools/hand/model1/model0_info.json"
            keypoint_desc = json.load(open(tool_info_path))["keypoint_descriptions"]
            keypoint_desc.update(
                json.load(open(asset_info_path))["keypoint_descriptions"]
            )
            keypoint_desc_replacement_str = g_utils.format_dict_prompt(keypoint_desc)
            plan_prompt_text = open(
                f"{self.prompt_folder}/prompt_task_config_visual.txt"
            ).read()
            plan_prompt_text = plan_prompt_text.replace(
                "TASK_NAME_TEMPLATE", self.new_task["task-name"]
            )
            plan_prompt_text = plan_prompt_text.replace(
                "TASK_DESCRIPTION_TEMPLATE", self.new_task["task-description"]
            )
            plan_prompt_text = plan_prompt_text.replace(
                "KEYPOINT_DESCRIPTION_TEMPLATE", keypoint_desc_replacement_str
            )
            plan_prompt_text = plan_prompt_text.replace("SOLVER_TRIALS", str(n))
            print(plan_prompt_text)
            res = g_utils.generate_feedback_visual(
                plan_prompt_text, images, temperature=0.5, interaction_txt=self.chat_log
            )
            self.new_task["task-solution"] = res  # keep the string form for reward
            task_configs = g_utils.extract_opt_config(
                res
            )  # keep the yaml form for planner
            print(task_configs)
            return task_configs
        except Exception as e:
            print(str(e))
            print("task config creation failure")
            return ""

    def propose_task_solution_stage1(self, n=1, t=0.0, task=None):
        # save self.chat_log to a file
        self.save_chatlog()
        if task is None:
            task = self.new_task

        try:
            """prompt to write optimization config."""
            g_utils.add_to_txt(
                self.chat_log,
                "================= Task Solution Stage 1 Design!",
                with_print=True,
            )
            asset_info_path = self.get_asset_info(task["assets-used"][0])
            tool_info_path = "assets/tools/hand/model1/model0_info.json"
            keypoint_desc = json.load(open(tool_info_path))["keypoint_descriptions"]
            keypoint_desc.update(
                json.load(open(asset_info_path))["keypoint_descriptions"]
            )
            keypoint_desc_replacement_str = g_utils.format_dict_prompt(keypoint_desc)
            plan_prompt_text = open(
                f"{self.prompt_folder}/prompt_task_config_stage1.txt"
            ).read()
            plan_prompt_text = plan_prompt_text.replace(
                "TASK_NAME_TEMPLATE", self.new_task["task-name"]
            )
            plan_prompt_text = plan_prompt_text.replace(
                "TASK_DESCRIPTION_TEMPLATE", self.new_task["task-description"]
            )
            plan_prompt_text = plan_prompt_text.replace(
                "KEYPOINT_DESCRIPTION_TEMPLATE", keypoint_desc_replacement_str
            )
            res = g_utils.generate_feedback(
                plan_prompt_text, temperature=t, interaction_txt=self.chat_log
            )
            task_config_stage1 = g_utils.extract_opt_config(
                res
            )  # keep the yaml form for planner
            print(task_config_stage1)
            return task_config_stage1
        except Exception as e:
            print(str(e))
            print("task config creation failure at constraint stage")
            return ""

    def propose_task_solution_stage1_visual(self, images, n=1, t=0.0, task=None):
        # save self.chat_log to a file
        self.save_chatlog()
        if task is None:
            task = self.new_task
        try:
            """prompt to write optimization config."""
            g_utils.add_to_txt(
                self.chat_log,
                "================= Task Solution Stage 1 Design Visual!",
                with_print=True,
            )
            asset_info_path = self.get_asset_info(task["assets-used"][0])
            tool_info_path = "assets/tools/hand/model1/model0_info.json"
            keypoint_desc = json.load(open(tool_info_path))["keypoint_descriptions"]
            keypoint_desc.update(
                json.load(open(asset_info_path))["keypoint_descriptions"]
            )
            keypoint_desc_replacement_str = g_utils.format_dict_prompt(keypoint_desc)
            plan_prompt_text = open(
                f"{self.prompt_folder}/prompt_task_config_stage1_visual.txt"
            ).read()
            plan_prompt_text = plan_prompt_text.replace(
                "TASK_NAME_TEMPLATE", task["task-name"]
            )
            plan_prompt_text = plan_prompt_text.replace(
                "TASK_DESCRIPTION_TEMPLATE", task["task-description"]
            )
            plan_prompt_text = plan_prompt_text.replace(
                "KEYPOINT_DESCRIPTION_TEMPLATE", keypoint_desc_replacement_str
            )
            plan_prompt_text = plan_prompt_text.replace("SOLVER_TRIALS", str(n))

            res = g_utils.generate_feedback_visual(
                plan_prompt_text, images, temperature=t, interaction_txt=self.chat_log
            )

            task_configs_stage1 = g_utils.extract_opt_config(
                res
            )  # keep the yaml form for planner
            print(task_configs_stage1)
            return task_configs_stage1
        except Exception as e:
            print(str(e))
            print("task config creation failure at constraint stage")
            return ""

    def propose_task_solution_stage2(self, n=1, t=0.0, task=None):
        # save self.chat_log to a file
        self.save_chatlog()
        if task is None:
            task = self.new_task
        try:
            """prompt to write optimization config."""
            g_utils.add_to_txt(
                self.chat_log,
                "================= Task Solution Stage 2 Design!",
                with_print=True,
            )

            plan_prompt_text = open(
                f"{self.prompt_folder}/prompt_task_config_stage2.txt"
            ).read()
            plan_prompt_text = plan_prompt_text.replace(
                "TASK_NAME_TEMPLATE", self.new_task["task-name"]
            )
            plan_prompt_text = plan_prompt_text.replace(
                "TASK_DESCRIPTION_TEMPLATE", self.new_task["task-description"]
            )
            print(plan_prompt_text)
            res = g_utils.generate_feedback(
                plan_prompt_text, temperature=0, interaction_txt=self.chat_log
            )
            task_config_stage2 = g_utils.extract_opt_config(
                res
            )  # keep the yaml form for planner
            print(task_config_stage2)
            return task_config_stage2
        except Exception as e:
            print(str(e))
            print("task config creation failure at pre/post-actuation constraint stage")
            return ""

    def propose_task_solution_stage2_visual(self, images, n=1, t=0.0, task=None):
        # save self.chat_log to a file
        self.save_chatlog()
        if task is None:
            task = self.new_task
        try:
            """prompt to write optimization config."""
            g_utils.add_to_txt(
                self.chat_log,
                "================= Task Solution Stage 2 Design Visual!",
                with_print=True,
            )

            plan_prompt_text = open(
                f"{self.prompt_folder}/prompt_task_config_stage2_visual.txt"
            ).read()
            plan_prompt_text = plan_prompt_text.replace(
                "TASK_NAME_TEMPLATE", task["task-name"]
            )
            plan_prompt_text = plan_prompt_text.replace(
                "TASK_DESCRIPTION_TEMPLATE", task["task-description"]
            )
            plan_prompt_text = plan_prompt_text.replace("SOLVER_TRIALS", str(n))

            res = g_utils.generate_feedback_visual(
                plan_prompt_text, images, temperature=t, interaction_txt=self.chat_log
            )

            task_configs_stage2 = g_utils.extract_opt_config(
                res
            )  # keep the yaml form for planner
            print(task_configs_stage2)

            return task_configs_stage2
        except Exception as e:
            print(str(e))
            print("task config creation failure at pre/post-actuation constraint stage")
            return ""

    def propose_task_solution_stage3(self, n=1, t=0.0, task=None):
        # save self.chat_log to a file
        self.save_chatlog()
        if task is None:
            task = self.new_task
        try:
            """prompt to write optimization config."""
            g_utils.add_to_txt(
                self.chat_log,
                "================= Task Solution Stage 3 Design Visual!",
                with_print=True,
            )

            plan_prompt_text = open(
                f"{self.prompt_folder}/prompt_task_config_stage3.txt"
            ).read()
            plan_prompt_text = plan_prompt_text.replace(
                "TASK_NAME_TEMPLATE", task["task-name"]
            )
            plan_prompt_text = plan_prompt_text.replace(
                "TASK_DESCRIPTION_TEMPLATE", task["task-description"]
            )
            plan_prompt_text = plan_prompt_text.replace("SOLVER_TRIALS", str(n))

            res = g_utils.generate_feedback(
                plan_prompt_text, temperature=t, interaction_txt=self.chat_log
            )

            task_configs_stage3 = g_utils.extract_opt_config(
                res
            )  # keep the yaml form for planner
            print(task_configs_stage3)

            return task_configs_stage3
        except Exception as e:
            print(str(e))
            print("task config creation failure at pre/post-actuation constraint stage")
            return ""

    def propose_task_solution_stage3_visual(self, images, n=1, t=0.0, task=None):
        # save self.chat_log to a file
        self.save_chatlog()
        if task is None:
            task = self.new_task
        try:
            """prompt to write optimization config."""
            g_utils.add_to_txt(
                self.chat_log,
                "================= Task Solution Stage 3 Design Visual!",
                with_print=True,
            )

            plan_prompt_text = open(
                f"{self.prompt_folder}/prompt_task_config_stage3_visual.txt"
            ).read()
            plan_prompt_text = plan_prompt_text.replace(
                "TASK_NAME_TEMPLATE", task["task-name"]
            )
            plan_prompt_text = plan_prompt_text.replace(
                "TASK_DESCRIPTION_TEMPLATE", task["task-description"]
            )
            plan_prompt_text = plan_prompt_text.replace("SOLVER_TRIALS", str(n))

            res = g_utils.generate_feedback_visual(
                plan_prompt_text, images, temperature=t, interaction_txt=self.chat_log
            )

            task_configs_stage3 = g_utils.extract_opt_config(
                res
            )  # keep the yaml form for planner
            print(task_configs_stage3)

            return task_configs_stage3
        except Exception as e:
            print(str(e))
            print("task config creation failure at pre/post-actuation constraint stage")
            return ""

    def regenerate_task_solution(self, reason, images, n=1, t=0.0, stage=1, task=None):
        if task is None:
            task = self.new_task
        try:
            g_utils.add_to_txt(
                self.chat_log,
                "================= Solution Regeneration!",
                with_print=True,
            )
            # asset_info_path = self.get_asset_info(task["assets-used"][0], task["assets_id"][0])
            asset_info_path = self.get_asset_info(task["assets-used"][0])
            tool_info_path = "assets/tools/hand/model1/model0_info.json"
            keypoint_desc = json.load(open(tool_info_path))["keypoint_descriptions"]
            keypoint_desc.update(
                json.load(open(asset_info_path))["keypoint_descriptions"]
            )
            keypoint_desc_replacement_str = g_utils.format_dict_prompt(keypoint_desc)
            regenerate_prompt_text = open(
                f"{self.prompt_folder}/prompt_task_config_stage{str(stage)}_regenerate.txt"
            ).read()
            regenerate_prompt_text = regenerate_prompt_text.replace(
                "REGENERATE_REASON_TEMPLATE", reason
            )
            regenerate_prompt_text = regenerate_prompt_text.replace(
                "SOLVER_TRIALS", str(n)
            )
            regenerate_prompt_text = regenerate_prompt_text.replace(
                "TASK_DESCRIPTION_TEMPLATE", task["task-description"]
            )
            regenerate_prompt_text = regenerate_prompt_text.replace(
                "KEYPOINT_DESCRIPTION_TEMPLATE", keypoint_desc_replacement_str
            )
            if self.cfg["visual_solver_generation"]:
                res = g_utils.generate_feedback_visual(
                    regenerate_prompt_text,
                    images,
                    temperature=t,
                    interaction_txt=self.chat_log,
                )
            else:
                res = g_utils.generate_feedback(
                    regenerate_prompt_text, temperature=t, interaction_txt=self.chat_log
                )

            task_configs = g_utils.extract_opt_config(
                res
            )  # keep the yaml form for planner
            print(task_configs)

            return task_configs
        except Exception as e:
            print(str(e))
            print("task config creation failure at constraint stage")
            return ""

    def get_asset_info(self, asset):
        if asset in ALL_ARTICULATED_OBJECTS:
            asset_root = ARTICULATED_OBJECTS_ROOT / asset
            asset_id = get_asset_id(asset)
            return asset_root / asset_id / "info.json"
        elif asset in ALL_RIGIDBODY_OBJECTS:
            asset_root = RIGIDBODY_OBJECTS_ROOT / asset
            return asset_root / "info.json"
