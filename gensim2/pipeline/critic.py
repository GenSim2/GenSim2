import numpy as np
import os
import IPython

import traceback
import json
from gensim2.pipeline.utils.utils import (
    save_text,
    add_to_txt,
    extract_dict,
    format_dict_prompt,
    generate_feedback,
)
import copy
import random

import gensim2.pipeline.utils.utils as g_utils


class Critic:
    """
    class that reflects and criticizes new task for improvement
    """

    def __init__(self, cfg, memory):
        self.prompt_folder = f"prompts/{cfg['prompt_folder']}"
        self.memory = memory
        self.chat_log = self.memory.chat_log
        self.cfg = cfg
        self.model_output_dir = cfg["model_output_dir"]

    def reflection(
        self,
        task,
        code=None,
        stage="task_creation",
        images=None,
        current_tasks=None,
        include_reason=False,
    ):
        """reflect on all generated tasks and assets."""
        pass_reflection = True

        if self.cfg["language_reflection"]:
            pass_reflection = pass_reflection & self.language_task_reflection(
                task, code, stage, current_tasks=None
            )

        if self.cfg["visual_reflection"]:
            pass_flag, reason = self.visual_task_reflection(
                task, code, images, stage, current_tasks=None
            )
            pass_reflection = pass_reflection & pass_flag

        if self.cfg["reject_sampling"]:
            pass_flag, reason = self.human_task_reflection(
                task, code, stage, current_tasks=None, include_reason=include_reason
            )
            pass_reflection = pass_reflection & pass_flag

        return pass_reflection, reason

    def visual_asset_reflect(self, urdf, image):
        """Given the urdf code and renderd the image image of the asset, determine if it passes the tests and can be saved."""
        pass

    def visual_task_reflection(
        self, new_task, new_code, images, stage, current_tasks=None
    ):
        """Given the goal image and task descriptions and reflect on if the new task should be added"""

        if stage == "task_creation":
            pass

        elif stage == "solver_creation_stage1":
            if os.path.exists(
                f"{self.prompt_folder}/prompt_task_config_stage1_reflection_visual.txt"
            ):
                self.chat_log = add_to_txt(
                    self.chat_log,
                    "================= Visual Reflect for Solver Config!",
                    with_print=True,
                )
                visual_reflection_prompt_text = open(
                    f"{self.prompt_folder}/prompt_task_config_stage1_reflection_visual.txt"
                ).read()
                visual_reflection_prompt_text = visual_reflection_prompt_text.replace(
                    "TASK_STRING_TEMPLATE", new_task["task-name"]
                )
                visual_reflection_prompt_text = visual_reflection_prompt_text.replace(
                    "TASK_DESCRIPTION_TEMPLATE", new_task["task-description"]
                )

                # replace template with new task and new code
                res = g_utils.generate_feedback_visual(
                    visual_reflection_prompt_text,
                    images,
                    temperature=0.4,
                    interaction_txt=self.chat_log,
                    n=self.cfg["reflection_agreement_num"],
                )

                pass_flag = True
                reason = ""
                for idx, r in enumerate(res):
                    # iterate through for agreement
                    print(r)
                    reflection_def_cmd = extract_dict(r, prefix="task_reflection")
                    exec(reflection_def_cmd, globals())
                    print(f"critic {idx}:", task_reflection)
                    if task_reflection["pass"] == "False":
                        pass_flag = False
                        reason = task_reflection["reason"]
                        print(f"critic {idx} suggests regenerating the constraints! ")
                        break

                return pass_flag, reason
        elif stage == "solver_creation_stage2":
            if os.path.exists(
                f"{self.prompt_folder}/prompt_task_config_stage2_reflection_visual.txt"
            ):
                self.chat_log = add_to_txt(
                    self.chat_log,
                    "================= Visual Reflect for Solver Config!",
                    with_print=True,
                )
                visual_reflection_prompt_text = open(
                    f"{self.prompt_folder}/prompt_task_config_stage2_reflection_visual.txt"
                ).read()
                visual_reflection_prompt_text = visual_reflection_prompt_text.replace(
                    "TASK_STRING_TEMPLATE", new_task["task-name"]
                )
                visual_reflection_prompt_text = visual_reflection_prompt_text.replace(
                    "TASK_DESCRIPTION_TEMPLATE", new_task["task-description"]
                )

                # replace template with new task and new code
                res = g_utils.generate_feedback_visual(
                    visual_reflection_prompt_text,
                    images,
                    temperature=0.4,
                    interaction_txt=self.chat_log,
                    n=self.cfg["reflection_agreement_num"],
                )

                pass_flag = True
                reason = ""
                for idx, r in enumerate(res):
                    # iterate through for agreement
                    reflection_def_cmd = extract_dict(r, prefix="task_reflection")
                    exec(reflection_def_cmd, globals())
                    print(f"critic {idx}:", task_reflection)
                    if task_reflection["pass"] == "False":
                        pass_flag = False
                        reason = task_reflection["reason"]
                        print(f"critic {idx} suggests regenerating the constraints! ")
                        break

                return pass_flag, reason

        elif stage == "solver_creation_stage3":
            if os.path.exists(
                f"{self.prompt_folder}/prompt_task_config_stage3_reflection_visual.txt"
            ):
                self.chat_log = add_to_txt(
                    self.chat_log,
                    "================= Visual Reflect for Solver Config!",
                    with_print=True,
                )
                visual_reflection_prompt_text = open(
                    f"{self.prompt_folder}/prompt_task_config_stage3_reflection_visual.txt"
                ).read()
                visual_reflection_prompt_text = visual_reflection_prompt_text.replace(
                    "TASK_STRING_TEMPLATE", new_task["task-name"]
                )
                visual_reflection_prompt_text = visual_reflection_prompt_text.replace(
                    "TASK_DESCRIPTION_TEMPLATE", new_task["task-description"]
                )

                # replace template with new task and new code
                res = g_utils.generate_feedback_visual(
                    visual_reflection_prompt_text,
                    images,
                    temperature=0.4,
                    interaction_txt=self.chat_log,
                    n=self.cfg["reflection_agreement_num"],
                )

                pass_flag = True
                reason = ""
                for idx, r in enumerate(res):
                    # iterate through for agreement
                    reflection_def_cmd = extract_dict(r, prefix="task_reflection")
                    exec(reflection_def_cmd, globals())
                    print(f"critic {idx}:", task_reflection)
                    if task_reflection["pass"] == "False":
                        pass_flag = False
                        reason = task_reflection["reason"]
                        print(f"critic {idx} suggests regenerating the constraints! ")
                        break

                return pass_flag, reason

    def language_task_reflection(self, new_task, new_code, current_tasks=None):
        """Given the goal image, reflect on if the new task should be added"""
        all_add_to_the_task_list_flag = True

        if os.path.exists(f"{self.prompt_folder}/prompt_task_reflection.txt"):
            # only consider successful task
            self.chat_log = add_to_txt(
                self.chat_log, "================= Code Reflect!", with_print=True
            )
            total_tasks = copy.deepcopy(self.memory.online_task_buffer)
            if current_tasks is not None:
                # adding all the origin_tasks in the current run. at least should not overlap with those
                for t in current_tasks:
                    total_tasks[t["task-name"]] = t

            # need to load more
            total_tasks = self.memory.online_task_buffer
            MAX_NUM = 40
            if len(total_tasks) > MAX_NUM:
                total_tasks = dict(random.sample(total_tasks.items(), MAX_NUM))

            print("reflection history task num:", len(total_tasks))
            task_descriptions_replacement_str = format_dict_prompt(total_tasks, -1)

            # append current new task
            code_reflection_prompt_text = open(
                f"{self.prompt_folder}/prompt_task_reflection.txt"
            ).read()
            code_reflection_prompt_text = code_reflection_prompt_text.replace(
                "CURRENT_TASK_NAME_TEMPLATE", str(task_descriptions_replacement_str)
            )
            code_reflection_prompt_text = code_reflection_prompt_text.replace(
                "TASK_STRING_TEMPLATE", str(new_task)
            )
            code_reflection_prompt_text = code_reflection_prompt_text.replace(
                "TASK_CODE_TEMPLATE", str(new_code)
            )
            if len(self.cfg["target_task_name"]) > 0:
                code_reflection_prompt_text = code_reflection_prompt_text.replace(
                    "TARGET_TASK_NAME", self.cfg["target_task_name"]
                )

            # no matter
            total_tasks[new_task["task-name"].replace("-", "_")] = str(new_task)
            res = generate_feedback(
                code_reflection_prompt_text,
                temperature=0.4,
                interaction_txt=self.chat_log,
                n=int(self.cfg["reflection_agreement_num"]),
            )

            all_add_to_the_task_list_flag = True
            for idx, r in enumerate(res):
                # iterate through for agreement
                reflection_def_cmd = extract_dict(r, prefix="task_reflection")
                exec(reflection_def_cmd, globals())
                print(f"critic {idx}:", task_reflection)
                if task_reflection["add_to_the_task_list"] == "False":
                    all_add_to_the_task_list_flag = False
                    print(f"critic {idx} suggests not adding this task to the buffer! ")

            save_text(
                self.model_output_dir,
                new_task["task-name"] + "_reflection_output",
                str(task_reflection),
            )

        return all_add_to_the_task_list_flag

    def human_task_reflection(
        self, new_task, new_code, stage, current_tasks=None, include_reason=True
    ):

        if stage == "solver_creation_stage1":
            pass_flag = input("Does the actuation pose look good? (y/n)")
            if pass_flag == "y":
                return True, ""
            elif pass_flag == "n" and include_reason:
                reason = input("Why? ")
                return False, reason
            else:
                return False, ""

        else:
            pass_flag = input("Do the pre/post actuation motions look good? (y/n)")
            if pass_flag == "y":
                return True, ""
            elif pass_flag == "n" and include_reason:
                reason = input("Why? ")
                return False, reason
            else:
                return False, ""
