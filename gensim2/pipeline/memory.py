import numpy as np
import os
from collections import OrderedDict
import random
import json

from gensim2.pipeline.utils.utils import save_text


class Memory:
    """
    class that maintains a buffer of generated origin_tasks and codes
    """

    def __init__(self, cfg):
        self.prompt_folder = f"prompts/{cfg['prompt_folder']}"
        self.data_folder = f"prompts/{cfg['prompt_data_folder']}"
        self.cfg = cfg

        # a chat history is a list of strings
        self.chat_log = []
        self.online_task_buffer = OrderedDict()
        self.online_code_buffer = OrderedDict()
        self.online_asset_buffer = OrderedDict()

        # directly load current offline memory into online memory
        base_tasks, base_assets, base_task_codes = self.load_offline_memory()
        self.online_task_buffer.update(base_tasks)
        self.online_asset_buffer.update(base_assets)

        # load each code file
        for task_file in base_task_codes:
            # the original task path
            if os.path.exists("gensim2/env/task/origin_tasks/" + task_file):
                self.online_code_buffer[task_file] = open(
                    "gensim2/env/task/origin_tasks/" + task_file
                ).read()

            # the generated task path
            elif os.path.exists("gensim2/env/task/generated_tasks/" + task_file):
                self.online_code_buffer[task_file] = open(
                    "gensim2/env/task/generated_tasks/" + task_file
                ).read()

        print(
            f"load {len(self.online_code_buffer)} origin_tasks for memory from offline to online:"
        )

    def save_run(self, new_task):
        """save chat history and potentially save base memory"""
        print("save all interaction to :", f'{new_task["task-name"]}_full_output')
        unroll_chatlog = ""
        for chat in self.chat_log:
            unroll_chatlog += chat
        save_text(
            self.cfg["model_output_dir"],
            f'{new_task["task-name"]}_full_output',
            unroll_chatlog,
        )

    def save_task_to_online(self, task, code, solution):
        """(not dumping the task offline). save the task information for online bootstrapping."""
        self.online_task_buffer[task["task-name"]] = task
        code_file_name = task["task-name"].replace("-", "_") + ".py"

        # code file name: actual code in contrast to offline code files format.
        self.online_code_buffer[code_file_name] = code

    def save_task_to_offline(self, task, code, solution):
        """save the current task descriptions, assets, and code, if it passes reflection and environment test"""
        generated_task_code_path = os.path.join(
            self.data_folder, "generated_task_codes.json"
        )
        generated_task_codes = json.load(open(generated_task_code_path))
        new_file_path = task["task-name"].replace("-", "_") + ".py"

        if new_file_path not in generated_task_codes:
            generated_task_codes.append(new_file_path)

            python_file_path = "gensim2/env/task/generated_tasks/" + new_file_path
            print(f"save {task['task-name']} to ", python_file_path)

            with open(python_file_path, "w") as fhandle:
                fhandle.write(code)

            with open(generated_task_code_path, "w") as outfile:
                json.dump(generated_task_codes, outfile, indent=4)
        else:
            print(f"{new_file_path}.py already exists.")

        # save task descriptions
        generated_task_path = os.path.join(self.data_folder, "generated_tasks.json")
        generated_tasks = json.load(open(generated_task_path))
        generated_tasks[task["task-name"]] = task

        # save task solution
        config_file_path = (
            "gensim2/env/solver/generated_solutions/"
            + new_file_path.replace("py", "yaml")
        )
        with open(config_file_path, "w") as fhandle:
            fhandle.write(solution)

    def load_offline_memory(self):
        """get the current task descriptions, assets, and code"""
        base_task_path = os.path.join(self.data_folder, "base_tasks.json")
        base_asset_path = os.path.join(self.data_folder, "base_assets.json")
        base_task_code_path = os.path.join(self.data_folder, "base_task_codes.json")

        base_tasks = json.load(open(base_task_path))
        base_assets = json.load(open(base_asset_path))
        base_task_codes = json.load(open(base_task_code_path))

        if self.cfg["load_memory"]:
            generated_task_path = os.path.join(self.data_folder, "generated_tasks.json")
            generated_task_code_path = os.path.join(
                self.data_folder, "generated_task_codes.json"
            )

            print("original base task num:", len(base_tasks))
            base_tasks.update(json.load(open(generated_task_path)))

            for task in json.load(open(generated_task_code_path)):
                if task not in base_task_codes:
                    base_task_codes.append(task)

            print("current base task num:", len(base_tasks))
        return base_tasks, base_assets, base_task_codes
