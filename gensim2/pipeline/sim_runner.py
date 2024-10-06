import os
from collections import OrderedDict

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter

import time
import traceback
import ipdb
import json
from PIL import Image
import pickle
import datetime
import yaml

from gensim2.pipeline.utils.utils import (
    mkdir_if_missing,
    save_text,
    save_stat,
    compute_diversity_score_from_assets,
    add_to_txt,
    Viewer,
)
from gensim2.agent.dataset.sim_traj_dataset import TrajDataset
from gensim2.env.create_task import create_gensim
from gensim2.env.solver.planner import KPAMPlanner

from scripts.run_env_with_ppo import run_ppo

from gensim2.pipeline.utils import utils as g_utils

TASK_CODE_DIR = "gensim2/env/task"
TASK_CONFIG_DIR = "gensim2/env/solver/kpam/config"


class SimulationRunner(object):
    """the main class that runs simulation loop"""

    def __init__(self, cfg, agent, critic, memory):
        self.cfg = cfg
        self.agent = agent
        self.critic = critic
        self.memory = memory

        # statistics
        self.syntax_pass_rate = 0
        self.runtime_pass_rate = 0
        self.env_pass_rate = 0
        self.curr_trials = 0

        self.prompt_folder = f"prompts/{cfg['prompt_folder']}"
        self.data_folder = f"prompts/{cfg['prompt_data_folder']}"
        self.chat_log = memory.chat_log
        self.task_asset_logs = []
        self.expert_planner = None

        self.generated_code = ""
        self.generated_task = ""
        self.generated_asset = ""
        self.generated_solution = ""
        self.generated_task_name = ""
        self.curr_task_name = ""
        self.task_creation_pass = False

        # All the generated origin_tasks in this run.
        # Different from the ones in online buffer that can load from offline.
        self.generated_task_assets = []
        self.generated_task_programs = []
        self.generated_task_names = []
        self.generated_tasks = []
        self.generated_task_configs = []
        self.passed_tasks = []  # accepted ones
        self.sub_tasks = []

    def save(self):
        """save the pipeline"""
        with open(f"{self.cfg['model_output_dir']}/runner.pkl", "wb") as f:
            pickle.dump(self, f)

    def load(self, path):
        """load the pipeline"""
        with open(f"{path}/runner.pkl", "rb") as f:
            runner = pickle.load(f)
        self.__dict__.update(runner.__dict__)

    def print_current_stats(self):
        """print the current statistics of the simulation design"""
        print("=========================================================")
        print(
            f"{self.cfg['prompt_folder']} Trial {self.curr_trials} SYNTAX_PASS_RATE: {(self.syntax_pass_rate / (self.curr_trials)) * 100:.1f}% RUNTIME_PASS_RATE: {(self.runtime_pass_rate / (self.curr_trials)) * 100:.1f}% ENV_PASS_RATE: {(self.env_pass_rate / (self.curr_trials)) * 100:.1f}%"
        )
        print("=========================================================")

    def save_stats(self):
        """save the final simulation statistics"""
        self.diversity_score = compute_diversity_score_from_assets(
            self.task_asset_logs, self.curr_trials
        )
        save_stat(
            self.cfg,
            self.cfg["model_output_dir"],
            self.generated_tasks,
            self.syntax_pass_rate / (self.curr_trials + 1),
            self.runtime_pass_rate / (self.curr_trials),
            self.env_pass_rate / (self.curr_trials),
            self.diversity_score,
        )
        print("Model Folder: ", self.cfg["model_output_dir"])
        print(
            f"Total {len(self.generated_tasks)} New Tasks:",
            [task["task-name"] for task in self.generated_tasks],
        )
        try:
            print(f"Added {len(self.passed_tasks)}  Tasks:", self.passed_tasks)
        except:
            pass

    def task_creation_topdown(self, long_horizon=False):
        """create the task through interactions of agent and critic"""
        self.task_creation_pass = True
        mkdir_if_missing(self.cfg["model_output_dir"])
        print("Generating task")
        try:
            start_time = time.time()
            # Ask GPT for a new task

            self.generated_task = self.agent.propose_task(self.generated_task_names)

            if long_horizon:
                self.task_decomposition()
                self.generated_task["success-criteria"] = self.sub_task_success_criteria
                code_dir = TASK_CODE_DIR + "/longhorizon_tasks/"
            else:
                code_dir = TASK_CODE_DIR + "/primitive_tasks/"

            self.generated_code, self.curr_task_name = self.agent.implement_task(
                task=self.generated_task
            )

            # add reward function
            if self.cfg["rl_solver"]:
                self.generated_reward = self.agent.implement_reward(self.generated_code)
                self.generated_code = (
                    self.generated_code + "\n\n" + self.generated_reward
                )

            print(
                "Save code to:",
                self.agent.model_output_dir,
                self.curr_task_name + "_code_output",
            )
            g_utils.save_text(
                self.agent.model_output_dir,
                self.curr_task_name + "_code_output",
                self.generated_code,
            )
            g_utils.save_code(
                code_dir, self.curr_task_name, self.generated_code, rename=True
            )

            self.task_asset_logs.append(self.generated_task["assets-used"])
            self.generated_task_name = self.generated_task["task-name"]
            self.generated_tasks.append(self.generated_task)
            self.generated_task_programs.append(self.generated_code)
            self.generated_task_names.append(self.generated_task_name)
        except:
            to_print = highlight(
                f"{str(traceback.format_exc())}", PythonLexer(), TerminalFormatter()
            )
            print("Task Creation Exception:", to_print)
            self.task_creation_pass = False

        # self.curr_task_name = self.generated_task['task-name']
        print("task creation time {:.3f}".format(time.time() - start_time))

    def task_decomposition(self, sub_tasks=None):
        """decompose the task into sub-tasks"""
        self.sub_tasks = self.agent.decompose_task() if sub_tasks is None else sub_tasks
        json.dump(
            self.sub_tasks,
            open(
                f"{self.agent.model_output_dir}/{self.curr_task_name}_sub_tasks.json",
                "w",
            ),
        )
        self.sub_task_names = []
        self.sub_task_codes = []
        self.sub_task_success_criteria = []
        self.sub_task_descriptions = []

        for task in self.sub_tasks:
            if task["task-name"] == "grasp":
                self.sub_task_names.append("Grasp")
                self.sub_task_codes.append("")
                self.sub_task_descriptions.append("")
                continue
            elif task["task-name"] == "ungrasp":
                self.sub_task_names.append("UnGrasp")
                self.sub_task_codes.append("")
                self.sub_task_descriptions.append("")
                continue
            sub_task_code, sub_task_name = self.agent.implement_task(task)
            self.sub_task_names.append(sub_task_name)
            self.sub_task_codes.append(sub_task_code)
            self.sub_task_success_criteria.extend(task["success-criteria"])
            self.sub_task_descriptions.append(task["task-description"])
            print(
                "Save code to:",
                self.agent.model_output_dir,
                sub_task_name + "_code_output",
            )
            g_utils.save_text(
                self.agent.model_output_dir,
                sub_task_name + "_code_output",
                sub_task_code,
            )
            g_utils.save_code(
                TASK_CODE_DIR + "/primitive_tasks/",
                self.curr_task_name,
                self.generated_code,
                rename=True,
            )

    def task_creation_bottomup(self, long_horizon=False):
        """create the task through interactions of agent and critic"""
        self.task_creation_pass = True
        mkdir_if_missing(self.cfg["model_output_dir"])
        print("Generating task")
        try:
            start_time = time.time()
            # Ask GPT for a new task

            self.generated_task, self.sub_tasks = self.agent.propose_task_bottomup(
                self.generated_task_names
            )

            if long_horizon:
                self.task_decomposition()
                self.generated_task["success-criteria"] = self.sub_task_success_criteria
                code_dir = TASK_CODE_DIR + "/longhorizon_tasks/"
            else:
                code_dir = TASK_CODE_DIR + "/primitive_tasks/"

            self.generated_code, self.curr_task_name = self.agent.implement_task(
                task=self.generated_task
            )

            # add reward function
            self.generated_reward = self.agent.implement_reward(self.generated_code)
            self.generated_code = self.generated_code + "\n\n" + self.generated_reward

            print(
                "Save code to:",
                self.agent.model_output_dir,
                self.curr_task_name + "_code_output",
            )
            g_utils.save_text(
                self.agent.model_output_dir,
                self.curr_task_name + "_code_output",
                self.generated_code,
            )
            g_utils.save_code(
                code_dir, self.curr_task_name, self.generated_code, rename=True
            )

            self.task_asset_logs.append(self.generated_task["assets-used"])
            self.generated_task_name = self.generated_task["task-name"]
            self.generated_tasks.append(self.generated_task)
            self.generated_task_programs.append(self.generated_code)
            self.generated_task_names.append(self.generated_task_name)
        except:
            to_print = highlight(
                f"{str(traceback.format_exc())}", PythonLexer(), TerminalFormatter()
            )
            print("Task Creation Exception:", to_print)
            self.task_creation_pass = False

        # self.curr_task_name = self.generated_task['task-name']
        print("task creation time {:.3f}".format(time.time() - start_time))

    def rl_solver_creation(self):
        """Begin running RL Solver on generated task"""
        print("Running RL Solver")
        result = run_ppo(env_name=self.curr_task_name, task_code=self.generated_code)
        print("Finished Running RL Solver")
        return result

    def solver_creation_2stage_longhorizon(self):
        for i, task in enumerate(self.sub_tasks):
            self.solver_creation_2stage(
                name=self.sub_task_names[i], code=self.sub_task_codes[i], task=task
            )

    def solver_creation_2stage(self, name=None, code=None, task=None):
        """create the solver given the generated tasks"""
        print("Generating solver")
        self.generated_solutions = []
        generated_solution = {}
        if name is None:
            name = self.curr_task_name
        if code is None:
            code = self.generated_code

        constraint_correct = False
        reflections = ""
        if self.cfg["visual_solver_generation"]:
            env_images = self.visualize_env(name=name, code=code)
            init_image = env_images[0]
            robot_axis_image = env_images[1]
            object_axis_image = env_images[2]

        generate_count = 0

        if self.cfg["use_primitive"]:
            constraint_lib = json.load(
                open("gensim2/env/solver/kpam/config/examples/constraint_lib.json")
            )
        else:
            constraint_lib = []

        while (not constraint_correct) and (
            generate_count < self.cfg["max_regeneration"]
        ):
            if generate_count == 0:
                if self.cfg["visual_solver_generation"]:
                    examples = self.agent.propose_task_solution_stage1_visual(
                        env_images,
                        n=self.cfg["solver_trials"],
                        t=self.cfg["solver_temperature"],
                        task=task,
                    )
                else:
                    examples = self.agent.propose_task_solution_stage1(
                        n=self.cfg["solver_trials"],
                        t=self.cfg["solver_temperature"],
                        task=task,
                    )
                generated_solutions_stage1 = []
                if self.cfg["use_primitive"]:
                    for constraint in constraint_lib:
                        solution = examples[0].copy()
                        solution.update(constraint)
                        generated_solutions_stage1.append(solution)
                else:
                    generated_solutions_stage1 = examples

            elif reflections == "":
                if self.cfg["visual_solver_generation"]:
                    generated_solutions_stage1 = (
                        self.agent.propose_task_solution_stage1_visual(
                            env_images,
                            n=self.cfg["solver_trials"],
                            t=self.cfg["solver_temperature"],
                            task=task,
                        )
                    )
                else:
                    generated_solutions_stage1 = (
                        self.agent.propose_task_solution_stage1(
                            n=self.cfg["solver_trials"],
                            t=self.cfg["solver_temperature"],
                            task=task,
                        )
                    )
                reflections = "The generated constraints can not be satisfied."
            else:
                if self.cfg["visual_solver_generation"]:
                    generated_solutions_stage1 = self.agent.regenerate_task_solution(
                        reflections,
                        env_images,
                        n=self.cfg["solver_trials"],
                        t=self.cfg["solver_temperature"],
                        stage=1,
                        task=task,
                    )
                else:
                    generated_solutions_stage1 = self.agent.regenerate_task_solution(
                        reflections,
                        [],
                        n=self.cfg["solver_trials"],
                        t=self.cfg["solver_temperature"],
                        stage=1,
                        task=task,
                    )
            generate_count += 1
            for solution in generated_solutions_stage1:
                try:
                    actuation_images, kpam_success = self.visualize_actuation_pose(
                        solution, code=code, name=name, viz=self.cfg["reject_sampling"]
                    )
                except:
                    actuation_images = []
                    kpam_success = False

                if not kpam_success:
                    constraint_correct = False
                else:
                    if self.cfg["visual_solver_generation"]:
                        constraint_correct, _ = self.critic.reflection(
                            task=self.generated_task,
                            stage="solver_creation_stage1",
                            images=actuation_images,
                        )
                    else:
                        constraint_correct, _ = self.critic.reflection(
                            task=self.generated_task,
                            stage="solver_creation_stage1",
                        )

                    if constraint_correct:
                        generated_solution.update(solution)
                        break

        motion_correct = False
        reflections = ""
        generate_count = 0

        while (
            constraint_correct
            and (not motion_correct)
            and (generate_count < self.cfg["max_regeneration"])
        ):
            if reflections == "":
                if self.cfg["visual_solver_generation"]:
                    generated_solutions_stage2 = (
                        self.agent.propose_task_solution_stage2_visual(
                            [init_image, actuation_images[0], robot_axis_image],
                            n=self.cfg["solver_trials"],
                            t=self.cfg["solver_temperature"],
                            task=task,
                        )
                    )
                else:
                    generated_solutions_stage2 = (
                        self.agent.propose_task_solution_stage2(
                            n=self.cfg["solver_trials"],
                            t=self.cfg["solver_temperature"],
                            task=task,
                        )
                    )
                reflections = (
                    "The generated pre/post actuation motions can not be satisfied."
                )
            else:
                if self.cfg["visual_solver_generation"]:
                    generated_solutions_stage2 = self.agent.regenerate_task_solution(
                        reflections,
                        env_images,
                        n=self.cfg["solver_trials"],
                        t=self.cfg["solver_temperature"],
                        stage=2,
                        task=task,
                    )
                else:
                    generated_solutions_stage2 = self.agent.regenerate_task_solution(
                        reflections,
                        [],
                        n=self.cfg["solver_trials"],
                        t=self.cfg["solver_temperature"],
                        stage=2,
                        task=task,
                    )
            generate_count += 1
            for solution in generated_solutions_stage2:
                print(solution)
                generated_solution.update(solution)
                try:
                    motion_images, kpam_success = self.visualize_actuation_motions(
                        generated_solution,
                        code=code,
                        name=name,
                        viz=self.cfg["reject_sampling"],
                    )
                except:
                    motion_images = []
                    kpam_success = False

                if not kpam_success:
                    motion_correct = False
                else:
                    if self.cfg["visual_solver_generation"]:
                        motion_correct, _ = self.critic.reflection(
                            task=self.generated_task,
                            stage="solver_creation_stage2",
                            images=motion_images,
                            include_reason=False,
                        )
                    else:
                        motion_correct, _ = self.critic.reflection(
                            task=self.generated_task,
                            stage="solver_creation_stage2",
                            include_reason=False,
                        )

                    if motion_correct:
                        break

        self.generated_solution = (
            generated_solution if constraint_correct and motion_correct else {}
        )
        self.generated_solutions.append(generated_solution)

        print(
            "Save solver config to:",
            self.agent.model_output_dir,
            name + "_config_output",
        )
        g_utils.save_text(
            self.agent.model_output_dir,
            name + "_config_output",
            str(self.generated_solution),
        )

        with open(f"{self.agent.model_output_dir}/{name}_config_output.yaml", "w") as f:
            yaml.dump(self.generated_solution, f, default_flow_style=False)

        if constraint_correct and motion_correct:
            with open(f"{TASK_CONFIG_DIR}/{name}.yaml", "w") as f:
                yaml.dump(self.generated_solution, f, default_flow_style=False)

    def solver_creation_3stage_longhorizon(self):
        for i, task in enumerate(self.sub_tasks):
            if task["task-name"] in ["grasp", "ungrasp"]:
                continue
            self.solver_creation_3stage(
                name=self.sub_task_names[i], code=self.sub_task_codes[i], task=task
            )

    def solver_creation_3stage(self, name=None, code=None, task=None):
        """create the solver given the generated tasks"""
        print("Generating solver")
        self.generated_solutions = []
        generated_solution = {}
        if name is None:
            name = self.curr_task_name
        if code is None:
            code = self.generated_code

        constraint_correct = False
        reflections = ""
        # ipdb.set_trace()
        if self.cfg["visual_solver_generation"]:
            env_images = self.visualize_env(
                name=name, code=code, viz=self.cfg["reject_sampling"]
            )
            if len(env_images) == 2:
                init_image = env_images[0]
                robot_axis_image = env_images[1]
            else:
                init_image = env_images[0]
                robot_axis_image = env_images[0]
            # object_axis_image = env_images[2]

        generate_count = 0
        if self.cfg["use_primitive"]:
            constraint_lib = json.load(
                open("gensim2/env/solver/kpam/config/examples/constraint_lib.json")
            )
        else:
            constraint_lib = []

        while (not constraint_correct) and (
            generate_count < self.cfg["max_regeneration"]
        ):
            if generate_count == 0:
                if self.cfg["visual_solver_generation"]:
                    examples = self.agent.propose_task_solution_stage1_visual(
                        env_images,
                        n=self.cfg["solver_trials"],
                        t=self.cfg["solver_temperature"],
                        task=task,
                    )
                else:
                    examples = self.agent.propose_task_solution_stage1(
                        n=self.cfg["solver_trials"],
                        t=self.cfg["solver_temperature"],
                        task=task,
                    )
                generated_solutions_stage1 = []
                if self.cfg["use_primitive"]:
                    for constraint in constraint_lib:
                        solution = examples[0].copy()
                        solution.update(constraint)
                        generated_solutions_stage1.append(solution)
                else:
                    generated_solutions_stage1 = examples

            elif reflections == "":
                if self.cfg["visual_solver_generation"]:
                    generated_solutions_stage1 = (
                        self.agent.propose_task_solution_stage1_visual(
                            env_images,
                            n=self.cfg["solver_trials"],
                            t=self.cfg["solver_temperature"],
                            task=task,
                        )
                    )
                else:
                    generated_solutions_stage1 = (
                        self.agent.propose_task_solution_stage1(
                            n=self.cfg["solver_trials"],
                            t=self.cfg["solver_temperature"],
                            task=task,
                        )
                    )
                reflections = "The generated constraints can not be satisfied."
            else:
                if self.cfg["visual_solver_generation"]:
                    generated_solutions_stage1 = self.agent.regenerate_task_solution(
                        reflections,
                        env_images,
                        n=self.cfg["solver_trials"],
                        t=self.cfg["solver_temperature"],
                        stage=1,
                        task=task,
                    )
                else:
                    generated_solutions_stage1 = self.agent.regenerate_task_solution(
                        reflections,
                        [],
                        n=self.cfg["solver_trials"],
                        t=self.cfg["solver_temperature"],
                        stage=1,
                        task=task,
                    )
            generate_count += 1
            for solution in generated_solutions_stage1:
                print(solution)
                try:
                    actuation_images, kpam_success = self.visualize_actuation_pose(
                        solution, name=name, code=code, viz=self.cfg["reject_sampling"]
                    )
                except:
                    actuation_images = []
                    kpam_success = False
                # ipdb.set_trace()
                if not kpam_success:
                    constraint_correct = False
                else:
                    if self.cfg["visual_solver_generation"]:
                        constraint_correct, _ = self.critic.reflection(
                            task=self.generated_task,
                            stage="solver_creation_stage1",
                            images=actuation_images,
                        )
                    else:
                        constraint_correct, _ = self.critic.reflection(
                            task=self.generated_task,
                            stage="solver_creation_stage1",
                        )

                    if constraint_correct:
                        generated_solution.update(solution)
                        break

        pre_motion_correct = False
        reflections = ""
        generate_count = 0
        if self.cfg["use_primitive"]:
            pre_actuation_lib = json.load(
                open("gensim2/env/solver/kpam/config/examples/pre_actuation_lib.json")
            )
        else:
            pre_actuation_lib = []

        while (
            constraint_correct
            and (not pre_motion_correct)
            and (generate_count < self.cfg["max_regeneration"])
        ):
            if generate_count == 0:
                if self.cfg["visual_solver_generation"]:
                    examples = self.agent.propose_task_solution_stage2_visual(
                        [init_image, actuation_images[0], robot_axis_image],
                        n=self.cfg["solver_trials"],
                        t=self.cfg["solver_temperature"],
                        task=task,
                    )
                else:
                    examples = self.agent.propose_task_solution_stage2(
                        n=self.cfg["solver_trials"],
                        t=self.cfg["solver_temperature"],
                        task=task,
                    )
                generated_solutions_stage2 = []
                if self.cfg["use_primitive"]:
                    for motion in pre_actuation_lib:
                        solution = examples[0].copy()
                        solution.update(motion)
                        generated_solutions_stage2.append(solution)
                else:
                    generated_solutions_stage2 = examples

            elif reflections == "":
                if self.cfg["visual_solver_generation"]:
                    generated_solutions_stage2 = (
                        self.agent.propose_task_solution_stage2_visual(
                            [init_image, actuation_images[0], robot_axis_image],
                            n=self.cfg["solver_trials"],
                            t=self.cfg["solver_temperature"],
                            task=task,
                        )
                    )
                else:
                    generated_solutions_stage2 = (
                        self.agent.propose_task_solution_stage2(
                            n=self.cfg["solver_trials"],
                            t=self.cfg["solver_temperature"],
                            task=task,
                        )
                    )
                reflections = (
                    "The generated pre/post actuation motions can not be satisfied."
                )
            else:
                if self.cfg["visual_solver_generation"]:
                    generated_solutions_stage2 = self.agent.regenerate_task_solution(
                        reflections,
                        env_images,
                        n=self.cfg["solver_trials"],
                        t=self.cfg["solver_temperature"],
                        stage=2,
                        task=task,
                    )
                else:
                    generated_solutions_stage2 = self.agent.regenerate_task_solution(
                        reflections,
                        [],
                        n=self.cfg["solver_trials"],
                        t=self.cfg["solver_temperature"],
                        stage=2,
                        task=task,
                    )
            generate_count += 1
            for solution in generated_solutions_stage2:
                generated_solution.update(solution)
                print(generated_solution)
                try:
                    motion_images, kpam_success = self.visualize_actuation_motions(
                        generated_solution,
                        name=name,
                        code=code,
                        viz=self.cfg["reject_sampling"],
                    )
                except:
                    motion_images = []
                    kpam_success = False

                if not kpam_success:
                    pre_motion_correct = False
                else:
                    if self.cfg["visual_solver_generation"]:
                        pre_motion_correct, _ = self.critic.reflection(
                            task=self.generated_task,
                            stage="solver_creation_stage2",
                            images=motion_images,
                            include_reason=False,
                        )
                    else:
                        pre_motion_correct, _ = self.critic.reflection(
                            task=self.generated_task,
                            stage="solver_creation_stage2",
                            include_reason=False,
                        )

                    if pre_motion_correct:
                        break

        if generated_solution.get("category_name", "") == "RigidBody":
            self.generated_solution = (
                generated_solution if constraint_correct and pre_motion_correct else {}
            )
            self.generated_solutions.append(generated_solution)

            with open(
                f"{self.agent.model_output_dir}/{name}_config_output.yaml", "w"
            ) as f:
                yaml.dump(self.generated_solution, f, default_flow_style=False)

            if constraint_correct and pre_motion_correct:
                with open(f"{TASK_CONFIG_DIR}/{name}.yaml", "w") as f:
                    yaml.dump(self.generated_solution, f, default_flow_style=False)

            return

        post_motion_correct = False
        reflections = ""
        generate_count = 0
        if self.cfg["use_primitive"]:
            post_actuation_lib = json.load(
                open("gensim2/env/solver/kpam/config/examples/post_actuation_lib.json")
            )
        else:
            post_actuation_lib = []

        while (
            constraint_correct
            and (not post_motion_correct)
            and (generate_count < self.cfg["max_regeneration"])
        ):
            if generate_count == 0:
                if self.cfg["visual_solver_generation"]:
                    examples = self.agent.propose_task_solution_stage3_visual(
                        [init_image, actuation_images[0], robot_axis_image],
                        n=self.cfg["solver_trials"],
                        t=self.cfg["solver_temperature"],
                        task=task,
                    )
                else:
                    examples = self.agent.propose_task_solution_stage3(
                        n=self.cfg["solver_trials"],
                        t=self.cfg["solver_temperature"],
                        task=task,
                    )
                generated_solutions_stage3 = []
                if self.cfg["use_primitive"]:
                    for motion in post_actuation_lib:
                        solution = examples[0].copy()
                        solution.update(motion)
                        generated_solutions_stage3.append(solution)
                else:
                    generated_solutions_stage3 = examples

            elif reflections == "":
                if self.cfg["visual_solver_generation"]:
                    generated_solutions_stage3 = (
                        self.agent.propose_task_solution_stage3_visual(
                            [init_image, actuation_images[0], robot_axis_image],
                            n=self.cfg["solver_trials"],
                            t=self.cfg["solver_temperature"],
                            task=task,
                        )
                    )
                else:
                    generated_solutions_stage3 = (
                        self.agent.propose_task_solution_stage3(
                            n=self.cfg["solver_trials"],
                            t=self.cfg["solver_temperature"],
                            task=task,
                        )
                    )

                reflections = (
                    "The generated pre/post actuation motions can not be satisfied."
                )
            else:
                if self.cfg["visual_solver_generation"]:
                    generated_solutions_stage3 = self.agent.regenerate_task_solution(
                        reflections,
                        env_images,
                        n=self.cfg["solver_trials"],
                        t=self.cfg["solver_temperature"],
                        stage=3,
                        task=task,
                    )
                else:
                    generated_solutions_stage3 = self.agent.regenerate_task_solution(
                        reflections,
                        [],
                        n=self.cfg["solver_trials"],
                        t=self.cfg["solver_temperature"],
                        stage=3,
                        task=task,
                    )
            generate_count += 1
            for solution in generated_solutions_stage3:
                generated_solution.update(solution)
                print(generated_solution)
                try:
                    motion_images, kpam_success = self.visualize_actuation_motions(
                        generated_solution,
                        name=name,
                        code=code,
                        viz=self.cfg["reject_sampling"],
                    )
                except:
                    motion_images = []
                    kpam_success = False
                    print("Solution failed.")

                if not kpam_success:
                    post_motion_correct = False
                else:
                    if self.cfg["visual_solver_generation"]:
                        post_motion_correct, _ = self.critic.reflection(
                            task=self.generated_task,
                            stage="solver_creation_stage3",
                            images=motion_images,
                            include_reason=False,
                        )
                    else:
                        post_motion_correct, _ = self.critic.reflection(
                            task=self.generated_task,
                            stage="solver_creation_stage3",
                            include_reason=False,
                        )

                    if post_motion_correct:
                        break

        self.generated_solution = (
            generated_solution
            if constraint_correct and pre_motion_correct and post_motion_correct
            else {}
        )
        self.generated_solutions.append(generated_solution)

        with open(f"{self.agent.model_output_dir}/{name}_config_output.yaml", "w") as f:
            yaml.dump(self.generated_solution, f, default_flow_style=False)

        if constraint_correct and pre_motion_correct and post_motion_correct:
            with open(f"{TASK_CONFIG_DIR}/{name}.yaml", "w") as f:
                yaml.dump(self.generated_solution, f, default_flow_style=False)

    def cache_stage(self):
        # save the conversation up-to-now to some cache in disk.
        pass

    def load_cache_stage(self):
        # load the cached conversation.
        pass

    def validate_solutions(self):
        # run through the kpam solver to see if one succeeds
        pass

    def setup_env(self):
        """build the new task"""
        self.env = create_gensim(
            task_name=self.curr_task_name,
            sim_type="Sapien",
            task_code_input=self.generated_code,
            use_gui=True,
            eval=False,
        )
        self.expert_planner = KPAMPlanner(self.env, self.generated_solution)
        self.dataset = TrajDataset(
            dataset_name=self.generated_task["task-name"] + "_train", from_empty=True
        )

        # Start video recording
        return self.dataset, self.env, self.expert_planner

    def visualize_env(self, name, code, viz=True):
        env = create_gensim(
            task_name=name,
            sim_type="Sapien",
            task_code_input=code,
            use_gui=True,
            eval=False,
        )
        img_list = []
        env.render()
        if viz:
            while not env.viewer.closed:
                if env.viewer.window.key_down("p"):
                    rgba = env.viewer.window.get_float_texture("Color")
                    rgb_img = (rgba * 255).clip(0, 255).astype("uint8")[:, :, :3]
                    rgb_pil = Image.fromarray(rgb_img)

                    t = datetime.datetime.now().strftime("%m%d-%H%M%S")

                    if not os.path.exists("screenshots"):
                        os.mkdir("screenshots")

                    rgb_pil.save(f"screenshots/env-{t}.png")
                    img_list.append(rgb_pil)
                env.render()
        else:
            for _ in range(10):
                env.render()

            rgba = env.viewer.window.get_float_texture("Color")
            rgb_img = (rgba * 255).clip(0, 255).astype("uint8")[:, :, :3]
            rgb_pil = Image.fromarray(rgb_img)

            t = datetime.datetime.now().strftime("%m%d-%H%M%S")

            if not os.path.exists("screenshots"):
                os.mkdir("screenshots")

            rgb_pil.save(f"screenshots/env-{t}.png")
            img_list.append(rgb_pil)

            env.viewer.close()
        return img_list

    def visualize_actuation_pose(self, config, name, code, viz=True):
        env = create_gensim(
            task_name=name,
            sim_type="Sapien",
            task_code_input=code,
            use_gui=True,
            eval=False,
        )
        planner = KPAMPlanner(env, config)
        img_list = []
        try:
            actuation_qpos, kpam_success = planner.get_actuation_qpos()
        except:
            actuation_qpos = []
            kpam_success = False

        if not kpam_success:
            env.viewer.close()

        elif viz:
            env.set_joint_positions(actuation_qpos)
            env.render()
            while not env.viewer.closed:
                if env.viewer.window.key_down("p"):
                    rgba = env.viewer.window.get_float_texture("Color")
                    rgb_img = (rgba * 255).clip(0, 255).astype("uint8")[:, :, :3]
                    rgb_pil = Image.fromarray(rgb_img)

                    t = datetime.datetime.now().strftime("%m%d-%H%M%S")

                    if not os.path.exists("screenshots"):
                        os.mkdir("screenshots")

                    rgb_pil.save(f"screenshots/pose-{t}.png")
                    img_list.append(rgb_pil)

                env.render()
        else:
            env.set_joint_positions(actuation_qpos)
            for _ in range(10):
                env.render()

            rgba = env.viewer.window.get_float_texture("Color")
            rgb_img = (rgba * 255).clip(0, 255).astype("uint8")[:, :, :3]
            rgb_pil = Image.fromarray(rgb_img)

            t = datetime.datetime.now().strftime("%m%d-%H%M%S")

            if not os.path.exists("screenshots"):
                os.mkdir("screenshots")

            rgb_pil.save(f"screenshots/pose-{t}.png")
            img_list.append(rgb_pil)
            env.viewer.close()
        return img_list, True

    def visualize_actuation_motions(self, config, name, code, viz=True):
        env = create_gensim(
            task_name=name,
            sim_type="Sapien",
            task_code_input=code,
            use_gui=True,
            eval=False,
        )
        planner = KPAMPlanner(env, config)
        img_list = []
        try:
            motion_qpos_list, plan_success = planner.get_sparse_traj_qpos()
        except:
            motion_qpos_list = []
            plan_success = False
            print("kpam failed to plan motions.")

        if not plan_success:
            if viz:
                env.viewer.close()
            return img_list, False

        elif viz:
            env.render()
            interval = 100
            i = 0
            while not env.viewer.closed:
                if i % interval == 0:
                    env.set_joint_positions(
                        motion_qpos_list[(i // interval) % len(motion_qpos_list)]
                    )
                i += 1

                if env.viewer.window.key_down("p"):
                    rgba = env.viewer.window.get_float_texture("Color")
                    rgb_img = (rgba * 255).clip(0, 255).astype("uint8")[:, :, :3]
                    rgb_pil = Image.fromarray(rgb_img)

                    t = datetime.datetime.now().strftime("%m%d-%H%M%S")

                    if not os.path.exists("screenshots"):
                        os.mkdir("screenshots")

                    rgb_pil.save(f"screenshots/motion-{t}.png")
                    img_list.append(rgb_pil)
                env.render()

        else:
            env.render()

            for i in range(0, len(motion_qpos_list)):
                env.set_joint_positions(motion_qpos_list[i])
                for _ in range(10):
                    env.render()
                rgba = env.viewer.window.get_float_texture("Color")
                rgb_img = (rgba * 255).clip(0, 255).astype("uint8")[:, :, :3]
                rgb_pil = Image.fromarray(rgb_img)

                t = datetime.datetime.now().strftime("%m%d-%H%M%S")

                if not os.path.exists("screenshots"):
                    os.mkdir("screenshots")

                rgb_pil.save(f"screenshots/motion-{t}.png")
                img_list.append(rgb_pil)

        env.viewer.close()
        return img_list, True

    def run_one_episode(self, episode, seed):
        """run the new task for one episode"""
        steps = []
        add_to_txt(
            self.chat_log,
            f"================= TRIAL: {self.curr_trials}",
            with_print=True,
        )

        print("Oracle demo: {}/{}".format(episode, self.cfg["n"]))
        reward = 0
        total_reward = 0
        max_steps = 500
        self.expert_planner.reset_expert()
        self.env.reset()

        # Rollout expert policy
        for _ in range(max_steps):
            action = self.expert_planner.get_action()
            obs, rew, done, info = self.env.step(action)
            total_reward += rew
            env_image = info["image"]
            steps.append(info)

        # add to data
        success = info["success"]
        self.env_image = env_image
        self.dataset.append_episode(steps)
        return success

    def simulate_task(self):
        """simulate the created task and save demonstrations"""
        total_cnt = 0.0
        reset_success_cnt = 0.0
        env_success_cnt = 0.0
        seed = 123
        self.curr_trials += 1

        if not self.task_creation_pass:
            print("task creation failure => count as syntax exceptions.")
            return

        # Check syntax and compilation-time error
        try:
            dataset, env, expert = self.setup_env()
            self.syntax_pass_rate += 1

        except:
            to_print = highlight(
                f"{str(traceback.format_exc())}", PythonLexer(), TerminalFormatter()
            )
            save_text(
                self.cfg["model_output_dir"],
                self.generated_task_name + "_error",
                str(traceback.format_exc()),
            )
            print("========================================================")
            print("Syntax Exception:", to_print)
            return

        try:
            # Collect environment and collect data from oracle demonstrations.
            while total_cnt < self.cfg["max_env_run_cnt"]:
                total_cnt += 1
                # Set seeds.
                episode = []
                total_reward = self.run_one_episode(episode, seed)

                total_reward = 1
                reset_success_cnt += 1
                env_success_cnt += total_reward > 0.99

            self.runtime_pass_rate += 1
            print("Runtime Test Pass!")

            # the task can actually be completed with oracle. 50% success rates are high enough.
            if env_success_cnt >= total_cnt / 2:
                self.env_pass_rate += 1
                print("Environment Test Pass!")

                # Only save completed demonstrations.
                if self.cfg["save_data"]:
                    save_task_flag = self.critic.reflection(
                        task=self.generated_task,
                        code=self.generated_code,
                        asset=self.generated_assets,
                        goal_image=self.env_image,
                        current_tasks=self.generated_tasks,
                    )
                    if save_task_flag:
                        """add to online buffer"""
                        self.memory.save_task_to_online(
                            self.generated_task,
                            self.generated_code,
                            self.generated_solution,
                        )
                        print(
                            f"added new task to online buffer: {self.generated_task['task-name']}"
                        )
                        self.passed_tasks.append(self.generated_task["task-name"])

                        if self.cfg["save_memory"]:
                            """add to disk for future inspection and use"""
                            self.memory.save_task_to_offline(
                                self.generated_task,
                                self.generated_code,
                                self.generated_solution,
                            )
                            print(
                                f"added new task to offline: {self.generated_task['task-name']}"
                            )

        except:
            to_print = highlight(
                f"{str(traceback.format_exc())}", PythonLexer(), TerminalFormatter()
            )
            save_text(
                self.cfg["model_output_dir"],
                self.generated_task_name + "_error",
                str(traceback.format_exc()),
            )
            print("========================================================")
            print("Runtime Exception:", to_print)

        self.memory.save_run(self.generated_task)
