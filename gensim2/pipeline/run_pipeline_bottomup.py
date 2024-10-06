import numpy as np
import os
import hydra
import random

import re
import openai
import pathlib
import textwrap
import IPython
import time
import traceback
from datetime import datetime
from pprint import pprint
import cv2
import re
import random
import json
import ipdb


from gensim2.pipeline.agent import Agent
from gensim2.pipeline.critic import Critic
from gensim2.pipeline.sim_runner import SimulationRunner
from gensim2.pipeline.memory import Memory
from gensim2.pipeline.utils.utils import set_llm_model, clear_messages


@hydra.main(config_path="experiments/config", config_name="config", version_base="1.2")
def main(cfg):
    openai.api_key = cfg["openai_key"]
    model_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    cfg["model_output_dir"] = os.path.join(
        cfg["output_folder"], cfg["prompt_folder"] + "_" + model_time
    )
    if "seed" in cfg:
        cfg["model_output_dir"] = cfg["model_output_dir"] + f"_{cfg['seed']}"

    print(f"model_output_dir = {cfg['model_output_dir']}")

    set_llm_model(model_name=cfg["gpt_model"])

    memory = Memory(cfg)
    agent = Agent(cfg, memory)
    critic = Critic(cfg, memory)
    simulation_runner = SimulationRunner(cfg, agent, critic, memory)

    if cfg["load_pipeline"]:
        simulation_runner.load(cfg["load_pipeline_path"])

    for trial_i in range(cfg["num_tasks"]):
        if cfg["create_task"]:
            simulation_runner.task_creation()
        if cfg["create_solver"]:
            simulation_runner.solver_creation_3_stage()
        if cfg["save_pipeline"]:
            simulation_runner.save()
        if cfg["simulation_task"]:
            simulation_runner.simulate_task()
        # simulation_runner.print_current_stats()
        # clear_messages()

    # simulation_runner.save_stats()


if __name__ == "__main__":
    main()
