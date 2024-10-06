import unittest
from unittest import TestCase
import icecream

import numpy as np
import argparse

from gensim2.env.task.origin_tasks import *
from gensim2.env.task.primitive_tasks import *
from gensim2.env.solver.planner import KPAMPlanner
from gensim2.env.create_task import create_gensim

TEST_ENV_NAMES = ["HammerHit", "TurnOnFaucet"]


class TryTesting(TestCase):
    def test_always_passes(self):
        self.assertTrue(True)


class TestingEnv(TestCase):
    def test_create_env(self):
        for env_name in TEST_ENV_NAMES:
            env = create_gensim(
                task_name=env_name, sim_type="Sapien", use_gui=False, eval=False
            )

    def test_run_kpam_env(self):
        for env_name in TEST_ENV_NAMES:
            env = create_gensim(
                task_name=env_name, sim_type="Sapien", use_gui=False, eval=False
            )
            icecream.ic(env.horizon)
            config_path = f"gensim2/env/solver/kpam/config/{env_name}.yaml"
            load_file = open(config_path, mode="r")
            expert_planner = KPAMPlanner(env, config_path)

            max_steps = 500
            for eps in range(2):
                print(f"episode {eps}")
                env.reset()
                for i in range(max_steps):
                    action = expert_planner.get_action()
                    obs, reward, done, info = env.step(action)
                    if done:
                        break
