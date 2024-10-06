import os

from gensim2.env.sapien.env import SapienEnv


def create_sapien(env_params):
    env = SapienEnv(**env_params)

    return env
