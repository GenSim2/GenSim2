import os
from gensim2.env.wrappers import GenSimTaskWrapper
from gensim2.env.task import *
from gensim2.env.sapien import create_sapien


def create_gensim(task_name, task_code_input=None, **kwargs):
    env_params = dict(**kwargs)

    env = create_sapien(env_params)

    # Specify rendering device if the computing device is given
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"

    # evaluate the task definition
    if task_code_input is not None:
        exec(task_code_input, globals())

    task = eval(task_name)(env, asset_id=env_params.get("asset_id", ""))
    return GenSimTaskWrapper(task)


if __name__ == "__main__":
    create_gensim("TurnOnFaucet", "Sapien")
    # create_gensim("TurnOnFaucet", "IssacGym")
