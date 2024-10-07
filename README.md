# 🦾 GenSim2: Scaling Robot Data Generation with Multi-modal and Reasoning LLMs
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)
[![Paper](https://badgen.net/badge/icon/arXiv?icon=awesome&label&color=red&style=flat-square)]()
[![Website](https://img.shields.io/badge/Website-gensim2-blue?style=flat-square)](https://gensim2.github.io)
[![Python](https://img.shields.io/badge/Python-%3E=3.8-blue?style=flat-square)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E=2.0-orange?style=flat-square)]()

![](doc/gensim2.gif)

[Pu Hua](https://piao-0429.github.io/)$^{1*}$, [Minghuan Liu](minghuanliu.com)$^{2,3*}$, [Annabella Macaluso](https://github.com/AnnabellaMacaluso)$^{2^*}$, [Yunfeng Lin](https://github.com/CreeperLin)$^{3}$, [Weinan Zhang](wnzhang.net)$^{3}$, [Huazhe Xu](http://hxu.rocks/)$^{1}$, [Lirui Wang](https://liruiw.github.io/)$^{4}$ 

$^1$ Tsinghua University, $^2$ UCSD, $^3$ Shanghai Jiao Tong University, $^4$ MIT CSAIL
\* equal contribution 

[Project Page](https://gensim2.github.io/) | [Arxiv](https://arxiv.org/abs/2408)

Conference on Robot Learning, 2024


This repo explores using an LLM code generation pipeline to generate task codes & demonstrations for zero-shot and few-shot sim2real transfer.

<hr style="border: 2px solid gray;"></hr>


## ⚙️ Install

0. Clone the repository
    ```shell
    git clone https://github.com/GenSim2/GenSim2.git --recursive
    cd gensim2
    ```

1. Create a conda environment
    ```shell
    conda create -n gensim2 python=3.9 -y
    conda activate gensim2
    ```
2. Install PyTorch which **matches your cuda version** (check with `nvcc --version`), or you may meet with errors when installing pytorch3d later. Please refer to the [PyTorch website](https://pytorch.org/get-started/locally/) for the installation commands. For example, for cuda 11.8 :
    ```shell

    conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
    ```

3. Install other dependencies
   ```shell
   bash install.sh
   ```
   If you meet with errors with the installation above, you can refer to a detailed step-by-step installation [here](doc/installation.md).

## 🚶Getting Started

### 1. Generate Simulated Tasks with GenSim2: [A Video Tutorial](https://www.youtube.com/watch?v=PZW_iuHXNOg)
First you should add your OpenAI API key to the environment variable by running:
```shell
export OPENAI_KEY=your_openai_api_key
```

To run the GenSim2 pipeline, you can use the following commands:
```python
# Generate a primitive task
python gensim2/pipeline/run_pipeline.py prompt_folder=keypoint_pipeline_articulated_3stage prompt_data_folder=data_articulated/

# Generate a long-horizon task with the top-down approach
python gensim2/pipeline/run_pipeline.py prompt_folder=keypoint_pipeline_longhorizon_topdown prompt_data_folder=data_longhorizon/

# Generate a long-horizon task with the bottom-up approach
# Remember to store adequate tasks in the prompt_data_folder
python gensim2/pipeline/run_pipeline.py prompt_folder=keypoint_pipeline_longhorizon_bottomup prompt_data_folder=data_longhorizon/ mode=bottomup
```


For more details of how to create a kPAM solver with GenSim2 (especially with multi-modal LLM and rejection sampling), please refer to [solver_creation](doc/solver_creation.md).

<details>
<summary><span style="font-weight: bold;">Common command line arguments for run_pipeline.py </span></summary>

  **prompt_folder:**  
  Name of the prompt folder in ```prompts/``` to use for the pipeline.

  **prompt_data_folder:**   
  Name of the data folder in ```prompts/``` to use for the pipeline, including the asset library and initial task libraty.

  **output_folder:**  
  Name of the output folder to save the generated results. (Default to be ```logs/```).

  **num_tasks:**  
  Number of tasks to generate. (Default to be 1).

  **solver_trials:**  
  Number of solver configs to output in each generation iteration. (Default to be 3).

  **max_regeneration:**  
  Maximum number of times to regenerate a task before giving up. (Default to be 5).

  **gpt_model:**  
  GPT model to use for task proposal and task decomposition. (Default to be "gpt-4-1106-preview").

  **gpt_temperature:**  
  GPT temperature for task proposal and task decomposition. (Default to be 0.3 to ensure stability).

  **visual_solver_generation:**  
  Whether to use multi-modal LLM (GPT-4V) in solver generation. You need a monitor to load a GUI. (Default to be False).

  **solver_temperature:**  
  GPT-4V temperature for solver generation. (Default to be 0.8 to encourage diversity).

  **reject_sampling:**  
  Whether to use rejection sampling in the pipeline. You need a monitor to load a GUI. (Default to be True).

  **target_task_name:**  
  Name of the target task to generate. (Default to be None).

  **target_object_name:**  
  Name of the target object to use for generation. (Default to be None).

  **mode:**  
  Mode of the long-horizon task generation. (Default to be "topdown").

  For more arguments, please refer to the [pipeline config](gensim2/pipeline/experiments/configs/config.yaml).

</details>
<br>


To run a generated task with a kPAM solver, you can use the following commands:
```python
# Run the task "OpenBox"
python scripts/run_env_with_kpam.py --env OpenBox
```

<details>
<summary><span style="font-weight: bold;">Common command line arguments for run_env_with_kpam.py </span></summary>

  **--env**  
  Name of the environment to run. (Default to be "OpenBox").

  **--asset_id**  
  ID of the asset to use for the environment. It can be an id (number) in the asset folder ``assets/articulated_objs/ARTICULATED_NAME/``, or "" to represent a pre-defined instance, or "random" represents a randomly chosen id from the folder. (Default to be "").

  **--random**  
  Whether to randomize the initial poses of the objects. (Add this flag to set true).

  **--render**  
  Whether to render the environment. You need a monitor to load a GUI. (Add this flag to set true).

  **--num_episode**  
  Number of episodes to run. (Default to be 5).

  **--max_steps**  
  Maximum number of steps to run in each episode. (Default to be 500).

  **--video**  
  Whether to save the video of the environment. (Add this flag to set true).

  **--early_stop**  
  Whether to early stop the episode if the task is completed. (Add this flag to set true).



</details>
<br>
  

### 2. Generate Demonstrations
To generate demonstrations for generated tasks, you can use the following commands
```python
# Collect data for multiple tasks with multi-processing
# Remember to modify envs in kpam_data_collection_mp.py to assign tasks for demonstration collection
python scripts/kpam_data_collection_mp.py --dataset gensim2 --random --asset_id random --obs_mode pointcloud --save

# Collect data for a given task (e.g. "OpenBox") without multi-processing
python scripts/kpam_data_collection.py --env OpenBox --dataset gensim2 --random --asset_id random --obs_mode pointcloud --save
```
<details>
<summary><span style="font-weight: bold;">Common command line arguments for kpam_data_collection.py </span></summary>

  **--env**  
  Name of the environment to run. If not None, the variable "envs" in the script will be overwritten by the given task. (Default to be None).

  **--dataset**  
  Name of the collected dataset. It will appear in folder ``gensim2/agent/data/``.

  **--asset_id**  
  ID of the asset to use for the environment. It can be an id (number) in the asset folder ``assets/articulated_objs/ARTICULATED_NAME/``, or "" to represent a pre-defined instance, or "random" represents a randomly chosen id from the folder. (Default to be "").

  **--random**  
  Whether to randomize the initial poses of the objects. (Add this flag to set true).

  **--render**  
  Whether to render the environment. You need a monitor to load a GUI. (Add this flag to set true).

  **--num_episode**  
  Number of episodes to run. (Default to be 5).

  **--max_steps**  
  Maximum number of steps to run in each episode. (Default to be 500).

  **--obs_mode**  
  The modality of your observation, supporting "state", "image", and "pointcloud". (Default to be "pointcloud").

  **--save**  
  Whether to save the collected data. (Add this flag to set true).

  **--nprocs**  
  Number of processes to use for data collection. Only for ``kpam_data_collection_mp.py``. (Default to be 20).


</details>
<br>

For a collected dataset, you can try the following commands to check its size:
```python
# Check the size of a dataset, e.g. gensim2
python gensim2/agent/dataset/sim_traj_dataset.py --dataset gensim2
```

### 3. Train&Test Multi-Task Policy
Download the pretrained models from [Google Drive](https://drive.google.com/drive/folders/1cPM85xqboBsGETY79eK1EWG8Z8fhM3iK?usp=sharing) and put them in the `gensim2/agent/experiments/pretrained_weights/` folder.
After generating demonstrations, you can train (and test) a multi-task policy with the following commands:
```python
cd gensim2/agent/
# Train
python run.py suffix=gensim2_multitask domains=gensim2

# Test
python run.py suffix=gensim2_multitask domains=gensim2 train.total_epochs=0 train.pretrained_dir=dir_or_path_to_the_model(.pth)
```
<details>
<summary><span style="font-weight: bold;">Common command line arguments for run.py </span></summary>

  **suffix:**  
  Name of the current run.

  **domains:**  
  Name of the dataset to use for training.

  **env:**
  Name of the environment to use for training. Select from ``gensim2/agent/experiments/configs/env``. (Default to be "gensim2").

  **dataset.action_horizon:**  
  Number of predicted action sequences. Should be set to 1 if you use MLP as policy head. (Default to be 4).

  **dataset.observation_horizon:**  
  Number of historical observation sequences. (Default to be 3).

  **train.total_epochs:**  
  Number of training epochs. Set to 0 if you aim to evaluate a trained policy. (Default to be 250).

  **train.pretrained_dir:**  
  Directory or path to the pretrained model to load for evaluation. (Default to be None).

  **rollout_runner.env_names:**  
  Names of the environments to use for testing. You need to modify this argument in the env config, e.g. Line 42 in [gensim2 config](gensim2/agent/experiments/configs/env/gensim2.yaml).
  
  For more arguments, please refer to the [training config](gensim2/agent/experiments/configs/config.yaml) and [env config](gensim2/agent/experiments/configs/env/gensim2.yaml).
</details>
<br>

If you are interested in trying PPT in [RLBench](https://github.com/stepjam/RLBench), please follow this [instruction](doc/rlbench_exp.md).

### 4. Sim2Real transfer
For steps on performing Sim2Real transfer, please read the following [README](doc/sim2real.md).

## Task List
Please refer to [task list](doc/env.md) for a full list of supported tasks.


## Acknowledgements


- We would like to thank Professor [Xiaolong Wang](xiaolonw.github.io) for his kind support and discussion of this project. We thank [Yuzhe Qin](https://yzqin.github.io/) and [Fanbo Xiang](https://www.fbxiang.com/) for their generous help in sapien development. We thank [Mazeyu Ji](https://www.linkedin.com/in/jimazeyu/en) for his help on real-world experiments.

- The dataset and modeling codes are referred to [HPT](https://liruiw.github.io/hpt).

### Citation
If you find GenSim2 useful, please consider citing:


```bibtex
@inproceedings{gensim2,
      title={GenSim2: Scaling Robot Data Generation with Multi-modal and Reasoning LLMs}, 
      author={Pu, Hua and Minghuan, Liu and Annabella, Macaluso and Yunfeng, Lin and Weinan, Zhang and Huazhe, Xu and Lirui, Wang},
      year={2024},
      eprint={2308.},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

If you have any questions, consider to contact [Pu Hua](https://piao-0429.github.io/), [Lirui Wang](https://liruiw.github.io/) or [Minghuan Liu](https://minghuanliu.com/).
