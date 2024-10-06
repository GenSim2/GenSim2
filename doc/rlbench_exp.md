## 🚶Evaluate Proprioceptive Pointcloud Transformer (PPT) only on RLBench

First you should install RLBench dependencies following this [repository](https://github.com/stepjam/RLBench?tab=readme-ov-file#install).
### Generate demonstrations
1. Change the environments in `gensim2/env/utils/rlbench.py`
2. Run `python scripts/rlbench_data_collection_mp.py --nprocs 32 --obs_mode pointcloud --random --asset_id 'random' ---save --num_episodes 100` (use `--save` to save collected data onto disks, `--random` is for randomization)

### Train&Test multi-task policies
1. Choose the config in `gensim2/agent/experiments/configs/env` and replace the name in `gensim2/agent/experiments/configs/config.yaml`, Line 14. Parameter `domains` determine the name of dataset.
2. Run 
```python
python agent/run.py  \  # Train
  suffix=name_of_the_run \  
  dataset.action_horizon=4 \ # predicted action sequence, has to be 1 for MLP head
  dataset.observation_horizon=3 \ # historical observation horizon
  train.total_epochs=0 # training epochs

python agent/run.py  \  # Test
  suffix=name_of_the_run \  
  dataset.observation_horizon=3 \ 
  train.total_epochs=0 \ # to prevent from re-training
  train.pretrained_dir=relative_path_to_the_model.pth \
  network.openloop_steps=${openloop_steps} \  # openloop actions
  temporal_agg=False; # trick as in ACT, this will make openloop_steps=1 only
```