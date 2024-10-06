set -x
set -e

FOR horizon in {2, 3, 4, 6}; do
	FOR action_horizon in {4, 8}; do
# train
HYDRA_FULL_ERROR=1  time  python   -m agent.run  \  
		suffix=all-horizon${horizon}-nblock4-nhead8-diff${action_horizon}-randomdemo50 \  
		dataset.action_horizon=${action_horizon} \  # predicted action sequence, has to be 1 for MLP head
		dataset.observation_horizon=${horizon};   # historical obs length

		FOR openloop_step in {1,4,8}; do
		# eval, same parameter but total_epochs=0!
		HYDRA_FULL_ERROR=1  time  python   -m agent.run  \  
				suffix=all-horizon${horizon}-nblock4-nhead8-diff${action_horizon}-randomdemo50 \
				dataset.action_horizon=${action_horizon} \  
				dataset.observation_horizon=${horizon} \ 
				train.total_epochs=0 \  
				train.pretrained_dir=outputs/gensim_articulated_tasks/all-horizon${horizon}-nblock4-nhead8-diff${action_horizon}-randomdemo50 \ 
				network.openloop_steps=${openloop_steps} \  # This take a sequence of actions
				temporal_agg=False; # this will make openloop_steps=1 only
		done
	done
done