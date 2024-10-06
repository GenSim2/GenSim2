set -x
set -e

HYDRA_FULL_ERROR=1  time  python   -m agent.run_real  \  
	suffix="test" \  
	train.pretrained_dir=${1} \
	+prompt=${2}