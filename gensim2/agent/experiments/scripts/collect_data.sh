set -x
set -e

# python misc/kpam_data_collection_mp.py --obs_mode pointcloud --random --asset_id 'random' --nproc 20 --save --num_episodes 
python misc/kpam_data_collection_mp.py --num_pcd 10240 --obs_mode pointcloud --random --asset_id 'random' --nproc 20 --save --num_episodes 100 --dataset "fr3-11task-100eps-10240pcd" 
# python misc/kpam_data_collection_mp.py --dataset "fr3-sim" --num_pcd 4096 --obs_mode pointcloud --random --asset_id 'random' --save --num_episodes 20