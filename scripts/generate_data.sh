#! /bin/bash

ENVS=(
	"OpenBox"
    "CloseBox"
    "OpenLaptop"
    "CloseLaptop"
    "OpenDrawer"
    "PushDrawerClose"
    "SwingBucketHandle"
    "LiftBucketUpright"
    "PressToasterLever"
    "PushToasterForward"
    "MoveBagForward"
    "OpenSafe"
    "CloseSafe"
    "RotateMicrowaveDoor"
    "CloseMicrowave"
    "CloseSuitcaseLid"
    "SwingSuitcaseLidOpen"
    "RelocateSuitcase"
    "TurnOnFaucet"
    "TurnOffFaucet"
    "SwingDoorOpen"
    "ToggleDoorClose"
    "CloseRefrigeratorDoor"
    "OpenRefrigeratorDoor"
	)

dataset="gensim2"

for env in "${ENVS[@]}";
do
	python scripts/kpam_data_collection_mp.py --asset_id "random" --random --dataset "$dataset" --env "$env" --num_episodes 100 --save --obs_mode pointcloud --nprocs 20
done