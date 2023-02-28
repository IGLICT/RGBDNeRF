#!/bin/bash
DATASET=./sample_data/scan_plant
save_dir=./logs/test_scan_plant_save
mkdir -p "${save_dir}"
python -u train.py "${DATASET}" --mesh-data ./sample_data/scan_plant_mesh \
	--gpu-id "0" \
	--train-views "0..9" --mesh-train-views "0..160" \
	--view-resolution "540x720" --view-per-batch 1 --pixel-per-view 1024 \
	--down-pixels-per-view-at "9000" --pixel-per-view-down "0.125" \
	--no-preload --sampling-on-mask 1.0 --no-sampling-at-reader \
	--valid-views "9..17" --mesh-valid-views "0..160:20" \
	--valid-view-resolution "180x240" --valid-view-per-batch 1 \
	--transparent-background "1.0,1.0,1.0" --background-stop-gradient \
	--use-octree --raymarching-stepsize-ratio 0.125 --discrete-regularization \
	--color-weight 128.0 --alpha-weight 1.0 --lr 0.001 \
	--mesh-pretrain-num 9000 --total-num-update 150000 --save-interval-updates 500 \
	--reduce-step-size-at "5000,25000,75000" \
	--save-dir "${save_dir}" --tensorboard-logdir "${save_dir}"/tensorboard --chunk-size 8 \
	--voxel-path "${DATASET}"/OccuVoxel_low.ply --octree-path "${DATASET}"/octree_low.npz --voxel-size 0.2 \
	--dis --dis-views "9..323" --gan-weight 2.0 --gan-norm-layer "instance" --n-layers 3 --patch-size 32 \
	--load-pc --pc-path "${DATASET}"/pc_color.ply --voxel-color-path "${DATASET}"/voxel_color.txt --pc-pose-dim 3 \
	| tee -a "${save_dir}"/train.log
