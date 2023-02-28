#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python render.py ./sample_data/scan_plant --path ./logs/test_scan_plant_save/checkpoint_last.pt --model-overrides '{"valid_chunk_size":64,"chunk_size":8,"raymarching_tolerance":0.01}' --render-save-fps 24 --render-resolution "540x720" --render-beam 1 --render-camera-poses ./sample_data/scan_plant_all/extrinsic --render-depth-rawoutput --render-views "0..339" --render-output ./logs/test_scan_plant_save/output_all --render-output-types "color" #

