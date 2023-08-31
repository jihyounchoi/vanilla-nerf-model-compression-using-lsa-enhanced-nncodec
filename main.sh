#!/bin/bash

set -e

CKPT_PATH="/home/gbang/jihyoun/NeRF/nerf_lsa/model_zoo/blender_paper_lego/lego_200000.tar"
CKPT_NICKNAME='lego_200K'
BASE_PATH_TO_SAVE='/home/gbang/jihyoun/NeRF/nerf_lsa'
DATASET_PATH='~'

cd /home/gbang/jihyoun/NeRF/test_nerf
# nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep python | awk -F ',' '{print $1}' | xargs -I{} kill -9 {}

CUDA_VISIBLE_DEVICES=0 python /home/gbang/jihyoun/NeRF/nerf_lsa/compress_nerf.py \
    --ckpt_path $CKPT_PATH \
    --ckpt_nickname $CKPT_NICKNAME \
    --base_path_to_save $BASE_PATH_TO_SAVE \
    --qp -20 \
    --lsa True \
    --epochs 2 \
    --learning_rate 0.0001 \
    --task_type NeRF \
    --dataset_type llff \
    --N_iters 50001 \
    --learning_rate_decay 0.1 \
    --i_save 50000 \
    --dataset_path $DATASET_PATH