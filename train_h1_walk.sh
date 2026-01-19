#!/bin/bash

# Training script for H1 Robot Imitation Learning (Walking)

python legged_gym/scripts/train.py \
    --task=h1 \
    --seed=42 \
    --max_iterations=40000 \
    --sim_device=cuda:0 \
    --rl_device=cuda:0 \
    --num_envs=4096 \
    --experiment_name=h1 \
    --run_name=AMASS_CMU_07_01_walk \
    # --headless
    # --experiment_name=h1_walk
    # --run_name=imitation_v1
    # --resume
    # --load_run=-1
    # --checkpoint=-1
