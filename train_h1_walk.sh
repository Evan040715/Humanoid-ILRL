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
    --run_name=h1_141_16_wavehello_19dof_03 \
    --save_interval=500 \
    --reference_motion_file=resources/motions/output/h1_141_16_wavehello_19dof_03.npy \
    --reference_loop=true
    # --headless
    # --experiment_name=h1_walk
    # --run_name=imitation_v1
    # --resume
    # --load_run=-1
    # --checkpoint=-1
    # --reference_motion_file=resources/motions/output/XX/h1_other_motion.npy
    # --reference_loop=false
