#!/bin/bash

# Training script for G1 Robot Imitation Learning (23 DoF)

python legged_gym/scripts/train.py \
    --task=g1 \
    --seed=42 \
    --max_iterations=40000 \
    --save_interval=500 \
    --sim_device=cuda:0 \
    --rl_device=cuda:0 \
    --num_envs=4096 \
    --experiment_name=g1 \
    --run_name=g1_07_01_walk_23dof \
    --reference_motion_file=resources/motions/output/g1_07_01_walk_23dof.npy \
    --reference_loop=true
    # --headless
v

