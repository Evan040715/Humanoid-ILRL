#!/bin/bash

# Testing script for H1 Robot Imitation Learning

python legged_gym/scripts/play.py \
    --task=h1 \
    --num_envs=5 \
    --experiment_name=h1 \
    --load_run=Jan20_13-11-05_AMASS_CMU_07_01_walk \
    --checkpoint=500 \
    # --run_name=AMASS_CMU_01_01_jump
    # --reference_motion_file=resources/motions/output/01/h1_cmu_jump_10dof.npy
    # --reference_loop=true


