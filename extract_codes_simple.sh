#!/bin/bash

# Simple script to extract codes from multiple LIBERO trajectories
export CUDA_VISIBLE_DEVICES=3

# Checkpoint path
CHECKPOINT="/ssd2/UniVLA/latent_action_model/logs/task_centric_lam_stage2/epoch=2-step=18000.ckpt"

# Run the analysis with multiple episodes
python libero_trajectory_analyzer.py \
    --checkpoint "$CHECKPOINT" \
    --data_root "/ssd1/openpi_official/datasets/libero_raw" \
    --task_name "libero_goal_no_noops/1.0.0" \
    --episode_range "0-9" \
    --save_video "codes_video" \
    --save_plot "codes_plot" \
    --fps 10 \
    --output_dir "./visualize/stage2"