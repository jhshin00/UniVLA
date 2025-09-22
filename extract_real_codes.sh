#!/bin/bash

# Extract real codes from LIBERO trajectories using actual model inference
# Supports both Stage-1 and Stage-2 models

export CUDA_VISIBLE_DEVICES=3

echo "=== Real LIBERO Trajectory Code Analysis ==="
echo "Using actual model inference (not dummy codes)"
echo ""

# Create output directory
mkdir -p ./visualize

# Check if stage argument is provided
STAGE=${1:-"both"}
EPISODE_RANGE=${2:-"0-2"}  # Default: episodes 0-2

if [ "$STAGE" = "stage1" ] || [ "$STAGE" = "both" ]; then
    echo "Processing Stage-1 model..."
    python libero_real_codes_analyzer.py \
        --checkpoint latent_action_model/logs/task_centric_lam_stage1/epoch=1-step=9000.ckpt \
        --data_root /ssd1/openpi_official/datasets/libero_raw \
        --task_name libero_goal_no_noops/1.0.0 \
        --episode_range $EPISODE_RANGE \
        --save_video ./visualize/stage1/real_codes_video_stage1 \
        --save_plot ./visualize/stage1/real_codes_plot_stage1 \
        --fps 10
    
    echo ""
fi

if [ "$STAGE" = "stage2" ] || [ "$STAGE" = "both" ]; then
    echo "Processing Stage-2 model..."
    python libero_real_codes_analyzer.py \
        --checkpoint latent_action_model/logs/task_centric_lam_stage2/epoch=2-step=18000.ckpt \
        --data_root /ssd1/openpi_official/datasets/libero_raw \
        --task_name libero_goal_no_noops/1.0.0 \
        --episode_range $EPISODE_RANGE \
        --save_video ./visualize/stage2/real_codes_video_stage2 \
        --save_plot ./visualize/stage2/real_codes_plot_stage2 \
        --fps 10
    
    echo ""
fi

echo "=== Analysis Complete ==="
echo "Real codes extracted using actual model inference!"
echo "Check the generated files:"
echo "  - ./visualize/real_codes_stage1_episode0.npz - Stage-1 codes data"
echo "  - ./visualize/real_codes_video_stage1_episode0.mp4 - Stage-1 visualization video"
echo "  - ./visualize/real_codes_plot_stage1_episode0.png - Stage-1 code sequence plot"
echo "  - ./visualize/real_codes_stage2_episode0.npz - Stage-2 codes data"
echo "  - ./visualize/real_codes_video_stage2_episode0.mp4 - Stage-2 visualization video"
echo "  - ./visualize/real_codes_plot_stage2_episode0.png - Stage-2 code sequence plot"
echo ""
echo "Usage:"
echo "  bash extract_real_codes.sh stage1 [episode_range]   # Run only Stage-1"
echo "  bash extract_real_codes.sh stage2 [episode_range]   # Run only Stage-2"
echo "  bash extract_real_codes.sh both [episode_range]     # Run both stages (default)"
echo ""
echo "Episode range examples:"
echo "  bash extract_real_codes.sh stage1 0-2     # Episodes 0, 1, 2"
echo "  bash extract_real_codes.sh stage1 0,2,4   # Episodes 0, 2, 4"
echo "  bash extract_real_codes.sh stage1 5       # Single episode 5"
