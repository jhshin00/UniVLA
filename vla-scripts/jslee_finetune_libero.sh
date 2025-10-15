#!/bin/bash

export CUDA_VISIBLE_DEVICES=1,2,3,5,6
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO

GPUS_PER_NODE=5
NNODES=1
MASTER_PORT=${MASTER_PORT:-28597}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
RANK=${RANK:-0}

# Set paths according to your setup
# IMPORTANT: No conversion needed!
# This script now uses Prismatic native format directly.
# Just point to your jslee_train_lora.py checkpoint (.pt file)

VLA_MODEL_ID="prism-dinosiglip-224px+7b"  # Prismatic model ID (will be downloaded from HF Hub)
LORA_CHECKPOINT="/ssd1/UniVLA/vla_log/prism-dinosiglip-224px+mx-libero+n0+b8+x42--image_aug+lora-r32+dropout-0.0+vlora-attn_mlp-LIBERO-Latent-Action-Pretraining-ws-16/checkpoints/step-000400-epoch-00-loss=0.7765.pt"  # Your jslee_train_lora.py checkpoint (.pt file)
LAM_PATH="/ssd1/UniVLA/latent_action_model/logs/task_centric_lam_stage2/epoch=2-step=18000.ckpt"
DATA_ROOT="/ssd4/libero_raw"
DATASET_NAME="libero_combined"

torchrun --nproc_per_node ${GPUS_PER_NODE} \
         --nnodes ${NNODES} \
         --node_rank ${RANK} \
         --master_addr ${MASTER_ADDR} \
         --master_port ${MASTER_PORT} \
         vla-scripts/jslee_finetune_libero.py \
         --vla_model_id "${VLA_MODEL_ID}" \
         --lora_pretrained_path "${LORA_CHECKPOINT}" \
         --lam_path "${LAM_PATH}" \
         --data_root_dir "${DATA_ROOT}" \
         --dataset_name "${DATASET_NAME}" \
         --run_root_dir "finetune_runs" \
         --freeze_base_model true \
         --lora_only_training true \
         --lora_vision true \
         --lora_vision_target "attn_mlp" \
         --lora_target "attn_mlp" \
         --lora_rank 32 \
         --lora_dropout 0.0 \
         --batch_size 1 \
         --grad_accumulation_steps 1 \
         --max_steps 20000 \
         --save_steps 5000 \
         --learning_rate 5e-5 \
         --window_size 16 \
         --wandb_entity "jlee24" \
         --wandb_project "jslee-finetune-libero" \
         --run_id_note "lora-only-real-action" \
         --image_aug true \