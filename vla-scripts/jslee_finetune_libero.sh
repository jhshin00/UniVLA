#!/bin/bash

export CUDA_VISIBLE_DEVICES=1,2,3,5,6,7
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO

GPUS_PER_NODE=6
NNODES=1
MASTER_PORT=${MASTER_PORT:-28597}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
RANK=${RANK:-0}

# Set paths according to your setup
# IMPORTANT: You need to export base VLM to HF format first using:
#   python vla-scripts/export_base_vlm_to_hf.py --model_id prism-dinosiglip-224px+7b --output_dir /ssd1/UniVLA/hf_models/prism-dinosiglip-224px+7b
# Then convert your .pt checkpoint to PEFT format using:
#   python vla-scripts/convert_checkpoint_to_peft.py --checkpoint_path <your-checkpoint.pt> --base_model_path <hf-base-model> --output_dir <output-dir>

VLA_BASE_PATH="/ssd1/UniVLA/hf_models/prism-dinosiglip-224px+7b"  # Base VLA model in HF format (export using export_base_vlm_to_hf.py)
LORA_PRETRAINED_PATH="/ssd1/UniVLA/lora_adapters/step-004000"  # LoRA adapters in PEFT format (convert using convert_checkpoint_to_peft.py)
LAM_PATH="/ssd1/UniVLA/latent_action_model/logs/task_centric_lam_stage2/epoch=2-step=18000.ckpt"
DATA_ROOT="/ssd4/libero_raw"
DATASET_NAME="libero_combined"

torchrun --nproc_per_node ${GPUS_PER_NODE} \
         --nnodes ${NNODES} \
         --node_rank ${RANK} \
         --master_addr ${MASTER_ADDR} \
         --master_port ${MASTER_PORT} \
         vla-scripts/jslee_finetune_libero.py \
         --vla_path "${VLA_BASE_PATH}" \
         --lora_pretrained_path "${LORA_PRETRAINED_PATH}" \
         --lam_path "${LAM_PATH}" \
         --data_root_dir "${DATA_ROOT}" \
         --dataset_name "${DATASET_NAME}" \
         --run_root_dir "finetune_runs" \
         --freeze_base_model true \
         --lora_only_training true \
         --batch_size 4 \
         --grad_accumulation_steps 2 \
         --max_steps 20000 \
         --save_steps 5000 \
         --learning_rate 5e-5 \
         --window_size 6 \
         --wandb_entity "jlee24" \
         --wandb_project "jslee-finetune-libero" \
         --run_id_note "lora-only-real-action" \
         --image_aug true