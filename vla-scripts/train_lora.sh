#!/usr/bin/env bash
set -euo pipefail

# =========[ GPU / CUDA / HF ]=========
export CUDA_VISIBLE_DEVICES=2,3
export LD_LIBRARY_PATH="/home/pai/envs/openvla/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
export HUGGINGFACE_HUB_TOKEN=

export WANDB_ENTITY="jhshin406"
export WANDB_PROJECT="univla-lora-libero"

# (선택) 캐시/토크나이저
export HF_HOME="${HF_HOME:-/ssd2/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export TOKENIZERS_PARALLELISM=false

# =========[ torch.distributed ]=========
GPUS_PER_NODE=2
NNODES=1
MASTER_PORT=${MASTER_PORT:-28596}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
RANK=${RANK:-0}

# 메모리 단편화 완화 (PyTorch 권장)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# (성능/안정화)
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_BLOCKING_WAIT=1

# =========[ Paths / Logs ]=========
RUN_ROOT_DIR="vla_log"
ADAPTER_ROOT_DIR="adapter-tmp"   # train_lora.py에서 사용
mkdir -p "$RUN_ROOT_DIR" "$ADAPTER_ROOT_DIR"

# =========[ Run ]=========

torchrun \
  --nproc_per_node ${GPUS_PER_NODE} \
  --nnodes ${NNODES} \
  --node_rank ${RANK} \
  --master_addr ${MASTER_ADDR} \
  --master_port ${MASTER_PORT} \
  train_lora.py \
    --vla.type prism-dinosiglip-224px+mx-libero \
    --vla.expected_world_size ${GPUS_PER_NODE} \
    --vla.global_batch_size 4 \
    --vla.per_device_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --run_root_dir "${RUN_ROOT_DIR}" \
    --use_lora true \
    --lora_rank 8 \
    --lora_dropout 0.0 \
    --dataset_name "libero_combined" \
    --window_size 6 \
    --shuffle_buffer_size 4000 \
    --lam_path "/ssd2/UniVLA/latent_action_model/logs/task_centric_lam_stage2/epoch=2-step=18000.ckpt" \
    --hf_token ${HUGGINGFACE_HUB_TOKEN} \
    --wandb_entity ${WANDB_ENTITY} \
    --wandb_project ${WANDB_PROJECT} \
    --use_flash_attention true \
    --lora_target "attn" \
    --vla.enable_mixed_precision_training true \
    --vla.enable_gradient_checkpointing true \
    # --clamp_seq_len 1024 \