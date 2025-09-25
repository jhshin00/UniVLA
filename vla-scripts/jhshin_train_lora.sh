# Load HuggingFace token from file
if [ -f "../hf_token.txt" ]; then
    export HUGGINGFACE_HUB_TOKEN=$(cat ../hf_token.txt | grep -v "^#" | head -1 | tr -d '\n\r')
    echo "Loaded HuggingFace token from file"
else
    echo "Warning: hf_token.txt not found. Make sure to set HUGGINGFACE_HUB_TOKEN environment variable."
fi

export CUDA_VISIBLE_DEVICES=1,2,3
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO

GPUS_PER_NODE=3
NNODES=1
MASTER_PORT=${MASTER_PORT:-28596}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
RANK=${RANK:-0}

torchrun --nproc_per_node ${GPUS_PER_NODE} --nnodes ${NNODES} --node_rank ${RANK} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} vla-scripts/train_lora.py \
                                    --vla.type prism-dinosiglip-224px+mx-libero \
                                    --run_root_dir "vla_log" \
                                    --use_lora true \
                                    --lora_rank 8 \
                                    --lora_dropout 0.0 \
                                    --lora_vision true \
                                    --lora_vision_target "attn" \
                                    --lora_target "attn" \
                                    --dataset_name "libero_combined" \
                                    --window_size 6 \
                                    --lam_path "/ssd2/UniVLA/latent_action_model/logs/task_centric_lam_stage2/epoch=2-step=18000.ckpt" \
                                    --vla.expected_world_size ${GPUS_PER_NODE} \
                                    --wandb_entity "jhshin406" \
                                    --wandb_project "univla-lora-libero" \
                                    --vla.global_batch_size 3 \
                                    --vla.per_device_batch_size 1 \
                                    --gradient_accumulation_steps 1 \
                                    --vla.enable_mixed_precision_training true \
                                    --vla.enable_gradient_checkpointing true \