export LD_LIBRARY_PATH=/home/pai/envs/openvla/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=3
GPUS_PER_NODE=1
NNODES=1
MASTER_PORT=${MASTER_PORT:-28596}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
RANK=${RANK:-0}

# Run your fine-tuning script with torchrun
torchrun --nproc_per_node ${GPUS_PER_NODE} --nnodes ${NNODES} --node_rank ${RANK} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} finetune_libero_lora.py \
                                 --vla_path "/path/to/your/pretrained-univla-7b" \
                                 --pretrained_lora_path "/path/to/pretrained/lora/adapter" \
                                 --run_root_dir "vla_finetune_log" \
                                 --use_lora True \
                                 --lora_rank 32 \
                                 --lora_dropout 0.0 \
                                 --dataset_name "libero_combined" \
                                 --window_size 12 \
                                 --shuffle_buffer_size 16000 \
                                 --batch_size 8 \
                                 --max_steps 30000 \
                                 --learning_rate 3.5e-4 \
                                 --lam_path "latent_action_model/logs/task_centric_lam_stage2/epoch=2-step=18000.ckpt" \
                                 --hf_token 