export LD_LIBRARY_PATH=/home/pai/envs/openvla/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=3
#export HUGGINGFACE_HUB_TOKEN=
GPUS_PER_NODE=1
NNODES=1
MASTER_PORT=${MASTER_PORT:-28596}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
RANK=${RANK:-0}


# Run your training script with torchrun
torchrun --nproc_per_node ${GPUS_PER_NODE} --nnodes ${NNODES} --node_rank ${RANK} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} train_lora.py \
                                 --vla.type prism-dinosiglip-224px+mx-libero \
                                 --vla.expected_world_size 1 \
                                 --vla.global_batch_size 32 \
                                 --vla.per_device_batch_size 32 \
                                 --run_root_dir "vla_log" \
                                 --use_lora True \
                                 --lora_rank 32 \
                                 --lora_dropout 0.0 \
                                 --dataset_name "libero_combined" \
                                 --window_size 12 \
                                 --shuffle_buffer_size 16000 \
                                 --lam_path "latent_action_model/logs/task_centric_lam_stage2/epoch=2-step=18000.ckpt"    