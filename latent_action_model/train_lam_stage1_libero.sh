export CUDA_VISIBLE_DEVICES=0
torchrun --standalone --nnodes 1 --nproc-per-node 1 main.py fit \
    --config config/lam-stage-1_libero.yaml \
    --ckpt_path ./logs/task_centric_lam_stage1/last.ckpt \
    2>&1 | tee lam-stage-1.log