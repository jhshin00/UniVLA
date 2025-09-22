export CUDA_VISIBLE_DEVICES=3
torchrun --standalone --nnodes 1 --nproc-per-node 1 main.py fit \
    --config config/lam-stage-2_libero.yaml \
    2>&1 | tee lam-stage-2.log