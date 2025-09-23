# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

UniVLA is a Vision-Language-Action (VLA) model that learns to act anywhere with task-centric latent actions. The repository implements a two-stage training approach: first learning task-centric latent actions using a VQ-VAE based latent action model (LAM), then training a generalist policy using these latent actions.

## Key Architecture Components

### 1. Latent Action Model (LAM) - `latent_action_model/`
- **Stage 1**: Learns task-irrelevant latent actions using VQ-VAE (`lam-stage-1.yaml`)
- **Stage 2**: Learns task-centric latent actions on top of stage-1 results (`lam-stage-2.yaml`)
- Main entry point: `latent_action_model/main.py` using Lightning CLI
- Model implementations: `latent_action_model/genie/model.py` (DINO_LAM, LAPA_LAM)
- Dataset handling: `latent_action_model/genie/dataset.py` (LightningOpenX)

### 2. Prismatic VLA Framework - `prismatic/`
- **Models**: Vision-language-action models built on Prismatic framework
  - Base VLM: `prismatic/models/vlms/prismatic.py`
  - Vision backbones: `prismatic/models/backbones/vision/` (CLIP, SigLIP, etc.)
- **VLA Training**: `prismatic/vla/` contains dataset handling and training logic
- **Configuration**: `prismatic/conf/` for model and training configurations

### 3. Training Scripts - `vla-scripts/`
- **Pretraining**: `train.py` for generalist policy training with latent actions
- **Fine-tuning**: Task-specific scripts (`finetune_libero.py`, `finetune_calvin.py`, etc.)
- **Real-world deployment**: `real_world_deployment.py`

### 4. Evaluation - `experiments/`
- **Robot tasks**: LIBERO, CALVIN, SimplerEnv evaluation scripts
- **Navigation**: Room2Room (R2R) VLN evaluation

## Common Development Commands

### Environment Setup
```bash
# Create conda environment
conda create -n univla python=3.10 -y
conda activate univla

# Install PyTorch (check https://pytorch.org for your CUDA version)
pip install torch torchvision

# Install repository
pip install -e .

# Install Flash Attention for training
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
```

### Latent Action Model Training
```bash
# Stage 1: Task-irrelevant latent actions
torchrun --standalone --nnodes 1 --nproc-per-node 8 main.py fit \
    --config config/lam-stage-1.yaml \
    2>&1 | tee lam-stage-1.log

# Stage 2: Task-centric latent actions (update stage_one_ckpt path first)
torchrun --standalone --nnodes 1 --nproc-per-node 8 main.py fit \
    --config config/lam-stage-2.yaml \
    2>&1 | tee lam-stage-2.log
```

### VLA Pretraining
```bash
# Multi-GPU pretraining (adjust GPUS_PER_NODE and NNODES)
bash ./vla-scripts/train.sh
# Or directly:
torchrun --nproc_per_node 8 --nnodes 4 train.py \
    --vla.type prism-dinosiglip-224px+mx-oxe-magic-soup-plus \
    --run_root_dir "vla_log"
```

### Task-Specific Fine-tuning

#### LIBERO
```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 finetune_libero.py \
    --dataset_name "libero_10_no_noops" \
    --run_root_dir "libero_log"
```

#### CALVIN
```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 finetune_calvin.py \
    --vla_path /path/to/univla-7b \
    --lam_path /path/to/lam-stage-2.ckpt \
    --calvin_root /path/to/calvin_root \
    --max_steps 100000
```

### Model Evaluation

#### LIBERO Evaluation
```bash
python experiments/robot/libero/run_libero_eval.py \
    --task_suite_name libero_10 \
    --action_decoder_path /path/to/action_decoder.pt \
    --pretrained_checkpoint /path/to/finetuned_model \
    --num_trials_per_task 50
```

#### CALVIN Evaluation
```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 \
    experiments/robot/calvin/run_calvin_eval_ddp.py \
    --calvin_root /path/to/calvin_root \
    --action_decoder_path /path/to/action_decoder.pt \
    --pretrained_checkpoint /path/to/finetuned_model
```

## Code Quality Tools

### Linting and Formatting
```bash
# Code formatting (Black)
black --line-length 121 .

# Linting (Ruff)
ruff check .
```

### Testing
Check individual experiment directories for specific test commands. Most evaluations include built-in validation through rollout success rates.

## Key Configuration Files

- **LAM Training**: `latent_action_model/config/lam-stage-{1,2}.yaml`
- **VLA Models**: `prismatic/conf/vla.py` - defines model architectures and data mixtures
- **Data Mixtures**: `prismatic/vla/datasets/rlds/oxe/mixtures.py` - dataset combinations
- **Training Configs**: Various `finetune_*.py` scripts contain embedded configurations

## Data Requirements

### For LAM Training
- OpenX datasets in RLDS format
- Ego4D dataset (optional, for human videos)
- Data root specified in config files (`data_root`)

### For Task-Specific Training
- **LIBERO**: Download from HuggingFace (`openvla/modified_libero_rlds`)
- **CALVIN**: Follow official CALVIN installation
- **Real-world**: HDF5 format with action, observation, and proprioceptive data

## Model Checkpoints

Pre-trained models available on HuggingFace:
- `qwbu/univla-latent-action-model` (LAM stage-1 & stage-2)
- `qwbu/univla-7b` (Base pretrained model)
- Task-specific models: `qwbu/univla-7b-224-sft-{libero,calvin,r2r,simpler-bridge}`

## Development Notes

- The codebase uses PyTorch Lightning for LAM training and native PyTorch for VLA training
- Multi-GPU training is supported via `torchrun`
- Configuration management uses both YAML files and Python dataclasses
- The current branch `univla_libero` appears to have LIBERO-specific modifications