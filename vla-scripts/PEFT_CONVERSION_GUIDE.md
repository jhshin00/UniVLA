# PEFT Format Conversion Guide

This guide explains how to convert your LoRA training checkpoints to HuggingFace PEFT format for easy finetuning.

## Overview

When training with LoRA using `jslee_train_lora.py`, checkpoints are saved in two formats:

1. **`.pt` format** (legacy): Contains state_dicts for trainable parameters only
2. **PEFT format** (new): Automatically saved alongside `.pt` files in `checkpoints/peft-step-XXXXXX/` directory

For existing `.pt` checkpoints trained before this update, you need to manually convert them to PEFT format.

## Step 1: Export Base VLM to HuggingFace Format

First, export your base VLM (without LoRA) to HuggingFace format:

```bash
python vla-scripts/export_base_vlm_to_hf.py \
    --model_id prism-dinosiglip-224px+7b \
    --output_dir /ssd1/UniVLA/hf_models/prism-dinosiglip-224px+7b
```

**Parameters:**
- `--model_id`: Prismatic model identifier (e.g., `prism-dinosiglip-224px+7b`)
- `--output_dir`: Where to save the HuggingFace format model

**Output:**
- Directory with `config.json`, `model.safetensors`, `preprocessor_config.json`, etc.
- This only needs to be done **once per base model**

## Step 2: Convert Your LoRA Checkpoint to PEFT Format

Convert your existing `.pt` checkpoint to PEFT format:

```bash
python vla-scripts/convert_checkpoint_to_peft.py \
    --checkpoint_path /ssd1/UniVLA/vla_log/YOUR-RUN-ID/checkpoints/step-004000-epoch-00-loss=0.5093.pt \
    --base_model_path /ssd1/UniVLA/hf_models/prism-dinosiglip-224px+7b \
    --output_dir /ssd1/UniVLA/lora_adapters/step-004000 \
    --lora_rank 32 \
    --lora_dropout 0.0 \
    --lora_target attn \
    --lora_vision True \
    --lora_vision_target attn_mlp
```

**Parameters (must match your training config):**
- `--checkpoint_path`: Path to your `.pt` checkpoint
- `--base_model_path`: Path to HF format base model (from Step 1)
- `--output_dir`: Where to save converted PEFT adapters
- `--lora_rank`: LoRA rank used in training (default: 32)
- `--lora_dropout`: LoRA dropout used in training (default: 0.0)
- `--lora_target`: LLM LoRA target (`attn` or `attn_mlp`)
- `--lora_vision`: Whether Vision LoRA was used (default: True)
- `--lora_vision_target`: Vision LoRA target (`attn` or `attn_mlp`)

**Output structure:**
```
/ssd1/UniVLA/lora_adapters/step-004000/
├── llm_lora/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── vision_lora/
│   ├── dino/
│   │   ├── adapter_config.json
│   │   └── adapter_model.safetensors
│   └── siglip/
│       ├── adapter_config.json
│       └── adapter_model.safetensors
├── projector.pt
└── conversion_metadata.json
```

## Step 3: Update Finetuning Script

Update `jslee_finetune_libero.sh` with the correct paths:

```bash
VLA_BASE_PATH="/ssd1/UniVLA/hf_models/prism-dinosiglip-224px+7b"
LORA_PRETRAINED_PATH="/ssd1/UniVLA/lora_adapters/step-004000"
```

## Step 4: Run Finetuning

Now you can run finetuning as usual:

```bash
bash vla-scripts/jslee_finetune_libero.sh
```

The finetuning script will automatically detect the PEFT format and load:
- LLM LoRA adapters from `llm_lora/`
- Vision LoRA adapters from `vision_lora/` (if present)
- Projector weights from `projector.pt`

## Future Training Runs

For new training runs using the updated `jslee_train_lora.py`, checkpoints will be automatically saved in both formats:

- **`.pt` format**: `checkpoints/step-XXXXXX-epoch-XX-loss=X.XXXX.pt`
- **PEFT format**: `checkpoints/peft-step-XXXXXX-epoch-XX-loss=X.XXXX/`

You can use the PEFT format directly for finetuning without conversion!

**Latest checkpoint shortcut:**
```bash
LORA_PRETRAINED_PATH="/ssd1/UniVLA/vla_log/YOUR-RUN-ID/checkpoints/latest-peft"
```

## Verification

To verify the conversion worked correctly:

1. Check the output directory structure matches the expected format
2. Run the finetuning script and check the logs for:
   - `✅ LLM LoRA loaded`
   - `✅ DINOv2 LoRA loaded`
   - `✅ SigLIP LoRA loaded`
   - `✅ Projector loaded`
   - No warnings about missing Vision LoRA modules

## Troubleshooting

### Issue: "Missing keys" or "Unexpected keys" during conversion

**Solution:** Make sure your LoRA configuration parameters (`--lora_rank`, `--lora_target`, etc.) exactly match the configuration used during training.

### Issue: "Vision LoRA not loaded (legacy format)"

**Solution:** Your checkpoint doesn't contain Vision LoRA weights. Either:
- Re-train with Vision LoRA enabled, or
- Set `--lora_vision False` in the conversion script

### Issue: Conv2D LoRA modules detected

**Solution:** This indicates patch_embed layers were incorrectly targeted. The conversion script filters these out, but verify your training script doesn't target Conv2D layers.

### Issue: Model size mismatch

**Solution:** Ensure the base model path points to the correct model that was used for training (e.g., `prism-dinosiglip-224px+7b`).

## Example: Complete Workflow

```bash
# 1. Export base VLM (once per model)
python vla-scripts/export_base_vlm_to_hf.py \
    --model_id prism-dinosiglip-224px+7b \
    --output_dir /ssd1/UniVLA/hf_models/prism-dinosiglip-224px+7b

# 2. Convert your checkpoint
python vla-scripts/convert_checkpoint_to_peft.py \
    --checkpoint_path /ssd1/UniVLA/prism-dinosiglip-224px+mx-libero+n0+b16+x42--image_aug+lora-r32+dropout-0.0+vlora-attn_mlp-LIBERO-Latent-Action-Pretraining-ws-16/checkpoints/step-004000-epoch-00-loss=0.5093.pt \
    --base_model_path /ssd1/UniVLA/hf_models/prism-dinosiglip-224px+7b \
    --output_dir /ssd1/UniVLA/lora_adapters/step-004000 \
    --lora_rank 32 \
    --lora_dropout 0.0 \
    --lora_target attn \
    --lora_vision True \
    --lora_vision_target attn_mlp

# 3. Update and run finetuning
# Edit jslee_finetune_libero.sh with:
#   VLA_BASE_PATH="/ssd1/UniVLA/hf_models/prism-dinosiglip-224px+7b"
#   LORA_PRETRAINED_PATH="/ssd1/UniVLA/lora_adapters/step-004000"

bash vla-scripts/jslee_finetune_libero.sh
```

## Notes

- The conversion process does not modify your original `.pt` checkpoint
- PEFT format takes up slightly more disk space but is more compatible with HuggingFace ecosystem
- You can delete intermediate conversion outputs after verifying finetuning works
- For multi-GPU training, only run conversion on rank 0 (or outside of torchrun)