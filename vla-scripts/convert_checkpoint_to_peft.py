#!/usr/bin/env python3
"""
Convert Prismatic .pt checkpoint (with LoRA weights) to HuggingFace PEFT format.

This script:
1. Loads base VLM in HuggingFace format
2. Applies LoRA configuration matching training setup
3. Loads LoRA weights from .pt checkpoint
4. Saves in HuggingFace PEFT format

Usage:
    python convert_checkpoint_to_peft.py \
        --checkpoint_path /path/to/step-004000-epoch-00-loss=0.5093.pt \
        --base_model_path /ssd1/UniVLA/hf_models/prism-dinosiglip-224px+7b \
        --output_dir /ssd1/UniVLA/lora_adapters/step-004000 \
        --lora_rank 32 \
        --lora_dropout 0.0 \
        --lora_target attn \
        --lora_vision True \
        --lora_vision_target attn_mlp
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class ConvertConfig:
    # Checkpoint to convert
    checkpoint_path: Path = Path("/ssd1/UniVLA/prism-dinosiglip-224px+mx-libero+n0+b16+x42--image_aug+lora-r32+dropout-0.0+vlora-attn_mlp-LIBERO-Latent-Action-Pretraining-ws-16/checkpoints/step-004000-epoch-00-loss=0.5093.pt")

    # Base model in HF format (without LoRA)
    base_model_path: Path = Path("/ssd1/UniVLA/hf_models/prism-dinosiglip-224px+7b")

    # Output directory for PEFT format
    output_dir: Path = Path("/ssd1/UniVLA/lora_adapters/step-004000")

    # LoRA Configuration (must match training config)
    lora_rank: int = 32
    lora_dropout: float = 0.0
    lora_target: str = "attn"  # "attn" or "attn_mlp"

    # Vision LoRA
    lora_vision: bool = True
    lora_vision_target: str = "attn_mlp"  # "attn" or "attn_mlp"


@draccus.wrap()
def convert_checkpoint(cfg: ConvertConfig) -> None:
    """Convert .pt checkpoint to HuggingFace PEFT format."""
    print("=" * 80)
    print("Converting Prismatic .pt Checkpoint to HuggingFace PEFT Format")
    print("=" * 80)
    print(f"Checkpoint: {cfg.checkpoint_path}")
    print(f"Base model: {cfg.base_model_path}")
    print(f"Output dir: {cfg.output_dir}")
    print(f"LoRA config: rank={cfg.lora_rank}, dropout={cfg.lora_dropout}, target={cfg.lora_target}")
    print(f"Vision LoRA: {cfg.lora_vision}, target={cfg.lora_vision_target}")
    print()

    # Create output directory
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Register OpenVLA model classes
    print("Registering OpenVLA classes...")
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load base model
    print(f"Loading base model from {cfg.base_model_path}...")
    base_model = AutoModelForVision2Seq.from_pretrained(
        str(cfg.base_model_path),
        torch_dtype=torch.float32,  # Load in FP32 for PEFT
        trust_remote_code=True,
    )

    print(f"Base model loaded: {type(base_model)}")

    # Load checkpoint
    print(f"\nLoading checkpoint from {cfg.checkpoint_path}...")
    checkpoint = torch.load(cfg.checkpoint_path, map_location="cpu")

    print(f"Checkpoint keys: {checkpoint.keys()}")
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
        print(f"Model state_dict keys: {state_dict.keys()}")
    else:
        raise ValueError("Checkpoint does not contain 'model' key!")

    # === Apply LoRA to LLM ===
    print("\n" + "=" * 80)
    print("Applying LoRA to LLM Backbone")
    print("=" * 80)

    if cfg.lora_target == "attn":
        target_modules_llm = ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif cfg.lora_target == "attn_mlp":
        target_modules_llm = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"]
    else:
        raise ValueError(f"Unsupported lora_target={cfg.lora_target}")

    lora_config_llm = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=min(cfg.lora_rank, 16),
        lora_dropout=cfg.lora_dropout,
        target_modules=target_modules_llm,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    print(f"Applying PEFT to language_model with target modules: {target_modules_llm}")
    base_model.language_model = get_peft_model(base_model.language_model, lora_config_llm)
    print("LLM LoRA applied successfully")
    base_model.language_model.print_trainable_parameters()

    # Load LLM LoRA weights from checkpoint
    print("\nLoading LLM LoRA weights from checkpoint...")
    if "llm_backbone" in state_dict:
        llm_state = state_dict["llm_backbone"]
        # Remove "llm." prefix if present (from DDP wrapper)
        llm_state_cleaned = {}
        for k, v in llm_state.items():
            if k.startswith("llm."):
                llm_state_cleaned[k[4:]] = v  # Remove "llm." prefix
            else:
                llm_state_cleaned[k] = v

        # Load into language_model
        missing, unexpected = base_model.language_model.load_state_dict(llm_state_cleaned, strict=False)
        print(f"  Missing keys: {len(missing)} (expected for non-LoRA params)")
        print(f"  Unexpected keys: {len(unexpected)}")
        if unexpected:
            print(f"    Unexpected: {unexpected[:5]}...")
    else:
        print("  ⚠️  Warning: 'llm_backbone' not found in checkpoint!")

    # === Apply LoRA to Vision Backbone ===
    if cfg.lora_vision:
        print("\n" + "=" * 80)
        print("Applying LoRA to Vision Backbone")
        print("=" * 80)

        # Extract vision modules from base model
        def _extract_vision_target_modules(vision_backbone, target_type):
            """Extract attention/MLP module names from vision backbone."""
            attention_modules = []
            mlp_modules = []

            for name, module in vision_backbone.named_modules():
                # Skip patch_embed
                if 'patch_embed' in name:
                    continue

                # Check for transformer blocks
                if 'blocks' in name or 'layers' in name:
                    if 'attn' in name and not ('norm' in name or 'drop' in name):
                        if hasattr(module, 'weight') and len(module.weight.shape) == 2:
                            attention_modules.append(name)
                    elif 'mlp' in name and not ('norm' in name or 'drop' in name):
                        if hasattr(module, 'weight') and len(module.weight.shape) == 2:
                            mlp_modules.append(name)

            return attention_modules, mlp_modules

        # Handle DinoSigLIP (dual featurizers) or single featurizer
        vision_backbone = base_model.vision_backbone
        target_modules_vision = []

        if hasattr(vision_backbone, 'dino_featurizer') and hasattr(vision_backbone, 'siglip_featurizer'):
            print("Detected DinoSigLIP backbone with dual featurizers")

            # Extract from both featurizers
            dino_attn, dino_mlp = _extract_vision_target_modules(vision_backbone.dino_featurizer, "DINOv2")
            siglip_attn, siglip_mlp = _extract_vision_target_modules(vision_backbone.siglip_featurizer, "SigLIP")

            if cfg.lora_vision_target == "attn":
                target_modules_vision = list(set(dino_attn + siglip_attn))
            elif cfg.lora_vision_target == "attn_mlp":
                target_modules_vision = list(set(dino_attn + dino_mlp + siglip_attn + siglip_mlp))

            print(f"  DINOv2 attention modules: {len(dino_attn)}")
            print(f"  DINOv2 MLP modules: {len(dino_mlp)}")
            print(f"  SigLIP attention modules: {len(siglip_attn)}")
            print(f"  SigLIP MLP modules: {len(siglip_mlp)}")

        elif hasattr(vision_backbone, 'featurizer'):
            print("Detected single featurizer backbone")
            attn_modules, mlp_modules = _extract_vision_target_modules(vision_backbone.featurizer, "Vision")

            if cfg.lora_vision_target == "attn":
                target_modules_vision = attn_modules
            elif cfg.lora_vision_target == "attn_mlp":
                target_modules_vision = attn_modules + mlp_modules

        else:
            raise ValueError("Could not identify vision backbone structure!")

        print(f"\nTotal vision LoRA targets: {len(target_modules_vision)}")
        print(f"Sample targets: {target_modules_vision[:5]}...")

        # Create Vision LoRA config
        vision_lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules=target_modules_vision,
            task_type=TaskType.FEATURE_EXTRACTION,
            bias="none",
        )

        # Apply to featurizers
        if hasattr(vision_backbone, 'dino_featurizer') and hasattr(vision_backbone, 'siglip_featurizer'):
            print("\nApplying LoRA to DINOv2 featurizer...")
            base_model.vision_backbone.dino_featurizer = get_peft_model(
                base_model.vision_backbone.dino_featurizer, vision_lora_config
            )
            base_model.vision_backbone.dino_featurizer.print_trainable_parameters()

            print("\nApplying LoRA to SigLIP featurizer...")
            base_model.vision_backbone.siglip_featurizer = get_peft_model(
                base_model.vision_backbone.siglip_featurizer, vision_lora_config
            )
            base_model.vision_backbone.siglip_featurizer.print_trainable_parameters()

        elif hasattr(vision_backbone, 'featurizer'):
            print("\nApplying LoRA to featurizer...")
            base_model.vision_backbone.featurizer = get_peft_model(
                base_model.vision_backbone.featurizer, vision_lora_config
            )
            base_model.vision_backbone.featurizer.print_trainable_parameters()

        # Load Vision LoRA weights from checkpoint
        print("\nLoading Vision LoRA weights from checkpoint...")
        if "vision_backbone" in state_dict:
            vision_state = state_dict["vision_backbone"]

            # Separate dino and siglip states
            if hasattr(vision_backbone, 'dino_featurizer'):
                dino_state = {k[len("dino_featurizer."):]: v for k, v in vision_state.items()
                              if k.startswith("dino_featurizer.")}
                siglip_state = {k[len("siglip_featurizer."):]: v for k, v in vision_state.items()
                                if k.startswith("siglip_featurizer.")}

                print(f"  Loading DINOv2 weights: {len(dino_state)} keys")
                missing_dino, unexpected_dino = base_model.vision_backbone.dino_featurizer.load_state_dict(
                    dino_state, strict=False
                )
                print(f"    Missing: {len(missing_dino)}, Unexpected: {len(unexpected_dino)}")

                print(f"  Loading SigLIP weights: {len(siglip_state)} keys")
                missing_siglip, unexpected_siglip = base_model.vision_backbone.siglip_featurizer.load_state_dict(
                    siglip_state, strict=False
                )
                print(f"    Missing: {len(missing_siglip)}, Unexpected: {len(unexpected_siglip)}")

            elif hasattr(vision_backbone, 'featurizer'):
                featurizer_state = {k[len("featurizer."):]: v for k, v in vision_state.items()
                                   if k.startswith("featurizer.")}
                print(f"  Loading featurizer weights: {len(featurizer_state)} keys")
                missing, unexpected = base_model.vision_backbone.featurizer.load_state_dict(
                    featurizer_state, strict=False
                )
                print(f"    Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        else:
            print("  ⚠️  Warning: 'vision_backbone' not found in checkpoint!")

    # === Load Projector weights (always trainable) ===
    print("\n" + "=" * 80)
    print("Loading Projector Weights")
    print("=" * 80)
    if "projector" in state_dict:
        projector_state = state_dict["projector"]
        missing, unexpected = base_model.projector.load_state_dict(projector_state, strict=True)
        print(f"  Projector weights loaded successfully")
        print(f"    Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    else:
        print("  ⚠️  Warning: 'projector' not found in checkpoint!")

    # === Save in PEFT format ===
    print("\n" + "=" * 80)
    print(f"Saving PEFT Model to {cfg.output_dir}")
    print("=" * 80)

    # Save LLM LoRA adapters
    print("Saving LLM LoRA adapters...")
    llm_adapter_dir = cfg.output_dir / "llm_lora"
    os.makedirs(llm_adapter_dir, exist_ok=True)
    base_model.language_model.save_pretrained(llm_adapter_dir)
    print(f"  ✅ Saved to {llm_adapter_dir}")

    # Save Vision LoRA adapters
    if cfg.lora_vision:
        print("Saving Vision LoRA adapters...")
        if hasattr(base_model.vision_backbone, 'dino_featurizer'):
            dino_adapter_dir = cfg.output_dir / "vision_lora" / "dino"
            os.makedirs(dino_adapter_dir, exist_ok=True)
            base_model.vision_backbone.dino_featurizer.save_pretrained(dino_adapter_dir)
            print(f"  ✅ DINOv2 saved to {dino_adapter_dir}")

            siglip_adapter_dir = cfg.output_dir / "vision_lora" / "siglip"
            os.makedirs(siglip_adapter_dir, exist_ok=True)
            base_model.vision_backbone.siglip_featurizer.save_pretrained(siglip_adapter_dir)
            print(f"  ✅ SigLIP saved to {siglip_adapter_dir}")

        elif hasattr(base_model.vision_backbone, 'featurizer'):
            vision_adapter_dir = cfg.output_dir / "vision_lora"
            os.makedirs(vision_adapter_dir, exist_ok=True)
            base_model.vision_backbone.featurizer.save_pretrained(vision_adapter_dir)
            print(f"  ✅ Vision saved to {vision_adapter_dir}")

    # Save projector weights separately
    print("Saving projector weights...")
    projector_path = cfg.output_dir / "projector.pt"
    torch.save(base_model.projector.state_dict(), projector_path)
    print(f"  ✅ Saved to {projector_path}")

    # Save conversion metadata
    metadata = {
        "original_checkpoint": str(cfg.checkpoint_path),
        "base_model": str(cfg.base_model_path),
        "lora_rank": cfg.lora_rank,
        "lora_dropout": cfg.lora_dropout,
        "lora_target": cfg.lora_target,
        "lora_vision": cfg.lora_vision,
        "lora_vision_target": cfg.lora_vision_target,
    }

    import json
    with open(cfg.output_dir / "conversion_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 80)
    print("✅ Conversion Complete!")
    print("=" * 80)
    print(f"PEFT adapters saved to: {cfg.output_dir}")
    print(f"\nFor finetuning, use:")
    print(f"  VLA_BASE_PATH=\"{cfg.base_model_path}\"")
    print(f"  LORA_PRETRAINED_PATH=\"{cfg.output_dir}\"")


if __name__ == "__main__":
    convert_checkpoint()