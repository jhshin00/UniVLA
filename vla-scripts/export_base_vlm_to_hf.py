#!/usr/bin/env python3
"""
Export Prismatic VLM to HuggingFace format for PEFT compatibility.

Usage:
    python export_base_vlm_to_hf.py --model_id prism-dinosiglip-224px+7b \
                                     --output_dir /ssd1/UniVLA/hf_models/prism-dinosiglip-224px+7b
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models import load

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class ExportConfig:
    # Model to export
    model_id: str = "prism-dinosiglip-224px+7b"

    # Output directory for HF format model
    output_dir: Path = Path("/ssd1/UniVLA/hf_models/prism-dinosiglip-224px+7b")

    # HuggingFace settings
    hf_token: Optional[str] = None
    hf_cache_dir: Path = Path("ssd2/hf_cache")


@draccus.wrap()
def export_to_hf(cfg: ExportConfig) -> None:
    """Export Prismatic VLM to HuggingFace format."""
    print(f"Exporting Prismatic VLM `{cfg.model_id}` to HuggingFace format...")
    print(f"Output directory: {cfg.output_dir}")

    # Create output directory
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Setup HF token and cache
    hf_token = cfg.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    os.environ.setdefault("HF_HOME", str(cfg.hf_cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cfg.hf_cache_dir))

    # Register OpenVLA model classes
    print("Registering OpenVLA classes with HuggingFace AutoClasses...")
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load base VLM using Prismatic's native loader
    print(f"Loading base VLM `{cfg.model_id}` using Prismatic loader...")
    vlm = load(
        cfg.model_id,
        hf_token=hf_token,
        load_for_training=False,  # Load in eval mode
        cache_dir=str(cfg.hf_cache_dir)
    )

    # Move to CPU to save memory during export
    print("Moving model to CPU for export...")
    vlm = vlm.cpu()

    # Convert to HuggingFace OpenVLA model
    print("Converting to HuggingFace format...")

    # Create OpenVLA config
    hf_config = OpenVLAConfig(
        # Vision backbone config
        vision_model_id=vlm.vision_backbone.identifier,
        image_sizes=vlm.vision_backbone.default_image_resolution,

        # LLM backbone config
        llm_backbone_id=vlm.llm_backbone.identifier,
        vocab_size=len(vlm.llm_backbone.tokenizer),

        # Architecture
        arch_specifier=vlm.arch_specifier,

        # Inference settings
        pad_token_id=vlm.llm_backbone.tokenizer.pad_token_id,
        eos_token_id=vlm.llm_backbone.tokenizer.eos_token_id,
    )

    # Create HF model instance
    hf_model = OpenVLAForActionPrediction(hf_config)

    # Transfer weights from Prismatic model to HF model
    print("Transferring weights...")

    # Vision backbone
    hf_model.vision_backbone.load_state_dict(vlm.vision_backbone.state_dict())

    # Projector
    hf_model.projector.load_state_dict(vlm.projector.state_dict())

    # LLM backbone
    hf_model.language_model.load_state_dict(vlm.llm_backbone.llm.state_dict())

    # Save model in HF format
    print(f"Saving model to {cfg.output_dir}...")
    hf_model.save_pretrained(cfg.output_dir, safe_serialization=True)

    # Create and save processor
    print("Creating and saving processor...")
    processor = PrismaticProcessor(
        image_processor=vlm.vision_backbone.image_transform,
        tokenizer=vlm.llm_backbone.tokenizer,
        image_size=vlm.vision_backbone.default_image_resolution,
    )

    # Save tokenizer
    vlm.llm_backbone.tokenizer.save_pretrained(cfg.output_dir)

    # Save image processor config
    image_processor_config = {
        "image_mean": vlm.vision_backbone.get_image_transform().transforms[0].mean.tolist()
                      if hasattr(vlm.vision_backbone.get_image_transform().transforms[0], 'mean')
                      else [0.5, 0.5, 0.5],
        "image_std": vlm.vision_backbone.get_image_transform().transforms[0].std.tolist()
                     if hasattr(vlm.vision_backbone.get_image_transform().transforms[0], 'std')
                     else [0.5, 0.5, 0.5],
        "size": {"height": vlm.vision_backbone.default_image_resolution[1],
                 "width": vlm.vision_backbone.default_image_resolution[2]},
        "do_normalize": True,
        "do_resize": True,
    }

    import json
    with open(cfg.output_dir / "preprocessor_config.json", "w") as f:
        json.dump(image_processor_config, f, indent=2)

    print(f"\n✅ Successfully exported model to {cfg.output_dir}")
    print(f"Model can now be loaded with:")
    print(f"  model = AutoModelForVision2Seq.from_pretrained('{cfg.output_dir}')")
    print(f"  processor = AutoProcessor.from_pretrained('{cfg.output_dir}')")


if __name__ == "__main__":
    export_to_hf()