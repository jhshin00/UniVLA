"""
ddp.py

Core class definition for a strategy implementing Torch native Distributed Data Parallel Training; note that on most
GPU hardware and LLM backbones >= 5-7B parameters, DDP training will OOM, which is why we opt for FSDP.
"""

import shutil
from pathlib import Path
from typing import Optional

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from transformers.optimization import get_constant_schedule, get_cosine_schedule_with_warmup

from prismatic.overwatch import initialize_overwatch
from prismatic.training.strategies.base_strategy import TrainingStrategy

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class DDPStrategy(TrainingStrategy):
    @overwatch.rank_zero_only
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None:
        """Save a checkpoint to the `run_dir` only containing the state_dicts for trainable parameters by default."""
        assert isinstance(self.vlm, DDP), "save_checkpoint assumes VLM is already wrapped in DDP!"

        # Splinter State Dictionary by Top-Level Submodules (or subset, if `only_trainable`)
        model_state_dicts = {
            mkey: getattr(self.vlm.module, mkey).state_dict()
            for mkey in (self.trainable_module_keys if only_trainable else self.all_module_keys)
        }
        optimizer_state_dict = self.optimizer.state_dict()

        # Set Checkpoint Path =>> Embed *minimal* training statistics!
        checkpoint_dir = run_dir / "checkpoints"
        if train_loss is None:
            checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss=inf.pt"
        else:
            checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss={train_loss:.4f}.pt"

        # Save Checkpoint & Copy Latest to `latest-checkpoint.pt`
        torch.save({"model": model_state_dicts, "optimizer": optimizer_state_dict}, checkpoint_path)
        shutil.copy(checkpoint_path, checkpoint_dir / "latest-checkpoint.pt")

        # TODO jslee add
        # === Additional PEFT Format Save for LoRA Training ===
        # Check if model has LoRA adapters and save in HuggingFace PEFT format for easy loading
        self._save_peft_adapters_if_present(run_dir, global_step, epoch, train_loss)

    # TODO jslee add 
    def _save_peft_adapters_if_present(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
    ) -> None:
        """Save LoRA adapters in HuggingFace PEFT format if present."""
        vlm = self.vlm.module  # Unwrap DDP

        # Check if LLM has LoRA adapters (PEFT wrapper)
        has_llm_lora = hasattr(vlm.llm_backbone.llm, 'peft_config')
        has_vision_lora = False

        # Check if Vision backbone has LoRA adapters
        if hasattr(vlm.vision_backbone, 'dino_featurizer'):
            has_vision_lora = hasattr(vlm.vision_backbone.dino_featurizer, 'peft_config')
        elif hasattr(vlm.vision_backbone, 'featurizer'):
            has_vision_lora = hasattr(vlm.vision_backbone.featurizer, 'peft_config')

        if not (has_llm_lora or has_vision_lora):
            # No LoRA adapters to save
            return

        # Create PEFT adapter directory
        if train_loss is None:
            adapter_dir_name = f"peft-step-{global_step:06d}-epoch-{epoch:02d}-loss=inf"
        else:
            adapter_dir_name = f"peft-step-{global_step:06d}-epoch-{epoch:02d}-loss={train_loss:.4f}"

        adapter_dir = run_dir / "checkpoints" / adapter_dir_name
        adapter_dir.mkdir(parents=True, exist_ok=True)

        overwatch.info(f"Saving PEFT adapters to {adapter_dir}")

        # Save LLM LoRA adapters
        if has_llm_lora:
            llm_adapter_dir = adapter_dir / "llm_lora"
            llm_adapter_dir.mkdir(exist_ok=True)
            vlm.llm_backbone.llm.save_pretrained(llm_adapter_dir)
            overwatch.info(f"  ✅ LLM LoRA saved to {llm_adapter_dir}")

        # Save Vision LoRA adapters
        if has_vision_lora:
            vision_lora_dir = adapter_dir / "vision_lora"
            vision_lora_dir.mkdir(exist_ok=True)

            if hasattr(vlm.vision_backbone, 'dino_featurizer') and hasattr(vlm.vision_backbone, 'siglip_featurizer'):
                # Dual featurizers (DinoSigLIP)
                dino_dir = vision_lora_dir / "dino"
                siglip_dir = vision_lora_dir / "siglip"
                dino_dir.mkdir(exist_ok=True)
                siglip_dir.mkdir(exist_ok=True)

                if hasattr(vlm.vision_backbone.dino_featurizer, 'peft_config'):
                    vlm.vision_backbone.dino_featurizer.save_pretrained(dino_dir)
                    overwatch.info(f"  ✅ DINOv2 LoRA saved to {dino_dir}")

                if hasattr(vlm.vision_backbone.siglip_featurizer, 'peft_config'):
                    vlm.vision_backbone.siglip_featurizer.save_pretrained(siglip_dir)
                    overwatch.info(f"  ✅ SigLIP LoRA saved to {siglip_dir}")

            elif hasattr(vlm.vision_backbone, 'featurizer'):
                # Single featurizer
                if hasattr(vlm.vision_backbone.featurizer, 'peft_config'):
                    vlm.vision_backbone.featurizer.save_pretrained(vision_lora_dir)
                    overwatch.info(f"  ✅ Vision LoRA saved to {vision_lora_dir}")

        # Save projector weights (always trainable in LoRA training)
        projector_path = adapter_dir / "projector.pt"
        torch.save(vlm.projector.state_dict(), projector_path)
        overwatch.info(f"  ✅ Projector saved to {projector_path}")

        # Create symlink to latest PEFT checkpoint
        latest_link = run_dir / "checkpoints" / "latest-peft"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(adapter_dir_name, target_is_directory=True)

        overwatch.info(f"PEFT adapters saved successfully to {adapter_dir}")

    def run_setup(self, run_dir: Path, n_train_examples: int) -> None:
        # Gradient Checkpointing Setup
        if self.enable_gradient_checkpointing:
            # For Gradient Checkpointing --> we make the assumption that the "bulk" of activation memory is taken up
            #     by the LLM; because we also make the explicit assumption that each LLM is derived from a HF
            #     pretrained model, the only thing we *need* to do (technically) is call `gradient_checkpoint_enable`
            #     on `self.llm_backbone`.
            #
            # What does it actually do? --> runs the *generic* custom_forward + torch.utils.checkpoint.checkpoint logic
            #   => github.com/huggingface/transformers/.../models/llama/modeling_llama.py#L692-L706
            #
            # Additional Reference (to better understand gradient checkpointing in PyTorch writ large)
            #   => github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb
            overwatch.info("Enabling Gradient Checkpointing on LLM Backbone", ctx_level=1)
            self.vlm.llm_backbone.gradient_checkpointing_enable()

        # Move to Device =>> Note parameters are in full precision (*mixed precision* will only autocast as appropriate)
        overwatch.info("Placing Entire VLM (Vision Backbone, LLM Backbone, Projector Weights) on GPU", ctx_level=1)
        self.vlm.to(self.device_id)

        # Wrap with Distributed Data Parallel
        #   => Note: By default, wrapping naively with DDP(self.vlm) will initialize a *separate* buffer on GPU that
        #            is the same size/dtype as the model parameters; this will *double* GPU memory!
        # - stackoverflow.com/questions/68949954/model-takes-twice-the-memory-footprint-with-distributed-data-parallel
        overwatch.info("Wrapping VLM with Distributed Data Parallel", ctx_level=1)
        self.vlm = DDP(self.vlm, device_ids=[self.device_id], gradient_as_bucket_view=True)

        # Create Optimizer and LR Scheduler =>> note that most of the LR Schedulers we use require `max_steps/epochs`
        #   => Optimizer should only operate on parameters that are *unfrozen* / trainable!
        trainable_params = [param for param in self.vlm.parameters() if param.requires_grad]
        if self.max_steps is None:
            num_training_steps = (n_train_examples * self.epochs) // self.global_batch_size
        else:
            num_training_steps = self.max_steps

        if self.lr_scheduler_type == "linear-warmup+cosine-decay":
            # Set warmup steps (floor) based on `warmup_ratio` (should be 0.03 - 0.05)
            num_warmup_steps = int(num_training_steps * self.warmup_ratio)

            assert self.weight_decay == 0, "DDP training does not currently support `weight_decay` > 0!"
            self.optimizer = AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
            self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = 0.0

        elif self.lr_scheduler_type == "constant":
            num_warmup_steps = 0

            assert self.weight_decay == 0, "DDP training does not currently support `weight_decay` > 0!"
            self.optimizer = AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
            self.lr_scheduler = get_constant_schedule(self.optimizer)

        else:
            raise ValueError(f"Learning Rate Schedule with type `{self.lr_scheduler_type}` is not supported!")

        # Finalize Setup =>> Log
        overwatch.info(
            "DDP Strategy =>> Finalized Training Setup:\n"
            f"         |-> Global (Effective) Batch Size = {self.global_batch_size}\n"
            f"         |-> Per-Device Batch Size = {self.per_device_batch_size}\n"
            f"         |-> Distributed World Size = {overwatch.world_size()}\n"
            f"         |-> Gradient Accumulation Steps = {self.grad_accumulation_steps}\n\n"
            f"         |-> LLM Backbone Gradient Checkpointing = {self.enable_gradient_checkpointing}\n"
            f"         |-> Use Native AMP = {self.enable_mixed_precision_training} ({self.mixed_precision_dtype})\n\n"
            f"         |-> Default AdamW LR = {self.learning_rate}\n"
            f"         |-> AdamW Weight Decay = {self.weight_decay}\n"
            f"         |-> LR Scheduler Type = {self.lr_scheduler_type}\n"
            f"         |-> LR Scheduler Warmup Steps (Ratio) = {num_warmup_steps} ({self.warmup_ratio})\n"
            f"         |-> Dataset Size = {n_train_examples} Examples\n"
            f"         |-> Max Steps = {num_training_steps}\n"
        )

    def clip_grad_norm(self) -> None:
        torch.nn.utils.clip_grad_norm_(self.vlm.parameters(), max_norm=self.max_grad_norm)
