import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import PeftModel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers import AutoConfig, AutoImageProcessor

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction_LIBERO
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransformLIBERO_withHis, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from prismatic.models.policy.transformer_utils import MAPBlock

class ActionDecoder(torch.nn.Module):
    def __init__(self, window_size = 12, hidden_dim = 512):
        super().__init__()
        self.latent_action_pool = MAPBlock(n_latents = 1, vis_dim = 4096, embed_dim = hidden_dim, n_heads = hidden_dim // 64)
        self.visual_pool = MAPBlock(n_latents = 1, vis_dim = 4096, embed_dim = hidden_dim, n_heads = hidden_dim // 64)

        self.proj = nn.Sequential(
                                nn.Linear(hidden_dim, 7 * window_size),
                                nn.Tanh(),
                    )

    def forward(self, latent_action_tokens, visual_embed):
        visual_embed = self.visual_pool(visual_embed)
        latent_action_tokens = latent_action_tokens[:, -4:]
        action_token = self.latent_action_pool(latent_action_tokens, init_embed = visual_embed)

        action = self.proj(action_token)

        return action

class Wrapped_Model(torch.nn.Module):
    def __init__(self, vla, freeze_vla = False, window_size = 12):
        super().__init__()
        self.vla = vla
        self.window_size = window_size
        self.action_decoder = ActionDecoder(window_size=window_size)

        if freeze_vla:
            self.vla.requires_grad_(False)

    def forward(self, batch):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vla_output = self.vla(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
                output_hidden_states = True,        # Return intermediate tokens of all layers
            )
        loss, loss_one_step, latent_action_tokens = self.action_decoder_forward(batch, vla_output)

        return vla_output, loss, loss_one_step, latent_action_tokens

    def action_decoder_forward(self, batch, vla_output):
        visual_embed = vla_output.hidden_states[-1][:, : self.vla.vision_backbone.featurizer.patch_embed.num_patches ].to(torch.float)
        latent_tokens = vla_output.hidden_states[-1][:, self.vla.vision_backbone.featurizer.patch_embed.num_patches : ]
        action_gt = batch["labels"].to(latent_tokens.device)
        mask = action_gt > 32000

        latent_action_tokens = []
        for idx, per_sample_latent_tokens in enumerate(latent_tokens):
            per_sample_latent_action_tokens = per_sample_latent_tokens[mask[idx], :]
            latent_action_tokens.append(per_sample_latent_action_tokens)
        latent_action_tokens = torch.stack(latent_action_tokens).to(torch.float)

        pred_action = self.action_decoder(latent_action_tokens, visual_embed).reshape(-1, self.window_size, 7)
        loss = torch.nn.functional.l1_loss(pred_action, batch['actions'], reduction='none')
        loss_one_step = loss[:,0].mean()
        loss = loss.mean()

        return loss, loss_one_step, latent_action_tokens



@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "/path/to/your/base-vla"                        # Path to base VLA model (without LoRA)
    lora_pretrained_path: str = "/path/to/your/lora-checkpoint"     # Path to LoRA pre-trained checkpoint directory
    lam_path: str = "latent_action_model/logs/task_centric_lam_stage2/epoch=0-step=200000.ckpt"

    # Directory Paths
    data_root_dir: Path = Path("/LIBERO/modified_libero_rlds")      # Path to Open-X dataset directory
    dataset_name: str = "libero_spatial_no_noops"                   # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # LoRA Fine-tuning Settings
    freeze_base_model: bool = True                                  # Whether to freeze base VLA parameters
    lora_only_training: bool = True                                 # Whether to train only LoRA adapters

    # Fine-tuning Parameters
    batch_size: int = 8                                             # Fine-tuning batch size
    max_steps: int = 30000                                          # Max number of fine-tuning steps
    save_steps: int = 30000                                         # Interval for checkpoint saving
    learning_rate: float = 1e-4                                     # Fine-tuning learning rate (lower for LoRA)
    grad_accumulation_steps: int = 2                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 16000                                # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)
    # LAM setting
    codebook_size: int = 16
    lam_model_dim: int = 768
    lam_latent_dim: int = 128
    lam_patch_size: int = 14
    lam_enc_blocks: int = 12
    lam_dec_blocks: int = 12
    lam_num_heads: int = 12
    window_size: int = 12

    # LoRA Arguments (should match pre-trained LoRA config)
    freeze_vla: bool = False                                        # We'll use selective freezing instead
    use_lora: bool = True                                           # Always True for this script
    lora_rank: int = 32                                             # Should match pre-trained LoRA rank
    lora_dropout: float = 0.0                                       # Should match pre-trained LoRA dropout
    use_quantization: bool = False                                  # Not recommended for LoRA fine-tuning

    # Vision LoRA Arguments (should match jslee_train_lora.py config)
    lora_vision: bool = True                                        # Apply LoRA to Vision backbone
    lora_vision_target: str = "attn_mlp"                           # Vision LoRA target: "attn" or "attn_mlp"

    # Tracking Parameters
    wandb_project: str = "jslee-finetune-LIBERO"                    # Name of W&B project to log to
    wandb_entity: str = "jlee24"                                    # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases



@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning LoRA Model from `{cfg.lora_pretrained_path}` on `{cfg.dataset_name}`")
    print(f"Vision LoRA enabled: {cfg.lora_vision} (target: {cfg.lora_vision_target})")
    print(f"LoRA config: rank={cfg.lora_rank}, dropout={cfg.lora_dropout}")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Validation: Check Vision LoRA compatibility
    if cfg.lora_vision:
        assert cfg.lora_vision_target in ["attn", "attn_mlp"], \
            f"Invalid lora_vision_target='{cfg.lora_vision_target}'. Must be 'attn' or 'attn_mlp'"
        print(f"✅ Vision LoRA validation passed: target='{cfg.lora_vision_target}'")

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.lora_pretrained_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.lora_only_training:
        exp_id += "+lora-only"
    if cfg.freeze_base_model:
        exp_id += "+freeze-base"
    if cfg.lora_vision:
        exp_id += f"+vlora-{cfg.lora_vision_target}"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    exp_id += f'=w-LowLevelDecoder-ws-{cfg.window_size}'

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)

    print("Loading base VLA model...")
    # Load Base VLA Model (without LoRA)
    base_vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device_id)

    print("Loading pre-trained LoRA adapters...")

    # Check if using new PEFT format (with separate LLM and Vision LoRA)
    lora_path = Path(cfg.lora_pretrained_path)
    llm_lora_path = lora_path / "llm_lora"
    vision_lora_path = lora_path / "vision_lora"
    projector_path = lora_path / "projector.pt"

    if llm_lora_path.exists():
        print(f"  Detected new PEFT format at {cfg.lora_pretrained_path}")

        # Load LLM LoRA adapters
        print(f"  Loading LLM LoRA from {llm_lora_path}...")
        from peft import PeftModel
        base_vla.language_model = PeftModel.from_pretrained(base_vla.language_model, str(llm_lora_path))
        print("  ✅ LLM LoRA loaded")

        # Load Vision LoRA adapters if present
        if cfg.lora_vision and vision_lora_path.exists():
            print(f"  Loading Vision LoRA from {vision_lora_path}...")

            # Check for dual featurizers (DinoSigLIP) or single featurizer
            dino_lora_path = vision_lora_path / "dino"
            siglip_lora_path = vision_lora_path / "siglip"

            if dino_lora_path.exists() and siglip_lora_path.exists():
                print("    Loading DINOv2 LoRA...")
                base_vla.vision_backbone.dino_featurizer = PeftModel.from_pretrained(
                    base_vla.vision_backbone.dino_featurizer, str(dino_lora_path)
                )
                print("    ✅ DINOv2 LoRA loaded")

                print("    Loading SigLIP LoRA...")
                base_vla.vision_backbone.siglip_featurizer = PeftModel.from_pretrained(
                    base_vla.vision_backbone.siglip_featurizer, str(siglip_lora_path)
                )
                print("    ✅ SigLIP LoRA loaded")
            elif vision_lora_path.exists():
                print("    Loading single featurizer LoRA...")
                base_vla.vision_backbone.featurizer = PeftModel.from_pretrained(
                    base_vla.vision_backbone.featurizer, str(vision_lora_path)
                )
                print("    ✅ Vision LoRA loaded")

        # Load projector weights
        if projector_path.exists():
            print(f"  Loading projector from {projector_path}...")
            base_vla.projector.load_state_dict(torch.load(projector_path, map_location=device_id))
            print("  ✅ Projector loaded")

        vla = base_vla
    else:
        # Legacy format: single PEFT adapter (LLM only)
        print(f"  Detected legacy PEFT format at {cfg.lora_pretrained_path}")
        print(f"  Loading LLM LoRA adapters...")
        vla = PeftModel.from_pretrained(base_vla, cfg.lora_pretrained_path)
        print("  ⚠️  Note: Vision LoRA not loaded (legacy format)")

    # Verify Vision LoRA loading
    if cfg.lora_vision and distributed_state.is_main_process:
        print("\n🔍 Checking Vision LoRA loading...")
        vision_lora_modules = []
        vision_conv2d_lora_modules = []

        # Check for Vision LoRA modules
        for name, module in vla.named_modules():
            if 'vision_backbone' in name and (hasattr(module, 'lora_A') or 'lora_' in str(type(module))):
                vision_lora_modules.append(name)
                if 'patch_embed' in name:
                    vision_conv2d_lora_modules.append(name)
                    print(f"  ❌ CRITICAL: Conv2D LoRA detected: {name}")
                else:
                    print(f"  ✅ Vision LoRA loaded: {name}")

        print(f"Total Vision LoRA modules found: {len(vision_lora_modules)}")
        if vision_conv2d_lora_modules:
            print(f"⚠️  WARNING: {len(vision_conv2d_lora_modules)} Conv2D LoRA modules found!")
        else:
            print("✅ No Conv2D LoRA modules found")

    # Print model info
    if distributed_state.is_main_process:
        print("\n=== LoRA Model Information ===")
        vla.print_trainable_parameters()

    # Apply selective freezing for LoRA-only training
    if cfg.freeze_base_model and cfg.lora_only_training:
        print("\n🔧 Applying selective freezing strategy...")
        print("  - Base model parameters: FROZEN")
        print("  - LLM LoRA adapters: TRAINABLE")
        if cfg.lora_vision:
            print("  - Vision LoRA adapters: TRAINABLE")

        # Freeze all base model parameters
        for param in vla.base_model.parameters():
            param.requires_grad = False

        # Ensure all LoRA parameters (LLM + Vision) are trainable
        llm_lora_count = 0
        vision_lora_count = 0
        for name, param in vla.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
                if 'vision_backbone' in name:
                    vision_lora_count += param.numel()
                    if distributed_state.is_main_process:
                        print(f"  ✅ Vision LoRA trainable: {name}")
                else:
                    llm_lora_count += param.numel()
                    if distributed_state.is_main_process:
                        print(f"  ✅ LLM LoRA trainable: {name}")

        if distributed_state.is_main_process:
            print(f"\nLoRA Parameters Summary:")
            print(f"  - LLM LoRA parameters: {llm_lora_count:,}")
            if cfg.lora_vision:
                print(f"  - Vision LoRA parameters: {vision_lora_count:,}")
            print(f"  - Total LoRA parameters: {llm_lora_count + vision_lora_count:,}")

    # Create wrapped model with action decoder
    wrapped_model = Wrapped_Model(vla=vla, freeze_vla=cfg.freeze_vla, window_size=cfg.window_size).to(device_id)

    # Print comprehensive parameter information
    total_params = sum(p.numel() for p in wrapped_model.parameters())
    trainable_params = sum(p.numel() for p in wrapped_model.parameters() if p.requires_grad)

    # Break down by component
    vla_total = sum(p.numel() for p in wrapped_model.vla.parameters())
    vla_trainable = sum(p.numel() for p in wrapped_model.vla.parameters() if p.requires_grad)
    action_decoder_total = sum(p.numel() for p in wrapped_model.action_decoder.parameters())
    action_decoder_trainable = sum(p.numel() for p in wrapped_model.action_decoder.parameters() if p.requires_grad)

    if distributed_state.is_main_process:
        print(f"\n=== Final Parameter Information ===")
        print(f"VLA Model:")
        print(f"  - Total: {vla_total:,}")
        print(f"  - Trainable: {vla_trainable:,} ({100*vla_trainable/vla_total:.2f}%)")
        print(f"Action Decoder:")
        print(f"  - Total: {action_decoder_total:,}")
        print(f"  - Trainable: {action_decoder_trainable:,} ({100*action_decoder_trainable/action_decoder_total:.2f}%)")
        print(f"Overall:")
        print(f"  - Total Parameters: {total_params:,}")
        print(f"  - Trainable Parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

        # Verify Vision LoRA is being used if enabled
        if cfg.lora_vision:
            vision_trainable = sum(
                p.numel() for name, p in wrapped_model.vla.named_parameters()
                if p.requires_grad and 'vision_backbone' in name and 'lora_' in name
            )
            if vision_trainable > 0:
                print(f"  ✅ Vision LoRA active: {vision_trainable:,} trainable parameters")
            else:
                print(f"  ⚠️  WARNING: Vision LoRA enabled but no trainable Vision LoRA parameters found!")

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    wrapped_model = DDP(wrapped_model, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Create Optimizer =>> Only for trainable parameters
    trainable_params = [param for param in wrapped_model.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=1e-4)  # Lower weight decay for LoRA
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(cfg.max_steps * 0.8), gamma=0.1)

    # Load LAM model
    from latent_action_model.genie.modules.lam import ControllableDINOLatentActionModel

    latent_action_model = ControllableDINOLatentActionModel(
        in_dim=3,
        model_dim=cfg.lam_model_dim,
        latent_dim=cfg.lam_latent_dim,
        num_latents=cfg.codebook_size,
        patch_size=cfg.lam_patch_size,
        enc_blocks=cfg.lam_enc_blocks,
        dec_blocks=cfg.lam_dec_blocks,
        num_heads=cfg.lam_num_heads,
        dropout=0.,
    )

    lam_ckpt = torch.load(cfg.lam_path)['state_dict']
    new_ckpt = {}
    for key in lam_ckpt.keys():
        new_ckpt[key.replace("lam.", "")] = lam_ckpt[key]

    latent_action_model.load_state_dict(new_ckpt, strict=True)
    latent_action_model = latent_action_model.to(device_id).eval()

    batch_transform = RLDSBatchTransformLIBERO_withHis(
        latent_action_model,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        image_transform_lam=transforms.ToTensor(),
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
        window_size=cfg.window_size
    )


    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(wrapped_model.module.vla.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        window_size=cfg.window_size + 1,        # for constructing history latent actions
        training_phase='post-training',
    )

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction_LIBERO(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)

    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        wrapped_model.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            batch["input_ids"] = batch["input_ids"].to(device_id)
            batch["attention_mask"] = batch["attention_mask"].to(device_id)
            batch["labels"] = batch["labels"].to(device_id)
            batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16).to(device_id)
            batch['actions'] = batch['actions'].to(device_id)
            batch['latent_action_idx'] = batch['latent_action_idx'].to(device_id)

            # Forward pass
            output, act_loss, loss_one_step, _ = wrapped_model(batch)

            # For LoRA-only training, we might want to focus more on action loss
            if cfg.lora_only_training:
                loss = act_loss  # Focus on action decoder training
            else:
                loss = act_loss + output.loss

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps
            torch.nn.utils.clip_grad_norm_(wrapped_model.parameters(), max_norm=1.)

            # Backward pass
            normalized_loss.backward()

            # Compute Accuracy and L1 Loss for Logging
            action_logits = output.logits[:, wrapped_model.module.vla.vision_backbone.featurizer.patch_embed.num_patches : -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > 32000

            # Compute Accuracy
            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()

            # Store recent train metrics
            recent_losses.append(loss.item())
            recent_action_accuracies.append(action_accuracy.item())

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)

            # Push Metrics to W&B (every 5 gradient steps)
            if distributed_state.is_main_process and gradient_step_idx % 5 == 0:
                wandb.log(
                    {
                        "train_loss": smoothened_loss,
                        "latent_action_accuracy": smoothened_action_accuracy,
                        "action_loss": act_loss.item(),
                        "action_loss_1step": loss_one_step.item(),
                        "lr": optimizer.state_dict()['param_groups'][0]['lr'],
                    },
                    step=gradient_step_idx,
                )

            # Optimizer Step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                progress.update()

            # Save Model Checkpoint
            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                if distributed_state.is_main_process:
                    print(f"Saving LoRA Model Checkpoint for Step {gradient_step_idx}")

                    # Save LoRA adapters only
                    if cfg.lora_only_training:
                        adapter_save_dir = adapter_dir / f"step_{gradient_step_idx}"
                        os.makedirs(adapter_save_dir, exist_ok=True)

                        # Save LoRA adapters
                        wrapped_model.module.vla.save_pretrained(adapter_save_dir)
                        processor.save_pretrained(adapter_save_dir)

                        print(f"Saved LoRA adapters to: {adapter_save_dir}")

                    # Save action decoder
                    torch.save(
                        wrapped_model.module.action_decoder.state_dict(),
                        str(run_dir) + f'/action_decoder-{gradient_step_idx}.pt'
                    )

                # Wait for main process to save
                dist.barrier()

            # Stop training when max_steps is reached
            if gradient_step_idx == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()