import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import draccus
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
import yaml
from transformers import BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

from prismatic.conf import VLAConfig, VLARegistry
from prismatic.models import load, load_vla
from prismatic.overwatch import initialize_overwatch
from prismatic.training import VLAMetrics, get_train_strategy
from prismatic.util import set_global_seed
from prismatic.vla import get_latent_vla_dataset_and_collator
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from prismatic.vla.datasets import RLDSBatchTransformLIBERO_withHis, RLDSDataset
from prismatic.util.data_utils import PaddedCollatorForActionPrediction_LIBERO
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


@dataclass
class TrainConfig:
    # fmt: off

    # VLAConfig (`prismatic/conf/vla.py`); override with --vla.type `VLARegistry.<VLA>.vla_id`
    vla: VLAConfig = field(
        default_factory=VLAConfig.get_choice_class(VLARegistry.DINOSIGLIP_224PX_MX_LIBERO.vla_id)
    )
    pretrain_vlm: str = 'prism-dinosiglip-224px+7b'
    lam_path: str = "latent_action_model/logs/task_centric_lam_stage2/epoch=2-step=18000.ckpt"

    # LAM setting
    codebook_size: int = 16
    lam_model_dim: int = 768
    lam_latent_dim: int = 128
    lam_patch_size: int = 14
    lam_enc_blocks: int = 12
    lam_dec_blocks: int = 12
    lam_num_heads: int = 12

    # Directory Paths
    data_root_dir: Path = Path(                                     # Path to Open-X dataset directory
        "/ssd1/openpi_official/datasets/libero_raw"
    )
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Resume Run Parameters
    pretrained_checkpoint: Optional[Path] = None                    # Absolute Path to Checkpoint
    is_resume: bool = True                                          # Whether we are continuing a prior training run
                                                                    #   (only applicable given pretrained checkpoint)
    resume_step: Optional[int] = None                               # Global Step to Resume (should match checkpoint)
    resume_epoch: Optional[int] = None                              # Epoch to Resume (should match checkpoint)

    # Run Arguments
    run_id: Optional[str] = None                                    # Run ID for logging, Weights & Biases
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases
    save_interval: int = 5000                                      # Interval for saving checkpoints (in steps)
    image_aug: bool = True                                          # Whether to enable image augmentations
    seed: int = 42                                                  # Random seed (for reproducibility)

    # HF Hub Credentials (for any gated models)
    hf_token: Union[str, Path] = ''                

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # LIBERO Dataset Settings
    dataset_name: str = "libero_combined"                           # LIBERO dataset name
    window_size: int = 12                                           # Window size for LIBERO training
    shuffle_buffer_size: int = 16000                                # Dataloader shuffle buffer size

    # Tracking Parameters
    trackers: Tuple[str, ...] = ("jsonl", "wandb")                  # Trackers to initialize (if W&B, add config!)
    wandb_project: str = "latent-action-pretrain-libero"            # Name of W&B project to log to (use default!)
    wandb_entity: str = "opendrivelab"                              # Name of entity to log under

    def __post_init__(self) -> None:
        """Lift optimization parameters from `self.vla` for ease of use =>> validate on `expected_world_size`"""
        self.epochs = self.vla.epochs
        self.max_steps = self.vla.max_steps
        self.global_batch_size = self.vla.global_batch_size
        self.per_device_batch_size = self.vla.per_device_batch_size

        self.learning_rate = self.vla.learning_rate
        self.weight_decay = self.vla.weight_decay
        self.max_grad_norm = self.vla.max_grad_norm
        self.lr_scheduler_type = self.vla.lr_scheduler_type
        self.warmup_ratio = self.vla.warmup_ratio

        self.train_strategy = self.vla.train_strategy

        # [Validate] Assert on `expected_world_size`
        assert (
            self.vla.expected_world_size == overwatch.world_size()
        ), f"Expected World Size = {self.vla.expected_world_size} but Found {overwatch.world_size()} GPUs!"

    # fmt: on


def _run_lora_training_loop(
    train_strategy, vla_dataset, collator, action_tokenizer, metrics, 
    save_interval, vlm, lora_config, adapter_dir, run_dir
):
    """Custom training loop with LoRA checkpoint saving support."""
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    
    # Create DataLoader
    dataloader = DataLoader(
        vla_dataset,
        batch_size=train_strategy.per_device_batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )
    
    # Training loop
    with tqdm(total=train_strategy.max_steps, leave=False) as progress:
        train_strategy.vlm.train()
        train_strategy.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(train_strategy.device_id) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = train_strategy.vlm(**batch)
                loss = output.loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_strategy.vlm.parameters(), max_norm=1.0)
            
            # Optimizer step
            train_strategy.optimizer.step()
            train_strategy.lr_scheduler.step()
            train_strategy.optimizer.zero_grad()
            
            # Update metrics
            epoch = (metrics.global_step + 1) // (len(vla_dataset) // train_strategy.global_batch_size)
            metrics.commit(global_step=metrics.global_step + 1, epoch=epoch, lr=train_strategy.lr_scheduler.get_last_lr()[0])
            status = metrics.push()
            
            # Save checkpoint with LoRA support
            if (terminate := (train_strategy.max_steps is not None and metrics.global_step >= train_strategy.max_steps)) or (
                (metrics.global_step % save_interval) == 0
            ):
                _save_lora_checkpoint(
                    vlm, lora_config, adapter_dir, run_dir, 
                    metrics.global_step, epoch, loss.item()
                )
                dist.barrier()
                
                if terminate:
                    return
            
            # Update progress
            progress.update()
            progress.set_description(status)


def _save_lora_checkpoint(vlm, lora_config, adapter_dir, run_dir, global_step, epoch, train_loss):
    """Save LoRA checkpoint with merging support (similar to finetune_libero.py)."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Saving LoRA Checkpoint for Step {global_step}")
        
        # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
        save_dir = adapter_dir if lora_config is not None else run_dir
        
        # Save adapter weights
        vlm.save_pretrained(save_dir)
        
        # Save merged model
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss={train_loss:.4f}.pt"
        
        # Merge LoRA weights for final model
        if lora_config is not None:
            from peft import PeftModel
            base_model = vlm.get_base_model() if hasattr(vlm, 'get_base_model') else vlm
            merged_model = PeftModel.from_pretrained(base_model, adapter_dir)
            merged_model = merged_model.merge_and_unload()
            torch.save({"model": merged_model.state_dict()}, checkpoint_path)
        else:
            torch.save({"model": vlm.state_dict()}, checkpoint_path)
        
        print(f"Saved LoRA Checkpoint for Step {global_step} at: {checkpoint_path}")


@draccus.wrap()
def train(cfg: TrainConfig) -> None:
    overwatch.info("OpenVLA Training :: Warming Up")

    # Note => Under `torchrun` initializing `overwatch` will automatically set up `torch.distributed`
    torch.cuda.set_device(device_id := overwatch.local_rank())
    torch.cuda.empty_cache()

    # Configure Unique Run Name & Save Directory
    vla_id = cfg.vla.vla_id
    cfg.run_id = (
        f"{vla_id}+n{cfg.vla.expected_world_size // 8}+b{cfg.per_device_batch_size}+x{cfg.seed}"
        if cfg.run_id is None
        else cfg.run_id
    )
    if cfg.run_id_note is not None:
        cfg.run_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        cfg.run_id += "--image_aug"

    if cfg.use_lora:
        cfg.run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        cfg.run_id += "+q-4bit"
    cfg.run_id += f'-LIBERO-Latent-Action-Pretraining-ws-{cfg.window_size}'
    # Start =>> Build Directories and Set Randomness
    overwatch.info('"Do or do not; there is no try."', ctx_level=1)
    # hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    hf_token = cfg.hf_token
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    os.makedirs(run_dir := (cfg.run_root_dir / cfg.run_id), exist_ok=True)
    os.makedirs(cfg.run_root_dir / cfg.run_id / "checkpoints", exist_ok=True)
    if cfg.use_lora:
        os.makedirs(adapter_dir := (cfg.adapter_tmp_dir / cfg.run_id), exist_ok=True)

    # Save Configuration =>> additionally save a JSON version for later HF Integration
    if overwatch.is_rank_zero():
        draccus.dump(cfg, open(run_dir / "config.yaml", "w"))
        with open(run_dir / "config.yaml", "r") as f_yaml, open(run_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)

    # Load VLA checkpoint (if resuming from training) or Base VLM otherwise (from `cfg.vla.base_vlm` ID or Path)
    #   =>> Note :: Verifies that all parameters are loaded in FP32 on load!
    overwatch.info(f"Loading Base VLM `{cfg.vla.base_vlm}` from ID/Path")
    if cfg.pretrained_checkpoint is not None:
        # [Validate] Pretrained Checkpoint `step` and `epoch` should match `resume_step` and `resume_epoch`
        #   =>> Note :: We make developers pass in `resume_*` arguments as an extra sanity check!
        if cfg.is_resume:
            assert int(re.search("step-(.+?)-", cfg.pretrained_checkpoint.name).group(1)) == cfg.resume_step
            assert int(re.search("epoch-(.+?)-", cfg.pretrained_checkpoint.name).group(1)) == cfg.resume_epoch

        vlm = load_vla(cfg.pretrained_checkpoint, hf_token=hf_token, load_for_training=True, cache_dir=cfg.pretrain_vlm)

    else:
        vlm = load(cfg.pretrain_vlm, hf_token=hf_token, load_for_training=True, cache_dir=cfg.pretrain_vlm)

    # [Validate] Model should be in Full Precision!
    for param in vlm.parameters():
        assert param.dtype == torch.float32, f"Loaded VLM parameter not in full precision: {param}"

    # [LoRA] Apply LoRA if enabled
    lora_config = None
    if cfg.use_lora:
        overwatch.info(f"Applying LoRA with rank={cfg.lora_rank}, dropout={cfg.lora_dropout}")
        
        # Quantization Config =>> only if LoRA fine-tuning
        quantization_config = None
        if cfg.use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
            )
            vlm = prepare_model_for_kbit_training(vlm)
        
        # Apply LoRA
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vlm = get_peft_model(vlm, lora_config)
        vlm.print_trainable_parameters()

    # Determine training "stage" based on frozen vs unfrozen parameters --> supports different fine-tuning schemes!
    if not cfg.vla.freeze_vision_backbone and not cfg.vla.freeze_llm_backbone:
        stage = "vla-full-train"  # Full fine-tuning
    elif cfg.vla.freeze_vision_backbone and not cfg.vla.freeze_llm_backbone:
        stage = "vla-train"  # Frozen vision encoder
    elif not cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone:
        assert cfg.vla.unfreeze_last_llm_layer, "You should unfreeze at least the last layer of your LLM!"
        stage = "vla-sandwich-train"  # Fine-tuning vision encoder, projector, and LLM last layer
    elif cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone:
        assert cfg.vla.unfreeze_last_llm_layer, "Need to unfreeze at least last LLM layer to train!"
        stage = "vla-last-layer-train"  # Fine-tuning LLM last layer only
    else:
        raise ValueError(
            "Weight freezing configuration not supported. VLA config has the following parameters: "
            f"freeze_vision_backbone: {cfg.vla.freeze_vision_backbone}"
            f"freeze_llm_backbone: {cfg.vla.freeze_llm_backbone}"
            f"unfreeze_last_llm_layer: {cfg.vla.unfreeze_last_llm_layer}"
        )

    # [Explicit] Call to `freeze_backbones` here for clarity =>> will log exactly what is/is not frozen
    overwatch.info(f"Invoking `VLM.freeze_backbones()` for `{vla_id}` => Stage: `{stage}`")
    vlm.freeze_backbones(stage)

    # Print number of total/trainable model parameters
    num_params = sum(p.numel() for p in vlm.parameters())
    num_trainable_params = sum(p.numel() for p in vlm.parameters() if p.requires_grad)
    overwatch.info(
        f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
    )
    
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

    # Get VLA Dataset & Collator for LIBERO
    overwatch.info(f"Creating VLA LIBERO Dataset with Mixture `{cfg.dataset_name}`")
    
    # Create LIBERO-specific batch transform
    batch_transform = RLDSBatchTransformLIBERO_withHis(
        latent_action_model,
        vlm.llm_backbone.get_tokenizer(),
        image_transform=vlm.vision_backbone.get_image_transform(),
        image_transform_lam=transforms.ToTensor(),
        prompt_builder_fn=PurePromptBuilder if "v01" not in str(cfg.pretrain_vlm) else VicunaV15ChatPromptBuilder,
        window_size=cfg.window_size
    )
    
    # Create LIBERO dataset
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=vlm.vision_backbone.default_image_resolution[1:],
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        window_size=cfg.window_size + 1,        # for constructing history latent actions
        training_phase='pre-training',
    )
    
    # Create LIBERO-specific collator
    collator = PaddedCollatorForActionPrediction_LIBERO(
        vlm.llm_backbone.get_tokenizer().model_max_length, 
        vlm.llm_backbone.get_tokenizer().pad_token_id, 
        padding_side="right"
    )
    
    # Add special tokens for latent actions
    special_tokens_dict = {'additional_special_tokens': [f'<ACT_{i}>' for i in range(cfg.codebook_size)]}
    num_added_toks = vlm.llm_backbone.get_tokenizer().add_special_tokens(special_tokens_dict)

    # Save dataset statistics for de-normalization at inference time
    if overwatch.is_rank_zero():
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Train Strategy
    overwatch.info(f"Initializing Train Strategy `{cfg.train_strategy}`")
    train_strategy = get_train_strategy(
        train_strategy=cfg.train_strategy,
        vlm=vlm,
        device_id=device_id,
        stage=stage,
        epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        global_batch_size=cfg.global_batch_size,
        per_device_batch_size=cfg.per_device_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        enable_gradient_checkpointing=cfg.vla.enable_gradient_checkpointing,
        enable_mixed_precision_training=cfg.vla.enable_mixed_precision_training,
        reduce_in_full_precision=cfg.vla.reduce_in_full_precision,
        worker_init_fn=worker_init_fn,
    )
    train_strategy.run_setup(run_dir=run_dir, n_train_examples=len(vla_dataset))

    # Create Metrics =>> Handles on the fly tracking, logging to specified trackers (e.g., JSONL, Weights & Biases)
    overwatch.info(f"Creating Metrics with Active Trackers => `{cfg.trackers}`")
    metrics = VLAMetrics(
        cfg.trackers,
        cfg.run_id,
        run_dir,
        draccus.encode(cfg),
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
        resume_step=cfg.resume_step,
        resume_epoch=cfg.resume_epoch,
    )

    # Run VLA Training
    overwatch.info("Starting VLA Latent Action Training Loop")
    
    # Custom training loop with LoRA checkpoint saving support
    if cfg.use_lora:
        _run_lora_training_loop(
            train_strategy, vla_dataset, collator, vlm.llm_backbone.get_tokenizer(), metrics, 
            cfg.save_interval, vlm, lora_config, adapter_dir, run_dir
        )
    else:
        train_strategy.run_latent_action_training(
            vla_dataset,
            collator,
            vlm.llm_backbone.get_tokenizer(),
            metrics,
            save_interval=cfg.save_interval,
        )

    # Finalize
    overwatch.info("Done with Training =>> Finalizing Metrics")
    metrics.finalize()

    # And... we're done!
    overwatch.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    train()