#!/usr/bin/env python3
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import draccus
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
import yaml

from peft import LoraConfig, get_peft_model, TaskType

from prismatic.conf import VLAConfig, VLARegistry
from prismatic.models import load
from prismatic.overwatch import initialize_overwatch
from prismatic.training import VLAMetrics, get_train_strategy
from prismatic.util import set_global_seed
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from prismatic.vla.datasets import RLDSBatchTransformLIBERO_withHis, RLDSDataset
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.util.data_utils import PaddedCollatorForActionPrediction_LIBERO

# Sane defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"
overwatch = initialize_overwatch(__name__)


@dataclass
class TrainConfig:
    # === VLA (LIBERO) 기본 설정 ===
    vla: VLAConfig = field(
        default_factory=VLAConfig.get_choice_class(VLARegistry.DINOSIGLIP_224PX_MX_LIBERO.vla_id)
    )
    pretrain_vlm: str = "prism-dinosiglip-224px+7b"

    # === LAM ===
    lam_path: str = "latent_action_model/logs/task_centric_lam_stage2/epoch=2-step=18000.ckpt"
    codebook_size: int = 16
    lam_model_dim: int = 768
    lam_latent_dim: int = 128
    lam_patch_size: int = 14
    lam_enc_blocks: int = 12
    lam_dec_blocks: int = 12
    lam_num_heads: int = 12

    # === 데이터/로그 경로 ===
    data_root_dir: Path = Path("/ssd1/openpi_official/datasets/libero_raw")
    dataset_name: str = "libero_combined"
    run_root_dir: Path = Path("vla_log")

    # === 재시작/체크포인트 ===
    pretrained_checkpoint: Optional[Path] = None
    is_resume: bool = False
    resume_step: Optional[int] = None
    resume_epoch: Optional[int] = None

    # === 런 설정 ===
    run_id: Optional[str] = None
    run_id_note: Optional[str] = None
    save_interval: int = 5000
    image_aug: bool = True
    seed: int = 42

    # === HF Hub ===
    hf_token: Union[str, Path] = ""
    hf_cache_dir: Path = Path("ssd2/hf_cache")

    # === LoRA ===
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    # (양자화는 prismatic.load 경로와 충돌 가능성 → 비활성 권장)

    # === LIBERO dataloader ===
    window_size: int = 12  # 8 -> 6으로 더 줄여서 메모리 사용량 감소
    shuffle_buffer_size: int = 16000  # 8000 -> 4000으로 더 줄여서 메모리 사용량 감소
    
    # === 메모리 최적화 ===
    gradient_accumulation_steps: int = 4  # gradient accumulation을 더 늘려서 effective batch size 유지
    max_memory_usage: float = 0.8  # GPU 메모리 사용량 제한 (80%)

    # === 로깅 ===
    trackers: Tuple[str, ...] = ("jsonl", "wandb")
    wandb_project: str = "univla-lora-libero"
    wandb_entity: str = "jhshin406"

    use_flash_attention: bool = True
    lora_target: str = "attn"  # "attn_mlp" -> "attn"으로 변경하여 메모리 사용량 감소
    clamp_seq_len: Optional[int] = None

    def __post_init__(self) -> None:
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
        assert (
            self.vla.expected_world_size == overwatch.world_size()
        ), f"Expected World Size = {self.vla.expected_world_size} but Found {overwatch.world_size()} GPUs!"


def _build_action_tokenizer_and_resize(vlm, codebook_size: int) -> ActionTokenizer:
    tok = vlm.llm_backbone.get_tokenizer()
    special_tokens = {"additional_special_tokens": [f"<ACT_{i}>" for i in range(codebook_size)]}
    tok.add_special_tokens(special_tokens)
    # PEFT 적용 전/후 어느 시점이든 한 번은 호출되어야 함
    vlm.llm_backbone.llm.resize_token_embeddings(len(tok))
    return ActionTokenizer(tok)


@draccus.wrap()
def train(cfg: TrainConfig) -> None:
    overwatch.info("OpenVLA LoRA Pre-Training on LIBERO :: Warmup")
    torch.cuda.set_device(device_id := overwatch.local_rank())
    torch.cuda.empty_cache()

    # --- Run ID 구성 ---
    vla_id = cfg.vla.vla_id
    run_id = f"{vla_id}+n{cfg.vla.expected_world_size // 8}+b{cfg.per_device_batch_size}+x{cfg.seed}"
    if cfg.run_id is not None:
        run_id = cfg.run_id
    if cfg.run_id_note:
        run_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        run_id += "--image_aug"
    if cfg.use_lora:
        run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    run_id += f"-LIBERO-Latent-Action-Pretraining-ws-{cfg.window_size}"
    cfg.run_id = run_id

    # --- 디렉토리/시드 ---
    run_dir = cfg.run_root_dir / cfg.run_id
    os.makedirs(run_dir / "checkpoints", exist_ok=True)
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)

    # --- 설정 저장 ---
    if overwatch.is_rank_zero():
        draccus.dump(cfg, open(run_dir / "config.yaml", "w"))
        with open(run_dir / "config.yaml", "r") as f_yaml, open(run_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)

    # --- HF 토큰/캐시 ---
    if isinstance(cfg.hf_token, Path):
        hf_token = cfg.hf_token.read_text().strip()
    else:
        hf_token = (cfg.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or None)
    os.environ.setdefault("HF_HOME", str(cfg.hf_cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cfg.hf_cache_dir))

    # --- 베이스 VLM 로드 (FP32) ---
    overwatch.info(f"Loading Base VLM `{cfg.pretrain_vlm}`")
    if cfg.pretrained_checkpoint is not None:
        raise NotImplementedError("Resume-from-checkpoint with LoRA pretrain is not wired here yet.")
    vlm = load(cfg.pretrain_vlm, hf_token=hf_token, load_for_training=True, cache_dir=str(cfg.hf_cache_dir))

    if cfg.use_flash_attention:
        # PyTorch 2.x scaled dot-product attention 커널 선택
        torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)

        # HF Llama 구현이 지원하면 FA2 요청
        try:
            vlm.llm_backbone.llm.config.attn_implementation = "flash_attention_2"
        except Exception:
            pass  # 설치가 안되어 있으면 SDP만 사용


    for p in vlm.parameters():
        assert p.dtype == torch.float32, "Model must be loaded in FP32 before PEFT."

    # --- LoRA: LLM 서브모듈에만 주입 ---
    if cfg.use_lora:
        if cfg.lora_target == "attn":
            target_modules_llm = ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif cfg.lora_target == "attn_mlp":
            target_modules_llm = ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"]
        else:
            raise ValueError(f"Unsupported lora_target={cfg.lora_target}. Use 'attn' or 'attn_mlp'.")

        lcfg_llm = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules=target_modules_llm,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        base_llm = vlm.llm_backbone.llm
        peft_llm = get_peft_model(base_llm, lcfg_llm)
        vlm.llm_backbone.llm = peft_llm
        if overwatch.is_rank_zero():
            vlm.llm_backbone.llm.print_trainable_parameters()

    # --- 동결 정책: projector+LLM 학습, vision freeze ---
    stage = "vla-train"  # projector + LLM train, vision freeze
    vlm.freeze_backbones(stage)

    # --- LAM 로드 ---
    from latent_action_model.genie.modules.lam import ControllableDINOLatentActionModel
    lam = ControllableDINOLatentActionModel(
        in_dim=3,
        model_dim=cfg.lam_model_dim,
        latent_dim=cfg.lam_latent_dim,
        num_latents=cfg.codebook_size,
        patch_size=cfg.lam_patch_size,
        enc_blocks=cfg.lam_enc_blocks,
        dec_blocks=cfg.lam_dec_blocks,
        num_heads=cfg.lam_num_heads,
        dropout=0.0,
    )
    lam_ckpt = torch.load(cfg.lam_path, map_location="cpu")["state_dict"]
    lam.load_state_dict({k.replace("lam.", ""): v for k, v in lam_ckpt.items()}, strict=True)
    lam = lam.to(device_id).eval()
    
    # LAM을 half precision으로 변환하여 메모리 사용량 감소
    lam = lam.half()

    # --- Dataset/Transform (LIBERO latent action) ---
    batch_tf = RLDSBatchTransformLIBERO_withHis(
        lam,
        vlm.llm_backbone.get_tokenizer(),
        image_transform=vlm.vision_backbone.get_image_transform(),
        image_transform_lam=transforms.ToTensor(),
        prompt_builder_fn=PurePromptBuilder if "v01" not in str(cfg.pretrain_vlm) else VicunaV15ChatPromptBuilder,
        window_size=cfg.window_size,
    )
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_tf,
        resize_resolution=vlm.vision_backbone.default_image_resolution[1:],
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        window_size=cfg.window_size + 1,   # (히스토리 포함)
        training_phase="post-training",    # chunk_act_obs_libero 사용을 위해 변경
    )

    # --- 액션 토큰 추가 & 임베딩 리사이즈 ---
    action_tokenizer = _build_action_tokenizer_and_resize(vlm, cfg.codebook_size)
    # if cfg.clamp_seq_len is not None:
    #     tok = vlm.llm_backbone.get_tokenizer()
    #     try:
    #         tok.model_max_length = min(getattr(tok, "model_max_length", cfg.clamp_seq_len), cfg.clamp_seq_len)
    #     except Exception:
    #         pass

    # --- collator ---
    collator = PaddedCollatorForActionPrediction_LIBERO(
        vlm.llm_backbone.get_tokenizer().model_max_length,
        vlm.llm_backbone.get_tokenizer().pad_token_id,
        padding_side="right",
    )

    # --- 통계 저장 ---
    if overwatch.is_rank_zero():
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # --- 트레인 전략 구성 & 셋업 ---
    train_strategy = get_train_strategy(
        train_strategy=cfg.train_strategy,
        vlm=vlm,
        device_id=device_id,
        stage=stage,
        epochs=cfg.epochs,
        max_steps=cfg.max_steps,  # None일 수 있음 → 아래에서 보정
        global_batch_size=cfg.global_batch_size,
        per_device_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,  # gradient accumulation 추가
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

    # === IterableDataset 안전: tqdm가 len(dataloader)에 의존하지 않도록 self.max_steps를 명시 ===
    # Strategy가 계산한 값 있으면 사용, 없으면 추정/기본값
    effective_max_steps = getattr(train_strategy, "max_steps", None)
    if effective_max_steps is None:
        try:
            n = len(vla_dataset)
            steps_per_epoch = max(n // cfg.global_batch_size, 1)
            effective_max_steps = steps_per_epoch * max(cfg.epochs, 1)
        except TypeError:
            # IterableDataset 길이 모르면 대략치
            effective_max_steps = 100000
    train_strategy.max_steps = int(effective_max_steps)

    # --- 메트릭 로거 ---
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

    # --- 학습 시작: Strategy 표준 루프 사용 / LoRA 어댑터만 저장 ---
    overwatch.info("[*] Starting VLA LoRA Latent Action Pre-Training Loop")
    train_strategy.run_latent_action_training(
        vla_dataset=vla_dataset,
        collator=collator,
        action_tokenizer=action_tokenizer,
        metrics=metrics,
        save_interval=cfg.save_interval,
        save_full_model=False,  # ← trainable-only 저장 (LoRA 어댑터)
    )

    overwatch.info("Finalize Metrics")
    metrics.finalize()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    train()