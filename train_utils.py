"""
Training utility functions for OmniVGGT

This module contains helper functions for training setup, including:
- Dataset building
- Model loading
- Optimizer and scheduler setup
- Loss criterion setup
- Logging configuration

License: MIT
"""

import os
import re
import json
import math
import random
import shutil
import logging
import contextlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import wandb
import numpy as np
import accelerate
import transformers
from safetensors.torch import load_file as load_safetensors
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from accelerate.logging import get_logger
from accelerate.utils import send_to_device

from omnivggt.loss import MultitaskLoss
from omnivggt.models.omnivggt import OmniVGGT
from omnivggt.models.omnivggt_omega import OmniVGGTOmega
from omnivggt.utils import weight_transfer
from omnivggt.utils.normalization import normalize_camera_extrinsics_and_points_batch
from omnivggt.datasets import get_data_loader
from omnivggt.datasets.utils.misc import merge_dicts

logger = get_logger(__name__, log_level="INFO")


def build_dataset(
    dataset: str,
    batch_size: int,
    num_workers: int,
    test: bool = False
) -> torch.utils.data.DataLoader:
    """
    Build data loader for training or testing.
    
    Args:
        dataset: Dataset configuration string
        batch_size: Batch size
        num_workers: Number of data loading workers
        test: Whether this is a test dataset
        
    Returns:
        DataLoader instance
    """
    split = 'Test' if test else 'Train'
    logger.info(f'Building {split} DataLoader for dataset: {dataset}')
    
    loader = get_data_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_mem=True,
        shuffle=not test,
        drop_last=not test
    )
    
    logger.info(f"{split} dataset length: {len(loader)}")
    return loader


def optimizer_steps_per_epoch(local_batches: int, accumulation_steps: int) -> int:
    """Optimizer steps per epoch; the epoch must hold a positive multiple of ``accumulation_steps`` micro-batches
    (a remainder would add an under-scaled extra step and shift the schedule and resume positions)."""
    if accumulation_steps < 1 or local_batches < accumulation_steps or local_batches % accumulation_steps:
        raise ValueError(f"{local_batches} micro-batches per epoch is not a positive multiple of "
                         f"gradient_accumulation_steps={accumulation_steps}; set steps_per_epoch to a multiple")
    return local_batches // accumulation_steps


def modality_rng(seed: int, rank: int, epoch: int, micro_step: int) -> np.random.Generator:
    """Generator for the training-time choice of views that get the auxiliary camera/depth. Keyed by the
    micro-batch, so a run replays from its seed and a resumed run sees the choices of an uninterrupted one."""
    return np.random.default_rng([int(seed), int(rank), int(epoch), int(micro_step)])


def start_epoch(train_dataloader, epoch: int) -> None:
    """Select the data order of ``epoch``. accelerate's DataLoaderShard re-applies its own epoch counter on
    every ``__iter__`` (it starts at 0 in a new process), so the epoch must be set on the loader itself;
    setting it on the dataset or sampler would be overridden and a resumed run would replay epoch 0."""
    train_dataloader.set_epoch(epoch)


def resume_skip(global_step: int, local_steps_per_epoch: int, accumulation_steps: int) -> Tuple[int, int]:
    """(optimizer steps already done in the current epoch, micro-batches to skip to resume after them)."""
    step_in_epoch = global_step % local_steps_per_epoch
    return step_in_epoch, step_in_epoch * accumulation_steps


def build_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    eta_min_factor: float = 0.05
) -> LambdaLR:
    """
    Build a learning rate scheduler with linear warmup and cosine decay.
    
    Args:
        optimizer: Optimizer instance
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        base_lr: Base learning rate
        eta_min_factor: Minimum learning rate factor (eta_min = eta_min_factor * base_lr)
        
    Returns:
        LambdaLR scheduler instance
    """
    def lr_lambda(current_step: int) -> float:
        # Linear warmup
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        # Cosine decay
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_factor + (1.0 - eta_min_factor) * cosine_decay
    
    return LambdaLR(optimizer, lr_lambda)


def setup_logging(accelerator: accelerate.Accelerator) -> None:
    """Setup logging configuration for all processes."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
    else:
        transformers.utils.logging.set_verbosity_error()


def setup_directories(cfg: Any) -> Tuple[str, str]:
    """
    Setup output and logging directories.
    
    Args:
        cfg: Configuration object
        
    Returns:
        Tuple of (save_dir, logging_dir)
    """
    save_dir = os.path.join(cfg.get("output_dir"), cfg.get("exp_name"))
    logging_dir = os.path.join(save_dir, cfg.get("logging_dir"))
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    return save_dir, logging_dir


def setup_wandb(cfg: Any, save_dir: str) -> None:
    """Setup Weights & Biases logging."""
    if cfg.get("wandb", False):
        wandb_dir = os.path.join(save_dir, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        
        wandb.init(
            project="OmniVGGT",
            name=cfg.get("exp_name"),
            config=cfg.to_dict(),
            dir=wandb_dir,
            settings=wandb.Settings(code_dir=".")
        )
        wandb.run.log_code(".")
        logger.info("WandB logging initialized")


def setup_tensorboard(cfg: Any, save_dir: str) -> Optional[SummaryWriter]:
    """
    Setup TensorBoard logging.
    
    Args:
        cfg: Configuration object
        save_dir: Output directory
        
    Returns:
        SummaryWriter instance or None
    """
    if cfg.get("tensorboard", True):
        tensorboard_log_dir = os.path.join(save_dir, "tensorboard")
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_log_dir)
        logger.info(f"TensorBoard logging initialized at {tensorboard_log_dir}")
        return writer
    return None


_repo_path = weight_transfer.repo_path


def resume_position(checkpoint_name: str, steps_per_epoch: int) -> Tuple[int, int]:
    """(epoch, global_step) encoded in a checkpoint directory name written by train_omnivggt.py:
    ``checkpoint-epoch-N`` (end of epoch N) or ``checkpoint-E-S`` (epoch E, step S)."""
    if checkpoint_name.startswith("checkpoint-epoch-"):
        epoch = int(checkpoint_name[len("checkpoint-epoch-"):])
        return epoch, epoch * steps_per_epoch
    parts = checkpoint_name[len("checkpoint-"):].split("-") if checkpoint_name.startswith("checkpoint-") else []
    if len(parts) == 2 and all(part.isdigit() for part in parts):
        return int(parts[0]), int(parts[1])
    raise ValueError(f"cannot infer epoch/step from checkpoint name {checkpoint_name!r}; "
                     "resume from checkpoint-epoch-N or checkpoint-E-S")


def build_model(cfg: Any) -> torch.nn.Module:
    """Instantiate the model named by ``cfg.model_name`` ("omnivggt" by default, or "omnivggt_omega")."""
    name = cfg.get("model_name", "omnivggt")
    causal, depth_norm = cfg.get("causal", False), cfg.get("depth_norm", "joint")
    if not isinstance(causal, bool):
        raise ValueError(f"causal must be true or false, got {causal!r}")
    if name == "omnivggt":
        if causal or depth_norm != "joint":
            raise ValueError("causal and depth_norm are options of model_name=omnivggt_omega only")
        return OmniVGGT(enable_point=cfg.get("enable_point", True),
                        enable_depth=cfg.get("enable_depth", True),
                        cam_drop_prob=cfg.get("cam_drop_prob", 0.1),
                        depth_drop_prob=cfg.get("depth_drop_prob", 0.1))
    if name == "omnivggt_omega":
        variant = cfg.get("omega_variant")
        if not variant:
            raise ValueError("model_name=omnivggt_omega needs omega_variant (a variant JSON)")
        torch.manual_seed(cfg.get("seed", weight_transfer.DEFAULT_SEED))  # initial values of parameters declared new
        return OmniVGGTOmega.from_variant(_repo_path(variant),
                                          cam_drop_prob=cfg.get("cam_drop_prob", 0.1),
                                          depth_drop_prob=cfg.get("depth_drop_prob", 0.1),
                                          depth_all_views=cfg.get("depth_all_views", False),
                                          causal=causal,
                                          depth_norm=depth_norm)
    raise ValueError(f"unknown model_name {name!r}")


def _write_initial_weights_report(report: dict, variant_path: Path, cfg: Any) -> None:
    """Record where the starting weights of an OmniVGGTOmega run came from (read back by the evaluation tool)."""
    report["variant"] = {"file": variant_path.name, "sha256": weight_transfer.sha256_of(variant_path),
                         "seed": cfg.get("seed", 42), "content": json.loads(variant_path.read_text())}
    output_dir = Path(cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "weight_transfer_report.json").write_text(json.dumps(report, indent=1) + "\n")


def load_variant_weights(model: torch.nn.Module, cfg: Any) -> str:
    """Non-strict, audited load of an OmniVGGTOmega variant from the checkpoints its weight map names."""
    variant_path = _repo_path(cfg.get("omega_variant"))
    weight_map = weight_transfer.load_map(weight_transfer.variant_map_path(variant_path))
    sources, files = weight_transfer.load_sources(weight_map)
    _write_initial_weights_report(weight_transfer.transfer_weights(model, weight_map, sources, files=files),
                                  variant_path, cfg)
    return f"weight_map:{weight_map['name']} (variant {variant_path.name})"


def load_omega_checkpoint(model: torch.nn.Module, cfg: Any) -> str:
    """Strict load of a trained OmniVGGTOmega checkpoint of the same variant (a checkpoint directory or its
    ``model.safetensors``), e.g. to continue training at another resolution; the weight map is not used."""
    variant_path = _repo_path(cfg.get("omega_variant"))
    path = Path(cfg.get("init_checkpoint"))
    if path.is_dir():
        path = path / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"init_checkpoint does not exist: {path}")
    # Variants with the same parameters (e.g. differing only in RoPE) load strictly into each other, so check
    # the variant recorded by the run that wrote the checkpoint (<output_dir>/<exp>/<checkpoint>/model.safetensors).
    source_report = path.parent.parent.parent / "weight_transfer_report.json"
    source_variant = "unverified"
    if source_report.is_file():
        recorded = json.loads(source_report.read_text())["variant"]
        source_variant = {"file": recorded["file"], "sha256": recorded["sha256"]}
        if source_variant["sha256"] != weight_transfer.sha256_of(variant_path):
            raise ValueError(f"init_checkpoint was trained with variant {recorded['file']}, not {variant_path.name}")
    state = load_safetensors(str(path))
    model.load_state_dict(state, strict=True)
    record = {"file": f"{path.parent.name}/{path.name}", "sha256": weight_transfer.sha256_of(path), "tensors": len(state),
              "source_variant": source_variant}
    _write_initial_weights_report({"init_checkpoint": record}, variant_path, cfg)
    return f"init_checkpoint:{record['file']} (variant {variant_path.name})"


def load_initial_weights(model: torch.nn.Module, cfg: Any) -> str:
    """Load the starting weights.

    ``init_checkpoint`` (a local ``.safetensors`` file, e.g. the released OmniVGGT
    weights) is loaded strictly and must exist; otherwise the original
    ``model_url`` behaviour (VGGT-1B from the hub) is used. OmniVGGTOmega variants
    are loaded through their weight map, or strictly from ``init_checkpoint`` when it
    names a trained OmniVGGTOmega checkpoint.
    """
    if cfg.get("model_name", "omnivggt") == "omnivggt_omega":
        return load_omega_checkpoint(model, cfg) if cfg.get("init_checkpoint") else load_variant_weights(model, cfg)
    init_checkpoint = cfg.get("init_checkpoint")
    if init_checkpoint:
        path = Path(init_checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"init_checkpoint does not exist: {path}")
        model.load_state_dict(load_safetensors(str(path)), strict=True)
        return f"init_checkpoint:{path.name}"

    model_url = cfg.get("model_url", "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt")
    logger.info(f"Loading pretrained weights from {model_url}")
    try:
        state_dict = torch.hub.load_state_dict_from_url(model_url)
        model.load_state_dict(state_dict, strict=cfg.get("model_load_strict", False))
        logger.info("Pretrained weights loaded successfully")
        return f"model_url:{model_url}"
    except Exception as e:
        logger.warning(f"Failed to load pretrained weights: {e}")
        logger.warning("Training from scratch...")
        return "scratch"


def load_model(cfg: Any, device: torch.device) -> Tuple[OmniVGGT, torch.dtype]:
    """
    Load and initialize the OmniVGGT model.
    
    Args:
        cfg: Configuration object
        device: Target device
        
    Returns:
        Tuple of (model, weight_dtype)
    """
    logger.info(f"Initializing {cfg.get('model_name', 'omnivggt')} model...")
    model = build_model(cfg)
    logger.info(f"cam_drop_prob={model.aggregator.cam_drop_prob} depth_drop_prob={model.aggregator.depth_drop_prob} "
                f"depth_all_views={getattr(model.aggregator, 'depth_all_views', False)} "
                f"causal={model.aggregator.causal} depth_norm={model.aggregator.depth_norm}")

    # Print network parameters and their indices
    # logger.info("Network parameters and their indices:")
    # for idx, (name, param) in enumerate(model.named_parameters()):
    #     logger.info(f"Parameter {idx}: {name} - Shape: {param.shape}")

    # Load pretrained weights
    logger.info(f"Initial weights: {load_initial_weights(model, cfg)}")

    # Set requires_grad
    model.requires_grad_(cfg.get("model_requires_grad", True))
    
    # Determine weight dtype
    weight_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    logger.info(f"Using weight dtype: {weight_dtype}")
    
    model.to(device)
    return model, weight_dtype


# AMUSE: Muon (orthogonalised momentum) for hidden-layer weight matrices, AdamW-style updates for the rest.
AMUSE_FALLBACK_WEIGHT_MODULES = (
    "aggregator.pose_embeddings.",  # input embeddings of the auxiliary camera
    "aggregator.depth_patch_embed.proj",  # input embedding of the auxiliary depth
    "camera_head.embed_pose",  # input embedding of the pose being refined
    "camera_head.pose_branch.fc2",  # output layer (pose encoding)
    "depth_head.scratch.output_conv2.2",  # output layer (depth and confidence)
)
_MATRIX_MODULES = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d)


def classify_amuse_parameters(model: torch.nn.Module) -> Tuple[list, list]:
    """Split the trainable parameters into (Muon names, AdamW-fallback names); every one appears exactly once."""
    matrices = {
        f"{name}.weight"
        for name, module in model.named_modules()
        if isinstance(module, _MATRIX_MODULES) and not name.startswith(AMUSE_FALLBACK_WEIGHT_MODULES)
    }
    muon, fallback = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (muon if name in matrices else fallback).append(name)
    if not muon or not fallback:
        raise ValueError(f"AMUSE needs both groups non-empty (muon={len(muon)}, fallback={len(fallback)})")
    return sorted(muon), sorted(fallback)


def _amuse_warmup_steps(cfg: Any, total_steps: int) -> int:
    return max(1, math.ceil(cfg.get("amuse_warmup_ratio", 0.05) * total_steps))


def _build_amuse(model: torch.nn.Module, cfg: Any) -> torch.optim.Optimizer:
    from omnivggt.optim.amuse import AMUSE

    if cfg.get("patch_embed_freeze", False):
        model.aggregator.patch_embed.requires_grad_(False)
        logger.info("patch_embed parameters are frozen.")
    for head in ("camera_head", "depth_head"):
        if cfg.get(f"{head}_freeze", False):
            getattr(model, head).requires_grad_(False)
    if cfg.get("enable_point", False) and getattr(model, "point_head", None) is None:
        raise ValueError("enable_point=True but the model has no point_head (use point_loss_mode='derived')")
    muon_names, fallback_names = classify_amuse_parameters(model)
    parameters = dict(model.named_parameters())
    weight_decay = cfg.get("amuse_weight_decay", 0.01)
    muon = {"use_muon": True, "lr": cfg.get("amuse_muon_lr", 1e-4), "momentum": cfg.get("amuse_momentum", 0.95),
            "aux_update_type": "adamw", "weight_decay": weight_decay, "name": "amuse_muon"}
    fallback = {"use_muon": False, "lr": cfg.get("amuse_aux_lr", 1e-5), "beta2": cfg.get("amuse_beta2", 0.999),
                "weight_decay": weight_decay, "name": "amuse_fallback"}
    # A trainable image encoder gets its own groups with the learning rates scaled down, as lr_patch_embed does for AdamW.
    encoder_scale = cfg.get("amuse_patch_embed_lr_scale", 1.0)
    groups = []
    for template, names in ((muon, muon_names), (fallback, fallback_names)):
        encoder = [n for n in names if n.startswith("aggregator.patch_embed.")]
        rest = [n for n in names if not n.startswith("aggregator.patch_embed.")]
        groups.append({**template, "params": [parameters[n] for n in rest]})
        if encoder:
            groups.append({**template, "params": [parameters[n] for n in encoder], "lr": template["lr"] * encoder_scale,
                           "name": f"{template['name']}_patch_embed"})
    estimated_steps = cfg.get("num_train_epochs", 1) * cfg.get("steps_per_epoch", 1) // cfg.get("gradient_accumulation_steps", 1)
    optimizer = AMUSE(
        groups,
        weight_decay_at_y=cfg.get("amuse_weight_decay_at_y", 0.0),
        beta1=cfg.get("amuse_beta1", 0.4),
        weight_lr_power=cfg.get("amuse_weight_lr_power", 2.0),
        warmup_steps=_amuse_warmup_steps(cfg, max(1, estimated_steps)),
        rho=cfg.get("amuse_rho", 0.3),
        r=cfg.get("amuse_r", 0.0),
    )
    for group in groups:
        logger.info(f"AMUSE group {group['name']}: {len(group['params'])} tensors, lr {group['lr']:g}")
    return optimizer


def configure_schedule(optimizer: torch.optim.Optimizer, cfg: Any, total_steps: int):
    """LR schedule for the run: AMUSE is schedule-free (only its warm-up length is set, returns None);
    other optimizers get the cosine warm-up scheduler."""
    inner = getattr(optimizer, "optimizer", optimizer)
    if hasattr(inner, "train_mode"):  # AMUSE (Schedule-Free family)
        warmup = _amuse_warmup_steps(cfg, total_steps)
        inner.warmup_steps = warmup
        for group in inner.param_groups:
            group["warmup_steps"] = warmup
        logger.info(f"AMUSE warm-up: {warmup} of {total_steps} steps; no external LR scheduler")
        return None
    return build_cosine_warmup_scheduler(
        optimizer=optimizer,
        warmup_steps=cfg.get("warmup_steps", 5000),
        total_steps=total_steps,
        eta_min_factor=cfg.get("eta_min_factor", 0.1),
    )


@contextmanager
def evaluation_weights(optimizer: torch.optim.Optimizer):
    """Expose the evaluation weights while the block runs (AMUSE: averaged x instead of the gradient point y),
    e.g. around checkpoint saving; restores the training state afterwards. No-op for other optimizers."""
    inner = getattr(optimizer, "optimizer", optimizer)
    if not hasattr(inner, "train_mode"):
        yield
        return
    was_training = inner.train_mode
    optimizer.eval()
    try:
        yield
    finally:
        if was_training:
            optimizer.train()


@contextmanager
def evaluation_mode(model: torch.nn.Module):
    """Run the block with every module of ``model`` in eval mode; afterwards each module is back in its own mode."""
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, training in modes:
            module.training = training


@contextmanager
def isolated_rng():
    """Run the block with the random states of Python, NumPy, torch (CPU) and every CUDA device restored afterwards,
    so that what the block draws (e.g. a validation pass) leaves the draws after it, and the training run, unchanged."""
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


# == k-best checkpoints ==

CHECKPOINTS_FILE = "checkpoints.json"
SMOKE_SPLIT = "smoke"  # the selection data; val is kept for the confirmation of the chosen checkpoint
VALIDATION_PROGRESS = 1.0  # the loss of every checkpoint is the end-of-training objective (e.g. predicted intrinsics)


def _rank(entry: dict) -> Tuple[bool, float, int]:
    """Sort key: lower score first, a non-finite score after every finite one; ties: the earlier step first."""
    score = entry["score"]
    finite = math.isfinite(score)
    return not finite, score if finite else 0.0, entry["step"]


class CheckpointKeeper:
    """Keeps the ``keep_best`` checkpoints of a run with the lowest score, and deletes the other checkpoint
    directories it recorded.

    ``record`` a checkpoint once its directory ``save_dir/name`` is written: while the run trains, the ``keep_best``
    best recorded checkpoints (ties: the earlier step) and the most recent one (to resume from) stay. ``finalize``
    with the final checkpoint, which always stays and counts towards ``keep_best``: afterwards only the final
    checkpoint and those among the ``keep_best`` best of all (the final one included) remain.

    Only directories recorded here are ever deleted (never the final checkpoint, nor any other file or directory of
    ``save_dir``); one missing at deletion is an error. ``save_dir/checkpoints.json`` is the audit trail: every
    recorded checkpoint (name, step, score, extra) with the checkpoints kept and removed after it. A keeper built on
    a ``save_dir`` holding one continues it (a resumed run).
    """

    def __init__(self, save_dir, keep_best: int):
        if isinstance(keep_best, bool) or not isinstance(keep_best, int) or keep_best < 0:
            raise ValueError(f"keep_best must be a non-negative integer, got {keep_best!r}")
        self.save_dir = Path(save_dir)
        self.keep_best = keep_best
        self.path = self.save_dir / CHECKPOINTS_FILE
        self.entries: List[dict] = []
        self._live: Dict[str, dict] = {}  # recorded and not removed, in recording order
        self.finalized = False
        if self.path.exists():
            self._resume(json.loads(self.path.read_text()))

    @property
    def kept(self) -> List[str]:
        """Names of the recorded checkpoints that are still on disk, in recording order."""
        return list(self._live)

    def record(self, name: str, score: float, step: int, extra: Optional[dict] = None) -> dict:
        """Record the written checkpoint ``name``; delete the recorded ones neither among the best nor the latest."""
        return self._record(name, score, step, extra, final=False)

    def finalize(self, final_name: str, score: float, step: int, extra: Optional[dict] = None) -> dict:
        """Record the final checkpoint; delete the recorded ones not among the best of all (the final one included)."""
        return self._record(final_name, score, step, extra, final=True)

    def _resume(self, audit: dict) -> None:
        if audit.get("finalized"):
            raise ValueError(f"{self.path} records a finished run: start a new run in a new output directory")
        for entry in audit["checkpoints"]:
            self._live.pop(entry["name"], None)
            self._live[entry["name"]] = entry
            for name in entry["removed"]:
                del self._live[name]
        self.entries = list(audit["checkpoints"])
        for name in self._live:
            if not self._directory(name).is_dir():
                raise FileNotFoundError(f"{self.path} keeps checkpoint {name!r}, which is not in {self.save_dir}")

    def _directory(self, name: str) -> Path:
        if not isinstance(name, str) or name in ("", ".", "..") or Path(name).name != name or os.sep in name:
            raise ValueError(f"a checkpoint is one directory inside {self.save_dir}, got {name!r}")
        return self.save_dir / name

    def _record(self, name: str, score: float, step: int, extra: Optional[dict], final: bool) -> dict:
        if self.finalized:
            raise RuntimeError(f"the final checkpoint is recorded: no checkpoint follows it ({name!r})")
        directory = self._directory(name)
        if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
            raise ValueError(f"a checkpoint is one real directory inside {self.save_dir}, got {name!r}")
        if not directory.is_dir():
            raise FileNotFoundError(f"checkpoint {name!r} is recorded after it is written to {directory}")
        entry = {"name": name, "step": int(step), "score": float(score), "final": final,
                 "keep_best": self.keep_best, "extra": dict(extra or {})}
        live = {key: value for key, value in self._live.items() if key != name}  # a rewritten name replaces its entry
        live[name] = entry
        # the best ones among the recorded checkpoints on disk (the final one included), and the one recorded now:
        # the latest while the run trains, or the final one
        keep = {value["name"] for value in sorted(live.values(), key=_rank)[:self.keep_best]} | {name}
        removed = [key for key in live if key not in keep]
        for key in removed:  # check every directory first: nothing is deleted when one is missing
            if not self._directory(key).is_dir() or self._directory(key).is_symlink():
                raise FileNotFoundError(f"recorded checkpoint {key!r} is not a directory in {self.save_dir}")
        for key in removed:
            shutil.rmtree(self._directory(key))
            del live[key]
        entry.update(kept=list(live), removed=removed)
        self._live, self.finalized = live, final
        self.entries.append(entry)
        self._write()
        return entry

    def _write(self) -> None:
        audit = {"keep_best": self.keep_best, "order": "lower score is better; ties: the earlier step",
                 "finalized": self.finalized, "checkpoints": self.entries}
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(audit, indent=1) + "\n")
        os.replace(temporary, self.path)


def smoke_split_spec(train_spec: str, samples: int) -> str:
    """The dataset spec of the selection data of a run: ``samples`` samples of the smoke split, drawn like those of
    ``train_spec`` (one ``N @ ColmapRgbd(..., split='train', ...)``: the same roots, views, resolution and sample seed)
    without augmentation (``aug_crop=0``, ``transform=ImgNorm``)."""
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError(f"the smoke split needs a positive number of samples, got {samples!r}")
    match = re.fullmatch(r"\s*\d[\d_]*\s*@\s*(ColmapRgbd\(.*\))\s*", train_spec, flags=re.S)
    body = match.group(1) if match else ""
    replacements = ((r"split\s*=\s*'train'", "split='smoke'"), (r"aug_crop\s*=\s*\d+", "aug_crop=0"),
                    (r"transform\s*=\s*\w+", "transform=ImgNorm"))
    for pattern, replacement in replacements:
        body, count = re.subn(pattern, replacement, body)
        if count != 1:
            raise ValueError(f"the smoke split is derived from one 'N @ ColmapRgbd(..., split='train', aug_crop=.., "
                             f"transform=..)' train_dataset, got {train_spec!r}")
    return f"{samples} @ {body}"


def build_validation_loader(cfg: Any) -> torch.utils.data.DataLoader:
    """The selection data of a batch-trainer run (train_omnivggt.py): ``cfg.val_samples`` training steps of the smoke
    split (``smoke_split_spec`` of ``cfg.train_dataset``), always the same samples in the same order."""
    spec = smoke_split_spec(cfg.get("train_dataset"), cfg.get("val_samples", 16))
    try:
        loader = get_data_loader(spec, batch_size=cfg.get("train_batch_images", 24), num_workers=cfg.get("num_workers", 8),
                                 shuffle=False, drop_last=False, full_clips=cfg.get("full_clips", False))
    except ValueError as err:
        raise ValueError(f"keep_best scores the checkpoints on the smoke split, which cannot be read: {err}") from err
    loader.dataset.set_epoch(0)  # fixed samples (the ResizedDataset draw) and order (the sampler draw)
    loader.sampler.set_epoch(0)
    logger.info(f"Validation (checkpoint selection) data: {len(loader)} steps of {spec}")
    return loader


def split_batch(batch: dict, target_scale: str) -> Tuple[dict, dict]:
    """(model inputs, loss targets) of a merged training batch: the targets hold the camera, points and depth
    normalised as the loss expects them; the inputs keep the raw ones."""
    new_extrinsics, _, new_world_points, new_depths = normalize_camera_extrinsics_and_points_batch(
        extrinsics=batch['extrinsic'],
        cam_points=None,
        world_points=batch['world_points'],
        depths=batch['depth'],
        point_masks=batch['valid_mask'],
        target_scale=target_scale,
    )
    inputs = {
        'images': batch['images'],
        'extrinsics': batch['extrinsic'].clone(),
        'intrinsics': batch['intrinsic'],
        'depth': batch['depth'].clone(),
        'mask': batch['valid_mask'].clone(),
    }
    targets = {**batch, 'extrinsic': new_extrinsics, 'world_points': new_world_points, 'depth': new_depths}
    return inputs, targets


def validation_modality_rng(seed: int, index: int) -> np.random.Generator:
    """Generator of the auxiliary camera/depth views of validation step ``index``: the same for every checkpoint,
    and keyed apart from the training draws (``modality_rng``)."""
    return np.random.default_rng([int(seed), int(index)])


def validation_objective(model: torch.nn.Module, loader, criterion: Callable, target_scale: str, seed: int,
                         device, autocast: Callable = contextlib.nullcontext) -> Dict[str, float]:
    """Mean of every loss term over the steps of ``loader`` (the smoke split), computed as in training (``split_batch``,
    the forward under ``autocast``, the loss without it) at ``VALIDATION_PROGRESS``, without gradients, with the
    model in eval mode and every random state restored afterwards: the training run continues as without it."""
    terms = []
    device = torch.device(device)
    with isolated_rng(), evaluation_mode(model), torch.no_grad():
        for index, batch in enumerate(loader):
            inputs, targets = split_batch(send_to_device(merge_dicts(batch), device), target_scale=target_scale)
            with autocast():
                predictions = model(**inputs, modality_rng=validation_modality_rng(seed, index))
            with torch.autocast(device.type, enabled=False):
                loss = criterion(predictions, targets, progress=VALIDATION_PROGRESS)
            terms.append({key: float(value) for key, value in loss.items()})
    if not terms:
        raise ValueError("the validation loader is empty")
    return {key: float(np.mean([term[key] for term in terms])) for key in terms[0]}


def build_optimizer(model: torch.nn.Module, cfg: Any) -> torch.optim.Optimizer:
    """
    Build optimizer with parameter groups for different learning rates.
    
    Args:
        model: Model instance
        cfg: Configuration object
        
    Returns:
        Optimizer instance
    """
    if cfg.get("optimizer_type", "adamw").lower() == "amuse":
        return _build_amuse(model, cfg)
    param_groups = []
    exclude_keys = ["aggregator.patch_embed"]
    
    if cfg.get("patch_embed_freeze", False):
        for param in model.aggregator.patch_embed.parameters():
            param.requires_grad = False
        logger.info("patch_embed parameters are frozen.")
    else:
        param_groups.append({
            "params": model.aggregator.patch_embed.parameters(),
            "lr": cfg.get("lr_patch_embed", cfg.get("lr")),
            "name": "patch_embed"
        })
        logger.info(f"patch_embed lr set to {cfg.get('lr_patch_embed', cfg.get('lr'))}")
        
    if cfg.get("enable_camera", False):
        exclude_keys.append("camera_head")
        if cfg.get("camera_head_freeze", False):
            for param in model.camera_head.parameters():
                param.requires_grad = False
            logger.info("camera_head parameters are frozen.")
        else:
            param_groups.append({
                "params": model.camera_head.parameters(),
                "lr": cfg.get("lr_camera_head", cfg.get("lr_head", cfg.get("lr"))),
                "name": "camera_head"
            })
            logger.info(f"camera_head lr set to {cfg.get('lr_camera_head', cfg.get('lr'))}")
    
    if cfg.get("enable_depth", False):
        exclude_keys.append("depth_head")
        if cfg.get("depth_head_freeze", False):
            for param in model.depth_head.parameters():
                param.requires_grad = False
            logger.info("depth_head parameters are frozen.")
        else:
            param_groups.append({
                "params": model.depth_head.parameters(),
                "lr": cfg.get("lr_depth_head", cfg.get("lr_head", cfg.get("lr"))),
                "name": "depth_head"
            })
            logger.info(f"depth_head lr set to {cfg.get('lr_depth_head', cfg.get('lr'))}")
            
    if cfg.get("enable_point", False):
        if getattr(model, "point_head", None) is None:
            raise ValueError("enable_point=True but the model has no point_head (use point_loss_mode='derived')")
        exclude_keys.append("point_head")
        if cfg.get("point_head_freeze", False):
            for param in model.point_head.parameters():
                param.requires_grad = False
            logger.info("point_head parameters are frozen.")
        else:
            param_groups.append({
                "params": model.point_head.parameters(),
                "lr": cfg.get("lr_point_head", cfg.get("lr_head", cfg.get("lr"))),
                "name": "point_head"
            })
            logger.info(f"point_head lr set to {cfg.get('lr_point_head', cfg.get('lr'))}")
    
    param_groups.append({
        "params": [
            p for n, p in model.named_parameters()
            if not any(k in n for k in exclude_keys)
        ],
        "lr": cfg.get("lr"),
        "name": "other"
    })
    
    optimizer_type = cfg.get("optimizer_type", "adamw").lower()
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(cfg.get("adam_beta1", 0.9), cfg.get("adam_beta2", 0.95)),
            eps=cfg.get("adam_epsilon", 1e-8),
            weight_decay=cfg.get("adam_weight_decay", 0.01)
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    logger.info(f"Optimizer created: {optimizer_type}")
    for i, pg in enumerate(param_groups):
        logger.info(f"  Group {i} ({pg['name']}): lr={pg['lr']}")
    
    return optimizer


def _point_loss_config(cfg: Any) -> dict:
    if cfg.get("point_loss_mode", "head") == "derived":
        return {"mode": "derived", "weight": cfg.get("point_loss_weight", 1.0),
                "intrinsics_warmup_ratio": cfg.get("point_intrinsics_warmup_ratio", 0.5)}
    return {"weight": cfg.get("point_loss_weight", 1.0),
            "gradient_loss_fn": cfg.get("point_gradient_loss_fn", "normal"),
            "valid_range": cfg.get("point_valid_range", 0.98)}


def build_loss_criterion(cfg: Any) -> MultitaskLoss:
    """
    Build multi-task loss criterion.
    
    Args:
        cfg: Configuration object
        
    Returns:
        MultitaskLoss instance
    """
    criterion = MultitaskLoss(
        camera={
            "weight": cfg.get("camera_loss_weight", 5.0),
            "loss_type": cfg.get("camera_loss_type", "l1")
        },
        depth={
            "weight": cfg.get("depth_loss_weight", 1.0),
            "gradient_loss_fn": cfg.get("depth_gradient_loss_fn", "grad"),
            "valid_range": cfg.get("depth_valid_range", 0.98)
        },
        point=_point_loss_config(cfg),
    )
    
    logger.info("Loss criterion initialized:")
    logger.info(f"  Camera loss weight: {cfg.get('camera_loss_weight', 5.0)}")
    logger.info(f"  Depth loss weight: {cfg.get('depth_loss_weight', 1.0)}")
    
    return criterion