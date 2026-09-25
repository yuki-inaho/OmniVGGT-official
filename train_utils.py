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
import json
import math
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import wandb
import numpy as np
import accelerate
import transformers
from safetensors.torch import load_file as load_safetensors
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from accelerate.logging import get_logger

from omnivggt.loss import MultitaskLoss
from omnivggt.models.omnivggt import OmniVGGT
from omnivggt.models.omnivggt_omega import OmniVGGTOmega
from omnivggt.utils import weight_transfer
from omnivggt.datasets import get_data_loader

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
    if name == "omnivggt":
        return OmniVGGT(enable_point=cfg.get("enable_point", True),
                        enable_depth=cfg.get("enable_depth", True),
                        cam_drop_prob=cfg.get("cam_drop_prob", 0.1),
                        depth_drop_prob=cfg.get("depth_drop_prob", 0.1))
    if name == "omnivggt_omega":
        if cfg.get("init_checkpoint"):
            raise ValueError("omnivggt_omega is initialised from its variant's weight map; do not set init_checkpoint")
        variant = cfg.get("omega_variant")
        if not variant:
            raise ValueError("model_name=omnivggt_omega needs omega_variant (a variant JSON)")
        torch.manual_seed(cfg.get("seed", weight_transfer.DEFAULT_SEED))  # initial values of parameters declared new
        return OmniVGGTOmega.from_variant(_repo_path(variant),
                                          cam_drop_prob=cfg.get("cam_drop_prob", 0.1),
                                          depth_drop_prob=cfg.get("depth_drop_prob", 0.1))
    raise ValueError(f"unknown model_name {name!r}")


def load_variant_weights(model: torch.nn.Module, cfg: Any) -> str:
    """Non-strict, audited load of an OmniVGGTOmega variant from the checkpoints its weight map names."""
    variant_path = _repo_path(cfg.get("omega_variant"))
    variant = json.loads(variant_path.read_text())
    weight_map = weight_transfer.load_map(weight_transfer.variant_map_path(variant_path))
    sources, files = weight_transfer.load_sources(weight_map)
    report = weight_transfer.transfer_weights(model, weight_map, sources, files=files)
    report["variant"] = {"file": variant_path.name, "sha256": weight_transfer.sha256_of(variant_path),
                         "seed": cfg.get("seed", 42), "content": variant}
    output_dir = Path(cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "weight_transfer_report.json").write_text(json.dumps(report, indent=1) + "\n")
    return f"weight_map:{weight_map['name']} (variant {variant_path.name})"


def load_initial_weights(model: torch.nn.Module, cfg: Any) -> str:
    """Load the starting weights.

    ``init_checkpoint`` (a local ``.safetensors`` file, e.g. the released OmniVGGT
    weights) is loaded strictly and must exist; otherwise the original
    ``model_url`` behaviour (VGGT-1B from the hub) is used. OmniVGGTOmega variants
    are loaded through their weight map instead.
    """
    if cfg.get("model_name", "omnivggt") == "omnivggt_omega":
        return load_variant_weights(model, cfg)
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
    logger.info(f"cam_drop_prob={model.aggregator.cam_drop_prob} depth_drop_prob={model.aggregator.depth_drop_prob}")

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