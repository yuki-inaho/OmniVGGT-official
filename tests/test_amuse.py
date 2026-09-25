"""AMUSE optimizer (vendored, Apache-2.0) and its integration into the training utilities."""

import hashlib
import math
from pathlib import Path

import pytest
import torch
from accelerate import PartialState

import train_utils
from omnivggt.models.omnivggt_omega import OmniVGGTOmega

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_BLOB_SHA1 = "144361bf100d0a3a07172fb007a6fb27ff58f046"  # kjeiun/amuse src/optim/AMUSE.py @ 48922743
TINY = dict(
    img_size=28,
    patch_size=14,
    embed_dim=32,
    num_register_tokens=16,
    register_attention_layers=(1,),
    global_rope=False,
    cached_layers=(0, 1, 2, 3),
    aggregator_kwargs=dict(depth=4, num_heads=2, patch_embed="conv"),
    camera_head_kwargs=dict(trunk_depth=1, num_heads=2),
    depth_head_kwargs=dict(features=16, out_channels=[8, 16, 32, 32], intermediate_layer_idx=[0, 1, 2, 3]),
)
AMUSE_CFG = {
    "optimizer_type": "amuse",
    "amuse_muon_lr": 1e-4,
    "amuse_aux_lr": 1e-5,
    "amuse_beta1": 0.4,
    "amuse_beta2": 0.999,
    "amuse_momentum": 0.95,
    "amuse_rho": 0.3,
    "amuse_r": 0.0,
    "amuse_weight_lr_power": 2.0,
    "amuse_warmup_ratio": 0.05,
    "amuse_weight_decay": 0.01,
    "amuse_weight_decay_at_y": 0.0,
    "enable_camera": True,
    "enable_depth": True,
    "enable_point": False,
    "patch_embed_freeze": True,
}


def _tiny():
    torch.manual_seed(0)
    return OmniVGGTOmega(**TINY)


def test_vendored_amuse_matches_upstream_blob():
    data = (ROOT / "omnivggt" / "optim" / "amuse.py").read_bytes()
    assert hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == UPSTREAM_BLOB_SHA1
    assert (ROOT / "omnivggt" / "optim" / "LICENSE").read_text().lstrip().startswith("Apache License")


def test_classify_amuse_parameters():
    model = _tiny()
    for parameter in model.aggregator.patch_embed.parameters():
        parameter.requires_grad = False
    muon, fallback = train_utils.classify_amuse_parameters(model)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert set(muon) | set(fallback) == trainable and not set(muon) & set(fallback)
    assert not any(name.startswith("aggregator.patch_embed.") for name in muon + fallback)
    for name in (
        "aggregator.frame_blocks.0.attn.qkv.weight",
        "aggregator.global_blocks.1.mlp.fc2.weight",
        "aggregator.camera_adapters.3.weight",
        "camera_head.poseLN_modulation.1.weight",
        "camera_head.pose_branch.fc1.weight",
        "depth_head.projects.0.weight",
        "depth_head.resize_layers.0.weight",
        "depth_head.scratch.output_conv1.weight",
    ):
        assert name in muon, name
    for name in (
        "aggregator.camera_token",
        "aggregator.register_token",
        "aggregator.depth_placeholder",
        "aggregator.pose_embeddings.0.weight",
        "aggregator.depth_patch_embed.proj.weight",
        "aggregator.frame_blocks.0.attn.qkv.bias",
        "aggregator.frame_blocks.0.norm1.weight",
        "aggregator.frame_blocks.0.ls1.gamma",
        "camera_head.embed_pose.weight",
        "camera_head.empty_pose_tokens",
        "camera_head.pose_branch.fc2.weight",
        "depth_head.scratch.output_conv2.2.weight",
    ):
        assert name in fallback, name


def test_build_optimizer_amuse_and_schedule():
    PartialState()
    from omnivggt.optim.amuse import AMUSE

    model = _tiny()
    optimizer = train_utils.build_optimizer(model, dict(AMUSE_CFG))
    assert isinstance(optimizer, AMUSE)
    groups = {g["use_muon"]: g for g in optimizer.param_groups}
    assert groups[True]["lr"] == 1e-4 and groups[False]["lr"] == 1e-5
    assert groups[True]["momentum"] == 0.95 and groups[False]["beta2"] == 0.999
    assert train_utils.configure_schedule(optimizer, dict(AMUSE_CFG), total_steps=3120) is None
    assert all(g["warmup_steps"] == math.ceil(0.05 * 3120) for g in optimizer.param_groups)
    adamw = train_utils.build_optimizer(_tiny(), {"lr": 1e-5, "enable_camera": True, "enable_depth": True})
    scheduler = train_utils.configure_schedule(adamw, {"warmup_steps": 100, "eta_min_factor": 0.1}, total_steps=3120)
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)


def _toy_amuse(steps=5):
    from omnivggt.optim.amuse import AMUSE

    torch.manual_seed(0)
    layer = torch.nn.Linear(8, 8)
    target = torch.randn(64, 8)
    inputs = torch.randn(64, 8)
    optimizer = AMUSE(
        [
            {"params": [layer.weight], "use_muon": True, "lr": 0.05, "weight_decay": 0.0},
            {"params": [layer.bias], "use_muon": False, "lr": 0.05, "weight_decay": 0.0},
        ],
        beta1=0.4,
        warmup_steps=2,
        rho=0.3,
    )
    optimizer.train()
    losses = []
    for _ in range(steps):
        loss = torch.nn.functional.mse_loss(layer(inputs), target)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return layer, optimizer, losses


def test_amuse_reduces_toy_loss():
    _, _, losses = _toy_amuse(steps=30)
    assert losses[-1] < 0.8 * losses[0]


def test_evaluation_weights_context():
    layer, optimizer, _ = _toy_amuse(steps=5)
    y = layer.weight.detach().clone()
    with train_utils.evaluation_weights(optimizer):
        assert optimizer.train_mode is False
        x = layer.weight.detach().clone()
        assert not torch.equal(x, y)
    assert optimizer.train_mode is True
    torch.testing.assert_close(layer.weight.detach(), y)
    adamw = torch.optim.AdamW(layer.parameters())
    with train_utils.evaluation_weights(adamw):
        pass  # no train/eval for AdamW: nothing to switch


def test_amuse_rejects_point_head_mismatch_like_adamw():
    PartialState()
    with pytest.raises(ValueError, match="point_head"):
        train_utils.build_optimizer(_tiny(), {**AMUSE_CFG, "enable_point": True})
