"""OmegaStyleAggregator: OmniVGGT's ZeroAggregator with VGGT-Omega style inter-frame (global) attention."""

from pathlib import Path

import pytest
import torch

from omnivggt.models import aggregator as aggregator_module
from omnivggt.models.omega_aggregator import OmegaStyleAggregator
from omnivggt.models.omnivggt_aggregator import ZeroAggregator

FIXTURE = Path(__file__).parent / "fixtures" / "zero_aggregator_golden.pt"
CONDITIONS = {
    "rgb": ([], []),
    "depth": ([0, 1, 2], []),
    "camera": ([], [0, 1, 2]),
    "depth+camera": ([0, 1, 2], [0, 1, 2]),
    "partial": ([1], [0, 1]),
}


@pytest.fixture(scope="module")
def golden():
    return torch.load(FIXTURE, weights_only=True)


def _build(golden, **kwargs):
    model = OmegaStyleAggregator(**golden["config"], **kwargs)
    return model


def _run(model, golden, condition="depth+camera"):
    depth_index, camera_index = CONDITIONS[condition]
    with torch.no_grad():
        return model.inference(**golden["inputs"], depth_gt_index=depth_index, camera_gt_index=camera_index)


@pytest.mark.parametrize("condition", list(CONDITIONS))
def test_v0_config_matches_zero_aggregator(golden, condition):
    model = _build(
        golden, num_register_tokens=4, register_attention_layers=(), global_rope=True, cached_layers=(0, 1, 2, 3)
    )
    model.load_state_dict(golden["state_dict"], strict=True)
    reference = ZeroAggregator(**golden["config"])
    reference.load_state_dict(golden["state_dict"], strict=True)
    layers, patch_start_idx = _run(model.eval(), golden, condition)
    want_layers, want_start = _run(reference.eval(), golden, condition)
    assert patch_start_idx == want_start == golden["outputs"]["patch_start_idx"]
    for got, want in zip(layers, want_layers, strict=True):
        assert torch.equal(got, want)


def _routed(golden, global_rope=True):
    model = _build(
        golden,
        num_register_tokens=4,
        register_attention_layers=(1,),
        global_rope=global_rope,
        cached_layers=(0, 1, 2, 3),
    )
    model.load_state_dict(golden["state_dict"], strict=True)
    return model.eval()


def test_register_layer_patch_tokens_bypass_block(golden):
    model = _routed(golden)
    layers, prefix = _run(model, golden)
    channels = layers[1].shape[-1] // 2
    frame_out, global_out = layers[1][..., :channels], layers[1][..., channels:]
    assert torch.equal(global_out[:, :, prefix:], frame_out[:, :, prefix:])
    assert not torch.equal(global_out[:, :, :prefix], frame_out[:, :, :prefix])


def test_register_layer_prefix_matches_prefix_only_block(golden):
    model = _routed(golden)
    layers, prefix = _run(model, golden)
    channels = layers[1].shape[-1] // 2
    frame_out, global_out = layers[1][..., :channels], layers[1][..., channels:]
    batch, frames = frame_out.shape[:2]
    prefix_tokens = frame_out[:, :, :prefix].reshape(batch, frames * prefix, channels)
    prefix_pos = torch.zeros(batch, frames * prefix, 2, dtype=torch.long)  # special tokens sit at position 0
    with torch.no_grad():
        expected = model.global_blocks[1](prefix_tokens, pos=prefix_pos).view(batch, frames, prefix, channels)
    torch.testing.assert_close(global_out[:, :, :prefix], expected, rtol=0, atol=1e-6)


def test_register_layer_is_active_in_forward(golden):
    """The training path (forward, random GT selection) routes the same layers."""
    model = _routed(golden).train()
    model.use_checkpoint = False
    layers, prefix = model(**golden["inputs"])
    channels = layers[1].shape[-1] // 2
    assert torch.equal(layers[1][:, :, prefix:, channels:], layers[1][:, :, prefix:, :channels])


def test_only_cached_layers_are_returned(golden):
    model = _build(
        golden, num_register_tokens=4, register_attention_layers=(1,), global_rope=True, cached_layers=(1, 3)
    )
    model.load_state_dict(golden["state_dict"], strict=True)
    layers, _ = _run(model.eval(), golden)
    assert len(layers) == 4
    assert layers[0] is None and layers[2] is None
    assert layers[1] is not None and layers[3] is not None


def test_global_rope_off_only_affects_global_blocks(golden):
    model = _build(
        golden, num_register_tokens=4, register_attention_layers=(1,), global_rope=False, cached_layers=(0, 1, 2, 3)
    )
    assert all(block.attn.rope is None for block in model.global_blocks)
    assert all(block.attn.rope is not None for block in model.frame_blocks)
    model.load_state_dict(golden["state_dict"], strict=True)
    layers, _ = _run(model.eval(), golden)
    assert all(torch.isfinite(layer).all() for layer in layers)


def test_register_count_16_sets_patch_start_idx_17(golden):
    model = _build(
        golden, num_register_tokens=16, register_attention_layers=(1,), global_rope=False, cached_layers=(0, 1, 2, 3)
    )
    channels = golden["config"]["embed_dim"]
    assert model.patch_start_idx == 17
    assert tuple(model.register_token.shape) == (1, 2, 16, channels)
    assert 5e-4 < model.register_token.detach().std().item() < 2e-3  # VGGT-Omega initialisation, std 1e-3
    layers, prefix = _run(model.eval(), golden)
    assert prefix == 17
    assert layers[-1].shape[2] == 17 + 2 * 3  # 28x42 image, 14-pixel patches


def test_encoder_register_count_stays_4_when_aggregator_has_16(golden, monkeypatch):
    seen = {}
    original = aggregator_module.Aggregator.__build_patch_embed__

    def spy(self, patch_embed, img_size, patch_size, num_register_tokens, **kwargs):
        seen["num_register_tokens"] = num_register_tokens
        return original(self, patch_embed, img_size, patch_size, num_register_tokens, **kwargs)

    monkeypatch.setattr(aggregator_module.Aggregator, "__build_patch_embed__", spy)
    _build(
        golden, num_register_tokens=16, register_attention_layers=(1,), global_rope=False, cached_layers=(0, 1, 2, 3)
    )
    assert seen["num_register_tokens"] == 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"register_attention_layers": (4,)},
        {"cached_layers": (0, 1, 2)},
        {"cached_layers": (0, 5, 3)},
    ],
)
def test_invalid_layer_indices_raise(golden, kwargs):
    base = {
        "num_register_tokens": 4,
        "register_attention_layers": (1,),
        "global_rope": True,
        "cached_layers": (0, 1, 2, 3),
    }
    with pytest.raises(ValueError):
        _build(golden, **{**base, **kwargs})
