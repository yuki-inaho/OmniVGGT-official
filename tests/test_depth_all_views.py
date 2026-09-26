"""Training option for RGB-D-only use: give the auxiliary depth to every view instead of a random subset."""

from pathlib import Path

import numpy as np
import pytest
import torch
from accelerate import PartialState

import train_utils
from omnivggt.models.omnivggt_omega import OmniVGGTOmega

FIXTURE = Path(__file__).parent / "fixtures" / "zero_aggregator_golden.pt"
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


def _tiny(**kwargs):
    torch.manual_seed(0)
    return OmniVGGTOmega(**TINY, **kwargs)


def test_all_views_gives_depth_to_every_view():
    aggregator = _tiny(depth_all_views=True, depth_drop_prob=0.3).aggregator
    assert aggregator.depth_all_views is True
    for frames in range(1, 13):
        for seed in range(30):
            assert aggregator.training_depth_gt_index(frames, rng=np.random.default_rng(seed)) == list(range(frames))


def test_default_keeps_the_random_subset():
    aggregator = _tiny(depth_drop_prob=0.3).aggregator
    assert aggregator.depth_all_views is False
    for frames in (2, 6, 12):
        got = [aggregator.training_depth_gt_index(frames, rng=np.random.default_rng(s)) for s in range(200)]
        want = [aggregator.select_depth_gt(frames, 0.3, rng=np.random.default_rng(s)) for s in range(200)]
        assert got == want
        assert any(len(index) < frames for index in got)


@pytest.mark.parametrize("all_views", [False, True])
def test_training_forward_uses_the_option(monkeypatch, all_views):
    golden = torch.load(FIXTURE, weights_only=True)
    model = _tiny(depth_all_views=all_views).train()
    calls = []
    original = model.aggregator.select_depth_gt
    monkeypatch.setattr(model.aggregator, "select_depth_gt", lambda *a, **k: calls.append(a) or original(*a, **k))
    model(**golden["inputs"])
    assert (len(calls) == 0) == all_views


def test_build_model_passes_the_option(monkeypatch):
    PartialState()
    built = {}

    def fake_from_variant(path, **kwargs):
        built.update(kwargs)
        return _tiny(**kwargs)

    monkeypatch.setattr(OmniVGGTOmega, "from_variant", staticmethod(fake_from_variant))
    cfg = {"model_name": "omnivggt_omega", "omega_variant": "configs/omnivggt_omega/variants/V5.json"}
    model = train_utils.build_model({**cfg, "depth_all_views": True})
    assert built["depth_all_views"] is True and model.aggregator.depth_all_views is True
    assert train_utils.build_model(cfg).aggregator.depth_all_views is False
