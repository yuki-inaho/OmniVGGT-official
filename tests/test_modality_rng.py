"""Training-time choice of the views that get the auxiliary camera/depth follows a keyed generator
(seed, rank, epoch, micro-batch), so runs replay and a resumed run sees the same choices."""

import inspect
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from test_depth_all_views import FIXTURE, _tiny

import train_utils
from omnivggt.models.omnivggt import OmniVGGT
from omnivggt.models.omnivggt_aggregator import ZeroAggregator

REPO = Path(train_utils.__file__).resolve().parent


def _selections(model, keys):
    seen = []
    agg = model.aggregator
    cam, dep = agg.select_camera_gt, agg.training_depth_gt_index
    agg.select_camera_gt = lambda *a, **k: seen.append(("cam", cam(*a, **k))) or seen[-1][1]
    agg.training_depth_gt_index = lambda *a, **k: seen.append(("dep", dep(*a, **k))) or seen[-1][1]
    golden = torch.load(FIXTURE, weights_only=True)
    with torch.no_grad():
        for key in keys:
            model(**golden["inputs"], return_points=False, modality_rng=train_utils.modality_rng(*key))
    return seen


@pytest.mark.parametrize("all_views", [False, True])
def test_selection_replays_for_the_same_key(all_views):
    keys = [(42, 0, epoch, step) for epoch in range(2) for step in range(10)]
    first = _selections(_tiny(depth_all_views=all_views, depth_drop_prob=0.3).train(), keys)
    again = _selections(_tiny(depth_all_views=all_views, depth_drop_prob=0.3).train(), keys)
    resumed = _selections(_tiny(depth_all_views=all_views, depth_drop_prob=0.3).train(), keys[10:])
    assert first == again
    assert first[20:] == resumed  # resume from checkpoint-epoch-1 replays epoch 2


def test_forward_draws_camera_then_depth_from_the_passed_generator():
    model = _tiny(depth_drop_prob=0.3).train()
    got = _selections(model, [(42, 0, 0, s) for s in range(20)])
    want = []
    for s in range(20):
        rng = train_utils.modality_rng(42, 0, 0, s)
        want += [("cam", ZeroAggregator.select_camera_gt(None, 3, 0.1, rng=rng)),
                 ("dep", ZeroAggregator.select_depth_gt(None, 3, 0.3, rng=rng))]
    assert got == want


def test_rank_and_step_change_the_stream():
    def draw(*key):
        return ZeroAggregator.select_camera_gt(None, 12, 0.1, rng=train_utils.modality_rng(*key))

    assert [draw(42, 0, 0, s) for s in range(16)] != [draw(42, 1, 0, s) for s in range(16)]
    assert [draw(42, 0, 0, s) for s in range(16)] != [draw(42, 0, 1, s) for s in range(16)]


@pytest.mark.parametrize("frames", [2, 3, 4, 6, 12])
def test_camera_statistics_unchanged(frames):
    n = 40_000
    lengths = Counter()
    for i in range(n):
        index = ZeroAggregator.select_camera_gt(None, frames, 0.1, rng=train_utils.modality_rng(42, 0, 0, i))
        assert index == list(range(len(index)))  # always a prefix
        lengths[len(index)] += 1
    for k in range(frames + 1):
        p = 0.1 + 0.9 / (frames + 1) if k == 0 else 0.9 / (frames + 1)
        assert abs(lengths[k] / n - p) < 5 * np.sqrt(p * (1 - p) / n)


def test_training_loop_passes_the_generator():
    source = (REPO / "train_omnivggt.py").read_text()
    assert "modality_rng=" in source and "accelerator.process_index" in source


def test_both_models_accept_the_generator():
    assert "modality_rng" in inspect.signature(OmniVGGT.forward).parameters  # same training loop for OmniVGGT
    assert "modality_rng" in inspect.signature(ZeroAggregator.forward).parameters
