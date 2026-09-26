"""Training on the stream: ``StreamingOmega(train=True)`` backpropagates through the current frame only, over a
KV cache that stores detached keys and values (one-frame BPTT, XStream-style)."""

import pytest
import torch
import torch.nn as nn
from test_kv_cache import _frame_kv, _queries
from test_stream_streaming import (
    DIM,
    HEADS,
    OUTPUTS,
    TOKENS,
    _block,
    _causal_model,
    _frame_inputs,
    _sequence,
    _stream,
)

from omnivggt.heads.camera_head import NUM_ITERATIONS
from omnivggt.layers.block import Block
from omnivggt.stream.kv_cache import CachePolicy, LayerKVCache
from omnivggt.stream.streaming import StreamingOmega, all_caches, stream_block

BOUNDED_INT8 = CachePolicy(recent=1, long_special=1, long_patch=4, selector="query", quant="int8")
POLICIES = {
    "full": CachePolicy.full(),
    "bounded_int8": BOUNDED_INT8,
    "long_frames_anchors": CachePolicy(recent=1, long_special=1, long_frames=1, selector="diversity",
                                       anchor_every=2, max_anchors=1),
    "retrieve": CachePolicy(recent=None, retrieve_frames=1, retrieve_camera=True),
}


# --- LayerKVCache: detached storage, the current frame read with its gradient ---------------------------------


def _grad_kv(frame_id, tokens):
    k, v = _frame_kv(frame_id, tokens)
    return k.requires_grad_(), v.requires_grad_()


def test_the_cache_stores_detached_keys_and_values():
    cache = LayerKVCache(CachePolicy.full(), 3, 1, torch.float64, layer_id=0)
    k, v = _grad_kv(1, 3)
    cache.append(k, v, 1)
    keys, values = cache.read()
    assert not keys.requires_grad and not values.requires_grad
    assert torch.equal(keys, k.detach()) and torch.equal(values, v.detach())


@pytest.mark.parametrize("policy", [CachePolicy.full(), BOUNDED_INT8])
def test_reading_the_current_frame_with_its_gradient(policy):
    tokens, special = 6, 2
    cache = LayerKVCache(policy, tokens, special, torch.float64, layer_id=0)
    for t in range(1, 5):
        k, v = _grad_kv(t, tokens)
        cache.append(k, v, t)
        stored = cache.read()
        keys, values = cache.read(current=(k, v))
        assert torch.equal(keys, stored[0]) and torch.equal(values, stored[1])  # the same rows, in the same order
        buffer = cache._keys.untyped_storage().data_ptr()
        assert keys.untyped_storage().data_ptr() != buffer  # a new tensor: later in-place writes cannot reach it
        (keys.sum() + 2 * values.sum()).backward()
        assert torch.equal(k.grad, torch.ones_like(k)) and torch.equal(v.grad, torch.full_like(v, 2.0))
        cache.commit(_queries(tokens), t, (2, 2))


def test_reading_selected_frames_with_the_current_gradient():
    cache = LayerKVCache(CachePolicy.full(), 3, 1, torch.float64, layer_id=0)
    for t in range(1, 4):
        cache.append(*_frame_kv(t, 3), t)
        cache.commit(_queries(3), t)
    k, v = _grad_kv(4, 3)
    cache.append(k, v, 4)
    frames = torch.tensor([1, 3, 4])
    keys, values = cache.read(frames, current=(k, v))
    stored = cache.read(frames)
    assert torch.equal(keys, stored[0]) and torch.equal(values, stored[1])
    keys[:, :, -3:].sum().backward()  # the current frame's rows are the last ones read
    assert torch.equal(k.grad, torch.ones_like(k))


def test_reading_the_current_frame_needs_a_pending_frame_of_its_shape():
    cache = LayerKVCache(CachePolicy.full(), 3, 1, torch.float64, layer_id=0)
    k, v = _frame_kv(1, 3)
    cache.append(k, v, 1)
    with pytest.raises(ValueError, match="shape"):
        cache.read(current=(k[:, :, :2], v[:, :, :2]))
    cache.commit(_queries(3), 1)
    with pytest.raises(RuntimeError, match="pending"):
        cache.read(current=(k, v))


# --- stream_block in training -------------------------------------------------------------------------------


def test_stream_block_training_runs_a_training_block_and_detaches_the_cache():
    block = _block().train()
    cache = LayerKVCache(CachePolicy.full(), TOKENS, 2, torch.float64, layer_id=0)
    torch.manual_seed(1)
    for t in (1, 2):
        x = torch.randn(1, TOKENS, DIM, dtype=torch.float64, requires_grad=True)
        out = stream_block(block, x, cache, t, grid_hw=(2, 2), train=True)
        out.square().sum().backward()
        assert x.grad is not None and bool(x.grad.abs().sum() > 0)
        assert not any(tensor.requires_grad for tensor in cache.read())


def test_stream_block_training_rejects_stochastic_depth():
    torch.manual_seed(0)
    block = Block(DIM, HEADS, init_values=0.01, qk_norm=True, drop_path=0.2).double().train()
    cache = LayerKVCache(CachePolicy.full(), TOKENS, 2, torch.float64, layer_id=0)
    with pytest.raises(NotImplementedError, match="stochastic depth"):
        stream_block(block, torch.zeros(1, TOKENS, DIM, dtype=torch.float64), cache, 1, train=True)


# --- StreamingOmega(train=True) --------------------------------------------------------------------------------


def _frame_loss(out):
    """A scalar that reads every per-frame output the training loss reads."""
    return (sum(out[key].square().mean() for key in OUTPUTS)
            + sum(pose.square().mean() for pose in out["pose_enc_list"]))


def _check_detached(stream):
    for cache in all_caches(stream.caches):
        assert not any(tensor.requires_grad for tensor in cache.read()), cache.layer_id


def test_training_step_returns_every_prediction_with_its_gradient():
    model, sequence = _causal_model().train(), _sequence(frames=2)
    stream = StreamingOmega(model, CachePolicy.full(), train=True)
    out = stream.step(*_frame_inputs(sequence, 0, "depth"))
    assert {*OUTPUTS, "pose_enc_list"} <= set(out)
    assert len(out["pose_enc_list"]) == NUM_ITERATIONS and torch.equal(out["pose_enc_list"][-1], out["pose_enc"])
    assert all(out[key].requires_grad for key in OUTPUTS)
    model.eval()
    inference = StreamingOmega(model, CachePolicy.full()).step(*_frame_inputs(sequence, 0, "depth"))
    assert set(inference) == set(OUTPUTS) and not any(value.requires_grad for value in inference.values())


@pytest.mark.parametrize("policy", list(POLICIES))
def test_backward_after_every_frame_over_a_detached_cache(policy):
    model, sequence = _causal_model().train(), _sequence(frames=3)
    stream = StreamingOmega(model, POLICIES[policy], train=True)
    for frame in range(3):
        _frame_loss(stream.step(*_frame_inputs(sequence, frame, "depth"))).backward()
        _check_detached(stream)
    aggregator, head = model.aggregator, model.camera_head
    for parameter in (aggregator.global_blocks[0].attn.qkv.weight, aggregator.global_blocks[1].attn.qkv.weight,
                      head.trunk[0].attn.qkv.weight, model.depth_head.scratch.output_conv2[2].weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert bool(parameter.grad.abs().sum() > 0)


@pytest.mark.parametrize("policy", ["full", "bounded_int8"])
def test_frame_gradient_equals_a_fresh_run_on_the_same_detached_cache(policy):
    model, sequence = _causal_model().train(), _sequence(frames=3)
    trained = StreamingOmega(model, POLICIES[policy], train=True)
    for frame in range(2):
        _frame_loss(trained.step(*_frame_inputs(sequence, frame, "depth"))).backward()
    before = [tuple(tensor.clone() for tensor in cache.read()) for cache in all_caches(trained.caches)]
    model.zero_grad(set_to_none=True)
    _frame_loss(trained.step(*_frame_inputs(sequence, 2, "depth"))).backward()
    got = {name: parameter.grad.clone() for name, parameter in model.named_parameters() if parameter.grad is not None}

    model.zero_grad(set_to_none=True)
    fresh = StreamingOmega(model, POLICIES[policy], train=True)
    with torch.no_grad():  # frames 1-2 build no graph at all
        for frame in range(2):
            fresh.step(*_frame_inputs(sequence, frame, "depth"))
    for (keys, values), cache in zip(before, all_caches(fresh.caches), strict=True):
        assert torch.equal(cache.read()[0], keys) and torch.equal(cache.read()[1], values)
    _frame_loss(fresh.step(*_frame_inputs(sequence, 2, "depth"))).backward()
    want = {name: parameter.grad for name, parameter in model.named_parameters() if parameter.grad is not None}
    assert got.keys() == want.keys() and len(got) > 0
    for name in got:
        assert torch.equal(got[name], want[name]), name


@pytest.mark.parametrize("condition", ["depth", "rgb"])
@pytest.mark.parametrize("policy", ["full", "bounded_int8"])
def test_training_stream_forward_equals_the_inference_stream(policy, condition):
    model, sequence = _causal_model(), _sequence()
    assert all(block.sample_drop_ratio == 0 for block in model.modules() if isinstance(block, Block))
    assert all(dropout.p == 0 for dropout in model.modules() if isinstance(dropout, nn.Dropout))
    inference = _stream(StreamingOmega(model, POLICIES[policy]), sequence, condition)
    model.train()
    training = _stream(StreamingOmega(model, POLICIES[policy], train=True), sequence, condition)
    for key in OUTPUTS:
        assert training[key].requires_grad, key
        assert torch.equal(training[key].detach(), inference[key]), key


def test_training_cache_bytes_stay_at_the_budget_over_48_frames():
    policy = CachePolicy(recent=1, long_special=1, long_patch=4, selector="query")
    model, sequence = _causal_model().train(), _sequence(frames=48)
    stream = StreamingOmega(model, policy, train=True)
    kv_bytes = []
    for frame in range(48):
        _frame_loss(stream.step(*_frame_inputs(sequence, frame, "depth"))).backward()
        kv_bytes.append(stream.kv_bytes())
        for cache in all_caches(stream.caches):
            assert cache.invariants()["ok"] and cache.size <= cache.budget, cache.layer_id
    _check_detached(stream)
    assert all(cache.size == cache.budget for cache in all_caches(stream.caches))  # every store is full
    assert len(set(kv_bytes[2:])) == 1 and kv_bytes[2] > kv_bytes[1]  # full from frame 3 on, then constant


def test_a_training_model_needs_a_training_stream():
    with pytest.raises(ValueError, match="train=True"):
        StreamingOmega(_causal_model().train(), CachePolicy.full())
