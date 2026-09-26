"""Streaming (one frame per step) frame-causal OmniVGGTOmega: ``stream_block`` and ``StreamingOmega``."""

import pytest
import torch
from test_stream_causal import _model

from omnivggt.heads.camera_head import NUM_ITERATIONS
from omnivggt.layers.block import Block
from omnivggt.layers.rope import PositionGetter, RotaryPositionEmbedding2D
from omnivggt.stream.kv_cache import CachePolicy, LayerKVCache
from omnivggt.stream.masks import frame_causal_mask
from omnivggt.stream.streaming import StreamingOmega, all_caches, stream_block

# --- stream_block --------------------------------------------------------------------------------------------

DIM, HEADS, SPECIAL, GRID = 16, 2, 2, (2, 2)
TOKENS = SPECIAL + GRID[0] * GRID[1]


def _block(rope=False):
    torch.manual_seed(0)
    return Block(DIM, HEADS, init_values=0.01, qk_norm=True,
                 rope=RotaryPositionEmbedding2D(100) if rope else None).double().eval()


def _frame_positions(batch, frames):
    """Positions as the aggregator builds them: 0 for the special tokens, grid position + 1 for the patches."""
    patches = PositionGetter()(batch * frames, *GRID, device="cpu") + 1
    special = torch.zeros(batch * frames, SPECIAL, 2, dtype=patches.dtype)
    return torch.cat([special, patches], dim=1).view(batch, frames * TOKENS, 2)


@pytest.mark.parametrize("rope", [False, True])
def test_stream_block_matches_masked_block(rope):
    frames, batch = 4, 2
    block = _block(rope)
    torch.manual_seed(1)
    x = torch.randn(batch, frames * TOKENS, DIM, dtype=torch.float64)
    pos = _frame_positions(batch, frames)
    with torch.no_grad():
        whole = block(x, pos=pos, attn_mask=frame_causal_mask(frames, TOKENS, "cpu"))
        cache = LayerKVCache(CachePolicy.full(), TOKENS, SPECIAL, torch.float64, layer_id=0)
        for t in range(1, frames + 1):
            rows = slice((t - 1) * TOKENS, t * TOKENS)
            out = stream_block(block, x[:, rows], cache, t, grid_hw=GRID, pos=pos[:, rows])
            torch.testing.assert_close(out, whole[:, rows], rtol=0, atol=1e-12)
    assert cache.last_frame == frames and cache.size == frames * TOKENS


def test_stream_block_needs_an_eval_block():
    block = _block().train()
    cache = LayerKVCache(CachePolicy.full(), TOKENS, SPECIAL, torch.float64, layer_id=0)
    with pytest.raises(NotImplementedError, match="eval"):
        stream_block(block, torch.zeros(1, TOKENS, DIM, dtype=torch.float64), cache, 1)


# --- StreamingOmega (full cache) -----------------------------------------------------------------------------

STREAM_FRAMES = 5
CONDITIONS = ("depth", "rgb")
OUTPUTS = ("pose_enc", "depth", "depth_conf", "world_points")


def _causal_model(**kwargs):
    return _model(causal=True, depth_norm="first_frame", **kwargs)


def _sequence(frames=STREAM_FRAMES, seed=0):
    """Images [1, S, 3, H, W], depth [1, S, H, W, 1] and mask [1, S, H, W] at the TINY resolution (2 x 3 patches)."""
    generator = torch.Generator().manual_seed(seed)
    shape = (1, frames, 28, 42)
    return {
        "images": torch.rand(1, frames, 3, 28, 42, generator=generator, dtype=torch.float64),
        "depth": 0.5 + torch.rand(*shape, 1, generator=generator, dtype=torch.float64),
        "mask": (torch.rand(*shape, generator=generator) > 0.2).double(),
    }


def _batch(model, sequence, condition):
    frames = sequence["images"].shape[1]
    with torch.no_grad():
        return model.inference(**sequence, depth_gt_index=list(range(frames)) if condition == "depth" else [],
                               camera_gt_index=[])


def _frame_inputs(sequence, frame, condition):
    if condition == "rgb":
        return (sequence["images"][:, frame],)
    return sequence["images"][:, frame], sequence["depth"][:, frame], sequence["mask"][:, frame]


def _stream(stream, sequence, condition, frames=None):
    """Outputs of ``stream.step`` over the frames of ``sequence``, concatenated along the frame axis."""
    steps = [stream.step(*_frame_inputs(sequence, frame, condition))
             for frame in range(frames or sequence["images"].shape[1])]
    return {key: torch.cat([step[key] for step in steps], dim=1) for key in OUTPUTS}


def _relative_error(got, want):
    return ((got - want).abs().max() / want.abs().max()).item()


def _contexts_unset(model):
    return model.aggregator._stream is None and model.camera_head._stream is None


@pytest.mark.parametrize("condition", CONDITIONS)
def test_full_cache_stream_equals_the_batch_causal_model(condition):
    model, sequence = _causal_model(), _sequence()
    batch = _batch(model, sequence, condition)
    streamed = _stream(StreamingOmega(model, CachePolicy.full()), sequence, condition)
    for key in OUTPUTS:
        assert streamed[key].shape == batch[key].shape, key
        assert _relative_error(streamed[key], batch[key]) <= 1e-10, key
    assert _contexts_unset(model)


def test_stream_uses_the_reference_special_tokens_only_at_t1():
    model, sequence = _causal_model(), _sequence(frames=3)
    aggregator = model.aggregator
    seen = []
    handle = aggregator.frame_blocks[0].register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
    _stream(StreamingOmega(model, CachePolicy.full()), sequence, "depth")
    handle.remove()
    prefix = aggregator.patch_start_idx
    assert len(seen) == 3
    for t, tokens in enumerate(seen, start=1):
        reference = 0 if t == 1 else 1
        assert torch.equal(tokens[0, 1:prefix], aggregator.register_token[0, reference]), t
        assert torch.equal(tokens[0, 0], aggregator.camera_token[0, reference, 0]), t  # the adapter input is zero


def test_cache_counts_follow_the_model_configuration():
    model = _causal_model()
    stream = StreamingOmega(model, CachePolicy.full())
    assert stream.caches is None and stream.kv_bytes() == 0
    _stream(stream, _sequence(frames=2), "depth")
    aggregator, head = model.aggregator, model.camera_head
    register_layers = sorted(aggregator.register_attention_layers)
    assert sorted(stream.caches["register"]) == register_layers == [1]
    assert sorted(stream.caches["global"]) == [i for i in range(aggregator.depth) if i not in register_layers]
    assert len(stream.caches["camera"]) == NUM_ITERATIONS == 4
    assert all(len(row) == head.trunk_depth == 1 for row in stream.caches["camera"])
    prefix = aggregator.patch_start_idx
    assert all(cache.tokens_per_frame == prefix + 2 * 3 for cache in stream.caches["global"].values())
    assert all(cache.tokens_per_frame == prefix for cache in stream.caches["register"].values())
    assert all(cache.tokens_per_frame == 1 for row in stream.caches["camera"] for cache in row)
    caches = [*stream.caches["global"].values(), *stream.caches["register"].values(),
              *(cache for row in stream.caches["camera"] for cache in row)]
    assert all(cache.last_frame == 2 for cache in caches)
    assert stream.kv_bytes() == sum(cache.nbytes() for cache in caches) > 0


def test_reset_restarts_the_stream_at_t1():
    model, sequence = _causal_model(), _sequence()
    fresh = _stream(StreamingOmega(model, CachePolicy.full()), sequence, "depth")
    stream = StreamingOmega(model, CachePolicy.full())
    _stream(stream, _sequence(seed=3), "depth", frames=3)
    stream.reset()
    assert stream.t == 0 and stream.caches is None
    again = _stream(stream, sequence, "depth")
    assert stream.t == STREAM_FRAMES
    for key in OUTPUTS:
        assert torch.equal(again[key], fresh[key]), key


def test_a_failing_step_unsets_the_context_and_needs_a_reset(monkeypatch):
    model, sequence = _causal_model(), _sequence(frames=3)
    stream = StreamingOmega(model, CachePolicy.full())
    stream.step(*_frame_inputs(sequence, 0, "depth"))

    def broken_head(*args, **kwargs):
        raise RuntimeError("depth head failure")

    monkeypatch.setattr(model.depth_head, "forward", broken_head)
    with pytest.raises(RuntimeError, match="depth head failure"):
        stream.step(*_frame_inputs(sequence, 1, "depth"))
    assert _contexts_unset(model)
    monkeypatch.undo()
    batch = _batch(model, sequence, "depth")  # the batch path works as before
    reference = _batch(_causal_model(), sequence, "depth")
    for key in OUTPUTS:
        assert torch.equal(batch[key], reference[key]), key
    with pytest.raises(RuntimeError, match="reset"):
        stream.step(*_frame_inputs(sequence, 1, "depth"))
    stream.reset()
    streamed = _stream(stream, sequence, "depth")
    for key in OUTPUTS:
        assert _relative_error(streamed[key], batch[key]) <= 1e-10, key


@pytest.mark.parametrize("options", [dict(causal=False, depth_norm="first_frame"),
                                     dict(causal=True, depth_norm="joint")])
def test_stream_needs_the_causal_first_frame_model(options):
    with pytest.raises(ValueError, match=r"causal|first_frame"):
        StreamingOmega(_model(**options), CachePolicy.full())


def test_stream_needs_an_eval_model():
    with pytest.raises(ValueError, match="eval"):
        StreamingOmega(_causal_model().train(), CachePolicy.full())


def test_depth_after_an_rgb_first_frame_raises():
    sequence = _sequence(frames=2)
    stream = StreamingOmega(_causal_model(), CachePolicy.full())
    stream.step(*_frame_inputs(sequence, 0, "rgb"))
    with pytest.raises(ValueError, match="frame 1"):
        stream.step(*_frame_inputs(sequence, 1, "depth"))


def test_first_frame_without_valid_depth_raises():
    sequence = _sequence(frames=1)
    sequence["mask"][:] = 0
    stream = StreamingOmega(_causal_model(), CachePolicy.full())
    with pytest.raises(ValueError, match="valid"):
        stream.step(*_frame_inputs(sequence, 0, "depth"))
    assert stream.t == 0 and _contexts_unset(stream.model)


def _cache_storages(stream):
    return {tensor.untyped_storage().data_ptr() for cache in all_caches(stream.caches) for tensor in cache.read()}


@pytest.mark.parametrize("condition", CONDITIONS)
def test_stream_outputs_ignore_later_frames(condition):
    model, sequence, other = _causal_model(), _sequence(), _sequence(seed=1)
    stream = StreamingOmega(model, CachePolicy.full())
    past = [stream.step(*_frame_inputs(sequence, frame, condition)) for frame in range(STREAM_FRAMES - 1)]
    saved = [{key: value.clone() for key, value in step.items()} for step in past]
    changed_last = stream.step(*_frame_inputs(other, STREAM_FRAMES - 1, condition))
    storages = _cache_storages(stream)
    for step, step_saved in zip(past, saved, strict=True):  # returned outputs are not views of the caches
        for key in OUTPUTS:
            assert torch.equal(step[key], step_saved[key]), key
            assert step[key].untyped_storage().data_ptr() not in storages, key
    runs = [[stream.step(*_frame_inputs(sequence, frame, condition)) for frame in range(STREAM_FRAMES)]
            for stream in (StreamingOmega(model, CachePolicy.full()), StreamingOmega(model, CachePolicy.full()))]
    for frame in range(STREAM_FRAMES):
        for key in OUTPUTS:  # the same input twice gives the same bits
            assert torch.equal(runs[0][frame][key], runs[1][frame][key]), (frame, key)
            if frame < STREAM_FRAMES - 1:  # another frame 5 leaves frames 1-4 alone
                assert torch.equal(runs[0][frame][key], saved[frame][key]), (frame, key)
    assert not torch.equal(runs[0][-1]["pose_enc"], changed_last["pose_enc"])


@pytest.mark.parametrize("selector, store", [
    ("query", "long_patch"), ("xstream", "long_patch"), ("recency", "long_patch"), ("random", "long_patch"),
    ("diversity", "long_patch"), ("query", "long_frames"), ("diversity", "long_frames"),
])
def test_bounded_cache_larger_than_the_history_equals_the_full_cache(selector, store):
    model, sequence = _causal_model(), _sequence()
    batch = _batch(model, sequence, "depth")
    roomy = CachePolicy(recent=2, long_special=100, selector=selector, **{store: 10_000})
    stream = StreamingOmega(model, roomy)
    streamed = _stream(stream, sequence, "depth")
    for key in OUTPUTS:
        assert _relative_error(streamed[key], batch[key]) <= 1e-10, key
    assert all(cache.size == STREAM_FRAMES * cache.tokens_per_frame for cache in all_caches(stream.caches))


@pytest.mark.parametrize("selector, quant", [("query", None), ("xstream", "int8"), ("recency", "int4"),
                                             ("random", None)])
def test_small_bounded_caches_stay_within_budget_over_20_frames(selector, quant):
    policy = CachePolicy(recent=1, long_special=1, long_patch=4, selector=selector, quant=quant)
    model, sequence = _causal_model(), _sequence(frames=20)
    stream = StreamingOmega(model, policy)
    tokens = model.aggregator.patch_start_idx + 2 * 3
    budgets = {"global": 2 * tokens + 17 + 4, "register": 17 * 3, "camera": 3}
    for frame in range(20):
        out = stream.step(*_frame_inputs(sequence, frame, "depth"))
        assert all(torch.isfinite(out[key]).all() for key in OUTPUTS)
        for kind in ("global", "register"):
            for cache in stream.caches[kind].values():
                invariants = cache.invariants()
                assert invariants["ok"] and cache.size <= cache.budget == budgets[kind], (kind, invariants)
        for cache in (cache for row in stream.caches["camera"] for cache in row):
            assert cache.invariants()["ok"] and cache.size <= cache.budget == budgets["camera"]
    assert stream.kv_bytes() == sum(cache.nbytes() for cache in all_caches(stream.caches))


@pytest.mark.parametrize("policy", [
    dict(recent=1, long_special=1, long_patch=4, selector="diversity", quant="int8"),
    dict(recent=1, long_special=1, long_frames=1, selector="query", quant="int8"),
    dict(recent=1, long_special=2, long_frames=2, selector="diversity", quant="int4"),
])
def test_phase5_bounded_caches_stay_within_budget_over_20_frames(policy):
    policy = CachePolicy(**policy)
    model, sequence = _causal_model(), _sequence(frames=20)
    stream = StreamingOmega(model, policy)
    for frame in range(20):
        out = stream.step(*_frame_inputs(sequence, frame, "depth"))
        assert all(torch.isfinite(out[key]).all() for key in OUTPUTS)
        for cache in all_caches(stream.caches):
            invariants = cache.invariants()
            assert invariants["ok"] and cache.size <= cache.budget, (cache.layer_id, invariants)
    tokens, special = model.aggregator.patch_start_idx + 2 * 3, model.aggregator.patch_start_idx
    assert stream.caches["global"][0].budget == policy.budget(tokens, special)
    assert stream.caches["global"][0].size == policy.budget(tokens, special)  # every store is full after 20 frames


def test_backbone_caches_store_the_given_dtype_and_camera_caches_the_head_dtype():
    model, sequence = _causal_model(), _sequence(frames=3)
    batch = _batch(model, sequence, "depth")
    stream = StreamingOmega(model, CachePolicy.full(), dtype=torch.float32)
    streamed = _stream(stream, sequence, "depth")
    backbone = [*stream.caches["global"].values(), *stream.caches["register"].values()]
    assert all(cache.read()[0].dtype == torch.float32 for cache in backbone)
    assert all(cache.read()[0].dtype == torch.float64 for row in stream.caches["camera"] for cache in row)
    assert streamed["pose_enc"].dtype == torch.float64  # the stored keys are cast to the queries' dtype
    for key in OUTPUTS:
        assert _relative_error(streamed[key], batch[key]) <= 1e-5, key


def test_frames_of_another_size_raise():
    stream = StreamingOmega(_causal_model(), CachePolicy.full())
    stream.step(torch.rand(1, 3, 28, 42, dtype=torch.float64))
    with pytest.raises(ValueError, match="size"):
        stream.step(torch.rand(1, 3, 28, 28, dtype=torch.float64))
