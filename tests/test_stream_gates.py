"""Correctness gates on a stream (equivalence, future invariance, cache invariants, quantize-once) and the
per-step efficiency bench, on the tiny frame-causal model."""

import bench_stream
import pytest
import stream_gates
import torch
from test_stream_causal import _inputs, _model, _with_other_last_frame

from omnivggt.stream.kv_cache import CachePolicy

BOUNDED = CachePolicy(recent=1, long_special=1, long_patch=3, selector="query", quant="int8")


def _causal():
    return _model(causal=True, depth_norm="first_frame")


def test_equivalence_gate_passes_for_the_causal_model():
    gate = stream_gates.equivalence(_causal(), _inputs())
    assert gate["ok"] and gate["pose_enc_max_abs"] <= 1e-10 and gate["depth_max_rel"] <= 1e-10


def test_future_gate_passes_for_the_causal_model_and_fails_without_the_frame_mask():
    inputs = _inputs()
    gate = stream_gates.future_invariance(_causal(), inputs, _with_other_last_frame(inputs))
    assert gate["ok"] and gate["past_max_abs"] == 0.0
    leaky = stream_gates.future_invariance_batch(_model(), inputs, _with_other_last_frame(inputs))
    assert not leaky["ok"] and leaky["past_max_abs"] > 0.0


def test_cache_gate_checks_invariants_and_quantize_once_every_step():
    gate = stream_gates.cache_invariants(_causal(), _inputs(), BOUNDED, quant_layers=(0, 3))
    assert gate["ok"] and gate["steps"] == 4
    assert gate["invariants_failed"] == [] and gate["requantized_rows"] == 0
    assert gate["compared_rows"] > 0  # long-patch rows that stayed across a step were compared


def test_quantize_once_comparison_detects_changed_rows():
    rows = {(2, 5): (torch.ones(3), torch.zeros(3))}
    assert stream_gates.changed_rows(rows, {(2, 5): (torch.ones(3), torch.zeros(3))}) == (1, 0)
    assert stream_gates.changed_rows(rows, {(2, 5): (torch.ones(3) * 1.001, torch.zeros(3))}) == (1, 1)
    assert stream_gates.changed_rows(rows, {(3, 5): (torch.ones(3), torch.zeros(3))}) == (0, 0)


def test_bench_records_the_requested_steps_and_growing_full_cache():
    model = _causal()
    rows = bench_stream.run(model, CachePolicy.full(), steps=(1, 2, 4), image_hw=(28, 42), dtype=torch.float64,
                            device="cpu")
    assert [row["t"] for row in rows] == [1, 2, 4]
    assert all(row["step_ms"] > 0 for row in rows)
    assert rows[0]["kv_bytes"] < rows[1]["kv_bytes"] < rows[2]["kv_bytes"]


def test_bench_bounded_cache_stops_growing():
    unquantized = CachePolicy(recent=1, long_special=1, long_patch=3, selector="query")
    rows = bench_stream.run(_causal(), unquantized, steps=(4, 6, 8), image_hw=(28, 42), dtype=torch.float64,
                            device="cpu")
    assert rows[0]["kv_bytes"] == rows[1]["kv_bytes"] == rows[2]["kv_bytes"]
    # quantised long patches keep one scale/offset block per source frame, so only that metadata varies
    full = bench_stream.run(_causal(), CachePolicy.full(), steps=(4, 8), image_hw=(28, 42), dtype=torch.float64,
                            device="cpu")
    quantized = bench_stream.run(_causal(), BOUNDED, steps=(4, 6, 8), image_hw=(28, 42), dtype=torch.float64,
                                 device="cpu")
    spread = max(row["kv_bytes"] for row in quantized) - min(row["kv_bytes"] for row in quantized)
    assert spread < 0.01 * (full[1]["kv_bytes"] - full[0]["kv_bytes"])


@pytest.mark.parametrize("text", ["1,2,4", "1, 32 ,128"])
def test_bench_step_list(text):
    assert bench_stream.parse_steps(text) == tuple(int(x) for x in text.split(","))


def test_eval_stream_runs_the_real_streamer_and_matches_its_batch_mode():
    import eval_stream

    model, inputs = _causal(), _inputs()
    meter = eval_stream.Meter("cpu")
    stream = eval_stream.make_predictor("stream", model, "full", "fp32", meter)(inputs, True)
    batch = eval_stream.make_predictor("bidir_f0", model, None, "fp32", meter)(inputs, True)
    assert stream["pose_enc"].shape == batch["pose_enc"].shape == (1, 4, 9)
    assert len(stream["step_ms"]) == 4 and stream["kv_bytes"] == sorted(stream["kv_bytes"])
    # the same causal model: streaming reproduces one batch forward (bidir_f0 only names the model options);
    # eval_stream keeps predictions in fp32, so the tolerance is fp32 rounding
    torch.testing.assert_close(stream["pose_enc"], batch["pose_enc"], atol=1e-6, rtol=0)


def test_full_cache_is_allocated_once_for_a_known_stream_length():
    from omnivggt.stream.streaming import StreamingOmega, all_caches

    model, inputs = _causal(), _inputs()
    streamer = StreamingOmega(model, CachePolicy.full())
    streamer.reset(max_frames=4)
    capacities = []
    for t in range(4):
        streamer.step(inputs["images"][:, t], inputs["depth"][:, t], inputs["mask"][:, t])
        capacities.append([cache.capacity for cache in all_caches(streamer.caches)])
    assert capacities[0] == capacities[-1]  # no reallocation while streaming
    assert all(cache.capacity == 4 * cache.tokens_per_frame for cache in all_caches(streamer.caches))
    with pytest.raises(ValueError, match="max_frames"):
        streamer.step(inputs["images"][:, 0], inputs["depth"][:, 0], inputs["mask"][:, 0])


def test_eval_stream_allocates_each_window_once():
    import eval_stream

    calls = []

    class Recorder:
        def reset(self, max_frames=None):
            calls.append(max_frames)

        def step(self, image, depth, mask):
            return {"pose_enc": torch.zeros(1, 1, 9), "depth": torch.ones(1, 1, 28, 42, 1)}

        def kv_bytes(self):
            return 0

    eval_stream.predict_stream(Recorder(), _inputs(), True, eval_stream.Meter("cpu"))
    assert calls == [4]


def test_cache_gate_does_not_count_demoted_anchors_as_requantized():
    """A demoted anchor enters the quantised long-patch store once, when it stops being an anchor: its rows were
    never long-patch rows before, so the quantize-once gate must not compare them."""
    from test_stream_streaming import _sequence

    policy = CachePolicy(recent=1, long_special=1, long_patch=100, selector="recency", quant="int8", anchor_every=2,
                         max_anchors=1)
    gate = stream_gates.cache_invariants(_causal(), _sequence(frames=8), policy, quant_layers=(0, 3))
    assert gate["ok"] and gate["invariants_failed"] == [] and gate["requantized_rows"] == 0
    assert gate["compared_rows"] > 0
