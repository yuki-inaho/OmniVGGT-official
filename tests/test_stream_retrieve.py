"""Frame retrieval of the streaming model (RetrieveVGGT-style): the full history is stored, and each step reads the
anchor, N-1 past frames chosen by Segment Sampling on the first global layer's scores, and the current frame."""

import pytest
import torch
from test_stream_causal import TINY
from test_stream_streaming import OUTPUTS, _causal_model, _frame_inputs, _sequence

from omnivggt.models.omnivggt_omega import OmniVGGTOmega
from omnivggt.stream.kv_cache import CachePolicy, LayerKVCache
from omnivggt.stream.streaming import StreamingOmega, all_caches


def _retrieve():
    from omnivggt.stream import retrieve

    return retrieve


# --- policy ------------------------------------------------------------------------------------------------------


def test_retrieve_policy_keeps_the_full_history():
    policy = CachePolicy(recent=None, retrieve_frames=8)
    assert policy.retrieve_frames == 8 and not policy.retrieve_camera and not policy.is_full
    assert policy.budget(605, 17) is None  # the stored history is not bounded
    assert CachePolicy(recent=None, retrieve_frames=8, retrieve_camera=True).retrieve_camera
    assert CachePolicy.full().retrieve_frames is None


@pytest.mark.parametrize("kwargs", [
    dict(recent=None, retrieve_frames=0),
    dict(recent=None, retrieve_frames=True),
    dict(recent=1, retrieve_frames=8),  # retrieval reads from the full history
    dict(recent=None, retrieve_camera=True),  # the camera caches follow a retrieval
    dict(recent=None, retrieve_frames=8, retrieve_camera=1),
    dict(recent=None, retrieve_frames=8, long_patch=4, selector="query"),
])
def test_invalid_retrieve_policies_raise(kwargs):
    with pytest.raises(ValueError, match=r"retriev|full"):
        CachePolicy(**kwargs)


def test_a_layer_cache_takes_no_retrieve_policy():
    with pytest.raises(ValueError, match="StreamingOmega"):
        LayerKVCache(CachePolicy(recent=None, retrieve_frames=4), 6, 2, torch.float64, layer_id=0)


# --- reading selected frames of a cache -------------------------------------------------------------------------


def _filled_cache(frames, tokens=3):
    cache = LayerKVCache(CachePolicy.full(), tokens, 1, torch.float64, layer_id=0)
    for t in range(1, frames + 1):
        k = torch.full((1, 2, tokens, 4), float(t), dtype=torch.float64)
        k[..., 1] = torch.arange(tokens, dtype=torch.float64)
        cache.append(k, -k, t)
        if t < frames:
            cache.commit(None, t)
    return cache


def test_read_of_selected_frames_gives_their_rows_in_time_order():
    cache = _filled_cache(6)
    keys, values = cache.read(torch.tensor([1, 4, 6]))
    assert keys[0, 0, :, 0].tolist() == [1.0] * 3 + [4.0] * 3 + [6.0] * 3
    assert keys[0, 0, :, 1].tolist() == [0.0, 1.0, 2.0] * 3 and torch.equal(values, -keys)
    every = cache.read(torch.arange(1, 7))  # every stored frame: the buffer views themselves
    assert all(torch.equal(a, b) and a.data_ptr() == b.data_ptr() for a, b in zip(every, cache.read(), strict=True))
    with pytest.raises(ValueError, match="frame"):
        cache.read(torch.tensor([1, 7]))


# --- relevance score -------------------------------------------------------------------------------------------


def test_relevance_score_matches_the_hand_computation():
    """r(t, i) = (1/H) sum_h <qbar_t[h], kbar_i[h]>, bars = means over the patch tokens (special tokens excluded)."""
    retrieve = _retrieve()
    generator = torch.Generator().manual_seed(0)
    special, tokens, heads, dim = 2, 7, 3, 4
    queries = torch.randn(1, heads, tokens, dim, generator=generator, dtype=torch.float64)
    keys = [torch.randn(1, heads, tokens, dim, generator=generator, dtype=torch.float64) for _ in range(5)]
    descriptors = torch.stack([retrieve.patch_mean(k, special)[0] for k in keys])  # [F, H, D]
    expected = torch.zeros(5, dtype=torch.float64)
    for i, k in enumerate(keys):
        for h in range(heads):
            q_bar = sum(queries[0, h, j] for j in range(special, tokens)) / (tokens - special)
            k_bar = sum(k[0, h, j] for j in range(special, tokens)) / (tokens - special)
            expected[i] += (q_bar @ k_bar) / heads
    got = retrieve.frame_scores(queries, descriptors, special)
    torch.testing.assert_close(got, expected, rtol=0, atol=1e-14)


# --- Segment Sampling ------------------------------------------------------------------------------------------


def _scores(length, values):
    scores = torch.zeros(length, dtype=torch.float64)
    for index, value in values.items():
        scores[index] = value
    return scores


# three segments over 24 frames: A = 2-4 (peak 9 at 3), B = 9-12 (runs 9 and 12 merged over a gap of 2 <= 3;
# peak 5 at 12), C = 17-20 (peak 8 at 19); tau = mu + 0.3 sigma = 2.90, so every non-zero score is in a segment
THREE_SEGMENTS = _scores(24, {2: 5, 3: 9, 4: 6, 9: 4, 12: 5, 17: 4, 18: 4, 19: 8, 20: 4})


@pytest.mark.parametrize("count, expected", [
    # cap int(0.35 * 6) = 2; quotas floor(6 * peak / 22) = A 2, B 1, C 2. A: peak 3 + the farther end (a tie:
    # the later, 4); B: 12; C: 19 + the farther end 17. One short: the best unselected frame, 2.
    (6, [2, 3, 4, 12, 17, 19]),
    # cap 3; quotas A 3, B 2, C 3. A: 3 + the ends of the rest (2, 4); B: 12 + the farther end 9; C: 19 + the
    # evenly spaced ends of the rest (17, 20). One short: the best unselected frame, 18.
    (9, [2, 3, 4, 9, 12, 17, 18, 19, 20]),
])
def test_segment_sampling_of_three_segments(count, expected):
    assert _retrieve().segment_sampling(THREE_SEGMENTS, count).tolist() == expected


def test_segment_sampling_caps_a_segment_and_fills_from_the_global_top():
    """A = 5-9 (peak 100) would take 5 of 8 frames; the cap int(0.35 * 8) = 2 leaves room that the global top fills
    with B's frames (45) before A's other frames (30)."""
    scores = _scores(30, {5: 30, 6: 30, 7: 100, 8: 30, 9: 30, 20: 45, 21: 45, 22: 50, 23: 45, 24: 45})
    assert _retrieve().segment_sampling(scores, 8).tolist() == [7, 8, 9, 20, 21, 22, 23, 24]


def test_segment_sampling_keeps_the_best_frames_when_the_segments_ask_for_too_many():
    """Four one-frame segments get a quota of at least 1 each; with 2 to choose, the two best of them stay."""
    scores = _scores(30, {2: 10, 9: 12, 16: 11, 23: 9})
    assert _retrieve().segment_sampling(scores, 2).tolist() == [9, 16]


def test_segment_sampling_falls_back_to_the_top_scores():
    retrieve = _retrieve()
    # no frame reaches tau = mu + 0.3 sigma (1.008): the top 3, ties to the newer frame
    no_segment = torch.tensor([0.0] + [1.0] * 14, dtype=torch.float64)
    assert retrieve.segment_sampling(no_segment, 3).tolist() == [12, 13, 14]
    # the segment peaks do not sum to a positive total: no quota is defined
    negative = torch.tensor([-5.0, -1.0, -3.0, -2.0, -4.0], dtype=torch.float64)
    assert retrieve.segment_sampling(negative, 2).tolist() == [1, 3]


def test_segment_sampling_of_few_candidates_keeps_them_all():
    retrieve = _retrieve()
    assert retrieve.segment_sampling(THREE_SEGMENTS[:5], 5).tolist() == [0, 1, 2, 3, 4]
    assert retrieve.segment_sampling(THREE_SEGMENTS[:5], 9).tolist() == [0, 1, 2, 3, 4]
    assert retrieve.segment_sampling(THREE_SEGMENTS, 0).tolist() == []


# --- the retriever ---------------------------------------------------------------------------------------------


def _frame_qk(value, special=1, patches=2, heads=2, dim=2):
    """q and k of one frame whose patch tokens all equal ``value`` (special tokens: a large decoy)."""
    x = torch.full((1, heads, special + patches, dim), float(value), dtype=torch.float64)
    x[:, :, :special] = 1e3
    return x, x.clone()


def test_the_anchor_and_the_current_frame_are_always_read():
    """Frame 1 has the lowest score, and still every selection holds it, the current frame and N-1 others."""
    retrieve = _retrieve()
    retriever = retrieve.FrameRetriever(retrieve_frames=3, special_count=1)
    values = [-5.0, 1.0, 2.0, 0.5, 3.0, 1.5, 2.5, 1.0]
    for t, value in enumerate(values, start=1):
        q, k = _frame_qk(value)
        frames = retriever.select(q, k, t).tolist()
        assert frames[0] == 1 and frames[-1] == t and frames == sorted(set(frames))
        assert len(frames) == min(t, 3 + 1)
        if t > 1:  # r(t, i) = <value_t, value_i> * dim (the patch means are constant), frames 1..t-1
            want = torch.tensor([2.0 * value * past for past in values[: t - 1]], dtype=torch.float64)
            torch.testing.assert_close(retriever.scores, want, rtol=0, atol=1e-12)
    assert retriever.frames.tolist() == frames


def test_the_retriever_selects_for_one_stream_in_order():
    retriever = _retrieve().FrameRetriever(retrieve_frames=3, special_count=1)
    q, k = _frame_qk(1.0)
    with pytest.raises(NotImplementedError, match="batch"):
        retriever.select(q.expand(2, -1, -1, -1), k.expand(2, -1, -1, -1), 1)
    with pytest.raises(ValueError, match="frame"):
        retriever.select(q, k, 2)
    with pytest.raises(ValueError, match="retrieve_frames"):
        _retrieve().FrameRetriever(retrieve_frames=0, special_count=1)


# --- streaming -------------------------------------------------------------------------------------------------

N = 4
FRAMES = 8


def _run(stream, sequence, frames=FRAMES, on_step=None):
    outputs = []
    for frame in range(frames):
        outputs.append(stream.step(*_frame_inputs(sequence, frame, "depth")))
        if on_step is not None:
            on_step(stream)
    return outputs


def test_with_at_most_n_past_frames_retrieval_equals_the_full_cache():
    """Steps t <= N + 1 have at most N past frames, all of them read: the same bits as the full cache (fp64)."""
    model, sequence = _causal_model(), _sequence(frames=FRAMES)
    full = _run(StreamingOmega(model, CachePolicy.full()), sequence)
    retrieved = _run(StreamingOmega(model, CachePolicy(recent=None, retrieve_frames=N)), sequence)
    for t in range(1, N + 2):
        for key in OUTPUTS:
            assert torch.equal(retrieved[t - 1][key], full[t - 1][key]), (t, key)
    assert not torch.equal(retrieved[-1]["pose_enc"], full[-1]["pose_enc"])  # later steps read a subset


@pytest.mark.parametrize("retrieve_camera", [False, True])
def test_the_selected_frames_are_shared_by_every_global_and_register_layer(monkeypatch, retrieve_camera):
    """The first global layer selects once per step; every global and register layer reads the same frames, and
    the camera trunk reads its full history (or, with retrieve_camera, the same frames)."""
    reads = []
    original = LayerKVCache.read

    def recorded(cache, frames=None):
        reads.append((cache, None if frames is None else frames.clone()))
        return original(cache, frames)

    monkeypatch.setattr(LayerKVCache, "read", recorded)
    model, sequence = _causal_model(), _sequence(frames=FRAMES)
    stream = StreamingOmega(model, CachePolicy(recent=None, retrieve_frames=N, retrieve_camera=retrieve_camera))
    steps = []

    def on_step(stream):
        steps.append((stream.t, list(reads), stream.retriever.frames.clone()))
        reads.clear()

    _run(stream, sequence, on_step=on_step)
    backbone = {id(cache) for kind in ("global", "register") for cache in stream.caches[kind].values()}
    camera = {id(cache) for row in stream.caches["camera"] for cache in row}
    assert all(cache.policy.is_full for cache in all_caches(stream.caches))  # the caches store the full history
    for t, step_reads, frames in steps:
        assert frames.tolist()[0] == 1 and frames.tolist()[-1] == t and len(frames) == min(t, N + 1)
        backbone_reads = [read for cache, read in step_reads if id(cache) in backbone]
        camera_reads = [read for cache, read in step_reads if id(cache) in camera]
        assert len(backbone_reads) == len(backbone) and len(camera_reads) == len(camera)
        assert all(torch.equal(read, frames) for read in backbone_reads), t
        if retrieve_camera:
            assert all(torch.equal(read, frames) for read in camera_reads), t
        else:
            assert all(read is None for read in camera_reads), t
    assert len(steps[-1][2]) == N + 1  # the last steps read a subset


def test_the_stream_score_is_the_first_global_layer_relevance(monkeypatch):
    """The stream's r(t, i) is computed from global block 0's current queries and the keys it stored for frame i
    (after q/k norm), averaged over the patch tokens."""
    model, sequence = _causal_model(), _sequence(frames=6)
    attn = model.aggregator.global_blocks[0].attn
    captured = []
    original = attn.qkv_heads

    def recorded(x, pos=None):
        q, k, v = original(x, pos=pos)
        captured.append((q.clone(), k.clone()))
        return q, k, v

    monkeypatch.setattr(attn, "qkv_heads", recorded)
    stream = StreamingOmega(model, CachePolicy(recent=None, retrieve_frames=2))
    special = model.aggregator.patch_start_idx
    scores = []
    _run(stream, sequence, frames=6, on_step=lambda stream: scores.append(stream.retriever.scores))
    assert len(captured) == 6 and scores[0] is None
    for t in range(2, 7):
        q_bar = captured[t - 1][0][0, :, special:].mean(dim=1)  # [H, D]
        expected = torch.stack([(q_bar * captured[i][1][0, :, special:].mean(dim=1)).sum(-1).mean()
                                for i in range(t - 1)])
        torch.testing.assert_close(scores[t - 1], expected, rtol=0, atol=1e-12)


def test_retrieval_stores_and_reports_the_full_history():
    model, sequence = _causal_model(), _sequence(frames=6)
    full, retrieved = (StreamingOmega(model, policy) for policy in
                       (CachePolicy.full(), CachePolicy(recent=None, retrieve_frames=2)))
    for stream in (full, retrieved):
        stream.reset(max_frames=6)
    capacities = []
    for frame in range(6):
        for stream in (full, retrieved):
            stream.step(*_frame_inputs(sequence, frame, "depth"))
        assert retrieved.kv_bytes() == full.kv_bytes() > 0
        capacities.append([cache.capacity for cache in all_caches(retrieved.caches)])
    assert capacities[0] == capacities[-1]  # allocated once for max_frames


def test_retrieval_needs_one_stream():
    stream = StreamingOmega(_causal_model(), CachePolicy(recent=None, retrieve_frames=2))
    with pytest.raises(NotImplementedError, match="batch"):
        stream.step(torch.rand(2, 3, 28, 42, dtype=torch.float64))


def test_retrieval_needs_a_first_inter_frame_layer_with_patches():
    torch.manual_seed(0)
    options = {**TINY, "register_attention_layers": (0,)}
    model = OmniVGGTOmega(**options, causal=True, depth_norm="first_frame").double().eval()
    StreamingOmega(model, CachePolicy.full())  # other policies do not select
    with pytest.raises(ValueError, match="first"):
        StreamingOmega(model, CachePolicy(recent=None, retrieve_frames=2))


def test_eval_stream_takes_a_retrieve_policy():
    import eval_stream
    from test_stream_causal import _inputs

    meter = eval_stream.Meter("cpu")
    policy = {"recent": None, "retrieve_frames": 2}
    out = eval_stream.make_predictor("stream", _causal_model(), policy, "fp32", meter)(_inputs(), True)
    assert out["pose_enc"].shape == (1, 4, 9) and torch.isfinite(out["pose_enc"]).all()
    assert out["kv_bytes"] == sorted(out["kv_bytes"]) and out["kv_bytes"][0] < out["kv_bytes"][-1]
