"""Per-layer KV cache of the streaming model (omnivggt.stream.kv_cache)."""

import numpy as np
import pytest
import torch

from omnivggt.stream.kv_cache import (
    SELECTOR_NAMES,
    CachePolicy,
    Candidates,
    LayerKVCache,
    query_groups,
    select_query,
    select_random,
    select_recency,
    select_xstream,
    top_frames,
    top_rows,
    xstream_rows,
)

HEADS, DIM = 2, 4


def _frame_kv(frame_id, tokens, batch=1, dtype=torch.float64):
    """Keys and values of one frame whose entries encode (frame, token, head, sample), so rows can be traced."""
    k = torch.zeros(batch, HEADS, tokens, DIM, dtype=dtype)
    k[..., 0] = frame_id
    k[..., 1] = torch.arange(tokens, dtype=dtype)
    k[..., 2] = torch.arange(HEADS, dtype=dtype)[:, None]
    k[..., 3] = torch.arange(batch, dtype=dtype)[:, None, None]
    return k, -1.0 - k


def _queries(tokens, batch=1, dtype=torch.float64):
    return torch.ones(batch, HEADS, tokens, DIM, dtype=dtype)


# --- full cache ----------------------------------------------------------------------------------------------


def test_full_cache_reads_every_frame_in_time_order():
    tokens = 5
    cache = LayerKVCache(CachePolicy.full(), tokens, 2, torch.float64, layer_id=0)
    keys, values = [], []
    for t in range(1, 7):
        k, v = _frame_kv(t, tokens)
        keys.append(k)
        values.append(v)
        cache.append(k, v, t)
        read_k, read_v = cache.read()  # the stored frames plus the current one
        assert torch.equal(read_k, torch.cat(keys, dim=2)) and torch.equal(read_v, torch.cat(values, dim=2))
        cache.commit(_queries(tokens), t)
        read_k, read_v = cache.read()
        assert torch.equal(read_k, torch.cat(keys, dim=2)) and torch.equal(read_v, torch.cat(values, dim=2))
    assert cache.size == 6 * tokens and cache.last_frame == 6


def test_full_cache_doubles_its_buffer_and_keeps_the_values():
    tokens = 3
    cache = LayerKVCache(CachePolicy.full(), tokens, 1, torch.float64, layer_id=0, initial_frames=2)
    keys = []
    capacities = []
    for t in range(1, 6):
        k, v = _frame_kv(t, tokens)
        keys.append(k)
        cache.append(k, v, t)
        cache.commit(_queries(tokens), t)
        capacities.append(cache.capacity)
        assert torch.equal(cache.read()[0], torch.cat(keys, dim=2))
    assert capacities == [2 * tokens, 2 * tokens, 4 * tokens, 4 * tokens, 8 * tokens]


@pytest.mark.parametrize("dtype, batch", [(torch.float64, 1), (torch.float32, 2), (torch.bfloat16, 1)])
def test_full_cache_nbytes_is_the_analytic_size(dtype, batch):
    tokens = 4
    cache = LayerKVCache(CachePolicy.full(), tokens, 1, dtype, layer_id=0)
    assert cache.nbytes() == 0
    for t in range(1, 4):
        k, v = _frame_kv(t, tokens, batch=batch, dtype=torch.float32)
        cache.append(k, v, t)  # stored in the cache dtype
        assert cache.read()[0].dtype == dtype
        cache.commit(_queries(tokens, batch=batch), t)
        itemsize = torch.finfo(dtype).bits // 8
        assert cache.nbytes() == 2 * batch * HEADS * (t * tokens) * DIM * itemsize  # K and V


def test_append_rejects_a_frame_of_another_size():
    cache = LayerKVCache(CachePolicy.full(), 4, 1, torch.float64, layer_id=0)
    with pytest.raises(ValueError, match="tokens"):
        cache.append(*_frame_kv(1, 3), 1)


def test_append_needs_consecutive_frames_from_1():
    cache = LayerKVCache(CachePolicy.full(), 4, 1, torch.float64, layer_id=0)
    with pytest.raises(ValueError, match="frame"):
        cache.append(*_frame_kv(2, 4), 2)
    cache.append(*_frame_kv(1, 4), 1)
    cache.commit(_queries(4), 1)
    with pytest.raises(ValueError, match="frame"):
        cache.append(*_frame_kv(3, 4), 3)


def test_append_and_commit_alternate():
    cache = LayerKVCache(CachePolicy.full(), 4, 1, torch.float64, layer_id=0)
    with pytest.raises(RuntimeError, match="append"):
        cache.commit(_queries(4), 1)
    cache.append(*_frame_kv(1, 4), 1)
    with pytest.raises(RuntimeError, match="commit"):
        cache.append(*_frame_kv(2, 4), 2)
    with pytest.raises(ValueError, match="frame"):
        cache.commit(_queries(4), 2)


# --- bounded cache: anchor, recent, long-special, long-patch -------------------------------------------------

N, M = 6, 2  # tokens per frame (2 special + 4 patch) of the synthetic global layer
W, K_U, B_P = 1, 2, 3
BOUNDED = CachePolicy(recent=W, long_special=K_U, long_patch=B_P, selector="recency")


def _stream_cache(cache, frames, tokens, on_commit=None):
    for t in range(1, frames + 1):
        cache.append(*_frame_kv(t, tokens), t)
        cache.commit(_queries(tokens), t)
        if on_commit is not None:
            on_commit(cache, t)


def _rows(cache):
    frame_id, token_id = cache.row_ids()
    return sorted(zip(frame_id.tolist(), token_id.tolist(), strict=True))


def _expected_recency_rows(t):
    """Anchor + current frame + specials of the 2 newest middle frames + patches 2, 3, 4 of the newest one."""
    rows = [(1, j) for j in range(N)]
    if t >= 2:
        rows += [(t, j) for j in range(N)]
    middle = list(range(2, t))
    for frame in middle[-K_U:]:
        rows += [(frame, j) for j in range(M)]
    if middle:
        rows += [(middle[-1], j) for j in (2, 3, 4)]  # ties inside a frame: the smaller token id first
    return sorted(rows)


def test_bounded_budget_of_each_layer_kind():
    assert BOUNDED.budget(N, M) == (1 + W) * N + M * K_U + B_P == 19
    assert BOUNDED.budget(M, M) == M * (1 + W + K_U) == 8  # register layer: special tokens only
    assert BOUNDED.budget(1, 1) == 1 + W + K_U == 4  # camera trunk: one token per frame
    assert CachePolicy.full().budget(N, M) is None


def test_bounded_cache_keeps_anchor_recent_and_long_stores_within_budget():
    def check(cache, t):
        invariants = cache.invariants()
        assert invariants["ok"], invariants
        assert cache.size <= cache.budget == 19
        assert _rows(cache) == _expected_recency_rows(t), t

    _stream_cache(LayerKVCache(BOUNDED, N, M, torch.float64, layer_id=3), 8, N, check)


def test_bounded_cache_rows_of_keys_values_and_ids_correspond():
    def check(cache, t):
        keys, values = cache.read()
        frame_id, token_id = cache.row_ids()
        assert keys.shape[2] == values.shape[2] == frame_id.numel() == token_id.numel() == cache.size
        assert torch.equal(keys[0, :, :, 0], frame_id.double().expand(HEADS, -1))
        assert torch.equal(keys[0, :, :, 1], token_id.double().expand(HEADS, -1))
        assert torch.equal(keys[0, :, :, 2], torch.arange(HEADS, dtype=torch.float64)[:, None].expand_as(keys[0, ..., 2]))
        assert torch.equal(values, -1.0 - keys)

    _stream_cache(LayerKVCache(BOUNDED, N, M, torch.float64, layer_id=3), 8, N, check)


def test_bounded_cache_selection_is_deterministic():
    caches = [LayerKVCache(BOUNDED, N, M, torch.float64, layer_id=3) for _ in range(2)]
    for t in range(1, 9):
        for cache in caches:
            cache.append(*_frame_kv(t, N), t)
            cache.commit(_queries(N), t)
        assert _rows(caches[0]) == _rows(caches[1])
        assert torch.equal(caches[0].read()[0], caches[1].read()[0])


@pytest.mark.parametrize("tokens, budget", [(M, 8), (1, 4)])
def test_bounded_register_and_camera_caches_keep_frames_of_special_tokens(tokens, budget):
    def check(cache, t):
        assert cache.invariants()["ok"]
        assert cache.size == min(t, 1 + W + K_U) * tokens <= budget
        frames = sorted(set(cache.row_ids()[0].tolist()))
        assert frames == sorted({1, *range(max(2, t - K_U), t + 1)})  # anchor, current, 2 newest long frames

    _stream_cache(LayerKVCache(BOUNDED, tokens, tokens, torch.float64, layer_id=0), 8, tokens, check)


def test_invariants_detect_a_missing_anchor_token_and_an_overfull_store():
    cache = LayerKVCache(BOUNDED, N, M, torch.float64, layer_id=0)
    _stream_cache(cache, 6, N)
    frame_id, token_id = cache.row_ids()
    cache._compact(~((frame_id == 1) & (token_id == 4)))  # evict one anchor token
    invariants = cache.invariants()
    assert not invariants["protected_complete"] and not invariants["ok"]

    roomy = LayerKVCache(CachePolicy(recent=W, long_special=10, long_patch=100, selector="recency"), N, M,
                         torch.float64, layer_id=0)
    _stream_cache(roomy, 6, N)
    roomy.policy, roomy.budget = BOUNDED, BOUNDED.budget(N, M)  # judged by the smaller policy
    invariants = roomy.invariants()
    assert not invariants["within_budget"] and not invariants["long_special"] and not invariants["long_patch"]
    assert invariants["protected_complete"] and not invariants["ok"]


def test_bounded_cache_keeps_every_candidate_under_the_budget():
    roomy = CachePolicy(recent=W, long_special=10, long_patch=100, selector="recency")

    def check(cache, t):
        assert cache.size == t * N and cache.invariants()["ok"]

    _stream_cache(LayerKVCache(roomy, N, M, torch.float64, layer_id=0), 8, N, check)


def test_bounded_nbytes_counts_the_stored_rows():
    cache = LayerKVCache(BOUNDED, N, M, torch.float64, layer_id=0)
    _stream_cache(cache, 6, N)
    assert cache.nbytes() == 2 * HEADS * 19 * DIM * 8


def test_policy_from_a_global_budget():
    policy = CachePolicy.for_budget(4096, 605, 17, recent=1, long_special=16, selector="query")
    assert policy == CachePolicy(recent=1, long_special=16, long_patch=2614, selector="query")
    assert policy.budget(605, 17) == 4096


@pytest.mark.parametrize("budget, recent, long_special", [(4096, 4, 64), (4096, 8, 16), (1209, 1, 0)])
def test_policy_from_an_infeasible_global_budget_raises(budget, recent, long_special):
    with pytest.raises(ValueError, match="budget"):
        CachePolicy.for_budget(budget, 605, 17, recent=recent, long_special=long_special, selector="query")


@pytest.mark.parametrize("kwargs", [
    dict(recent=0),
    dict(recent=1, long_special=-1, selector="query"),
    dict(recent=1, long_patch=-1, selector="query"),
    dict(recent=1, long_patch=4, selector="best"),
    dict(recent=1, long_patch=4),  # a long-term store needs a selector
    dict(recent=1, selector="query"),  # a selector without a long-term store selects nothing
    dict(recent=1, long_patch=4, selector="query", quant="int2"),
    dict(recent=1, long_special=4, selector="query", quant="int8"),  # quantisation is of long-patch only
    dict(recent=None, long_patch=4, selector="query"),  # the full cache keeps everything
    dict(recent=True),
])
def test_invalid_policies_raise(kwargs):
    with pytest.raises(ValueError):
        CachePolicy(**kwargs)


# --- selectors ------------------------------------------------------------------------------------------------


def _candidates(queries, keys, frame_id, token_id, special_count, grid_hw=None, layer_id=0, t=5):
    return Candidates(queries, keys, torch.as_tensor(frame_id), torch.as_tensor(token_id), special_count, grid_hw,
                      layer_id, t)


def _random_qk(tokens, rows, seed=0, heads=2, dim=2):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(1, heads, tokens, dim, generator=generator, dtype=torch.float64),
            torch.randn(1, heads, rows, dim, generator=generator, dtype=torch.float64))


def test_query_groups_of_the_g3_grid():
    groups = query_groups(17, (21, 28))  # 392 x 294 at patch 14: 21 rows x 28 columns
    sizes = torch.bincount(groups)
    assert groups.shape == (605,) and sizes.numel() == 171  # 17 special + 11 x 14 patch groups
    assert (sizes[:17] == 1).all() and sizes[17:].numel() == 154
    assert sorted(sizes[17:].tolist()) == [2] * 14 + [4] * 140  # the last (odd) row forms 1 x 2 groups
    torch.testing.assert_close((sizes.double() / 605).sum(), torch.tensor(1.0, dtype=torch.float64), rtol=0,
                               atol=1e-15)  # the query weights w_a = |G_a| / n
    for row, column in [(0, 0), (1, 1), (19, 27), (20, 0), (20, 27)]:  # patch (row, column) -> 2 x 2 block
        assert groups[17 + row * 28 + column] == 17 + (row // 2) * 14 + column // 2


def test_query_groups_of_an_odd_grid_and_without_patches():
    assert sorted(torch.bincount(query_groups(1, (3, 3)))[1:].tolist()) == [1, 2, 2, 4]
    assert query_groups(17, None).tolist() == list(range(17))


def test_query_selector_matches_the_formula_on_a_hand_example():
    """H=2, one special token and a 1 x 4 patch grid: groups {0}, {1, 2}, {3, 4} with weights 1/5, 2/5, 2/5."""
    queries, keys = _random_qk(5, 7)
    groups, weights = [[0], [1, 2], [3, 4]], [1 / 5, 2 / 5, 2 / 5]
    expected = torch.zeros(7, dtype=torch.float64)
    for h in range(2):
        for group, weight in zip(groups, weights, strict=True):
            q_bar = queries[0, h, group].mean(0)
            logits = torch.stack([q_bar @ keys[0, h, j] for j in range(7)]) / 2**0.5
            expected += weight * torch.softmax(logits, 0) / 2
    got = select_query(_candidates(queries, keys, [1] * 7, list(range(7)), 1, grid_hw=(1, 4)))
    torch.testing.assert_close(got, expected, rtol=0, atol=1e-14)


def test_query_selector_needs_the_grid_of_a_layer_with_patches():
    queries, keys = _random_qk(5, 7)
    with pytest.raises(ValueError, match="grid"):
        select_query(_candidates(queries, keys, [1] * 7, list(range(7)), 1, grid_hw=None))


def test_xstream_rows_of_the_g3_frame():
    rows = xstream_rows(17, 588)
    sizes = torch.bincount(rows)
    assert sizes.numel() == 17 + 37 and (sizes[:17] == 1).all()
    assert sizes[17:].tolist() == [16] * 36 + [12]  # raster chunks of 16 patches, the remainder is one row


def test_xstream_selector_matches_the_formula_on_a_hand_example():
    """Two special rows, patches 0-15 in one row and the remaining 4 patches in another: mu is the row mean of
    the head-mean pooled queries, and s_j = <mu, mean_h k_j>."""
    queries, keys = _random_qk(22, 9, heads=3, dim=4)
    rows = [[0], [1], list(range(2, 18)), list(range(18, 22))]
    mu = torch.stack([queries[0, :, row].mean(1).mean(0) for row in rows]).mean(0)
    expected = torch.stack([mu @ keys[0, :, j].mean(0) for j in range(9)])
    got = select_xstream(_candidates(queries, keys, [1] * 9, list(range(9)), 2, grid_hw=(4, 5)))
    torch.testing.assert_close(got, expected, rtol=0, atol=1e-14)


def test_recency_selector_keeps_the_newest_rows():
    frame_id, token_id = torch.tensor([2, 2, 3, 3, 4, 4]), torch.tensor([3, 2, 5, 4, 1, 0])
    scores = select_recency(_candidates(None, None, frame_id, token_id, 1))
    kept = top_rows(scores, frame_id, token_id, torch.ones(6, dtype=torch.bool), 3)
    assert kept.tolist() == [5, 4, 3]  # frame 4 (tokens 0, 1), then frame 3 token 4


def test_random_selector_is_keyed_by_layer_and_time():
    def scores(layer_id, t):
        return select_random(_candidates(None, None, [1] * 50, list(range(50)), 1, layer_id=layer_id, t=t))

    assert torch.equal(scores(3, 7), scores(3, 7))
    assert torch.equal(scores(3, 7), torch.from_numpy(np.random.default_rng([42, 3, 7]).random(50)))
    assert not torch.equal(scores(3, 7), scores(4, 7)) and not torch.equal(scores(3, 7), scores(3, 8))


def test_ties_go_to_the_newer_frame_then_the_smaller_token():
    frame_id, token_id = torch.tensor([2, 3, 3, 2, 3]), torch.tensor([1, 4, 2, 0, 3])
    equal = torch.zeros(5, dtype=torch.float64)
    everything = torch.ones(5, dtype=torch.bool)
    assert top_rows(equal, frame_id, token_id, everything, 4).tolist() == [2, 4, 1, 3]
    assert top_frames(equal, frame_id, everything, 1).tolist() == [3]


def test_long_special_frames_are_chosen_by_their_mean_score():
    frame_id = torch.tensor([2, 2, 3, 3, 4, 4])
    scores = torch.tensor([0.9, 0.0, 0.5, 0.5, 0.3, 0.3], dtype=torch.float64)  # means 0.45, 0.5, 0.3
    everything = torch.ones(6, dtype=torch.bool)
    assert top_frames(scores, frame_id, everything, 2).tolist() == [3, 2]
    assert top_rows(scores, frame_id, frame_id, everything, 1).tolist() == [0]  # per token, 0.9 wins


def test_query_selector_keeps_the_patches_the_current_queries_attend_to():
    """The patches of frame 2 align with every query, so a query-guided long-patch store keeps them over the
    newer frame 3 (which recency would keep)."""
    policy = CachePolicy(recent=1, long_special=1, long_patch=4, selector="query")
    cache = LayerKVCache(policy, N, M, torch.float64, layer_id=0)
    for t in range(1, 5):
        k, v = _frame_kv(t, N)
        k = 0.01 * k
        if t == 2:
            k[:, :, M:] = 10.0  # every channel aligned with the all-ones queries
        cache.append(k, v, t)
        cache.commit(_queries(N), t, grid_hw=(2, 2))
        assert cache.invariants()["ok"]
    rows = _rows(cache)
    assert [(2, j) for j in range(M, N)] == [row for row in rows if row[0] == 2 and row[1] >= M]
    assert not [row for row in rows if row[0] == 3 and row[1] >= M]


@pytest.mark.parametrize("selector", SELECTOR_NAMES)
def test_every_selector_streams_within_budget(selector):
    policy = CachePolicy(recent=1, long_special=2, long_patch=3, selector=selector)
    cache = LayerKVCache(policy, N, M, torch.float64, layer_id=1)
    for t in range(1, 9):
        k, v = _frame_kv(t, N)
        cache.append(k + torch.randn(k.shape, generator=torch.Generator().manual_seed(t), dtype=k.dtype), v, t)
        cache.commit(_queries(N), t, grid_hw=(2, 2))
        assert cache.invariants()["ok"] and cache.size <= 19
