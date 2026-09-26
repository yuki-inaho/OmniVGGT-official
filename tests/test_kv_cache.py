"""Per-layer KV cache of the streaming model (omnivggt.stream.kv_cache)."""

import pytest
import torch

from omnivggt.stream.kv_cache import CachePolicy, LayerKVCache

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
