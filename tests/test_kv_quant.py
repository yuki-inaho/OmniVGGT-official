"""INT8/INT4 storage of the long-patch store (omnivggt.stream.kv_cache): asymmetric, quantised once."""

import pytest
import torch

from omnivggt.stream.kv_cache import CachePolicy, LayerKVCache, dequantize, quantize_k, quantize_v

BITS = {"int8": 8, "int4": 4}
HEADS = 2


def _tensor(tokens, dim, seed=0):
    """[1, H, tokens, dim] with an outlier channel, as keys have."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(1, HEADS, tokens, dim, generator=generator, dtype=torch.float64)
    x[..., 3] *= 20.0
    return x


def _per_token(params, tokens):
    """K scale/offset [.., blocks, D] -> [.., tokens, D]: token i uses block i // 64."""
    return params.repeat_interleave(64, dim=2)[:, :, :tokens]


def _per_channel(params, dim):
    """V scale/offset [.., N, blocks] -> [.., N, dim]: channel c uses block c // 64."""
    return params.repeat_interleave(64, dim=3)[..., :dim]


@pytest.mark.parametrize("quant", list(BITS))
def test_quantisation_error_is_within_half_a_step(quant):
    bits = BITS[quant]
    k, v = _tensor(130, 80), _tensor(130, 80, seed=1)
    codes, scale, offset = quantize_k(k, bits)
    scale_k = _per_token(scale, 130)
    restored = dequantize(codes, scale_k, _per_token(offset, 130), bits)
    assert ((restored - k).abs() <= scale_k / 2 + 1e-12).all()
    assert not torch.equal(restored, k)
    codes, scale, offset = quantize_v(v, bits)
    scale_v = _per_channel(scale, 80)
    restored = dequantize(codes, scale_v, _per_channel(offset, 80), bits)
    assert ((restored - v).abs() <= scale_v / 2 + 1e-12).all()


@pytest.mark.parametrize("quant", list(BITS))
def test_keys_are_quantised_per_channel_over_64_tokens_and_values_per_token_over_64_channels(quant):
    bits = BITS[quant]
    levels = 2**bits - 1
    x = _tensor(130, 80)
    _, scale, offset = quantize_k(x, bits)
    assert scale.shape == offset.shape == (1, HEADS, 3, 80)  # blocks of tokens 0-63, 64-127, 128-129
    for block, tokens in enumerate((slice(0, 64), slice(64, 128), slice(128, 130))):
        low, high = x[:, :, tokens].amin(dim=2), x[:, :, tokens].amax(dim=2)
        torch.testing.assert_close(offset[:, :, block], low, rtol=0, atol=0)
        torch.testing.assert_close(scale[:, :, block], (high - low) / levels, rtol=1e-15, atol=0)
    _, scale, offset = quantize_v(x, bits)
    assert scale.shape == offset.shape == (1, HEADS, 130, 2)  # blocks of channels 0-63, 64-79
    for block, channels in enumerate((slice(0, 64), slice(64, 80))):
        low, high = x[..., channels].amin(dim=3), x[..., channels].amax(dim=3)
        torch.testing.assert_close(offset[..., block], low, rtol=0, atol=0)
        torch.testing.assert_close(scale[..., block], (high - low) / levels, rtol=1e-15, atol=0)


def test_int4_packs_two_codes_per_byte():
    x = _tensor(70, 64)
    codes4, _, _ = quantize_k(x, 4)
    codes8, _, _ = quantize_k(x, 8)
    assert codes4.dtype == codes8.dtype == torch.uint8
    assert codes4.shape == (1, HEADS, 70, 32) and codes8.shape == (1, HEADS, 70, 64)
    with pytest.raises(ValueError, match="even"):
        quantize_k(_tensor(4, 5), 4)
    with pytest.raises(ValueError, match="bits"):
        quantize_k(x, 2)


def test_a_constant_block_is_finite_and_exact():
    x = torch.full((1, HEADS, 10, 8), 3.25, dtype=torch.float64)
    for bits in BITS.values():
        for quantize in (quantize_k, quantize_v):
            codes, scale, offset = quantize(x, bits)
            assert torch.isfinite(scale).all() and (scale > 0).all()
            expand = _per_token if quantize is quantize_k else _per_channel
            size = 10 if quantize is quantize_k else 8
            assert torch.equal(dequantize(codes, expand(scale, size), expand(offset, size), bits), x)


# --- the long-patch store of a cache -------------------------------------------------------------------------

SPECIAL = 2


def _frame(t, patches, dim, seed_offset=0):
    generator = torch.Generator().manual_seed(1000 * seed_offset + t)
    k = torch.randn(1, HEADS, SPECIAL + patches, dim, generator=generator, dtype=torch.float64)
    return k, torch.randn(k.shape, generator=generator, dtype=torch.float64)


def _long_patch_rows(cache):
    frame_id, token_id = cache.row_ids()
    return (token_id >= SPECIAL) & (frame_id != 1) & (frame_id < cache.last_frame)  # w = 1


@pytest.mark.parametrize("quant", list(BITS))
def test_nbytes_is_the_payload_plus_scales_and_offsets(quant):
    """130 patches per frame enter as K blocks of 64, 64 and 2 tokens; recency keeps 200 long patches: the newest
    long frame whole and tokens 2-71 of the one before, whose last K block (tokens 130, 131) is released."""
    bits, patches, dim = BITS[quant], 130, 8
    tokens = SPECIAL + patches
    cache = LayerKVCache(CachePolicy(recent=1, long_patch=200, selector="recency", quant=quant), tokens, SPECIAL,
                         torch.float64, layer_id=0)
    for t in range(1, 7):
        cache.append(*_frame(t, patches, dim), t)
        cache.commit(torch.ones(1, HEADS, tokens, dim, dtype=torch.float64), t)
        invariants = cache.invariants()
        assert invariants["ok"], invariants
    frame_id, token_id = cache.row_ids()
    long_patch = _long_patch_rows(cache)
    assert int(long_patch.sum()) == 200
    blocks = sum(len({(j - SPECIAL) // 64 for j in token_id[long_patch & (frame_id == f)].tolist()})
                 for f in frame_id[long_patch].unique().tolist())
    assert blocks == 3 + 2
    full_precision_rows = cache.size - 200
    per_row = HEADS * dim  # values of one row of K (or V)
    expected = (2 * full_precision_rows * per_row * 8  # anchor and current frame, fp64 K and V
                + 2 * 200 * per_row * bits // 8  # K and V codes
                + 200 * HEADS * 1 * 2 * 8  # V scale and offset: per token and 64-channel block (dim 8: one block)
                + blocks * per_row * 2 * 8)  # K scale and offset: per channel and 64-token block
    assert cache.nbytes() == expected


@pytest.mark.parametrize("quant", list(BITS))
def test_long_patch_tokens_are_quantised_once(quant):
    """The patches of frame 2 align with every query, so the query selector keeps them for 100 commits while
    the other long patches come and go; their dequantised keys and values never change."""
    patches, dim = 70, 8
    tokens = SPECIAL + patches
    policy = CachePolicy(recent=1, long_patch=patches + 30, selector="query", quant=quant)
    cache = LayerKVCache(policy, tokens, SPECIAL, torch.float64, layer_id=0)
    queries = torch.ones(1, HEADS, tokens, dim, dtype=torch.float64)
    first = None
    for t in range(1, 104):
        k, v = _frame(t, patches, dim)
        if t == 2:
            k[:, :, SPECIAL:] += 10.0
            original = k[:, :, SPECIAL:].clone(), v[:, :, SPECIAL:].clone()
        cache.append(k, v, t)
        cache.commit(queries, t, grid_hw=(7, 10))
        assert cache.invariants()["ok"]
        if t < 3:
            continue
        frame_id, token_id = cache.row_ids()
        rows = ((frame_id == 2) & (token_id >= SPECIAL)).nonzero().squeeze(1)
        assert rows.numel() == patches and torch.equal(token_id[rows], torch.arange(SPECIAL, tokens))
        keys, values = cache.read()
        stored = keys[:, :, rows], values[:, :, rows]
        if first is None:
            first = stored
            for restored, exact in zip(stored, original, strict=True):
                assert not torch.equal(restored, exact)  # really quantised
                assert (restored - exact).abs().max() < 1.0
        assert torch.equal(stored[0], first[0]) and torch.equal(stored[1], first[1]), t
    others = cache.row_ids()[0][_long_patch_rows(cache)]
    others = others[others != 2]
    # the other 30 slots were full from frame 4 on, so the late frames that are in got there by eviction
    assert others.numel() == 30 and int(others.max()) > 90
