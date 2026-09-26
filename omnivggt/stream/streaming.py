"""Streaming inference of the frame-causal OmniVGGTOmega: one frame per step, with per-layer KV caches."""

from typing import Optional, Tuple

from torch import Tensor

from omnivggt.stream.kv_cache import LayerKVCache


def stream_block(block, x: Tensor, cache: LayerKVCache, t: int, grid_hw: Optional[Tuple[int, int]] = None,
                 pos: Optional[Tensor] = None) -> Tensor:
    """One inter-frame ``Block`` on the tokens of frame ``t`` ([B, n, C]), attending to the cached past and the frame.

    The residual structure of ``Block.forward`` in eval mode: the current keys and values are appended to
    ``cache`` before the attention (a frame sees all of its own tokens), and the cache commits (selects what it
    keeps for the next step) after the block. ``grid_hw`` is the patch grid of the frame and ``pos`` the token
    positions (for a block with RoPE).
    """
    if block.training:
        raise NotImplementedError("stream_block runs blocks in eval mode only (no dropout or stochastic depth)")
    attn = block.attn
    q, k, v = attn.qkv_heads(block.norm1(x), pos=pos)
    cache.append(k, v, t)
    keys, values = cache.read()
    x = x + block.ls1(attn.attend(q, keys.to(q.dtype), values.to(q.dtype)))
    x = x + block.ls2(block.mlp(block.norm2(x)))
    cache.commit(q, t, grid_hw)
    return x
