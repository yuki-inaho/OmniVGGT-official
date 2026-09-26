"""Streaming (one frame per step) frame-causal OmniVGGTOmega: ``stream_block`` and ``StreamingOmega``."""

import pytest
import torch

from omnivggt.layers.block import Block
from omnivggt.layers.rope import PositionGetter, RotaryPositionEmbedding2D
from omnivggt.stream.kv_cache import CachePolicy, LayerKVCache
from omnivggt.stream.masks import frame_causal_mask
from omnivggt.stream.streaming import stream_block

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
