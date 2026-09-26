"""Attention split into ``qkv_heads``/``attend`` and the optional ``attn_mask`` of Attention and Block."""

import pytest
import torch

from omnivggt.layers.attention import Attention
from omnivggt.layers.block import Block
from omnivggt.layers.rope import PositionGetter, RotaryPositionEmbedding2D
from omnivggt.layers.vision_transformer import vit_small

DIM, HEADS, FRAMES, TOKENS = 16, 2, 3, 4


def _frame_mask(frames, tokens):
    """Key frame <= query frame (True = may attend), written out independently of the library."""
    frame = torch.arange(frames * tokens) // tokens
    return frame[None, :] <= frame[:, None]


def _attention(rope=False):
    torch.manual_seed(0)
    return Attention(DIM, num_heads=HEADS, qk_norm=True, rope=RotaryPositionEmbedding2D(100) if rope else None
                     ).double().eval()


def _block(**kwargs):
    torch.manual_seed(0)
    return Block(DIM, HEADS, init_values=0.01, qk_norm=True, **kwargs).double()


def _tokens(batch=2):
    torch.manual_seed(1)
    return torch.randn(batch, FRAMES * TOKENS, DIM, dtype=torch.float64)


def _pos(batch=2):
    return PositionGetter()(batch * FRAMES, 2, 2, device="cpu").view(batch, FRAMES * TOKENS, 2)


@pytest.mark.parametrize("rope", [False, True])
def test_forward_is_attend_of_qkv_heads(rope):
    attn, x = _attention(rope), _tokens()
    pos = _pos() if rope else None
    with torch.no_grad():
        assert torch.equal(attn(x, pos=pos), attn.attend(*attn.qkv_heads(x, pos=pos)))


def test_all_true_mask_is_unmasked():
    attn, x = _attention(), _tokens()
    full = torch.ones(FRAMES * TOKENS, FRAMES * TOKENS, dtype=torch.bool)
    with torch.no_grad():
        torch.testing.assert_close(attn(x, attn_mask=full), attn(x), rtol=0, atol=1e-12)


def test_frame_mask_equals_attending_only_the_past():
    attn, x = _attention(), _tokens()
    with torch.no_grad():
        masked = attn(x, attn_mask=_frame_mask(FRAMES, TOKENS))
        q, k, v = attn.qkv_heads(x)
        for frame in range(FRAMES):
            rows, past = slice(frame * TOKENS, (frame + 1) * TOKENS), slice(0, (frame + 1) * TOKENS)
            expected = attn.attend(q[:, :, rows], k[:, :, past], v[:, :, past])
            torch.testing.assert_close(masked[:, rows], expected, rtol=0, atol=1e-12)


def test_unfused_attention_applies_the_mask_like_sdpa():
    fused, x = _attention(), _tokens()
    unfused = _attention()
    unfused.fused_attn = False
    mask = _frame_mask(FRAMES, TOKENS)
    with torch.no_grad():
        torch.testing.assert_close(unfused(x, attn_mask=mask), fused(x, attn_mask=mask), rtol=0, atol=1e-12)
        additive = torch.zeros(mask.shape, dtype=x.dtype).masked_fill(~mask, float("-inf"))
        torch.testing.assert_close(unfused(x, attn_mask=additive), fused(x, attn_mask=mask), rtol=0, atol=1e-12)


def test_qkv_heads_returns_normalised_heads():
    attn, x = _attention(), _tokens()
    batch, tokens, _ = x.shape
    with torch.no_grad():
        q, k, v = attn.qkv_heads(x)
        raw = attn.qkv(x).reshape(batch, tokens, 3, HEADS, DIM // HEADS).permute(2, 0, 3, 1, 4)
    assert q.shape == k.shape == v.shape == (batch, HEADS, tokens, DIM // HEADS)
    assert torch.equal(q, attn.q_norm(raw[0]))
    assert torch.equal(k, attn.k_norm(raw[1]))
    assert torch.equal(v, raw[2])
    assert not torch.equal(k, raw[1])  # qk_norm is on, so the norm really changed the keys


def test_block_passes_the_mask_to_attention():
    block, x = _block().eval(), _tokens()
    mask = _frame_mask(FRAMES, TOKENS)
    with torch.no_grad():
        h = x + block.ls1(block.attn(block.norm1(x), attn_mask=mask))
        expected = h + block.ls2(block.mlp(block.norm2(h)))
        torch.testing.assert_close(block(x, attn_mask=mask), expected, rtol=0, atol=1e-12)
        assert not torch.allclose(block(x), expected)  # the mask matters for this input


@pytest.mark.parametrize("drop_path", [0.05, 0.5])  # the two training branches (drop path, sample subset)
def test_training_drop_path_branches_keep_the_frame_mask(drop_path):
    block = _block(drop_path=drop_path).train()
    x, mask = _tokens(batch=4), _frame_mask(FRAMES, TOKENS)
    future = x.clone()
    future[:, -TOKENS:] += 1.0
    past = slice(0, (FRAMES - 1) * TOKENS)
    torch.manual_seed(5)
    out = block(x, attn_mask=mask)
    torch.manual_seed(5)
    out_future = block(future, attn_mask=mask)
    assert torch.equal(out[:, past], out_future[:, past])
    torch.manual_seed(5)
    unmasked = block(x)
    torch.manual_seed(5)
    assert not torch.equal(unmasked[:, past], block(future)[:, past])


def test_sample_subset_branch_rejects_a_per_sample_mask():
    block = _block(drop_path=0.5).train()
    x = _tokens(batch=4)
    per_sample = _frame_mask(FRAMES, TOKENS).expand(4, 1, -1, -1)
    with pytest.raises(NotImplementedError):
        block(x, attn_mask=per_sample)


def test_encoder_blocks_without_a_mask_are_unchanged():
    """The DINOv2 encoder (NestedTensorBlock + MemEffAttention) never receives an attn_mask argument."""
    torch.manual_seed(0)
    encoder = vit_small(img_size=28, patch_size=14, num_register_tokens=4, block_chunks=0).eval()
    images = torch.rand(2, 3, 28, 28)
    with torch.no_grad():
        tokens = encoder(images, is_training=True)["x_norm_patchtokens"]
    assert tokens.shape == (2, 4, 384) and torch.isfinite(tokens).all()
