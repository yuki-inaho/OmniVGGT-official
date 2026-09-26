"""Frame visibility of the inter-frame attention: which key frames each query frame of a batch attends to.

A frame visibility is an [S, S] bool matrix; ``visibility[a, b]`` is True when every token of query frame a may
attend to every token of key frame b. A frame always sees itself (the diagonal is True). The inter-frame layers
apply it without a dense [S*P, S*P] token mask where P is large: ``visible_frames_block`` gathers, per query
frame, the keys and values of its visible frames; ``frame_visibility_mask`` expands it to a token mask for layers
with few tokens per frame (register layers, the camera trunk).
"""

from typing import Optional

import torch
from torch import Tensor


def band_visibility(num_frames: int, width: int, device=None) -> Tensor:
    """[S, S] bool: frame a sees frame b when b == 0 (the anchor) or |a - b| <= ``width``."""
    if not isinstance(width, int) or isinstance(width, bool) or width < 1:
        raise ValueError(f"band width must be an int >= 1, got {width!r}")
    frame = torch.arange(num_frames, device=device)
    return ((frame[:, None] - frame[None, :]).abs() <= width) | (frame[None, :] == 0)


def is_lower_triangular(visibility: Tensor) -> bool:
    """No query frame sees a later frame."""
    return not visibility.triu(diagonal=1).any()


def check_frame_visibility(visibility: Tensor, num_frames: int, causal: bool) -> None:
    """Reject a visibility the model cannot apply: not an [S, S] bool matrix, a frame that does not see itself, or,
    for a frame-causal model, a query frame that sees a later frame."""
    if not isinstance(visibility, Tensor) or visibility.dtype != torch.bool:
        raise ValueError(f"frame_visibility must be a bool tensor, got {getattr(visibility, 'dtype', type(visibility))}")
    if visibility.shape != (num_frames, num_frames):
        raise ValueError(f"frame_visibility must have shape ({num_frames}, {num_frames}) for {num_frames} frames, "
                         f"got {tuple(visibility.shape)}")
    if not visibility.diagonal().all():
        raise ValueError("frame_visibility must let every frame see itself (a True diagonal)")
    if causal and not is_lower_triangular(visibility):
        raise ValueError("a frame-causal model (causal=True) takes only a lower-triangular frame_visibility: "
                         "no frame may see a later one")


def frame_visibility_mask(visibility: Tensor, tokens_per_frame: int) -> Tensor:
    """Token mask of frame-major tokens, [S*n, S*n] bool (``kron(visibility, ones(n, n))``); for a small n only."""
    return visibility.repeat_interleave(tokens_per_frame, dim=0).repeat_interleave(tokens_per_frame, dim=1)


def visible_frames_block(block, x: Tensor, visibility: Tensor, pos: Optional[Tensor] = None) -> Tensor:
    """``block(x, pos, attn_mask=frame_visibility_mask(visibility, P))`` in eval mode, without that mask.

    ``x`` holds the frame-major tokens of S frames, [B, S*P, C]. The queries of frame a attend to the keys and
    values of the frames b with ``visibility[a, b]``, gathered per query frame; the residual structure is that of
    ``Block.forward`` in eval mode.
    """
    if block.training:
        raise NotImplementedError("visible_frames_block runs blocks in eval mode only (no dropout or stochastic depth)")
    num_frames = len(visibility)
    batch, tokens, _ = x.shape
    if tokens % num_frames:
        raise ValueError(f"{tokens} tokens do not split into {num_frames} frames")
    per_frame = tokens // num_frames
    attn = block.attn
    q, k, v = attn.qkv_heads(block.norm1(x), pos=pos)  # [B, H, S*P, D]
    heads, dim = k.shape[1], k.shape[-1]
    k, v = (t.view(batch, heads, num_frames, per_frame, dim) for t in (k, v))
    outputs = []
    for frame, row in enumerate(visibility.cpu()):  # the indices on the host: no device sync per query frame
        visible = row.nonzero().flatten().to(x.device)
        keys, values = (t[:, :, visible].flatten(2, 3) for t in (k, v))
        outputs.append(attn.attend(q[:, :, frame * per_frame : (frame + 1) * per_frame], keys, values))
    x = x + block.ls1(torch.cat(outputs, dim=1))
    return x + block.ls2(block.mlp(block.norm2(x)))
