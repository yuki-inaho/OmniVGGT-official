"""Attention masks of the frame-causal model (True = the query may attend to the key)."""

import torch


def frame_causal_mask(num_frames: int, tokens_per_frame: int, device) -> torch.Tensor:
    """Frame-causal mask over frame-major tokens: ``[S*P, S*P]`` bool.

    A query token of frame a may attend to every token of a key frame b <= a, including all of its own frame;
    tokens are not causal within a frame. With one token per frame this is the lower-triangular mask.
    """
    if num_frames < 1 or tokens_per_frame < 1:
        raise ValueError(f"need at least one frame and one token per frame, got {num_frames} x {tokens_per_frame}")
    frame = torch.arange(num_frames, device=device).repeat_interleave(tokens_per_frame)
    return frame[None, :] <= frame[:, None]
