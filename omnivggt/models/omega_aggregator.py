"""OmniVGGT aggregator with VGGT-Omega style inter-frame (global) attention.

Only the inter-frame part changes; the image encoder, frame blocks and the GeoAdapter
(camera/depth injection) are inherited unchanged from ``ZeroAggregator``:

* register attention: in ``register_attention_layers`` only the camera and register tokens of all
  frames attend to each other; patch tokens skip the whole block (attention and FFN).
* ``num_register_tokens``: registers per frame (VGGT-Omega uses 16; OmniVGGT has 4). The image
  encoder keeps its own 4 DINOv2 registers either way.
* ``global_rope``: VGGT-Omega applies RoPE only in frame attention.
* ``cached_layers``: only the layers read by the heads are kept (the others are ``None``).
"""

import torch
import torch.nn as nn

from omnivggt.models.omnivggt_aggregator import ZeroAggregator
from omnivggt.stream.visibility import frame_visibility_mask

ENCODER_REGISTER_TOKENS = 4  # DINOv2 ViT-L/14-reg, as in the released OmniVGGT weights


class OmegaStyleAggregator(ZeroAggregator):
    def __init__(
        self,
        *args,
        num_register_tokens=16,
        register_attention_layers=(2, 6, 9, 14, 20),
        global_rope=False,
        cached_layers=(4, 11, 17, 23),
        **kwargs,
    ):
        super().__init__(*args, num_register_tokens=ENCODER_REGISTER_TOKENS, **kwargs)
        self.register_attention_layers = frozenset(register_attention_layers)
        self.cached_layers = frozenset(cached_layers)
        out_of_range = [i for i in (*self.register_attention_layers, *self.cached_layers) if not 0 <= i < self.depth]
        if out_of_range:
            raise ValueError(f"layer indices outside [0, {self.depth}): {sorted(out_of_range)}")
        if self.depth - 1 not in self.cached_layers:
            raise ValueError(f"the last layer ({self.depth - 1}) must be cached: the camera head reads it")
        if self.aa_block_size != 1 or self.aa_order != ["frame", "global"]:
            raise ValueError("register attention assumes aa_order=['frame', 'global'] and aa_block_size=1")
        if num_register_tokens != ENCODER_REGISTER_TOKENS:
            embed_dim = self.camera_token.shape[-1]
            self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim) * 1e-3)
            self.patch_start_idx = 1 + num_register_tokens
        if not global_rope:
            for block in self.global_blocks:
                block.attn.rope = None

    def _collect_layer(self, layer_idx, frame_out, global_out):
        return super()._collect_layer(layer_idx, frame_out, global_out) if layer_idx in self.cached_layers else None

    def _process_global_attention(
        self, tokens, B, S, P, C, global_idx, pos=None, pose_encoding=None, depth_encoding=None, attn_mask=None,
        visibility=None,
    ):
        if global_idx not in self.register_attention_layers:
            return super()._process_global_attention(
                tokens, B, S, P, C, global_idx, pos=pos, attn_mask=attn_mask, visibility=visibility
            )
        prefix = self.patch_start_idx
        tokens = tokens.reshape(B, S, P, C)
        special = tokens[:, :, :prefix].reshape(B, S * prefix, C)
        special_pos = None if pos is None else pos.reshape(B, S, P, 2)[:, :, :prefix].reshape(B, S * prefix, 2)
        special_mask = None
        if attn_mask is not None:  # the same inter-frame mask, restricted to the special tokens of every frame
            special_mask = attn_mask.view(S, P, S, P)[:, :prefix, :, :prefix].reshape(S * prefix, S * prefix)
        if visibility is not None:  # few tokens per frame: the dense [S*m, S*m] mask of the special tokens is small
            special_mask = frame_visibility_mask(visibility, prefix)
        special = self._run_global_block(global_idx, special, special_pos, special_mask)
        tokens = torch.cat([special.reshape(B, S, prefix, C), tokens[:, :, prefix:]], dim=2)
        return tokens.reshape(B, S * P, C), global_idx + 1, [tokens]
