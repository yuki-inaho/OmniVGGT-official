"""Streaming inference of the frame-causal OmniVGGTOmega: one frame per step, with per-layer KV caches.

``StreamingOmega.step`` runs ``model.inference`` on the current frame alone (S=1) while a ``_StreamContext`` is set
on the aggregator and the camera head. With the context set,

* the reference camera/register tokens are used at stream time t=1 only;
* the auxiliary depth of every frame is divided by gamma, the mean valid depth of frame 1 (fixed at t=1);
* every inter-frame block (global and register) runs ``stream_block`` on its own KV cache, and so does every
  camera-trunk block, with one cache per (refinement iteration, block).

With ``CachePolicy.full()`` step t equals frame t of the batch frame-causal inference (``causal=True``,
``depth_norm="first_frame"``); a bounded policy keeps, per cache, what ``LayerKVCache`` describes.
"""

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from omnivggt.heads.camera_head import NUM_ITERATIONS
from omnivggt.stream.kv_cache import CachePolicy, LayerKVCache

STEP_OUTPUTS = ("pose_enc", "depth", "depth_conf", "world_points")


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


class _StreamContext:
    """What the aggregator and the camera head read while ``StreamingOmega`` runs stream time ``t``."""

    def __init__(self, t: int, depth_scale: Optional[Tensor], grid_hw: Tuple[int, int], caches: Dict):
        self.t = t
        self.depth_scale = depth_scale  # gamma per sample, [B]; None when frame 1 had no depth
        self.grid_hw = grid_hw
        self.caches = caches

    @property
    def is_first(self) -> bool:
        return self.t == 1

    @property
    def camera_iterations(self) -> int:
        return len(self.caches["camera"])

    def run_global(self, layer_idx: int, block, x: Tensor, pos: Optional[Tensor] = None) -> Tensor:
        """Global block ``layer_idx`` on all tokens of the frame, or register block on its special tokens."""
        if layer_idx in self.caches["register"]:
            return stream_block(block, x, self.caches["register"][layer_idx], self.t, pos=pos)
        return stream_block(block, x, self.caches["global"][layer_idx], self.t, grid_hw=self.grid_hw, pos=pos)

    def run_camera(self, iteration: int, index: int, block, x: Tensor) -> Tensor:
        """Camera-trunk block ``index`` of refinement ``iteration`` on the frame's camera token."""
        return stream_block(block, x, self.caches["camera"][iteration][index], self.t)


def first_frame_depth_scale(depth: Tensor, mask: Tensor) -> Tensor:
    """gamma: the mean valid depth of each sample's frame ([B, H, W, 1] depth, [B, H, W] mask) -> [B]."""
    scales = []
    for b in range(depth.shape[0]):
        valid = depth[b, ..., 0][mask[b] > 0]  # the elements and order of ZeroAggregator.normalize_depth
        if valid.numel() == 0:
            raise ValueError("frame 1 of the stream has no valid depth pixel: the depth scale is undefined")
        scales.append(valid.mean())
    return torch.stack(scales)


class StreamingOmega:
    """Frame-by-frame inference of a frame-causal ``OmniVGGTOmega`` with per-layer KV caches.

    ``policy`` bounds every cache (see ``CachePolicy``). The backbone caches (global and register layers) store
    ``dtype`` (default: the model's parameter dtype); the camera caches store the camera head's parameter dtype,
    the dtype it computes in (``OmniVGGTOmega`` runs its heads without autocast). The caches are built at t=1
    from the model (one per aggregator layer, trunk_depth x refinement iterations for the camera head) and the
    frame size.
    """

    def __init__(self, model, policy: CachePolicy, dtype: Optional[torch.dtype] = None):
        aggregator, head = model.aggregator, model.camera_head
        if not (aggregator.causal and head.causal):
            raise ValueError("streaming runs the frame-causal model: build it with causal=True")
        if aggregator.depth_norm != "first_frame":
            raise ValueError("streaming fixes the depth scale at frame 1: build the model with depth_norm='first_frame'")
        if model.training:
            raise ValueError("streaming runs the model in eval mode: call model.eval() first")
        self.model = model
        self.policy = policy
        self.dtype = dtype if dtype is not None else aggregator.camera_token.dtype
        self.camera_dtype = head.empty_pose_tokens.dtype
        self.reset()

    def reset(self) -> None:
        """Forget the stream: the next step is t=1."""
        self.t = 0
        self.depth_scale = None
        self.image_hw = None
        self.caches = None
        self._failed = False

    @torch.no_grad()
    def step(self, image: Tensor, depth: Optional[Tensor] = None, mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Process the next frame: ``image`` [B, 3, H, W] in [0, 1], with ``depth`` [B, H, W, 1] and ``mask``
        [B, H, W] (RGB-D) or without both (RGB). Returns pose_enc [B, 1, 9], depth [B, 1, H, W, 1], depth_conf
        [B, 1, H, W] and world_points [B, 1, H, W, 3] of this frame.
        """
        if self._failed:
            raise RuntimeError("a previous step failed and left the caches inconsistent: call reset()")
        batch, image_hw, depth_scale = self._check_step_inputs(image, depth, mask)
        t = self.t + 1
        has_depth = depth is not None
        if has_depth and t == 1:
            depth_scale = first_frame_depth_scale(depth, mask)
        if not has_depth:  # unused placeholders: without depth views the aggregator reads only their device
            depth, mask = image.new_zeros(batch, *image_hw, 1), image.new_zeros(batch, *image_hw)
        patch = self.model.aggregator.patch_size
        grid_hw = (image_hw[0] // patch, image_hw[1] // patch)
        caches = self.caches if t > 1 else self._new_caches(grid_hw)
        context = _StreamContext(t, depth_scale, grid_hw, caches)
        aggregator, head = self.model.aggregator, self.model.camera_head
        aggregator._stream = head._stream = context
        try:
            predictions = self.model.inference(images=image[:, None], depth=depth[:, None], mask=mask[:, None],
                                               depth_gt_index=[0] if has_depth else [], camera_gt_index=[])
            _check_all_committed(caches, t)
        except BaseException:
            self._failed = True
            raise
        finally:
            aggregator._stream = head._stream = None
        self.t, self.depth_scale, self.image_hw, self.caches = t, depth_scale, image_hw, caches
        return {key: predictions[key] for key in STEP_OUTPUTS}

    def kv_bytes(self) -> int:
        """Bytes stored in all caches (``LayerKVCache.nbytes``)."""
        return 0 if self.caches is None else sum(cache.nbytes() for cache in all_caches(self.caches))

    def _check_step_inputs(self, image, depth, mask):
        if image.dim() != 4 or image.shape[1] != 3:
            raise ValueError(f"expected one frame per sample, [B, 3, H, W], got {tuple(image.shape)}")
        batch, _, height, width = image.shape
        if self.image_hw is not None and (height, width) != self.image_hw:
            raise ValueError(f"every frame of a stream has the same size {self.image_hw}, got {(height, width)}")
        if (depth is None) != (mask is None):
            raise ValueError("pass depth and mask together (RGB-D) or neither (RGB)")
        if depth is not None:
            if depth.shape != (batch, height, width, 1) or mask.shape != (batch, height, width):
                raise ValueError(f"expected depth [B, H, W, 1] and mask [B, H, W] of the image size, "
                                 f"got {tuple(depth.shape)} and {tuple(mask.shape)}")
            if self.t > 0 and self.depth_scale is None:
                raise ValueError("frame 1 of the stream had no depth, so later depth has no scale (gamma)")
        return batch, (height, width), self.depth_scale

    def _new_caches(self, grid_hw: Tuple[int, int]) -> Dict:
        aggregator, head = self.model.aggregator, self.model.camera_head
        special = aggregator.patch_start_idx
        tokens = special + grid_hw[0] * grid_hw[1]
        register_layers = aggregator.register_attention_layers

        def cache(tokens_per_frame, special_count, dtype, layer_id):
            return LayerKVCache(self.policy, tokens_per_frame, special_count, dtype, layer_id)

        return {
            "global": {i: cache(tokens, special, self.dtype, i)
                       for i in range(aggregator.depth) if i not in register_layers},
            "register": {i: cache(special, special, self.dtype, i) for i in sorted(register_layers)},
            "camera": [[cache(1, 1, self.camera_dtype, aggregator.depth + i * head.trunk_depth + j)
                        for j in range(head.trunk_depth)] for i in range(NUM_ITERATIONS)],
        }


def all_caches(caches: Dict):
    """Every ``LayerKVCache`` of ``StreamingOmega.caches``: global, register, then camera (iteration-major)."""
    yield from caches["global"].values()
    yield from caches["register"].values()
    for row in caches["camera"]:
        yield from row


def _check_all_committed(caches: Dict, t: int) -> None:
    stale = [cache.layer_id for cache in all_caches(caches) if cache.last_frame != t]
    if stale:
        raise RuntimeError(f"caches of layers {stale} were not updated at stream time {t}")
