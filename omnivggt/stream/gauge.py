"""First-frame output gauge (design doc §7.2): an optional post-processing of the predicted cameras and depth.

With the raw camera-from-world extrinsics [R_t | t_t] and depth D_t, and s_1 the median valid depth of frame 1,
the gauge returns [R_t R_1^T | (t_t - R_t R_1^T t_1) / s_1] and D_t / s_1: frame 1 gets R = I, c = 0 and a
median valid depth of 1 (``torch.median``: the lower middle value for an even count). Relative poses keep their
rotation, and translations and depth share the one scale s_1, so the unprojected points become
P'_t = R_1 (P_t - c_1) / s_1. The gauge is set once, at frame 1, and never recomputed; it fixes conventions and
does not correct drift.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor


class FirstFrameGauge:
    """The gauge of a stream: the first call sets it from its first frame, every call applies it."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.rotation = self.translation = self.scale = None  # R_1 [3, 3], t_1 [3] and s_1 of frame 1

    @property
    def initialized(self) -> bool:
        return self.scale is not None

    def __call__(self, extrinsics_w2c: Tensor, depth: Tensor, valid: Optional[Tensor] = None
                 ) -> Tuple[Tensor, Tensor]:
        """Gauged ``extrinsics_w2c`` [S, 3, 4] and ``depth`` [S, H, W] or [S, H, W, 1].

        ``valid`` [S, H, W] (bool) is read on the first call only, where its frame 0 is frame 1 of the stream: the
        pixels whose depth is also finite and positive give s_1.
        """
        _check_shapes(extrinsics_w2c, depth)
        if not self.initialized:
            if valid is None:
                raise ValueError("the first call sets the gauge from frame 1: it needs the valid mask")
            self._initialize(extrinsics_w2c[0], depth[0], valid[0])
        rotation = extrinsics_w2c[:, :, :3] @ self.rotation.T
        translation = (extrinsics_w2c[:, :, 3] - rotation @ self.translation) / self.scale
        return torch.cat([rotation, translation[..., None]], dim=-1), depth / self.scale

    def _initialize(self, extrinsic_w2c: Tensor, depth: Tensor, valid: Tensor) -> None:
        depth = depth[..., 0] if depth.dim() == 3 else depth
        values = depth[valid.bool() & depth.isfinite() & (depth > 0)]
        if values.numel() == 0:
            raise ValueError("frame 1 has no valid pixel with a finite positive depth: the gauge is undefined")
        self.rotation, self.translation = extrinsic_w2c[:, :3].clone(), extrinsic_w2c[:, 3].clone()  # own copies
        self.scale = values.median()


def apply_first_frame_gauge(extrinsics_w2c: Tensor, depth: Tensor, valid: Tensor) -> Tuple[Tensor, Tensor]:
    """The first-frame gauge of a whole sequence (frame 1 = index 0); see ``FirstFrameGauge``."""
    return FirstFrameGauge()(extrinsics_w2c, depth, valid)


def _check_shapes(extrinsics_w2c: Tensor, depth: Tensor) -> None:
    if extrinsics_w2c.dim() != 3 or extrinsics_w2c.shape[1:] != (3, 4):
        raise ValueError(f"expected camera-from-world extrinsics [S, 3, 4], got {tuple(extrinsics_w2c.shape)}")
    if depth.dim() not in (3, 4) or (depth.dim() == 4 and depth.shape[-1] != 1) or len(depth) != len(extrinsics_w2c):
        raise ValueError(f"expected depth [S, H, W] or [S, H, W, 1] for {len(extrinsics_w2c)} frames, "
                         f"got {tuple(depth.shape)}")
