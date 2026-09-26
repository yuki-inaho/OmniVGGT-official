"""Per-layer KV cache of the streaming (frame-causal) OmniVGGTOmega.

One ``LayerKVCache`` holds the keys and values (after q/k norm, as ``Attention.qkv_heads`` returns them) that one
inter-frame block, register block or camera-trunk block has seen. A stream step appends the current frame, reads
the stored past plus the current frame for the attention, then commits: the cache keeps what its ``CachePolicy``
allows for the next step. Frame ids start at 1 and are consecutive; token ids number the tokens of a frame, the
``special_count`` special (camera and register) tokens first.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class CachePolicy:
    """What a cache keeps. ``recent=None`` is the full cache: every frame is kept."""

    recent: Optional[int]
    long_special: int = 0
    long_patch: int = 0
    selector: Optional[str] = None
    quant: Optional[str] = None

    @classmethod
    def full(cls) -> "CachePolicy":
        return cls(recent=None)

    @property
    def is_full(self) -> bool:
        return self.recent is None


class LayerKVCache:
    """Keys and values of one layer, stored in ``dtype`` in a preallocated buffer that doubles when it is full.

    ``tokens_per_frame`` (n) tokens take part in the layer per frame, the first ``special_count`` (m) of them
    special; register blocks and the camera trunk have n == m (no patch tokens).
    """

    def __init__(self, policy: CachePolicy, tokens_per_frame: int, special_count: int, dtype: torch.dtype,
                 layer_id: int, initial_frames: int = 16):
        if not 1 <= special_count <= tokens_per_frame:
            raise ValueError(f"need 1 <= special_count <= tokens_per_frame, got {special_count}, {tokens_per_frame}")
        if initial_frames < 1:
            raise ValueError(f"initial_frames must be positive, got {initial_frames}")
        if not policy.is_full:
            raise NotImplementedError("only the full cache is implemented")
        self.policy = policy
        self.tokens_per_frame = tokens_per_frame
        self.special_count = special_count
        self.dtype = dtype
        self.layer_id = layer_id
        self.last_frame = 0  # the last committed frame id
        self._initial_rows = initial_frames * tokens_per_frame
        self._pending = None  # frame id appended but not committed yet
        self._keys = self._values = None  # [B, H, capacity, D], allocated on the first append
        self._frame_id = self._token_id = None  # [capacity] int64, row metadata
        self._rows = 0

    @property
    def capacity(self) -> int:
        return 0 if self._keys is None else self._keys.shape[2]

    @property
    def size(self) -> int:
        """Number of stored rows (tokens), the current frame included while it is not committed."""
        return self._rows

    def append(self, k: Tensor, v: Tensor, frame_id: int) -> None:
        """Write the current frame's keys and values ([B, H, n, D]) after the stored rows."""
        if self._pending is not None:
            raise RuntimeError(f"frame {self._pending} was appended but not committed: commit before the next append")
        if frame_id != self.last_frame + 1:
            raise ValueError(f"frames must be appended in order from 1: expected frame {self.last_frame + 1}, "
                             f"got {frame_id}")
        if k.shape != v.shape or k.dim() != 4 or k.shape[2] != self.tokens_per_frame:
            raise ValueError(f"expected keys and values of shape [B, H, {self.tokens_per_frame} tokens, D], "
                             f"got {tuple(k.shape)} and {tuple(v.shape)}")
        if self._keys is None:
            self._allocate(k)
        elif k.shape[:2] != self._keys.shape[:2] or k.shape[3] != self._keys.shape[3]:
            raise ValueError(f"keys of shape {tuple(k.shape)} do not match the cache {tuple(self._keys.shape)}")
        self._reserve(self._rows + self.tokens_per_frame)
        rows = slice(self._rows, self._rows + self.tokens_per_frame)
        self._keys[:, :, rows] = k.to(self.dtype)
        self._values[:, :, rows] = v.to(self.dtype)
        self._frame_id[rows] = frame_id
        self._token_id[rows] = torch.arange(self.tokens_per_frame, device=self._token_id.device)
        self._rows += self.tokens_per_frame
        self._pending = frame_id

    def read(self) -> Tuple[Tensor, Tensor]:
        """Keys and values of every stored row (views of the buffer), in the order of ``row_ids``."""
        if self._keys is None:
            raise RuntimeError("the cache is empty: append a frame first")
        return self._keys[:, :, : self._rows], self._values[:, :, : self._rows]

    def row_ids(self) -> Tuple[Tensor, Tensor]:
        """Frame id and token id of every row of ``read()``."""
        return self._frame_id[: self._rows], self._token_id[: self._rows]

    def commit(self, q: Tensor, t: int, grid_hw: Optional[Tuple[int, int]] = None) -> None:
        """End step ``t``: keep what the policy allows for the next step.

        ``q`` ([B, H, n, D]) are the current frame's queries and ``grid_hw`` its patch grid (rows, columns); the
        query-guided selectors read them.
        """
        if self._pending is None:
            raise RuntimeError("nothing to commit: append the current frame first")
        if t != self._pending:
            raise ValueError(f"commit of frame {t}, but frame {self._pending} was appended")
        self._pending = None
        self.last_frame = t

    def nbytes(self) -> int:
        """Bytes of the stored keys and values (the stored rows, not the buffer capacity)."""
        if self._keys is None:
            return 0
        batch, heads, _, dim = self._keys.shape
        return 2 * batch * heads * self._rows * dim * self._keys.element_size()

    def _allocate(self, k: Tensor) -> None:
        batch, heads, _, dim = k.shape
        self._keys = k.new_empty(batch, heads, self._initial_rows, dim, dtype=self.dtype)
        self._values = torch.empty_like(self._keys)
        self._frame_id = torch.empty(self._initial_rows, dtype=torch.int64, device=k.device)
        self._token_id = torch.empty_like(self._frame_id)

    def _reserve(self, rows: int) -> None:
        """Grow the buffer (doubling its capacity) until it holds ``rows`` rows, keeping the stored ones."""
        capacity = self.capacity
        if rows <= capacity:
            return
        while capacity < rows:
            capacity *= 2
        keys = self._keys.new_empty(*self._keys.shape[:2], capacity, self._keys.shape[3])
        values = torch.empty_like(keys)
        keys[:, :, : self._rows] = self._keys[:, :, : self._rows]
        values[:, :, : self._rows] = self._values[:, :, : self._rows]
        frame_id = self._frame_id.new_empty(capacity)
        token_id = self._token_id.new_empty(capacity)
        frame_id[: self._rows] = self._frame_id[: self._rows]
        token_id[: self._rows] = self._token_id[: self._rows]
        self._keys, self._values, self._frame_id, self._token_id = keys, values, frame_id, token_id
