"""Per-layer KV cache of the streaming (frame-causal) OmniVGGTOmega.

One ``LayerKVCache`` holds the keys and values (after q/k norm, as ``Attention.qkv_heads`` returns them) that one
inter-frame block, register block or camera-trunk block has seen. A stream step appends the current frame, reads
the stored past plus the current frame for the attention, then commits: the cache keeps what its ``CachePolicy``
allows for the next step. Frame ids start at 1 and are consecutive; token ids number the tokens of a frame, the
``special_count`` special (camera and register) tokens first, then the patches in raster order.

A bounded cache is partitioned (design doc §4; no row is in two parts) into

* anchor: every token of frame 1, always kept, and with ``anchor_every`` = A of the newest ``max_anchors`` frames
  1 + kA (k >= 1), each promoted at its own step; a demoted anchor becomes a long-term candidate;
* recent: every token of the ``recent`` newest frames (the current one included, the anchors excluded);
* long-special: the special tokens of at most ``long_special`` other frames, selected per frame;
* long-patch: at most ``long_patch`` other patch tokens, selected per token, or (``long_frames``) every patch
  token of at most ``long_frames`` other frames, selected per frame (IncVGGT-style whole frames).

The long-term stores are selected at the commit, from every candidate of the written cache, by the policy's
selector; ties go to the newer frame, then to the smaller token id. A frame is ranked by the mean score of its
candidate rows. The ``diversity`` selector (InfiniteVGGT-style, query-free) ranks the long-patch candidates only;
the long-special frames are then chosen by the ``query`` selector. With ``quant`` the long-patch store keeps
INT8/INT4 codes (asymmetric, design doc §6): a token is quantised once, when it enters the store, and keeps its
codes and block scale/offset until it is evicted; the other parts stay in the cache dtype.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

SELECTOR_NAMES = ("query", "xstream", "recency", "random", "diversity")
QUANT_BITS = {"int8": 8, "int4": 4}
QUANT_NAMES = tuple(QUANT_BITS)
QUANT_BLOCK = 64  # K: tokens of one channel per scale/offset; V: channels of one token
XSTREAM_POOL = 16  # patch queries per pooled row of XStreamVGGT's score
RANDOM_SEED = 42


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class CachePolicy:
    """What every cache of a stream keeps.

    * ``recent`` (w >= 1): the w newest frames besides the anchor. ``None`` is the full cache: every frame is kept,
      and nothing else may be set.
    * ``long_special`` (k_U): special tokens of at most k_U other frames.
    * ``long_patch`` (b_P): at most b_P other patch tokens (layers without patches ignore it).
    * ``long_frames`` (k_P): instead of ``long_patch``, every patch token of at most k_P other frames.
    * ``selector``: one of ``SELECTOR_NAMES``; required exactly when there is a long-term store.
    * ``quant``: ``"int8"`` or ``"int4"`` storage of the long-patch store.
    * ``anchor_every`` (A) and ``max_anchors`` (n_anchor), together or neither: frame 1 + kA becomes an anchor at
      step 1 + kA, and the newest n_anchor of them are kept whole besides frame 1.
    """

    recent: Optional[int]
    long_special: int = 0
    long_patch: int = 0
    selector: Optional[str] = None
    quant: Optional[str] = None
    long_frames: int = 0
    anchor_every: Optional[int] = None
    max_anchors: Optional[int] = None

    def __post_init__(self):
        if self.recent is None:
            others = (self.long_special, self.long_patch, self.long_frames, self.selector, self.quant,
                      self.anchor_every, self.max_anchors)
            if others != (0, 0, 0, None, None, None, None):
                raise ValueError("the full cache (recent=None) keeps every frame: it takes no long-term store, "
                                 "selector, quantisation or periodic anchor")
            return
        if not _is_int(self.recent) or self.recent < 1:
            raise ValueError(f"recent must be an int >= 1 (it includes the current frame), got {self.recent!r}")
        if (self.anchor_every is None) != (self.max_anchors is None):
            raise ValueError("periodic anchors take both anchor_every and max_anchors (or neither), got "
                             f"anchor_every={self.anchor_every!r}, max_anchors={self.max_anchors!r}")
        if self.anchor_every is not None:
            for name in ("anchor_every", "max_anchors"):
                value = getattr(self, name)
                if not _is_int(value) or value < 1:
                    raise ValueError(f"{name} must be an int >= 1, got {value!r}")
        for name in ("long_special", "long_patch", "long_frames"):
            value = getattr(self, name)
            if not _is_int(value) or value < 0:
                raise ValueError(f"{name} must be an int >= 0, got {value!r}")
        if self.long_patch > 0 and self.long_frames > 0:
            raise ValueError("the long-patch store keeps tokens (long_patch) or whole frames (long_frames), not both")
        has_long_patch = self.long_patch > 0 or self.long_frames > 0
        has_long_term = self.long_special > 0 or has_long_patch
        if has_long_term and self.selector not in SELECTOR_NAMES:
            raise ValueError(f"a long-term store needs a selector from {SELECTOR_NAMES}, got {self.selector!r}")
        if not has_long_term and self.selector is not None:
            raise ValueError(f"selector {self.selector!r} given, but there is no long-term store to select")
        if self.quant is not None and self.quant not in QUANT_NAMES:
            raise ValueError(f"quant must be None or one of {QUANT_NAMES}, got {self.quant!r}")
        if self.quant is not None and not has_long_patch:
            raise ValueError("quant applies to the long-patch store only, but long_patch=0 and long_frames=0")

    @classmethod
    def full(cls) -> "CachePolicy":
        return cls(recent=None)

    @classmethod
    def for_budget(cls, budget: int, tokens_per_frame: int, special_count: int, recent: int, long_special: int,
                   selector: Optional[str], quant: Optional[str] = None) -> "CachePolicy":
        """The policy whose global-layer budget is ``budget``: b_P = B - (1+w)n - m k_U."""
        long_patch = budget - (1 + recent) * tokens_per_frame - special_count * long_special
        if long_patch < 0:
            raise ValueError(f"budget {budget} is below the anchor, {recent} recent frames and {long_special} "
                             f"long-special frames ({budget - long_patch} tokens): change the resolution, recent "
                             "or long_special (the anchor is never trimmed)")
        return cls(recent, long_special, long_patch, selector, quant)

    @property
    def is_full(self) -> bool:
        return self.recent is None

    def budget(self, tokens_per_frame: int, special_count: int) -> Optional[int]:
        """Most tokens a cache keeps between steps: (1+n_anchor+w)n + m k_U + b_P (b_P = k_P (n-m) with whole
        frames), or m(1+n_anchor+w+k_U) without patches (register layers: n = m; camera trunk: n = m = 1). None
        for the full cache."""
        if self.is_full:
            return None
        return ((1 + self.extra_anchors + self.recent) * tokens_per_frame + special_count * self.long_special
                + self.long_patch_capacity(tokens_per_frame - special_count))

    @property
    def extra_anchors(self) -> int:
        """n_anchor: the most promoted anchors kept besides frame 1."""
        return 0 if self.anchor_every is None else self.max_anchors

    def anchor_frames(self, t: int) -> List[int]:
        """Anchors after the commit of step ``t``: frame 1 and the newest ``max_anchors`` frames 1 + kA <= t."""
        if self.anchor_every is None:
            return [1]
        return [1, *range(1 + self.anchor_every, t + 1, self.anchor_every)[-self.max_anchors:]]

    def protected_frames(self, t: int) -> List[int]:
        """Frames kept whole after the commit of step ``t`` (ascending): all of them for the full cache, else the
        anchors and the ``recent`` newest frames."""
        if self.is_full:
            return list(range(1, t + 1))
        return sorted({*self.anchor_frames(t), *range(max(2, t - self.recent + 1), t + 1)})

    def long_patch_capacity(self, patches: int) -> int:
        """Most long-patch tokens of a layer with ``patches`` patch tokens per frame: b_P, or k_P whole frames."""
        if patches == 0:
            return 0
        return self.long_patch + self.long_frames * patches


@dataclass(frozen=True)
class Candidates:
    """What a selector sees at the commit of stream time ``t``: the current queries and every written row."""

    queries: Tensor  # [1, H, n, D], the current frame's (after q norm)
    keys: Tensor  # [1, H, N, D], every row of the written cache, in ``LayerKVCache.row_ids`` order
    frame_id: Tensor  # [N]
    token_id: Tensor  # [N]
    special_count: int
    grid_hw: Optional[Tuple[int, int]]  # patch grid (rows, columns) of a frame; None without patches
    layer_id: int
    t: int
    pool: Optional[Tensor] = None  # [N] bool, the long-patch candidates (the diversity selector's pool)


def query_groups(special_count: int, grid_hw: Optional[Tuple[int, int]]) -> Tensor:
    """Query group of every token of a frame ([n] int64): each special token alone, then the patches (raster
    order) in 2 x 2 blocks of the grid; an odd last row or column gives 1 x 2, 2 x 1 or 1 x 1 blocks."""
    special = torch.arange(special_count)
    if grid_hw is None:
        return special
    rows, columns = grid_hw
    blocks = torch.arange(rows)[:, None] // 2 * ((columns + 1) // 2) + torch.arange(columns)[None, :] // 2
    return torch.cat([special, special_count + blocks.reshape(-1)])


def xstream_rows(special_count: int, patch_count: int) -> Tensor:
    """XStreamVGGT's pooled query rows of every token ([n] int64): each special token alone, then the patches in
    raster chunks of ``XSTREAM_POOL`` (the remainder is one more row)."""
    return torch.cat([torch.arange(special_count), special_count + torch.arange(patch_count) // XSTREAM_POOL])


def _pooled_queries(queries: Tensor, groups: Tensor) -> Tuple[Tensor, Tensor]:
    """Mean query of every group ([1, H, G, D]) and the group sizes ([G])."""
    if groups.numel() != queries.shape[2]:
        raise ValueError(f"{groups.numel()} grouped tokens for {queries.shape[2]} queries: check grid_hw")
    groups = groups.to(queries.device)
    sizes = torch.bincount(groups).to(queries.dtype)
    sums = queries.new_zeros(*queries.shape[:2], sizes.numel(), queries.shape[3]).index_add_(2, groups, queries)
    return sums / sizes[:, None], sizes


def _score_dtype(candidates: Candidates) -> torch.dtype:
    return torch.promote_types(candidates.keys.dtype, torch.float32)


def select_query(candidates: Candidates) -> Tensor:
    """Query-guided importance of every row (design doc §5): s_j = (1/H) sum_h sum_a w_a alpha_{h,a,j}.

    The current queries are pooled per ``query_groups`` (mean q_{h,a}, weight w_a = |G_a| / n) and alpha_{h,a,.}
    = softmax_j(q_{h,a} . k_{h,j} / sqrt(d)) over every row of the written cache.
    """
    queries, keys = candidates.queries, candidates.keys
    has_patches = queries.shape[2] > candidates.special_count
    if has_patches and candidates.grid_hw is None:
        raise ValueError("the query selector groups the patch queries on the grid: grid_hw is needed")
    dtype = _score_dtype(candidates)
    pooled, sizes = _pooled_queries(queries.to(dtype), query_groups(candidates.special_count, candidates.grid_hw))
    logits = pooled @ keys.to(dtype).transpose(-2, -1) / math.sqrt(queries.shape[3])  # [1, H, G, N]
    weights = sizes / queries.shape[2]
    return (logits.softmax(dim=-1) * weights[:, None]).sum(dim=2).mean(dim=1)[0]


def select_xstream(candidates: Candidates) -> Tensor:
    """XStreamVGGT's rank-1 score of every row: s_j = <mu, mean_h k_j>, mu the mean over the pooled query rows
    (``xstream_rows``) of their head-mean queries. Only the score is XStreamVGGT's, not its cache policy."""
    queries = candidates.queries
    dtype = _score_dtype(candidates)
    patch_count = queries.shape[2] - candidates.special_count
    pooled, _ = _pooled_queries(queries.to(dtype), xstream_rows(candidates.special_count, patch_count))
    mu = pooled.mean(dim=1).mean(dim=1)  # [1, D]: mean over heads, then over rows
    return (candidates.keys.to(dtype).mean(dim=1) @ mu[0])[0]


def select_recency(candidates: Candidates) -> Tensor:
    """Importance of every row (higher is kept first): the newer frame (ties: the smaller token id)."""
    return candidates.frame_id.double()


def select_random(candidates: Candidates) -> Tensor:
    """Random importance of every row, from ``numpy.random.default_rng([42, layer_id, t])``."""
    generator = np.random.default_rng([RANDOM_SEED, candidates.layer_id, candidates.t])
    return torch.from_numpy(generator.random(candidates.frame_id.numel())).to(candidates.frame_id.device)


def select_diversity(candidates: Candidates) -> Tensor:
    """InfiniteVGGT's query-free diversity of every row: s_j = -(1/H) sum_h cos(k_{h,j}, mu_h), mu_h the mean of
    the unit keys khat = k / |k| of the pool (the long-patch candidates). One score per row, so every head keeps
    the same rows; rows far from the pool's mean direction are kept first."""
    pool = candidates.pool
    if pool is None or not bool(pool.any()):
        raise ValueError("the diversity selector scores against the long-patch pool: give a non-empty pool")
    unit = F.normalize(candidates.keys.to(_score_dtype(candidates)), dim=-1)  # [1, H, N, D]
    direction = F.normalize(unit[:, :, pool].mean(dim=2, keepdim=True), dim=-1)  # [1, H, 1, D]
    return -(unit * direction).sum(dim=-1).mean(dim=1)[0]


SELECTORS = {"query": select_query, "xstream": select_xstream, "recency": select_recency, "random": select_random,
             "diversity": select_diversity}


def _ranked(scores: Tensor, frame_id: Tensor, token_id: Tensor) -> Tensor:
    """Order by score (descending), then frame id (newer first), then token id (smaller first)."""
    order = torch.argsort(token_id, stable=True)
    order = order[torch.argsort(frame_id[order], descending=True, stable=True)]
    return order[torch.argsort(scores[order], descending=True, stable=True)]


def top_rows(scores: Tensor, frame_id: Tensor, token_id: Tensor, candidates: Tensor, k: int) -> Tensor:
    """Indices of the (at most) ``k`` best rows among the ``candidates`` (bool mask)."""
    rows = candidates.nonzero().squeeze(1)
    return rows[_ranked(scores[rows], frame_id[rows], token_id[rows])[:k]]


def top_frames(scores: Tensor, frame_id: Tensor, candidates: Tensor, k: int) -> Tensor:
    """Ids of the (at most) ``k`` frames with the best mean score S_f over their candidate rows."""
    frames, inverse = torch.unique(frame_id[candidates], return_inverse=True)
    sums = torch.zeros(frames.numel(), dtype=scores.dtype, device=scores.device)
    sums.index_add_(0, inverse, scores[candidates])
    frame_scores = sums / torch.bincount(inverse, minlength=frames.numel())
    return frames[_ranked(frame_scores, frames, torch.zeros_like(frames))[:k]]


def _check_bits(bits: int) -> None:
    if bits not in QUANT_BITS.values():
        raise ValueError(f"bits must be one of {sorted(QUANT_BITS.values())}, got {bits}")


def _pack(codes: Tensor, bits: int) -> Tensor:
    """uint8 codes: one per byte (8 bits) or two per byte along the last axis (4 bits: low nibble first)."""
    if bits == 8:
        return codes
    if codes.shape[-1] % 2:
        raise ValueError(f"INT4 packs pairs of channels: the last axis must be even, got {codes.shape[-1]}")
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def _unpack(packed: Tensor, bits: int) -> Tensor:
    if bits == 8:
        return packed
    return torch.stack([packed & 0xF, packed >> 4], dim=-1).flatten(-2)


def _block_quantize(x: Tensor, dim: int, bits: int) -> Tuple[Tensor, Tensor, Tensor]:
    """Asymmetric quantisation of ``x`` [B, H, N, D] in blocks of ``QUANT_BLOCK`` consecutive entries along ``dim``.

    Per block, s = max((x_max - x_min) / (2^b - 1), eps) and q = clip(round((x - x_min) / s), 0, 2^b - 1), so
    x_min + s q is within s/2 of x. Returns packed uint8 codes and the scale and offset (x_min) per block, stored
    in ``x.dtype``; the codes are rounded against the stored scale and offset.
    """
    _check_bits(bits)
    levels = 2**bits - 1
    work = x.to(torch.promote_types(x.dtype, torch.float32))
    block_of = torch.arange(x.shape[dim], device=x.device) // QUANT_BLOCK
    index = block_of.view([-1 if axis == dim else 1 for axis in range(x.dim())]).expand_as(work)
    shape = list(x.shape)
    shape[dim] = int(block_of[-1]) + 1
    low = work.new_full(shape, float("inf")).scatter_reduce(dim, index, work, "amin")
    high = work.new_full(shape, float("-inf")).scatter_reduce(dim, index, work, "amax")
    scale = ((high - low) / levels).clamp_min(torch.finfo(x.dtype).tiny).to(x.dtype)
    offset = low.to(x.dtype)
    step = scale.to(work.dtype).index_select(dim, block_of)
    codes = torch.round((work - offset.to(work.dtype).index_select(dim, block_of)) / step).clamp_(0, levels)
    return _pack(codes.to(torch.uint8), bits), scale, offset


def quantize_k(k: Tensor, bits: int) -> Tuple[Tensor, Tensor, Tensor]:
    """Keys [B, H, N, D]: each channel of a head in blocks of 64 tokens; scale/offset [B, H, ceil(N/64), D]."""
    return _block_quantize(k, 2, bits)


def quantize_v(v: Tensor, bits: int) -> Tuple[Tensor, Tensor, Tensor]:
    """Values [B, H, N, D]: each token of a head in blocks of 64 channels; scale/offset [B, H, N, ceil(D/64)]."""
    return _block_quantize(v, 3, bits)


def dequantize(codes: Tensor, scale: Tensor, offset: Tensor, bits: int) -> Tensor:
    """offset + scale * code, with ``scale`` and ``offset`` given per entry (the shape of the unpacked codes)."""
    _check_bits(bits)
    work = torch.promote_types(scale.dtype, torch.float32)
    return (offset.to(work) + scale.to(work) * _unpack(codes, bits).to(work)).to(scale.dtype)


class _QuantizedRows:
    """Rows of a quantised long-patch store, each quantised once when it entered.

    K codes keep a scale/offset per (block of up to 64 tokens that entered together, channel); a block is
    released when none of its tokens is left. V codes keep theirs per (token, block of 64 channels).
    """

    def __init__(self, bits: int):
        self.bits = bits
        self.rows = 0
        self.k_codes = self.v_codes = None  # [1, H, R, D * bits / 8] uint8
        self.v_scale = self.v_offset = None  # [1, H, R, ceil(D / 64)]
        self.k_scale = self.k_offset = None  # [1, H, K blocks, D]
        self.block = self.frame_id = self.token_id = None  # [R] int64

    def add(self, keys: Tensor, values: Tensor, frame_id: Tensor, token_id: Tensor) -> None:
        """Quantise rows entering together ([1, H, R, D], in token order) and append them."""
        k_codes, k_scale, k_offset = quantize_k(keys, self.bits)
        v_codes, v_scale, v_offset = quantize_v(values, self.bits)
        first_block = 0 if self.k_scale is None else self.k_scale.shape[2]
        block = first_block + torch.arange(keys.shape[2], device=keys.device) // QUANT_BLOCK
        parts = {"k_codes": k_codes, "v_codes": v_codes, "v_scale": v_scale, "v_offset": v_offset,
                 "k_scale": k_scale, "k_offset": k_offset, "block": block, "frame_id": frame_id, "token_id": token_id}
        for name, part in parts.items():
            stored = getattr(self, name)
            axis = 0 if part.dim() == 1 else 2
            setattr(self, name, part if stored is None else torch.cat([stored, part], dim=axis))
        self.rows += keys.shape[2]

    def keep(self, keep: Tensor) -> None:
        """Keep the rows of ``keep`` (bool mask) in their order; release the K blocks none of them refers to."""
        if self.rows == 0 or bool(keep.all()):
            return
        rows = keep.nonzero().squeeze(1)
        for name in ("k_codes", "v_codes", "v_scale", "v_offset"):
            setattr(self, name, getattr(self, name)[:, :, rows])
        for name in ("frame_id", "token_id"):
            setattr(self, name, getattr(self, name)[rows])
        used, self.block = torch.unique(self.block[rows], return_inverse=True)
        self.k_scale, self.k_offset = self.k_scale[:, :, used], self.k_offset[:, :, used]
        self.rows = rows.numel()

    def dequantized(self) -> Tuple[Tensor, Tensor]:
        dim = self.k_scale.shape[3]
        keys = dequantize(self.k_codes, self.k_scale[:, :, self.block], self.k_offset[:, :, self.block], self.bits)
        v_scale = self.v_scale.repeat_interleave(QUANT_BLOCK, dim=3)[..., :dim]
        v_offset = self.v_offset.repeat_interleave(QUANT_BLOCK, dim=3)[..., :dim]
        return keys, dequantize(self.v_codes, v_scale, v_offset, self.bits)

    def nbytes(self) -> int:
        """Codes plus scales and offsets."""
        if self.rows == 0:
            return 0
        codes = self.k_codes.numel() + self.v_codes.numel()
        params = self.k_scale.numel() + self.k_offset.numel() + self.v_scale.numel() + self.v_offset.numel()
        return codes + params * self.k_scale.element_size()

    def consistent(self) -> bool:
        """Rows aligned across the arrays, and every K block referred to by a row (no stale metadata)."""
        if self.rows == 0:
            return self.k_scale is None or self.k_scale.shape[2] == 0
        lengths = {self.k_codes.shape[2], self.v_codes.shape[2], self.v_scale.shape[2], self.v_offset.shape[2],
                   self.block.numel(), self.frame_id.numel(), self.token_id.numel()}
        blocks = self.k_scale.shape[2]
        return lengths == {self.rows} and torch.unique(self.block).numel() == blocks == int(self.block.max()) + 1


class LayerKVCache:
    """Keys and values of one layer, stored in ``dtype`` in a preallocated buffer that doubles when it is full.

    ``tokens_per_frame`` (n) tokens take part in the layer per frame, the first ``special_count`` (m) of them
    special; register blocks and the camera trunk have n == m (no patch tokens). ``layer_id`` keys the random
    selector. A bounded policy needs one stream per cache (batch size 1).
    """

    def __init__(self, policy: CachePolicy, tokens_per_frame: int, special_count: int, dtype: torch.dtype,
                 layer_id: int, initial_frames: int = 16):
        if not 1 <= special_count <= tokens_per_frame:
            raise ValueError(f"need 1 <= special_count <= tokens_per_frame, got {special_count}, {tokens_per_frame}")
        if initial_frames < 1:
            raise ValueError(f"initial_frames must be positive, got {initial_frames}")
        self.policy = policy
        self.tokens_per_frame = tokens_per_frame
        self.special_count = special_count
        self.dtype = dtype
        self.layer_id = layer_id
        self.budget = policy.budget(tokens_per_frame, special_count)
        self.last_frame = 0  # the last committed frame id
        has_patches = tokens_per_frame > special_count
        # the long-patch store of a layer with patches, quantised (layers without patches have none)
        self._quant = _QuantizedRows(QUANT_BITS[policy.quant]) if policy.quant and has_patches else None
        # the buffer holds the rows in the cache dtype: a bounded cache holds at most its budget (less a quantised
        # long-patch store) plus the current frame, so its buffer never grows
        if policy.is_full:
            self._initial_rows = initial_frames * tokens_per_frame
        else:
            quantized = policy.long_patch_capacity(tokens_per_frame - special_count) if self._quant else 0
            self._initial_rows = self.budget + tokens_per_frame - quantized
        self._pending = None  # frame id appended but not committed yet
        self._keys = self._values = None  # [B, H, capacity, D], allocated on the first append
        self._frame_id = self._token_id = None  # [capacity] int64, row metadata
        self._rows = 0  # rows in the buffer

    @property
    def capacity(self) -> int:
        """Rows the buffer (cache dtype) holds before it grows."""
        return 0 if self._keys is None else self._keys.shape[2]

    @property
    def size(self) -> int:
        """Number of stored rows (tokens), the current frame included while it is not committed."""
        return self._rows + self._quantized_rows

    @property
    def _quantized_rows(self) -> int:
        return 0 if self._quant is None else self._quant.rows

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
        if not self.policy.is_full and k.shape[0] != 1:
            raise NotImplementedError(f"a bounded cache selects for one stream (batch size 1), got {k.shape[0]}")
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
        """Keys and values of every stored row, in the order of ``row_ids``: views of the buffer, followed by the
        dequantised long-patch store if there is one."""
        if self._keys is None:
            raise RuntimeError("the cache is empty: append a frame first")
        keys, values = self._keys[:, :, : self._rows], self._values[:, :, : self._rows]
        if not self._quantized_rows:
            return keys, values
        quantized_keys, quantized_values = self._quant.dequantized()
        return torch.cat([keys, quantized_keys], dim=2), torch.cat([values, quantized_values], dim=2)

    def row_ids(self) -> Tuple[Tensor, Tensor]:
        """Frame id and token id of every row of ``read()``."""
        frame_id, token_id = self._frame_id[: self._rows], self._token_id[: self._rows]
        if not self._quantized_rows:
            return frame_id, token_id
        return torch.cat([frame_id, self._quant.frame_id]), torch.cat([token_id, self._quant.token_id])

    def commit(self, q: Tensor, t: int, grid_hw: Optional[Tuple[int, int]] = None) -> None:
        """End step ``t``: keep what the policy allows for the next step.

        ``q`` ([B, H, n, D]) are the current frame's queries and ``grid_hw`` its patch grid (rows, columns); the
        query-guided selectors read them.
        """
        if self._pending is None:
            raise RuntimeError("nothing to commit: append the current frame first")
        if t != self._pending:
            raise ValueError(f"commit of frame {t}, but frame {self._pending} was appended")
        patches = self.tokens_per_frame - self.special_count
        if grid_hw is not None and grid_hw[0] * grid_hw[1] != patches:
            raise ValueError(f"grid {grid_hw} does not hold the {patches} patch tokens of a frame")
        if not self.policy.is_full:
            self._keep(*self._select(q, t, grid_hw))
        self._pending = None
        self.last_frame = t

    def invariants(self) -> dict:
        """Checks of the stored rows between steps; ``ok`` is their conjunction."""
        if self._pending is not None:
            raise RuntimeError("the invariants hold between steps: commit the current frame first")
        if self._keys is None:
            return {"size": 0, "budget": self.budget, "ok": True}
        n, m, t = self.tokens_per_frame, self.special_count, self.last_frame
        frame_id, token_id = self.row_ids()
        keys, values = self.read()
        protected_frames = self.policy.protected_frames(t)
        protected = torch.isin(frame_id, torch.tensor(protected_frames, device=frame_id.device))
        special = token_id < m
        long_special_frames, special_counts = torch.unique(frame_id[~protected & special], return_counts=True)
        long_patch = ~protected & ~special
        long_patch_tokens = int(long_patch.sum())
        unique = torch.unique(frame_id * n + token_id).numel() == self.size
        checks = {
            "aligned": keys.shape[2] == values.shape[2] == frame_id.numel() == token_id.numel() == self.size,
            "unique": unique,
            "within_budget": self.budget is None or self.size <= self.budget,
            # the rows are unique, so n rows per protected frame means that every token of it is there
            "protected_complete": unique and int(protected.sum()) == n * len(protected_frames),
            "long_special": long_special_frames.numel() <= self.policy.long_special
            and bool((special_counts == m).all()),
            "long_patch": long_patch_tokens <= self.policy.long_patch_capacity(n - m),
        }
        if self.policy.long_frames:  # whole frames: every patch of at most k_P frames
            long_patch_frames, patch_counts = torch.unique(frame_id[long_patch], return_counts=True)
            checks["long_frames"] = (long_patch_frames.numel() <= self.policy.long_frames
                                     and bool((patch_counts == n - m).all()))
        if self._quant is not None:  # the long-patch rows are exactly the quantised ones, and their blocks live
            quantized = torch.zeros_like(long_patch)
            quantized[self._rows:] = True  # row_ids lists the quantised store after the buffer
            checks["long_patch_quantized"] = torch.equal(long_patch, quantized)
            checks["quantized_blocks"] = self._quant.consistent()
        return {"size": self.size, "budget": self.budget, "long_special_frames": long_special_frames.numel(),
                "long_patch_tokens": long_patch_tokens, **checks, "ok": all(checks.values())}

    def nbytes(self) -> int:
        """Bytes of the stored keys and values (the stored rows, not the buffer capacity): the payload in the cache
        dtype, plus the codes, scales and offsets of a quantised long-patch store."""
        if self._keys is None:
            return 0
        batch, heads, _, dim = self._keys.shape
        full_precision = 2 * batch * heads * self._rows * dim * self._keys.element_size()
        return full_precision + (0 if self._quant is None else self._quant.nbytes())

    def _select(self, q: Tensor, t: int, grid_hw: Optional[Tuple[int, int]]) -> Tuple[Tensor, Tensor]:
        """Rows (bool masks over ``row_ids``) kept after the commit of step ``t``, and the long-patch candidates."""
        policy = self.policy
        frame_id, token_id = self.row_ids()
        protected = torch.isin(frame_id, torch.tensor(policy.protected_frames(t), device=frame_id.device))
        special = token_id < self.special_count
        long_special, long_patch = ~protected & special, ~protected & ~special
        select_special = torch.unique(frame_id[long_special]).numel() > policy.long_special
        if policy.long_frames:
            select_patch = torch.unique(frame_id[long_patch]).numel() > policy.long_frames
        else:
            select_patch = int(long_patch.sum()) > policy.long_patch
        keep = protected.clone()
        scores = {}  # per selector, computed at most once per commit, and only when a store must select

        def scored(selector: str) -> Tensor:
            if selector not in scores:
                scores[selector] = self._scores(selector, q, t, grid_hw, long_patch)
            return scores[selector]

        # a store keeps every candidate while they fit; a store of size 0 keeps none (nothing to score)
        if not select_special:
            keep |= long_special
        elif policy.long_special > 0:  # diversity ranks the patches only: the special tokens keep the query selector
            special_selector = "query" if policy.selector == "diversity" else policy.selector
            kept_frames = top_frames(scored(special_selector), frame_id, long_special, policy.long_special)
            keep |= long_special & torch.isin(frame_id, kept_frames)
        if not select_patch:
            keep |= long_patch
        elif policy.long_frames > 0:
            kept_frames = top_frames(scored(policy.selector), frame_id, long_patch, policy.long_frames)
            keep |= long_patch & torch.isin(frame_id, kept_frames)
        elif policy.long_patch > 0:
            keep[top_rows(scored(policy.selector), frame_id, token_id, long_patch, policy.long_patch)] = True
        return keep, long_patch

    def _keep(self, keep: Tensor, long_patch: Tensor) -> None:
        """Apply the selection; kept long-patch rows still in the buffer enter the quantised store (once)."""
        if self._quant is None:
            self._compact(keep)
            return
        keep_buffer = keep[: self._rows].clone()
        self._quant.keep(keep[self._rows:])
        entering = (keep_buffer & long_patch[: self._rows]).nonzero().squeeze(1)
        entering_frames = self._frame_id[entering]
        for frame in torch.unique(entering_frames).tolist():  # blocks of 64 tokens never span two frames
            rows = entering[entering_frames == frame]
            self._quant.add(self._keys[:, :, rows], self._values[:, :, rows], self._frame_id[rows],
                            self._token_id[rows])
        keep_buffer[entering] = False
        self._compact(keep_buffer)

    def _scores(self, selector: str, q: Tensor, t: int, grid_hw: Optional[Tuple[int, int]], pool: Tensor) -> Tensor:
        frame_id, token_id = self.row_ids()
        candidates = Candidates(q, self.read()[0], frame_id, token_id, self.special_count, grid_hw, self.layer_id, t,
                                pool)
        scores = SELECTORS[selector](candidates)
        if not torch.isfinite(scores).all():
            raise ValueError(f"selector {selector!r} gave non-finite scores at layer {self.layer_id}")
        return scores

    def _compact(self, keep: Tensor) -> None:
        """Keep the rows of ``keep`` (bool mask), in their order, at the front of the buffer."""
        rows = keep.nonzero().squeeze(1)
        if rows.numel() == self._rows:
            return
        kept = rows.numel()
        self._keys[:, :, :kept] = self._keys[:, :, rows]
        self._values[:, :, :kept] = self._values[:, :, rows]
        self._frame_id[:kept] = self._frame_id[rows]
        self._token_id[:kept] = self._token_id[rows]
        self._rows = kept

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
