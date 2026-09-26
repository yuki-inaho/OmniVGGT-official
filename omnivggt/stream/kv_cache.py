"""Per-layer KV cache of the streaming (frame-causal) OmniVGGTOmega.

One ``LayerKVCache`` holds the keys and values (after q/k norm, as ``Attention.qkv_heads`` returns them) that one
inter-frame block, register block or camera-trunk block has seen. A stream step appends the current frame, reads
the stored past plus the current frame for the attention, then commits: the cache keeps what its ``CachePolicy``
allows for the next step. Frame ids start at 1 and are consecutive; token ids number the tokens of a frame, the
``special_count`` special (camera and register) tokens first, then the patches in raster order.

A bounded cache is partitioned (design doc §4; no row is in two parts) into

* anchor: every token of frame 1, always kept;
* recent: every token of the ``recent`` newest frames (the current one included, frame 1 excluded);
* long-special: the special tokens of at most ``long_special`` other frames, selected per frame;
* long-patch: at most ``long_patch`` other patch tokens, selected per token.

The long-term stores are selected at the commit, from every candidate of the written cache, by the policy's
selector; ties go to the newer frame, then to the smaller token id.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

SELECTOR_NAMES = ("query", "xstream", "recency", "random")
QUANT_NAMES = ("int8", "int4")
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
    * ``selector``: one of ``SELECTOR_NAMES``; required exactly when there is a long-term store.
    * ``quant``: ``"int8"`` or ``"int4"`` storage of the long-patch store.
    """

    recent: Optional[int]
    long_special: int = 0
    long_patch: int = 0
    selector: Optional[str] = None
    quant: Optional[str] = None

    def __post_init__(self):
        if self.recent is None:
            if (self.long_special, self.long_patch, self.selector, self.quant) != (0, 0, None, None):
                raise ValueError("the full cache (recent=None) keeps every frame: it takes no long-term store, "
                                 "selector or quantisation")
            return
        if not _is_int(self.recent) or self.recent < 1:
            raise ValueError(f"recent must be an int >= 1 (it includes the current frame), got {self.recent!r}")
        for name in ("long_special", "long_patch"):
            value = getattr(self, name)
            if not _is_int(value) or value < 0:
                raise ValueError(f"{name} must be an int >= 0, got {value!r}")
        has_long_term = self.long_special > 0 or self.long_patch > 0
        if has_long_term and self.selector not in SELECTOR_NAMES:
            raise ValueError(f"a long-term store needs a selector from {SELECTOR_NAMES}, got {self.selector!r}")
        if not has_long_term and self.selector is not None:
            raise ValueError(f"selector {self.selector!r} given, but there is no long-term store to select")
        if self.quant is not None and self.quant not in QUANT_NAMES:
            raise ValueError(f"quant must be None or one of {QUANT_NAMES}, got {self.quant!r}")
        if self.quant is not None and self.long_patch == 0:
            raise ValueError("quant applies to the long-patch store only, but long_patch=0")

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
        """Most tokens a cache keeps between steps: (1+w)n + m k_U + b_P, or m(1+w+k_U) without patches
        (register layers: n = m; camera trunk: n = m = 1). None for the full cache."""
        if self.is_full:
            return None
        patches = self.long_patch if tokens_per_frame > special_count else 0
        return (1 + self.recent) * tokens_per_frame + special_count * self.long_special + patches


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


SELECTORS = {"query": select_query, "xstream": select_xstream, "recency": select_recency, "random": select_random}


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
        if policy.quant is not None:
            raise NotImplementedError("quantisation is not implemented")
        self.policy = policy
        self.tokens_per_frame = tokens_per_frame
        self.special_count = special_count
        self.dtype = dtype
        self.layer_id = layer_id
        self.budget = policy.budget(tokens_per_frame, special_count)
        self.last_frame = 0  # the last committed frame id
        # a bounded cache holds at most its budget plus the current frame, so its buffer never grows
        self._initial_rows = initial_frames * tokens_per_frame if policy.is_full else self.budget + tokens_per_frame
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
        patches = self.tokens_per_frame - self.special_count
        if grid_hw is not None and grid_hw[0] * grid_hw[1] != patches:
            raise ValueError(f"grid {grid_hw} does not hold the {patches} patch tokens of a frame")
        if not self.policy.is_full:
            self._compact(self._select(q, t, grid_hw))
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
        recent_from = 1 if self.policy.is_full else max(2, t - self.policy.recent + 1)
        protected_frames = [frame for frame in range(1, t + 1) if frame == 1 or frame >= recent_from]
        protected = torch.isin(frame_id, torch.tensor(protected_frames, device=frame_id.device))
        special = token_id < m
        long_special_frames, special_counts = torch.unique(frame_id[~protected & special], return_counts=True)
        long_patch_tokens = int((~protected & ~special).sum())
        unique = torch.unique(frame_id * n + token_id).numel() == self.size
        checks = {
            "aligned": keys.shape[2] == values.shape[2] == frame_id.numel() == token_id.numel() == self.size,
            "unique": unique,
            "within_budget": self.budget is None or self.size <= self.budget,
            # the rows are unique, so n rows per protected frame means that every token of it is there
            "protected_complete": unique and int(protected.sum()) == n * len(protected_frames),
            "long_special": long_special_frames.numel() <= self.policy.long_special
            and bool((special_counts == m).all()),
            "long_patch": long_patch_tokens <= (self.policy.long_patch if n > m else 0),
        }
        return {"size": self.size, "budget": self.budget, "long_special_frames": long_special_frames.numel(),
                "long_patch_tokens": long_patch_tokens, **checks, "ok": all(checks.values())}

    def nbytes(self) -> int:
        """Bytes of the stored keys and values (the stored rows, not the buffer capacity)."""
        if self._keys is None:
            return 0
        batch, heads, _, dim = self._keys.shape
        return 2 * batch * heads * self._rows * dim * self._keys.element_size()

    def _select(self, q: Tensor, t: int, grid_hw: Optional[Tuple[int, int]]) -> Tensor:
        """Rows (bool mask) kept after the commit of step ``t``."""
        policy = self.policy
        frame_id, token_id = self.row_ids()
        protected = (frame_id == 1) | (frame_id > t - policy.recent)
        special = token_id < self.special_count
        long_special, long_patch = ~protected & special, ~protected & ~special
        select_special = torch.unique(frame_id[long_special]).numel() > policy.long_special
        select_patch = int(long_patch.sum()) > policy.long_patch
        keep = protected.clone()
        if not (select_special or select_patch):  # fewer candidates than the budget: keep them all
            return keep | long_special | long_patch
        scores = self._scores(q, t, grid_hw)
        if select_special:
            kept_frames = top_frames(scores, frame_id, long_special, policy.long_special)
            keep |= long_special & torch.isin(frame_id, kept_frames)
        else:
            keep |= long_special
        if select_patch:
            keep[top_rows(scores, frame_id, token_id, long_patch, policy.long_patch)] = True
        else:
            keep |= long_patch
        return keep

    def _scores(self, q: Tensor, t: int, grid_hw: Optional[Tuple[int, int]]) -> Tensor:
        frame_id, token_id = self.row_ids()
        candidates = Candidates(q, self.read()[0], frame_id, token_id, self.special_count, grid_hw, self.layer_id, t)
        scores = SELECTORS[self.policy.selector](candidates)
        if not torch.isfinite(scores).all():
            raise ValueError(f"selector {self.policy.selector!r} gave non-finite scores at layer {self.layer_id}")
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
