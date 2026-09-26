"""Frame retrieval of the streaming model (RetrieveVGGT-style, reimplemented from the paper's description).

With ``CachePolicy(recent=None, retrieve_frames=N)`` every cache of ``StreamingOmega`` stores the full history, and
stream step t attends to the anchor (frame 1), N-1 past frames and the current frame only. ``FrameRetriever``
chooses the frames once per step, in the first global layer (it holds the patch tokens), and every global and
register layer (and, with ``retrieve_camera``, the camera trunk) reads the same frames:

* relevance r(t, i) = (1/H) sum_h <qbar_t[h], kbar_i[h]> of every past frame i, with qbar_t the mean of the current
  frame's queries and kbar_i the mean of frame i's keys over the patch tokens (special tokens excluded), both as
  ``Attention.qkv_heads`` returns them (after q/k norm; the Omega global blocks have no RoPE). kbar_i is taken
  when frame i is processed and kept in fp32 (or wider);
* the N-1 frames among the candidates 2..t-1 come from ``segment_sampling`` of their relevance.

While t-1 <= N every past frame is read, so those steps equal the full cache.
"""

import math
from typing import List, Optional, Sequence

import torch
from torch import Tensor

SEGMENT_THRESHOLD_STD = 0.3  # tau = mu + 0.3 sigma
SEGMENT_MERGE_GAP = 3  # runs of frames >= tau at most 3 frames apart form one segment
SEGMENT_CAP = 0.35  # a segment's quota is at most int(0.35 count)


def patch_mean(x: Tensor, special_count: int) -> Tensor:
    """Mean of ``x`` [B, H, n, D] over its patch tokens (the tokens after the ``special_count`` special ones):
    [B, H, D], in fp32 or wider."""
    if x.shape[2] <= special_count:
        raise ValueError(f"{x.shape[2]} tokens hold no patch token after {special_count} special tokens")
    return x[:, :, special_count:].to(torch.promote_types(x.dtype, torch.float32)).mean(dim=2)


def frame_scores(queries: Tensor, descriptors: Tensor, special_count: int) -> Tensor:
    """r_i = (1/H) sum_h <qbar[h], descriptors[i, h]> ([F]) of the current queries [1, H, n, D] and the frame
    descriptors (kbar) [F, H, D]."""
    q_bar = patch_mean(queries, special_count)[0]  # [H, D]
    return (descriptors.to(q_bar.dtype) * q_bar).sum(dim=-1).mean(dim=-1)


def _best(values: Sequence[float], indices, count: int) -> List[int]:
    """The ``count`` best ``indices`` by value, ties to the later index."""
    return sorted(indices, key=lambda index: (-values[index], -index))[:count]


def _sample_segment(values: Sequence[float], start: int, end: int, quota: int) -> List[int]:
    """``quota`` indices of the segment [start, end]: its peak (ties: the later index), then for a quota of 2 the
    end farther from the peak (ties: the later end), for a quota q >= 3 the q-1 indices of the rest of the segment
    at positions round(linspace(0, L-1, q-1)) (both ends of the rest included)."""
    peak = max(range(start, end + 1), key=lambda index: (values[index], index))
    if quota == 1:
        return [peak]
    if quota == 2:
        return [peak, start if peak - start > end - peak else end]
    rest = [index for index in range(start, end + 1) if index != peak]
    positions = torch.linspace(0, len(rest) - 1, quota - 1).round().long().tolist()
    return [peak, *(rest[position] for position in positions)]


def segment_sampling(scores: Tensor, count: int) -> Tensor:
    """Indices (ascending, int64) of ``count`` of the candidate ``scores`` ([C], in time order), by RetrieveVGGT's
    Segment Sampling:

    1. tau = mu + 0.3 sigma of the scores (population sigma). A segment is a maximal run of indices scoring
       >= tau; runs at most 3 indices apart merge (the indices between them join the segment).
    2. The quota of segment k is floor(count * peak_k / sum_j peak_j), at most int(0.35 count) and its size, and at
       least 1.
    3. Each segment gives its quota of indices (``_sample_segment``).
    4. Too many: the best of them by score; too few: the best unselected indices fill up (ties: the later index).

    Without a segment, or when the segment peaks do not sum to a positive total (no quota is defined), the best
    ``count`` indices. With ``count`` >= C, every index.
    """
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    total = scores.numel()
    if count >= total:
        return torch.arange(total)
    if count == 0:
        return torch.zeros(0, dtype=torch.long)
    values = scores.tolist()
    tau = float(scores.mean() + SEGMENT_THRESHOLD_STD * scores.std(correction=0))
    segments = []  # [start, end], inclusive
    for index, value in enumerate(values):
        if value < tau:
            continue
        if segments and index - segments[-1][1] - 1 <= SEGMENT_MERGE_GAP:
            segments[-1][1] = index
        else:
            segments.append([index, index])
    peaks = [max(values[start:end + 1]) for start, end in segments]
    if not segments or sum(peaks) <= 0:
        return torch.tensor(sorted(_best(values, range(total), count)), dtype=torch.long)
    cap = int(SEGMENT_CAP * count)
    chosen = set()
    for (start, end), peak in zip(segments, peaks, strict=True):
        quota = max(1, min(math.floor(count * peak / sum(peaks)), cap, end - start + 1))
        chosen.update(_sample_segment(values, start, end, quota))
    if len(chosen) > count:
        chosen = set(_best(values, chosen, count))
    else:
        chosen.update(_best(values, [index for index in range(total) if index not in chosen], count - len(chosen)))
    return torch.tensor(sorted(chosen), dtype=torch.long)


class FrameRetriever:
    """The frames of a stream step that the retrieving layers read (see the module docstring).

    ``retrieve_frames`` (N) past frames are read per step: the anchor and N-1 selected ones. ``special_count`` is
    the number of special tokens of a frame in the first global layer.
    """

    def __init__(self, retrieve_frames: int, special_count: int):
        if isinstance(retrieve_frames, bool) or not isinstance(retrieve_frames, int) or retrieve_frames < 1:
            raise ValueError(f"retrieve_frames must be an int >= 1, got {retrieve_frames!r}")
        self.retrieve_frames = retrieve_frames
        self.special_count = special_count
        self.descriptors: List[Tensor] = []  # kbar of frames 1..t, [H, D] each
        self.scores: Optional[Tensor] = None  # r(t, i) of the past frames 1..t-1 at the last step; None at t=1
        self.frames: Optional[Tensor] = None  # frame ids read at the last step: the anchor, the selected, the current

    def select(self, q: Tensor, k: Tensor, t: int) -> Tensor:
        """Frame ids (ascending) read at step ``t``, from the current frame's queries and keys [1, H, n, D] in the
        first global layer; keeps the frame's kbar for the later steps."""
        if q.shape[0] != 1:
            raise NotImplementedError(f"retrieval selects the frames of one stream (batch size 1), got {q.shape[0]}")
        if t != len(self.descriptors) + 1:
            raise ValueError(f"frames are selected in order from 1: expected frame {len(self.descriptors) + 1}, "
                             f"got {t}")
        self.descriptors.append(patch_mean(k, self.special_count)[0])
        self.scores = None if t == 1 else frame_scores(q, torch.stack(self.descriptors[:-1]), self.special_count)
        if t - 1 <= self.retrieve_frames:  # every past frame fits
            self.frames = torch.arange(1, t + 1, device=q.device)
        else:  # candidates 2..t-1: the anchor is always read
            chosen = segment_sampling(self.scores[1:], self.retrieve_frames - 1).to(q.device) + 2
            anchor, current = torch.tensor([1], device=q.device), torch.tensor([t], device=q.device)
            self.frames = torch.cat([anchor, chosen, current])
        return self.frames
