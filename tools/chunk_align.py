"""Chunked offline inference in the style of VGGT-Long (arXiv 2507.16443), written from the paper's description.

A window of S frames is cut into chunks of ``chunk`` frames that start every ``chunk - overlap`` frames; the last
chunk is moved back to end at the window end (its overlap may then be larger). Each chunk is predicted on its own;
chunk j is mapped onto chunk j-1, which is already in the frame of chunk 0, by a similarity (Sim(3)) fitted to the
predicted world points of their overlap frames:

* correspondences: the same pixel of an overlap frame in both chunks, every ``PIXEL_STEP``-th row and column,
  inside the valid mask; weight = the product of the two chunks' depth confidences; weights below
  ``MIN_WEIGHT_RATIO`` x their median are dropped;
* fit: weighted Umeyama, then ``IRLS_ITERATIONS`` Huber reweightings (weight = confidence x psi(r)/r) with delta
  fixed at the median residual of the first fit;
* the similarity moves the chunk's cameras (rotation and centre), its depth (x scale) and its world points.

Frame t is taken from the chunk containing it whose centre is closest to t (ties: the later chunk). There is no
loop closure and no global optimisation.
"""

from __future__ import annotations

import numpy as np
import torch

from omnivggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri

PIXEL_STEP = 4
MIN_WEIGHT_RATIO = 0.1
IRLS_ITERATIONS = 5
ALIGNMENT = {
    "pixel_step": PIXEL_STEP,
    "min_weight_ratio": MIN_WEIGHT_RATIO,
    "irls_iterations": IRLS_ITERATIONS,
    "huber_delta": "median_initial_residual",
}
COLLINEAR_RATIO = 1e-9  # second / first singular value of the weighted covariance below which a fit is refused
SPREAD_RATIO = 1e-24  # weighted variance / weighted mean squared norm of the source points below which it is refused


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def check_chunking(chunk, overlap) -> None:
    """``chunk`` an int >= 2 and ``overlap`` an int in [1, chunk - 1]."""
    if not _is_int(chunk) or chunk < 2:
        raise ValueError(f"chunk must be an int >= 2, got {chunk!r}")
    if not _is_int(overlap) or not 1 <= overlap < chunk:
        raise ValueError(f"overlap must be an int in [1, chunk - 1] = [1, {chunk - 1}], got {overlap!r}")


def chunk_starts(frames: int, chunk: int, overlap: int) -> list[int]:
    """First frame of every chunk: 0, s, 2s, ... (s = chunk - overlap), and ``frames - chunk`` when the last of
    these ends before the window end."""
    check_chunking(chunk, overlap)
    if frames < chunk:
        raise ValueError(f"a window of {frames} frames is shorter than one chunk of {chunk} frames")
    starts = list(range(0, frames - chunk + 1, chunk - overlap))
    if starts[-1] + chunk < frames:
        starts.append(frames - chunk)
    return starts


def chunk_owners(starts: list[int], chunk: int, frames: int) -> list[int]:
    """Index of the chunk each frame is taken from: among the chunks containing the frame, the one whose centre
    (start + (chunk - 1) / 2) is closest; on a tie the later chunk."""
    owners = []
    for frame in range(frames):
        containing = [index for index, start in enumerate(starts) if start <= frame < start + chunk]
        if not containing:
            raise ValueError(f"frame {frame} lies in no chunk of {starts} (chunk {chunk})")
        # twice the distance to the centre, in integers
        owners.append(min(containing, key=lambda index: (abs(2 * (frame - starts[index]) - (chunk - 1)), -index)))
    return owners


def weighted_umeyama(source: np.ndarray, target: np.ndarray, weights: np.ndarray):
    """(scale, rotation, translation) minimising sum_i w_i ||target_i - (scale R source_i + t)||^2 (Umeyama 1991
    with weights)."""
    source, target, weights = (np.asarray(x, dtype=np.float64) for x in (source, target, weights))
    if source.ndim != 2 or source.shape[1] != 3 or target.shape != source.shape or weights.shape != source.shape[:1]:
        raise ValueError(f"expected (N, 3) points and (N,) weights, got {source.shape}, {target.shape}, "
                         f"{weights.shape}")
    if len(source) < 3:
        raise ValueError(f"a similarity needs at least 3 points, got {len(source)}")
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("weights must be finite, non-negative and not all zero")
    weights = weights / weights.sum()
    mean_source, mean_target = weights @ source, weights @ target
    centred_source, centred_target = source - mean_source, target - mean_target
    variance = float(weights @ np.sum(centred_source**2, axis=1))
    if variance <= SPREAD_RATIO * float(weights @ np.sum(source**2, axis=1)):
        raise ValueError("the weighted source points have no spread")
    u, singular, vt = np.linalg.svd((centred_target * weights[:, None]).T @ centred_source)
    if singular[1] <= COLLINEAR_RATIO * singular[0]:
        raise ValueError("the weighted points are collinear: the rotation about their line is undetermined")
    sign = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2] = -1.0
    rotation = (u * sign) @ vt
    scale = float(singular @ sign / variance)
    return scale, rotation, mean_target - scale * rotation @ mean_source


def huber_weights(residuals: np.ndarray, delta: float) -> np.ndarray:
    """psi(r) / r of the Huber loss: 1 for r <= delta, delta / r above."""
    residuals = np.asarray(residuals, dtype=np.float64)
    return np.where(residuals <= delta, 1.0, delta / np.maximum(residuals, np.finfo(np.float64).tiny))


def _residuals(source, target, scale, rotation, translation) -> np.ndarray:
    return np.linalg.norm(scale * source @ rotation.T + translation - target, axis=1)


def irls_sim3(source: np.ndarray, target: np.ndarray, weights: np.ndarray, iterations: int = IRLS_ITERATIONS):
    """(scale, rotation, translation) with target ~ scale R source + t: a weighted Umeyama fit, then ``iterations``
    refits with the weights ``weights x huber_weights(r, delta)``, r the residuals of the previous fit and delta
    the median residual of the first fit."""
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    estimate = weighted_umeyama(source, target, weights)
    residuals = _residuals(source, target, *estimate)
    delta = float(np.median(residuals))
    for _ in range(iterations):
        estimate = weighted_umeyama(source, target, weights * huber_weights(residuals, delta))
        residuals = _residuals(source, target, *estimate)
    return estimate


def overlap_correspondences(target_points, target_conf, source_points, source_conf, valid,
                            step: int = PIXEL_STEP, min_weight_ratio: float = MIN_WEIGHT_RATIO):
    """(source, target, weights) of the overlap frames: every ``step``-th pixel (rows and columns) inside ``valid``,
    weight = target_conf x source_conf, weights below ``min_weight_ratio`` x their median dropped.

    ``*_points`` are (F, H, W, 3) world points of the same F frames in the two chunks, ``*_conf`` and ``valid``
    (F, H, W)."""
    target_points, source_points = np.asarray(target_points), np.asarray(source_points)
    target_conf, source_conf, valid = np.asarray(target_conf), np.asarray(source_conf), np.asarray(valid, bool)
    grid = valid.shape
    if (target_points.shape != source_points.shape or target_points.shape != (*grid, 3)
            or target_conf.shape != grid or source_conf.shape != grid or len(grid) != 3):
        raise ValueError(f"overlap shapes disagree: points {target_points.shape} / {source_points.shape}, "
                         f"conf {target_conf.shape} / {source_conf.shape}, valid {valid.shape}")
    sample = (slice(None), slice(None, None, step), slice(None, None, step))
    keep = valid[sample]
    if not keep.any():
        raise ValueError("no valid overlap pixel on the correspondence grid")
    weights = (np.asarray(target_conf[sample], np.float64) * np.asarray(source_conf[sample], np.float64))[keep]
    heavy = weights >= min_weight_ratio * np.median(weights)
    source = np.asarray(source_points[sample], np.float64)[keep][heavy]
    target = np.asarray(target_points[sample], np.float64)[keep][heavy]
    return source, target, weights[heavy]


def align_overlap(target_points, target_conf, source_points, source_conf, valid):
    """(scale, rotation, translation, correspondences): the IRLS similarity mapping the source chunk's overlap
    points onto the target chunk's (``overlap_correspondences``)."""
    source, target, weights = overlap_correspondences(target_points, target_conf, source_points, source_conf, valid)
    scale, rotation, translation = irls_sim3(source, target, weights)
    return scale, rotation, translation, len(weights)


def transform_prediction(prediction: dict, scale: float, rotation: np.ndarray, translation: np.ndarray,
                         image_size_hw) -> dict:
    """``prediction`` (pose_enc (B, S, 9), depth (B, S, H, W, 1), depth_conf, world_points (B, S, H, W, 3)) moved by
    x -> scale R x + t: camera rotations R_c2w -> R R_c2w, centres c -> scale R c + t, depth x scale, world points
    in float64. The pose encoding is rebuilt with the prediction's own intrinsics."""
    w2c, intrinsics = pose_encoding_to_extri_intri(prediction["pose_enc"].double(), image_size_hw)
    rotation_t = torch.as_tensor(rotation, dtype=torch.float64)
    translation_t = torch.as_tensor(translation, dtype=torch.float64)
    world_to_camera = w2c[..., :3, :3]
    centres = -torch.einsum("...ji,...j->...i", world_to_camera, w2c[..., :3, 3])
    moved_rotation = world_to_camera @ rotation_t.T
    moved_centres = scale * centres @ rotation_t.T + translation_t
    moved_w2c = torch.cat([moved_rotation, -torch.einsum("...ij,...j->...i", moved_rotation, moved_centres)[..., None]],
                          dim=-1)
    points = prediction["world_points"].double()
    return {
        "pose_enc": extri_intri_to_pose_encoding(moved_w2c, intrinsics.double(), image_size_hw),
        "depth": prediction["depth"] * scale,
        "depth_conf": prediction["depth_conf"],
        "world_points": scale * points @ rotation_t.T + translation_t,
    }
