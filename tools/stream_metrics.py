"""Trajectory and depth metrics for a window of streamed predictions (first-frame gauge).

Poses are camera_to_world 4x4 unless named ``*_w2c``; ground truth is in metres and every
trajectory metric is reported in millimetres. The prediction lives in an arbitrary gauge: the
metrics fix it from quantities known at the window's first frame (frame 1):

* ``s1`` = median(D_gt / D_pred) over the valid pixels of frame 1 (the only use of GT depth scale);
* ATE_g1: frame-1 poses made equal by one SE(3), prediction translations scaled by ``s1``;
* ATE_sim3: one orientation-aware Sim(3) over all frames (``align_pose_sim3``, no trimming);
* RPE_t: translation error of the Delta-frame relative motion, prediction scaled by ``s1``, for
  Delta in {1, 8, 16} and 32 only for windows of at least 64 frames (undefined Delta are omitted);
* pose scale drift: ``s1``-scaled predicted displacement over 8-frame steps / GT displacement;
* depth scale drift: s_t / s_1 with s_t = median(D_gt / D_pred) of frame t;
* depth AbsRel and delta<1.25 under one median scale per window (``eval_colmap_rgbd.depth_metrics``)
  and under ``s1``, also per time bin t in 1-8, 9-32, 33-64, 65+.
"""

from __future__ import annotations

import numpy as np
from eval_colmap_rgbd import pose_metrics
from rgbd_pose_pipeline.colmap_prior_ba import align_pose_sim3
from rgbd_pose_pipeline.se3 import as_homogeneous, invert

RPE_DELTAS = (1, 8, 16)
LONG_RPE_DELTA, LONG_RPE_MIN_FRAMES = 32, 64
DRIFT_DELTA = 8
T_BINS = ((1, 8), (9, 32), (33, 64), (65, None))
MM = 1e3


def depth_scale(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """median(gt / pred) over the valid pixels of one frame."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        raise ValueError("no valid depth pixel to estimate the scale from")
    return float(np.median(np.asarray(gt, float)[mask] / np.asarray(pred, float)[mask]))


def depth_scales(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-frame depth scale s_t of a (T, H, W) window."""
    return np.array([depth_scale(p, g, m) for p, g, m in zip(pred, gt, mask, strict=True)])


def ate_first_frame(pred_c2w: np.ndarray, gt_c2w: np.ndarray, scale: float) -> np.ndarray:
    """Per-frame centre error [m] after mapping predicted frame 1 onto GT frame 1 (SE(3)) with the
    prediction's translations scaled by ``scale``."""
    pred, gt = as_homogeneous(pred_c2w), as_homogeneous(gt_c2w)
    relative = invert(pred[0]) @ pred
    relative[:, :3, 3] *= scale
    placed = gt[0] @ relative
    return np.linalg.norm(placed[:, :3, 3] - gt[:, :3, 3], axis=1)


def ate_sim3(pred_c2w: np.ndarray, gt_c2w: np.ndarray) -> np.ndarray:
    """Per-frame centre error [m] after one orientation-aware Sim(3) fitted to every frame."""
    _, _, _, aligned = align_pose_sim3(as_homogeneous(pred_c2w), as_homogeneous(gt_c2w), trim_fraction=0.0)
    return np.linalg.norm(aligned[:, :3, 3] - as_homogeneous(gt_c2w)[:, :3, 3], axis=1)


def rpe_deltas(frames: int) -> tuple[int, ...]:
    """The pre-registered RPE frame gaps that a window of ``frames`` frames can hold."""
    deltas = tuple(delta for delta in RPE_DELTAS if delta < frames)
    return deltas + ((LONG_RPE_DELTA,) if frames >= LONG_RPE_MIN_FRAMES else ())


def rpe_translation(pred_c2w: np.ndarray, gt_c2w: np.ndarray, scale: float, delta: int) -> float:
    """RMSE [m] of the translation error of the ``delta``-frame relative motions."""
    pred, gt = as_homogeneous(pred_c2w), as_homogeneous(gt_c2w)
    if not 0 < delta < len(gt):
        raise ValueError(f"RPE delta {delta} is undefined for a window of {len(gt)} frames")
    moved_pred = invert(pred[:-delta]) @ pred[delta:]
    moved_pred[:, :3, 3] *= scale
    error = invert(invert(gt[:-delta]) @ gt[delta:]) @ moved_pred
    return float(np.sqrt(np.mean(np.sum(error[:, :3, 3] ** 2, axis=1))))


def pose_scale_ratios(pred_c2w: np.ndarray, gt_c2w: np.ndarray, scale: float, delta: int = DRIFT_DELTA) -> np.ndarray:
    """Scaled predicted / GT camera displacement over ``delta`` frames, one value per start frame."""
    pred, gt = as_homogeneous(pred_c2w)[:, :3, 3] * scale, as_homogeneous(gt_c2w)[:, :3, 3]
    if not 0 < delta < len(gt):
        raise ValueError(f"pose scale drift over delta {delta} is undefined for a window of {len(gt)} frames")
    gt_steps = np.linalg.norm(gt[delta:] - gt[:-delta], axis=1)
    if not (gt_steps > 0).all():
        raise ValueError(f"GT camera displacement over {delta} frames is zero")
    return np.linalg.norm(pred[delta:] - pred[:-delta], axis=1) / gt_steps


def depth_errors(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, scale: float) -> dict:
    """AbsRel and delta<1.25 of ``scale * pred`` pooled over the valid pixels."""
    p, g = np.asarray(pred, float)[mask], np.asarray(gt, float)[mask]
    p = p * scale
    return {"abs_rel": float(np.mean(np.abs(p - g) / g)), "delta<1.25": float(np.mean(np.maximum(p / g, g / p) < 1.25))}


def time_bins(frames: int) -> list[tuple[str, slice]]:
    """The pre-registered time bins (1-based t) that a window of ``frames`` frames reaches."""
    bins = []
    for first, last in T_BINS:
        if first > frames:
            break
        label = f"t{first}-{last}" if last is not None else f"t{first}+"
        bins.append((label, slice(first - 1, frames if last is None else min(last, frames))))
    return bins


def depth_report(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, s1: float) -> dict:
    """Depth errors of a (T, H, W) window under the window median scale and under ``s1``, per time bin."""
    mask = np.asarray(mask, bool)
    valid_pred, valid_gt = np.asarray(pred, float)[mask], np.asarray(gt, float)[mask]
    scalings = {"median": np.median(valid_gt) / np.median(valid_pred), "s1": s1}
    report = {}
    for name, scale in scalings.items():
        for key, value in depth_errors(pred, gt, mask, scale).items():
            report[f"{key}_{name}"] = value
        for label, frames in time_bins(len(gt)):
            for key, value in depth_errors(pred[frames], gt[frames], mask[frames], scale).items():
                report[f"{key}_{name}@{label}"] = value
    return report


def _drift_summary(prefix: str, ratios: np.ndarray) -> dict:
    return {
        f"{prefix}_first": float(ratios[0]),
        f"{prefix}_last": float(ratios[-1]),
        f"{prefix}_max_dev": float(np.max(np.abs(ratios - 1.0))),
    }


def window_metrics(
    pred_w2c: np.ndarray, gt_w2c: np.ndarray, pred_depth: np.ndarray, gt_depth: np.ndarray, mask: np.ndarray
) -> dict:
    """All pre-registered metrics of one window: ``metrics`` (flat, finite scalars) and per-frame ``series``."""
    pred_c2w, gt_c2w = invert(pred_w2c), invert(gt_w2c)
    mask = np.asarray(mask, bool)
    frames = len(gt_c2w)
    s1 = depth_scale(pred_depth[0], gt_depth[0], mask[0])
    ate_g1 = ate_first_frame(pred_c2w, gt_c2w, s1)
    depth_ratios = depth_scales(pred_depth, gt_depth, mask) / s1
    metrics = {
        "s1": s1,
        "ate_g1_rmse_mm": float(MM * np.sqrt(np.mean(ate_g1**2))),
        "ate_g1_final_mm": float(MM * ate_g1[-1]),
        "ate_sim3_rmse_mm": float(MM * np.sqrt(np.mean(ate_sim3(pred_c2w, gt_c2w) ** 2))),
    }
    for delta in rpe_deltas(frames):
        metrics[f"rpe_t_mm@{delta}"] = MM * rpe_translation(pred_c2w, gt_c2w, s1, delta)
    series = {"ate_g1_err_mm": (MM * ate_g1).tolist(), "depth_scale_ratio": depth_ratios.tolist()}
    if frames > DRIFT_DELTA:
        pose_ratios = pose_scale_ratios(pred_c2w, gt_c2w, s1)
        metrics.update(_drift_summary("pose_scale_ratio", pose_ratios))
        series["pose_scale_ratio"] = pose_ratios.tolist()
    metrics["depth_scale_ratio_last"] = float(depth_ratios[-1])
    metrics["depth_scale_ratio_max_dev"] = float(np.max(np.abs(depth_ratios - 1.0)))
    metrics.update(depth_report(pred_depth, gt_depth, mask, s1))
    metrics.update(pose_metrics(pred_w2c, gt_w2c))
    return {"metrics": metrics, "series": series}
