"""Lane (rail, one-axis) constraint for a camera trajectory on a straight track.

Camera centres are fit with a robust line (SVD + MAD outlier rejection).  The
one-axis hypothesis is adopted only when the trajectory is actually rail-like:
enough inliers and a small cross-rail spread relative to the travelled range.
When adopted, camera centres are moved towards the rail by
``projection_strength`` (rotations untouched), the same projection as
vggt-omega's ``inject_rail_pose_priors.project_to_rail``.  Otherwise the CLI
fails closed unless ``--allow-worse`` is given explicitly.

Per-step observation evidence (SE(3) vs one-axis motion on real RGB-D
correspondences) is produced later by the MambaGlue verification stage, which
uses rgbd_tracking_lab's motion estimator.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rgbd_pose_pipeline.se3 import check_poses, chordal_mean_rotation, rotation_angle_deg

MIN_INLIER_RATIO = 0.9
MAX_LINEARITY_RATIO = 0.05


@dataclass(frozen=True)
class RailFit:
    centroid: np.ndarray
    axis: np.ndarray
    inliers: np.ndarray
    cross_residual: np.ndarray


def _cross_residual(centers: np.ndarray, centroid: np.ndarray, axis: np.ndarray) -> np.ndarray:
    offset = centers - centroid
    return np.linalg.norm(offset - np.outer(offset @ axis, axis), axis=1)


def fit_rail(centers: np.ndarray, mad_k: float = 3.5, max_iterations: int = 10) -> RailFit:
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or len(centers) < 3:
        raise ValueError("need at least three 3D camera centres")
    inliers = np.ones(len(centers), dtype=bool)
    for _ in range(max_iterations):
        centroid = centers[inliers].mean(axis=0)
        axis = np.linalg.svd(centers[inliers] - centroid)[2][0]
        residual = _cross_residual(centers, centroid, axis)
        median = np.median(residual[inliers])
        mad = 1.4826 * np.median(np.abs(residual[inliers] - median))
        updated = residual <= median + mad_k * max(mad, 1e-9)
        if np.array_equal(updated, inliers):
            break
        inliers = updated
    if axis @ (centers[-1] - centers[0]) < 0:
        axis = -axis
    return RailFit(centroid, axis, inliers, _cross_residual(centers, centroid, axis))


def refine_to_rail(camera_to_world: np.ndarray, projection_strength: float) -> tuple[np.ndarray, dict]:
    if not 0 <= projection_strength < 1:
        raise ValueError("projection_strength must be in [0, 1)")
    poses = check_poses(camera_to_world, "input poses")
    centers = poses[:, :3, 3]
    fit = fit_rail(centers)
    along = (centers - fit.centroid) @ fit.axis
    on_rail = fit.centroid + np.outer(along, fit.axis)
    refined = poses.copy()
    refined[:, :3, 3] = centers + projection_strength * (on_rail - centers)
    after = _cross_residual(refined[:, :3, 3], fit.centroid, fit.axis)

    along_range = float(along.max() - along.min())
    cross_rms = float(np.sqrt(np.mean(fit.cross_residual**2)))
    linearity = cross_rms / max(along_range, 1e-9)
    inlier_ratio = float(fit.inliers.mean())
    rotation_spread = rotation_angle_deg(
        np.einsum("ji,njk->nik", chordal_mean_rotation(poses[:, :3, :3]), poses[:, :3, :3])
    )
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    adopted = inlier_ratio >= MIN_INLIER_RATIO and linearity <= MAX_LINEARITY_RATIO
    summary = {
        "frame_count": len(poses),
        "projection_strength": projection_strength,
        "rail_axis": fit.axis.tolist(),
        "rail_centroid_m": fit.centroid.tolist(),
        "rail_inlier_count": int(fit.inliers.sum()),
        "rail_inlier_ratio": inlier_ratio,
        "along_rail_range_m": along_range,
        "cross_rail_rms_before_m": cross_rms,
        "cross_rail_max_before_m": float(fit.cross_residual.max()),
        "cross_rail_rms_after_m": float(np.sqrt(np.mean(after**2))),
        "cross_rail_max_after_m": float(after.max()),
        "linearity_ratio": linearity,
        "rotation_spread_deg": {"median": float(np.median(rotation_spread)), "max": float(rotation_spread.max())},
        "step_m": {"mean": float(steps.mean()), "median": float(np.median(steps)), "max": float(steps.max())},
        "criteria": {"min_inlier_ratio": MIN_INLIER_RATIO, "max_linearity_ratio": MAX_LINEARITY_RATIO},
        "one_axis_adopted": bool(adopted),
    }
    return refined, summary


def _plot(centers_before: np.ndarray, centers_after: np.ndarray, summary: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axis = np.asarray(summary["rail_axis"])
    centroid = np.asarray(summary["rail_centroid_m"])
    helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    cross = np.cross(axis, helper)
    cross /= np.linalg.norm(cross)
    figure, ax = plt.subplots(figsize=(10, 4))
    for centers, label in ((centers_before, "input"), (centers_after, "lane refined")):
        offset = centers - centroid
        ax.plot(offset @ axis, offset @ cross * 1000.0, ".", ms=2, label=label)
    ax.set_xlabel("along rail [m]")
    ax.set_ylabel("cross rail [mm]")
    ax.legend()
    ax.set_title(f"one_axis_adopted={summary['one_axis_adopted']}")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--poses-npz", type=Path, required=True, help="npz with frame_stems and camera_to_global")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--projection-strength", type=float, required=True)
    parser.add_argument(
        "--allow-worse", action="store_true", help="write outputs even if the rail hypothesis is rejected"
    )
    args = parser.parse_args()
    data = np.load(args.poses_npz)
    refined, summary = refine_to_rail(data["camera_to_global"], args.projection_strength)
    summary["input"] = args.poses_npz.name
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "lane_refinement_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not summary["one_axis_adopted"] and not args.allow_worse:
        print(json.dumps(summary, indent=2))
        print("rail hypothesis rejected; no refined poses written (use --allow-worse to override explicitly)")
        return 2
    np.savez_compressed(args.output_dir / "lane_refined.npz", frame_stems=data["frame_stems"], camera_to_global=refined)
    _plot(data["camera_to_global"][:, :3, 3], refined[:, :3, 3], summary, args.output_dir / "lane_refinement.png")
    print(json.dumps({k: summary[k] for k in summary if k not in ("rail_axis", "rail_centroid_m")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
