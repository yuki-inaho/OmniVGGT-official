"""Pose check independent of feature matching: reproject measured depth between frames.

For a pair (i, j) the valid depth pixels of frame i (every ``stride`` pixels) are
back-projected, moved into camera j with the poses under test and projected;
the predicted depth is compared with the measured depth of frame j at that
pixel.  A pixel is an inlier when the difference is within
``max(abs_tolerance_m, rel_tolerance * depth)``.  Occlusions produce some
outliers for any pose, so the metric is used to compare pose sets on the same
pairs, not as an absolute pass/fail.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from rgbd_pose_pipeline.se3 import check_poses, relative


def pair_depth_consistency(
    depth_i: np.ndarray,
    depth_j: np.ndarray,
    k: np.ndarray,
    pose_i: np.ndarray,
    pose_j: np.ndarray,
    stride: int = 8,
    abs_tolerance_m: float = 0.005,
    rel_tolerance: float = 0.01,
) -> dict:
    rows, cols = np.mgrid[0 : depth_i.shape[0] : stride, 0 : depth_i.shape[1] : stride]
    z = depth_i[rows, cols]
    valid = z > 0
    pixels = np.stack([cols[valid], rows[valid], np.ones(valid.sum())], axis=1)
    points_i = (pixels @ np.linalg.inv(k).T) * z[valid][:, None]
    motion = relative(pose_i, pose_j)
    points_j = points_i @ motion[:3, :3].T + motion[:3, 3]
    in_front = points_j[:, 2] > 0
    projected = points_j[in_front] @ k.T
    u = np.round(projected[:, 0] / projected[:, 2]).astype(int)
    v = np.round(projected[:, 1] / projected[:, 2]).astype(int)
    inside = (u >= 0) & (u < depth_j.shape[1]) & (v >= 0) & (v < depth_j.shape[0])
    predicted = points_j[in_front][inside, 2]
    observed = depth_j[v[inside], u[inside]]
    both = observed > 0
    error = np.abs(predicted[both] - observed[both])
    if error.size == 0:
        return {"compared": 0, "inlier_ratio": None, "median_abs_error_m": None}
    tolerance = np.maximum(abs_tolerance_m, rel_tolerance * observed[both])
    return {
        "compared": int(error.size),
        "inlier_ratio": float(np.mean(error <= tolerance)),
        "median_abs_error_m": float(np.median(error)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--session-dir", type=Path, required=True, help="view with rgb/, mapped_depth/, camera_parameters/"
    )
    parser.add_argument("--poses", type=Path, nargs="+", required=True, help="pose npz files to compare")
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--pairs-per-step", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    camera = yaml.safe_load((args.session_dir / "camera_parameters" / "rgb_camera_param.yaml").read_text())
    k = np.asarray(camera["K"], dtype=np.float64).reshape(3, 3)
    sets = {}
    for path in args.poses:
        data = np.load(path)
        key = f"{path.parent.name}/{path.name}"
        if key in sets:
            raise ValueError(f"duplicate pose set key {key}")
        sets[key] = ([str(s) for s in data["frame_stems"]], check_poses(data["camera_to_global"], key))
    stems = next(iter(sets.values()))[0]
    if any(s != stems for s, _ in sets.values()):
        raise ValueError("pose sets refer to different frames")

    cache: dict[int, np.ndarray] = {}

    def depth(index):
        if index not in cache:
            image = Image.open(args.session_dir / "mapped_depth" / f"{stems[index]}_depth.png")
            cache[index] = np.asarray(image, dtype=np.float64) * 1e-3
        return cache[index]

    result = {"steps": {}}
    for step in args.steps:
        starts = np.linspace(0, len(stems) - 1 - step, args.pairs_per_step).round().astype(int)
        per_set = {}
        for name, (_, poses) in sets.items():
            stats = [pair_depth_consistency(depth(i), depth(i + step), k, poses[i], poses[i + step]) for i in starts]
            ratios = [s["inlier_ratio"] for s in stats if s["inlier_ratio"] is not None]
            errors = [s["median_abs_error_m"] for s in stats if s["median_abs_error_m"] is not None]
            per_set[name] = {
                "pairs": len(ratios),
                "inlier_ratio_mean": float(np.mean(ratios)),
                "median_abs_error_m_median": float(np.median(errors)),
            }
        result["steps"][str(step)] = per_set
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
