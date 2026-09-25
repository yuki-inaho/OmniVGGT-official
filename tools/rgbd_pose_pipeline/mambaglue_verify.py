"""Verify camera poses with RGB-D MambaGlue matches and RANSAC.

For every edge (i, i+k) of a subsampled RGB-D view:

1. RGB (INTER_LINEAR) and mapped depth (INTER_NEAREST) are resized to the long
   side ``--long-side`` and the RGB intrinsics are scaled by the same factor
   (the MambaGlue RGB-D staging contract).
2. SuperPoint-RGBD features are extracted once per frame and matched with the
   ONNX MambaGlue matcher (``mambaglue_rgbd_onnx.rgbd_onnx``).
3. ``rgbd_tracking_lab.camera_motion.estimate_rgbd_motion_from_correspondences``
   (USAC-MAGSAC epipolar inliers -> depth back-projection -> rigid RANSAC,
   SE(3) vs one-axis rail translation) measures the metric motion
   ``X_j = T X_i``.
4. The measured motion is compared with the motion implied by the poses under
   test; an edge passes when it is metric, well supported and within the
   rotation / translation tolerances.

Run inside the mambaglue_rgbd_onnx environment with ``mambaglue_rgbd_onnx``,
``rgbd_tracking_lab`` and this repository's ``tools`` directory on PYTHONPATH.
``--measurements-from`` re-evaluates stored measurements against other poses
without re-matching (used for before/after comparisons).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from rgbd_pose_pipeline.se3 import check_poses, relative, rotation_angle_deg

METRIC_STATUSES = ("metric", "metric_rotation")


@dataclass(frozen=True)
class EdgeThresholds:
    min_geometric_inliers: int = 30
    min_support: int = 10
    max_rotation_deg: float = 2.0
    min_translation_tolerance_m: float = 0.01
    relative_translation_tolerance: float = 0.1


def pose_relative_motion(c2w_i: np.ndarray, c2w_j: np.ndarray) -> np.ndarray:
    """Transform of camera-i coordinates into camera-j coordinates."""
    return relative(c2w_i, c2w_j)


def motion_errors(measured: np.ndarray, from_poses: np.ndarray) -> tuple[float, float, float]:
    rotation_error = float(rotation_angle_deg(measured[:3, :3].T @ from_poses[:3, :3]))
    translation_error = float(np.linalg.norm(measured[:3, 3] - from_poses[:3, 3]))
    return rotation_error, translation_error, float(np.linalg.norm(from_poses[:3, 3]))


def edge_passes(record: dict, thresholds: EdgeThresholds) -> bool:
    if record.get("status") not in METRIC_STATUSES:
        return False
    tolerance = max(
        thresholds.min_translation_tolerance_m, thresholds.relative_translation_tolerance * record["pose_translation_m"]
    )
    return (
        record["geometric_inlier_count"] >= thresholds.min_geometric_inliers
        and record["support_count"] >= thresholds.min_support
        and record["rotation_error_deg"] <= thresholds.max_rotation_deg
        and record["translation_error_m"] <= tolerance
    )


def rail_axis_in_camera(c2w: np.ndarray, axis_world: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis_world, dtype=np.float64)
    return c2w[:3, :3].T @ (axis / np.linalg.norm(axis))


def scale_intrinsics(k: np.ndarray, scale: float) -> np.ndarray:
    scaled = np.asarray(k, dtype=np.float64).copy()
    scaled[:2] *= scale
    return scaled


def _summary(edges: list[dict], frame_count: int, thresholds: EdgeThresholds) -> dict:
    by_step: dict[str, dict] = {}
    for step in sorted({e["step"] for e in edges}):
        subset = [e for e in edges if e["step"] == step]
        metric = [e for e in subset if e["status"] in METRIC_STATUSES]
        passed = [e for e in subset if e["passed"]]
        rot = np.array([e["rotation_error_deg"] for e in metric]) if metric else np.zeros(0)
        tra = np.array([e["translation_error_m"] for e in metric]) if metric else np.zeros(0)

        def pct(a, q):
            return float(np.percentile(a, q)) if a.size else None

        by_step[str(step)] = {
            "edges": len(subset),
            "metric": len(metric),
            "passed": len(passed),
            "pass_ratio": len(passed) / len(subset) if subset else None,
            "rotation_error_deg": {"median": pct(rot, 50), "p95": pct(rot, 95)},
            "translation_error_m": {"median": pct(tra, 50), "p95": pct(tra, 95)},
            "selected_model_counts": {
                m: sum(1 for e in metric if e["selected_model"] == m)
                for m in sorted({e["selected_model"] for e in metric})
            },
        }
    ratios: dict[str, list[float]] = {}
    for e in edges:
        if e["status"] in METRIC_STATUSES and e["measured_motion"] is not None and e["pose_translation_m"] > 2e-3:
            measured = np.linalg.norm(np.asarray(e["measured_motion"])[:3, 3])
            ratios.setdefault(e["selected_model"], []).append(measured / e["pose_translation_m"])
    scale_ratio = {
        model: {
            "count": len(v),
            "median": float(np.median(v)),
            "p25": float(np.percentile(v, 25)),
            "p75": float(np.percentile(v, 75)),
        }
        for model, v in sorted(ratios.items())
    }
    covered = {e["i"] for e in edges if e["passed"]} | {e["j"] for e in edges if e["passed"]}
    passed = sum(e["passed"] for e in edges)
    return {
        "thresholds": asdict(thresholds),
        "edge_count": len(edges),
        "passed": passed,
        "pass_ratio": passed / len(edges) if edges else None,
        "frames_with_passing_edge": len(covered),
        "frame_count": frame_count,
        "by_step": by_step,
        "translation_scale_ratio": scale_ratio,
    }


def _compare(edge: dict, poses: np.ndarray, thresholds: EdgeThresholds) -> dict:
    motion = pose_relative_motion(poses[edge["i"]], poses[edge["j"]])
    edge = dict(edge)
    if edge.get("measured_motion") is None:
        edge.update(
            rotation_error_deg=None,
            translation_error_m=None,
            pose_translation_m=float(np.linalg.norm(motion[:3, 3])),
            passed=False,
        )
        return edge
    rot, tra, norm = motion_errors(np.asarray(edge["measured_motion"]), motion)
    edge.update(rotation_error_deg=rot, translation_error_m=tra, pose_translation_m=norm)
    edge["passed"] = edge_passes(edge, thresholds)
    return edge


EMPTY_MEASUREMENT = {
    "selected_model": None,
    "geometric_inlier_count": 0,
    "support_count": 0,
    "se3_rms_m": None,
    "one_axis_rms_m": None,
    "measured_motion": None,
}


def measure_edge(estimator, source, target, *args, error_types=(), **kwargs) -> dict:
    """Run the RGB-D motion estimator on one edge; failures are recorded, not dropped."""
    if len(source) < 8:
        return {**EMPTY_MEASUREMENT, "status": "too_few_matches", "reason": "fewer_than_8_matches"}
    try:
        estimate = estimator(np.asarray(source), np.asarray(target), *args, **kwargs)
    except error_types as exc:
        return {**EMPTY_MEASUREMENT, "status": "estimator_error", "reason": f"{type(exc).__name__}: {exc}"[:300]}
    return {
        "status": estimate.status,
        "selected_model": estimate.selected_model,
        "geometric_inlier_count": estimate.geometric_inlier_count,
        "support_count": estimate.support_count,
        "se3_rms_m": estimate.se3_rms_m,
        "one_axis_rms_m": estimate.one_axis_rms_m,
        "reason": estimate.reason,
        "measured_motion": None if estimate.transform is None else np.asarray(estimate.transform).tolist(),
    }


FEATURE_KEYS = ("keypoints", "descriptors", "valid", "packed", "image_size")


def _features(args, stems, size, load_frame):
    """SuperPoint-RGBD features for every frame, cached in ``--feature-cache`` when given."""
    cache = args.feature_cache
    if cache is not None and cache.is_file():
        data = np.load(cache)
        if [str(s) for s in data["stems"]] != stems or int(data["long_side"]) != args.long_side:
            raise ValueError(f"feature cache {cache} belongs to other frames or settings")
        print(f"features loaded from cache ({len(stems)})", flush=True)
        return [{key: data[key][n] for key in FEATURE_KEYS} for n in range(len(stems))]

    from rgbd_onnx import build_extractor, extract_features, make_rgbd_tensor

    extractor = build_extractor(device=args.device)
    features, start = [], time.time()
    for index, stem in enumerate(stems):
        rgb, depth = load_frame(stem)
        valid = depth > 0
        rgbd, _ = make_rgbd_tensor(rgb, np.where(valid, depth * 1e-3, 0.0).astype(np.float32), valid)
        features.append(extract_features(extractor, rgbd, args.device))
        if index % 200 == 0:
            print(f"features {index}/{len(stems)} ({time.time() - start:.0f}s)", flush=True)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache,
            stems=np.asarray(stems),
            long_side=args.long_side,
            **{key: np.stack([f[key] for f in features]) for key in FEATURE_KEYS},
        )
    return features


def _measure(args, poses: np.ndarray, stems: list[str], rail_axis: np.ndarray) -> list[dict]:
    import cv2
    import yaml
    from rgbd_onnx import load_session, run_matcher
    from rgbd_tracking_lab.camera_motion import estimate_rgbd_motion_from_correspondences

    camera = yaml.safe_load((args.session_dir / "camera_parameters" / "rgb_camera_param.yaml").read_text())
    width, height = int(camera["width"]), int(camera["height"])
    scale = args.long_side / max(width, height)
    size = (round(width * scale), round(height * scale))
    k = scale_intrinsics(np.asarray(camera["K"], dtype=np.float64).reshape(3, 3), scale)

    def load_frame(stem):
        rgb = cv2.imread(str(args.session_dir / "rgb" / f"{stem}_rgb.png"), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(args.session_dir / args.depth_subdir / f"{stem}_depth.png"), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None or rgb.shape[:2] != (height, width):
            raise FileNotFoundError(f"cannot read a {width}x{height} RGB-D frame for {stem}")
        depth = cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST)
        depth = np.where((depth > 0) & (depth != 65535), depth, 0)
        return cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR), depth

    features = _features(args, stems, size, load_frame)
    depths = [load_frame(stem)[1] for stem in stems]
    session = load_session(provider=args.provider)
    edges, start = [], time.time()
    for step in args.edge_steps:
        for i in range(len(stems) - step):
            j = i + step
            matches, _scores = run_matcher(session, features[i], features[j])
            pairs = [
                (features[i]["keypoints"][index], features[j]["keypoints"][int(match)])
                for index, match in enumerate(matches)
                if features[i]["valid"][index] and 0 <= int(match) < int(features[j]["packed"])
            ]
            record = {"i": i, "j": j, "step": step, "stem_i": stems[i], "stem_j": stems[j], "matches": len(pairs)}
            record.update(
                measure_edge(
                    estimate_rgbd_motion_from_correspondences,
                    [p[0] for p in pairs],
                    [p[1] for p in pairs],
                    depths[i],
                    depths[j],
                    k,
                    error_types=(cv2.error, ValueError),
                    depth_scale_m_per_unit=1e-3,
                    one_axis=rail_axis_in_camera(poses[i], rail_axis),
                    magsac_threshold_px=args.magsac_threshold_px,
                    rigid_threshold_m=args.rigid_threshold_m,
                )
            )
            edges.append(record)
            if len(edges) % 250 == 0:
                print(f"edges {len(edges)} (step {step}, {time.time() - start:.0f}s)", flush=True)
    return edges


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--poses-npz", type=Path, required=True)
    parser.add_argument("--rail-json", type=Path, required=True, help="lane_refinement_summary.json (rail_axis)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--edge-steps", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--depth-subdir", default="mapped_depth")
    parser.add_argument("--long-side", type=int, default=640)
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--magsac-threshold-px", type=float, default=1.5)
    parser.add_argument("--rigid-threshold-m", type=float, default=0.015)
    parser.add_argument("--measurements-from", type=Path, help="reuse measured motions from a previous output")
    parser.add_argument("--feature-cache", type=Path, help="npz cache of per-frame features (created if missing)")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output exists: {args.output}")

    data = np.load(args.poses_npz)
    stems = [str(s) for s in data["frame_stems"]]
    poses = check_poses(data["camera_to_global"], "poses")
    rail_axis = np.asarray(json.loads(args.rail_json.read_text())["rail_axis"], dtype=np.float64)
    thresholds = EdgeThresholds()
    if args.measurements_from:
        previous = json.loads(args.measurements_from.read_text())
        if previous["stems"] != stems:
            raise ValueError("stored measurements belong to different frames")
        measured = [
            {
                k: v
                for k, v in e.items()
                if k not in ("rotation_error_deg", "translation_error_m", "pose_translation_m", "passed")
            }
            for e in previous["edges"]
        ]
    else:
        measured = _measure(args, poses, stems, rail_axis)
    edges = [_compare(edge, poses, thresholds) for edge in measured]
    output = {
        "poses": args.poses_npz.name,
        "measurements_from": args.measurements_from.name if args.measurements_from else None,
        "summary": _summary(edges, len(stems), thresholds),
        "stems": stems,
        "edges": edges,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output) + "\n")
    print(json.dumps(output["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
