"""Robust pose graph: fuse initial poses, measured RGB-D motions and the rail prior.

Variables: a correction (rotation vector, centre offset) for every frame except
the first, applied as ``R = exp(w) R0`` and ``c = c0 + d`` (world frame).

Residual blocks (each divided by its scale):

* measured edge ``X_j = Z_ij X_i`` vs the pose motion ``T_j^-1 T_i``:
  rotation log-error / ``edge_rotation_deg`` and translation / ``edge_translation_m``,
  robustified with block-wise Huber weights (IRLS);
* prior to the initial poses: rotation / ``prior_rotation_deg``, centre / ``prior_translation_m``;
* optional rail prior: cross-rail component of the centre / ``rail_cross_m``
  (the along-rail position is not constrained).

Edges are selected by measurement quality only (metric status and inlier
support), not by agreement with the initial poses, so wrong initial poses can
be corrected.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from rgbd_pose_pipeline.se3 import check_poses, relative, rotation_angle_deg

METRIC_STATUSES = ("metric", "metric_rotation")


@dataclass(frozen=True)
class PoseGraphConfig:
    # Defaults for the rail pipeline: COLMAP BA rotations are the most precise source (strong
    # rotation prior); metric translations come from RGB-D edges and the rail prior.
    edge_rotation_deg: float = 0.5
    edge_translation_m: float = 0.005
    prior_rotation_deg: float = 0.2
    prior_translation_m: float = 0.05
    rail_cross_m: float = 0.01
    min_geometric_inliers: int = 30
    min_support: int = 10
    irls_iterations: int = 5


def _usable_edges(edges: list[dict], config: PoseGraphConfig) -> list[tuple[int, int, np.ndarray]]:
    usable = []
    for edge in edges:
        if (
            edge.get("status") in METRIC_STATUSES
            and edge.get("measured_motion") is not None
            and edge["geometric_inlier_count"] >= config.min_geometric_inliers
            and edge["support_count"] >= config.min_support
        ):
            usable.append((int(edge["i"]), int(edge["j"]), np.asarray(edge["measured_motion"], dtype=np.float64)))
    return usable


def _apply(init: np.ndarray, x: np.ndarray) -> np.ndarray:
    poses = init.copy()
    delta = x.reshape(-1, 6)
    poses[1:, :3, :3] = Rotation.from_rotvec(delta[:, :3]).as_matrix() @ init[1:, :3, :3]
    poses[1:, :3, 3] = init[1:, :3, 3] + delta[:, 3:]
    return poses


def _edge_residuals(poses: np.ndarray, edges) -> np.ndarray:
    i = np.array([e[0] for e in edges])
    j = np.array([e[1] for e in edges])
    measured = np.stack([e[2] for e in edges])
    motion = relative(poses[i], poses[j])
    rotation_error = np.einsum("nji,njk->nik", measured[:, :3, :3], motion[:, :3, :3])
    return np.concatenate(
        [Rotation.from_matrix(rotation_error).as_rotvec(), motion[:, :3, 3] - measured[:, :3, 3]], axis=1
    )


def optimize_pose_graph(init_c2w: np.ndarray, edges: list[dict], rail: dict | None, config: PoseGraphConfig):
    init = check_poses(init_c2w, "initial poses")
    n = len(init)
    usable = _usable_edges(edges, config)
    if not usable:
        raise ValueError("no usable edges")
    edge_scale = np.array([np.radians(config.edge_rotation_deg)] * 3 + [config.edge_translation_m] * 3)
    prior_scale = np.array([np.radians(config.prior_rotation_deg)] * 3 + [config.prior_translation_m] * 3)
    if rail is not None:
        axis = np.asarray(rail["axis"], float) / np.linalg.norm(rail["axis"])
        cross_projector = np.eye(3) - np.outer(axis, axis)
        centroid = np.asarray(rail["centroid"], float)

    def residuals(x, weights):
        poses = _apply(init, x)
        parts = [(_edge_residuals(poses, usable) / edge_scale * np.sqrt(weights)[:, None]).ravel()]
        delta = x.reshape(-1, 6)
        parts.append((delta / prior_scale).ravel())
        if rail is not None:
            parts.append(((poses[1:, :3, 3] - centroid) @ cross_projector / config.rail_cross_m).ravel())
        return np.concatenate(parts)

    rows = len(usable) * 6 + (n - 1) * 6 + ((n - 1) * 3 if rail is not None else 0)
    sparsity = lil_matrix((rows, (n - 1) * 6), dtype=int)
    for row, (i, j, _) in enumerate(usable):
        for frame in (i, j):
            if frame > 0:
                sparsity[row * 6 : (row + 1) * 6, (frame - 1) * 6 : frame * 6] = 1
    offset = len(usable) * 6
    for frame in range(1, n):
        sparsity[offset + (frame - 1) * 6 : offset + frame * 6, (frame - 1) * 6 : frame * 6] = 1
    if rail is not None:
        offset += (n - 1) * 6
        for frame in range(1, n):
            sparsity[offset + (frame - 1) * 3 : offset + frame * 3, (frame - 1) * 6 + 3 : frame * 6] = 1

    x = np.zeros((n - 1) * 6)
    weights = np.ones(len(usable))
    for _ in range(config.irls_iterations):
        x = least_squares(residuals, x, jac_sparsity=sparsity, method="trf", args=(weights,), x_scale="jac").x
        block = np.linalg.norm(_edge_residuals(_apply(init, x), usable) / edge_scale, axis=1)
        weights = np.minimum(1.0, 1.0 / np.maximum(block, 1e-12))
    final = check_poses(_apply(init, x), "optimised poses")

    def edge_stats(poses):
        r = _edge_residuals(poses, usable)
        rot, tra = np.degrees(np.linalg.norm(r[:, :3], axis=1)), np.linalg.norm(r[:, 3:], axis=1)
        return {
            "rotation_deg": {"median": float(np.median(rot)), "p95": float(np.percentile(rot, 95))},
            "translation_m": {"median": float(np.median(tra)), "p95": float(np.percentile(tra, 95))},
        }

    moved_rot = rotation_angle_deg(np.einsum("nji,njk->nik", init[:, :3, :3], final[:, :3, :3]))
    moved_t = np.linalg.norm(final[:, :3, 3] - init[:, :3, 3], axis=1)
    summary = {
        "config": asdict(config),
        "frame_count": n,
        "edges_total": len(edges),
        "edges_used": len(usable),
        "downweighted_edges": int((weights < 1.0).sum()),
        "edge_residual_before": edge_stats(init),
        "edge_residual_after": edge_stats(final),
        "moved_from_initial": {
            "rotation_deg": {"median": float(np.median(moved_rot)), "max": float(moved_rot.max())},
            "translation_m": {"median": float(np.median(moved_t)), "max": float(moved_t.max())},
        },
    }
    if rail is not None:

        def cross_rms(poses):
            return float(np.sqrt(np.mean(np.sum(((poses[:, :3, 3] - centroid) @ cross_projector) ** 2, axis=1))))

        summary["rail_cross_rms_before_m"] = cross_rms(init)
        summary["rail_cross_rms_after_m"] = cross_rms(final)
    return final, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--init-npz", type=Path, required=True)
    parser.add_argument("--edges", type=Path, required=True, help="mambaglue_verify output JSON")
    parser.add_argument("--rail-json", type=Path, help="lane_refinement_summary.json; omit to disable the rail prior")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = np.load(args.init_npz)
    verification = json.loads(args.edges.read_text())
    if verification["stems"] != [str(s) for s in data["frame_stems"]]:
        raise ValueError("edges and initial poses refer to different frames")
    rail = None
    if args.rail_json:
        lane = json.loads(args.rail_json.read_text())
        rail = {"axis": lane["rail_axis"], "centroid": lane["rail_centroid_m"]}
    final, summary = optimize_pose_graph(data["camera_to_global"], verification["edges"], rail, PoseGraphConfig())
    summary.update(init=args.init_npz.name, edges=args.edges.name, rail_prior=rail is not None)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output_dir / "final_poses.npz", frame_stems=data["frame_stems"], camera_to_global=final)
    (args.output_dir / "pose_graph_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
