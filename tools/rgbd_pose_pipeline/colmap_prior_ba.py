"""COLMAP refinement of prior camera poses with a metric Sim(3) scale alignment.

Stages (each a sub-command, each COLMAP call logged to ``<workdir>/logs``):

``extract-match``   SIFT features (PINHOLE, one shared camera fixed to the RGB
                    YAML intrinsics) and sequential matching.
``triangulate-ba``  prior poses -> text model -> ``point_triangulator`` (poses
                    fixed) -> ``bundle_adjuster`` (poses + points, intrinsics fixed).
``align``           Sim(3) from the BA model to the prior poses (metric scale).
                    Rotation from camera orientations (a centres-only fit is
                    degenerate about the rail axis); scale/translation from
                    centres.  COLMAP ``model_aligner`` (centres only) is run as
                    a scale cross-check.

COLMAP runs in its own pixi environment:
``pixi run --manifest-path <colmap-repo>/pixi.toml colmap <command> ...``.
Poses are camera_to_world (OpenCV); COLMAP text models store world_to_camera
``QW QX QY QZ TX TY TZ``.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy.spatial.transform import Rotation

from rgbd_pose_pipeline.se3 import check_poses, chordal_mean_rotation, invert, rotation_angle_deg

RGB_SUFFIX = "_rgb.png"


# ----------------------------------------------------------------------------- pure helpers
def c2w_to_qvec_tvec(camera_to_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w2c = invert(camera_to_world)
    x, y, z, w = Rotation.from_matrix(w2c[:3, :3]).as_quat()
    return np.array([w, x, y, z]), w2c[:3, 3]


def qvec_tvec_to_c2w(qvec, tvec) -> np.ndarray:
    w, x, y, z = qvec
    w2c = np.eye(4)
    w2c[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    w2c[:3, 3] = tvec
    return invert(w2c)


def write_images_txt(path: Path, records) -> None:
    """records: iterable of (image_id, name, camera_id, camera_to_world)."""
    lines = ["# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME", "# POINTS2D[] (empty)"]
    for image_id, name, camera_id, pose in records:
        qvec, tvec = c2w_to_qvec_tvec(pose)
        values = " ".join(f"{v:.17g}" for v in (*qvec, *tvec))
        lines.extend([f"{image_id} {values} {camera_id} {name}", ""])
    Path(path).write_text("\n".join(lines) + "\n")


def read_images_txt(path: Path) -> dict[str, np.ndarray]:
    poses: dict[str, np.ndarray] = {}
    lines = [line for line in Path(path).read_text().splitlines() if not line.startswith("#")]
    for header in lines[0::2]:
        if not header.strip():
            continue
        fields = header.split()
        qvec = np.array([float(v) for v in fields[1:5]])
        tvec = np.array([float(v) for v in fields[5:8]])
        poses[fields[9]] = qvec_tvec_to_c2w(qvec, tvec)
    return poses


def umeyama_sim3(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """target ~= scale * R @ source + t (least squares, Umeyama 1991)."""
    source, target = np.asarray(source, float), np.asarray(target, float)
    mu_s, mu_t = source.mean(0), target.mean(0)
    xs, xt = source - mu_s, target - mu_t
    u, d, vt = np.linalg.svd(xt.T @ xs / len(source))
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1
    rotation = u @ sign @ vt
    scale = float(np.trace(np.diag(d) @ sign) / xs.var(0).sum())
    return scale, rotation, mu_t - scale * rotation @ mu_s


def align_pose_sim3(
    source_c2w: np.ndarray, target_c2w: np.ndarray, trim_fraction: float = 0.1, iterations: int = 5
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Similarity mapping source poses onto target poses: x_t = s * R @ x_s + t.

    Camera centres on a rail are nearly collinear, which leaves the rotation
    about the rail undetermined for a centres-only fit.  The rotation is
    therefore the robust chordal mean of R_target @ R_source^T; scale and
    translation then follow in closed form from the centres.  The worst
    ``trim_fraction`` of cameras is excluded from each estimate.
    """
    source, target = check_poses(source_c2w, "source"), check_poses(target_c2w, "target")
    keep = int(np.ceil(len(source) * (1.0 - trim_fraction)))
    relative = np.einsum("nij,nkj->nik", target[:, :3, :3], source[:, :3, :3])
    inliers = np.arange(len(source))
    for _ in range(iterations):
        rotation = chordal_mean_rotation(relative[inliers])
        errors = rotation_angle_deg(np.einsum("ji,njk->nik", rotation, relative))
        inliers = np.argsort(errors)[:keep]
    rotated = source[:, :3, 3] @ rotation.T
    centres = target[:, :3, 3]
    inliers = np.arange(len(source))
    for _ in range(iterations):
        mu_s, mu_t = rotated[inliers].mean(0), centres[inliers].mean(0)
        xs, xt = rotated[inliers] - mu_s, centres[inliers] - mu_t
        scale = float((xs * xt).sum() / (xs * xs).sum())
        translation = mu_t - scale * mu_s
        residual = np.linalg.norm(scale * rotated + translation - centres, axis=1)
        inliers = np.argsort(residual)[:keep]
    aligned = source.copy()
    aligned[:, :3, :3] = rotation @ source[:, :3, :3]
    aligned[:, :3, 3] = scale * rotated + translation
    return scale, rotation, translation, aligned


def depth_scale_from_observations(z_model: np.ndarray, measured: np.ndarray, mad_k: float = 3.5) -> dict:
    """Metric scale of a model from measured depth: robust median of measured / model depth."""
    z_model, measured = np.asarray(z_model, float), np.asarray(measured, float)
    valid = (measured > 0) & (z_model > 0)
    ratios = measured[valid] / z_model[valid]
    if ratios.size == 0:
        raise ValueError("no observation with valid measured depth")
    median = np.median(ratios)
    mad = 1.4826 * np.median(np.abs(ratios - median))
    inliers = ratios[np.abs(ratios - median) <= mad_k * max(mad, 1e-9)]
    return {
        "scale": float(np.median(inliers)),
        "valid_observations": int(valid.sum()),
        "inlier_observations": int(inliers.size),
        "ratio_p25": float(np.percentile(inliers, 25)),
        "ratio_p75": float(np.percentile(inliers, 75)),
    }


def align_with_fixed_scale(
    source_c2w: np.ndarray, target_c2w: np.ndarray, scale: float, rotation: np.ndarray, trim_fraction: float = 0.1
) -> np.ndarray:
    """Apply x_t = scale * R @ x_s + t with given scale and rotation; t from centres (trimmed mean)."""
    source, target = check_poses(source_c2w, "source"), check_poses(target_c2w, "target")
    rotated = source[:, :3, 3] @ rotation.T
    offsets = target[:, :3, 3] - scale * rotated
    keep = int(np.ceil(len(source) * (1.0 - trim_fraction)))
    translation = offsets.mean(0)
    for _ in range(5):
        translation = offsets[np.argsort(np.linalg.norm(offsets - translation, axis=1))[:keep]].mean(0)
    aligned = source.copy()
    aligned[:, :3, :3] = rotation @ source[:, :3, :3]
    aligned[:, :3, 3] = scale * rotated + translation
    return aligned


def read_images_txt_with_points(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """name -> (camera_to_world, points2D xy (M,2), point3D ids (M,)) from a COLMAP images.txt."""
    records = {}
    lines = [line for line in Path(path).read_text().splitlines() if not line.startswith("#")]
    for header, observations in zip(lines[0::2], lines[1::2], strict=True):
        fields = header.split()
        pose = qvec_tvec_to_c2w(np.array(fields[1:5], float), np.array(fields[5:8], float))
        values = np.array(observations.split(), dtype=float).reshape(-1, 3)
        records[fields[9]] = (pose, values[:, :2], values[:, 2].astype(np.int64))
    return records


def read_points3d_txt(path: Path) -> dict[int, np.ndarray]:
    points = {}
    for line in Path(path).read_text().splitlines():
        if line and not line.startswith("#"):
            fields = line.split()
            points[int(fields[0])] = np.array(fields[1:4], float)
    return points


_ANALYZER_KEYS = {
    "Cameras": "cameras",
    "Images": "images",
    "Registered images": "registered_images",
    "Points": "points",
    "Observations": "observations",
    "Mean track length": "mean_track_length",
    "Mean observations per image": "mean_observations_per_image",
    "Mean reprojection error": "mean_reprojection_error_px",
}


def parse_model_analyzer(text: str) -> dict:
    stats = {}
    for label, key in _ANALYZER_KEYS.items():
        match = re.search(rf"\] ?{re.escape(label)}: ([0-9.eE+-]+)|\b{re.escape(label)}: ([0-9.eE+-]+)", text)
        if match:
            value = float(match.group(1) or match.group(2))
            stats[key] = (
                int(value) if key in {"cameras", "images", "registered_images", "points", "observations"} else value
            )
    return stats


def read_pinhole_from_yaml(path: Path) -> tuple[int, int, list[float]]:
    data = yaml.safe_load(Path(path).read_text())
    k = np.asarray(data["K"], dtype=float).reshape(3, 3)
    if data.get("D"):
        raise ValueError("non-empty distortion is not supported by the PINHOLE model used here")
    return int(data["width"]), int(data["height"]), [k[0, 0], k[1, 1], k[0, 2], k[1, 2]]


# ----------------------------------------------------------------------------- COLMAP runner
class Colmap:
    def __init__(self, repo: Path, log_dir: Path):
        self.repo, self.log_dir = Path(repo), Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, command: str, **options) -> str:
        argv = ["pixi", "run", "--manifest-path", str(self.repo / "pixi.toml"), "colmap", command]
        for key, value in options.items():
            argv += [f"--{key}", str(value)]
        log = self.log_dir / f"{command}.log"
        with log.open("a") as handle:
            handle.write(f"\n# {shlex.join(argv)}\n")
            handle.flush()
            result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
            handle.write(result.stdout)
        if result.returncode != 0:
            raise RuntimeError(f"colmap {command} failed (exit {result.returncode}); see {log}")
        return result.stdout


def _db_images(database: Path) -> list[tuple[int, str, int]]:
    with sqlite3.connect(database) as connection:
        return connection.execute("SELECT image_id, name, camera_id FROM images ORDER BY image_id").fetchall()


def _db_counts(database: Path) -> dict:
    with sqlite3.connect(database) as connection:

        def count(sql):
            return int(connection.execute(sql).fetchone()[0])

        return {
            "images": count("SELECT COUNT(*) FROM images"),
            "keypoints_total": count("SELECT COALESCE(SUM(rows),0) FROM keypoints"),
            "matched_pairs": count("SELECT COUNT(*) FROM matches WHERE rows > 0"),
            "verified_pairs": count("SELECT COUNT(*) FROM two_view_geometries WHERE rows > 0"),
        }


def _load_prior(prior_npz: Path) -> tuple[list[str], np.ndarray]:
    data = np.load(prior_npz)
    names = [f"{stem}{RGB_SUFFIX}" for stem in data["frame_stems"]]
    return names, check_poses(data["camera_to_global"], "prior poses")


# ----------------------------------------------------------------------------- stages
def extract_match(colmap: Colmap, workdir: Path, images: Path, camera_yaml: Path, use_gpu: int, overlap: int) -> dict:
    database = workdir / "database.db"
    if database.exists():
        raise FileExistsError(f"database already exists: {database}")
    width, height, params = read_pinhole_from_yaml(camera_yaml)
    first = min(images.glob(f"*{RGB_SUFFIX}"), default=None)
    if first is None:
        raise FileNotFoundError(f"no *{RGB_SUFFIX} images in {images}")
    with Image.open(first) as image:
        if image.size != (width, height):
            raise ValueError(f"image size {image.size} != camera YAML {(width, height)}")
    colmap(
        "feature_extractor",
        database_path=database,
        image_path=images,
        **{
            "ImageReader.camera_model": "PINHOLE",
            "ImageReader.single_camera": 1,
            "ImageReader.camera_params": ",".join(f"{p:.10g}" for p in params),
            "FeatureExtraction.use_gpu": use_gpu,
        },
    )
    colmap(
        "sequential_matcher",
        database_path=database,
        **{"SequentialMatching.overlap": overlap, "FeatureMatching.use_gpu": use_gpu},
    )
    summary = {
        "camera_pinhole_fx_fy_cx_cy": params,
        "image_size_wh": [width, height],
        "sequential_overlap": overlap,
        **_db_counts(database),
    }
    (workdir / "extract_match_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def triangulate_ba(
    colmap: Colmap,
    workdir: Path,
    images: Path,
    prior_npz: Path,
    camera_yaml: Path,
    *,
    min_track_len: int,
    max_reproj_error: float,
    min_tri_angle: float,
    ba_max_iterations: int,
) -> dict:
    database = workdir / "database.db"
    names, prior = _load_prior(prior_npz)
    by_name = dict(zip(names, prior, strict=True))
    db_images = _db_images(database)
    if {name for _, name, _ in db_images} != set(names):
        raise ValueError("database image names do not match the prior poses")
    camera_ids = {camera_id for _, _, camera_id in db_images}
    if len(camera_ids) != 1:
        raise ValueError(f"expected one shared camera, got {sorted(camera_ids)}")
    width, height, params = read_pinhole_from_yaml(camera_yaml)
    prior_model = workdir / "prior_model"
    prior_model.mkdir()
    (prior_model / "cameras.txt").write_text(
        f"{camera_ids.pop()} PINHOLE {width} {height} " + " ".join(f"{p:.17g}" for p in params) + "\n"
    )
    write_images_txt(prior_model / "images.txt", [(i, n, c, by_name[n]) for i, n, c in db_images])
    (prior_model / "points3D.txt").write_text("")
    stages = ("triangulated", "triangulated_filtered", "ba_raw", "ba")
    for name in stages:
        (workdir / name).mkdir()
    filters = {"min_track_len": min_track_len, "max_reproj_error": max_reproj_error, "min_tri_angle": min_tri_angle}
    colmap(
        "point_triangulator",
        database_path=database,
        image_path=images,
        input_path=prior_model,
        output_path=workdir / "triangulated",
        refine_intrinsics=0,
    )
    # Degenerate (near-parallel / far) points make the unrobustified BA report explode; filter explicitly.
    colmap(
        "point_filtering", input_path=workdir / "triangulated", output_path=workdir / "triangulated_filtered", **filters
    )
    colmap(
        "bundle_adjuster",
        input_path=workdir / "triangulated_filtered",
        output_path=workdir / "ba_raw",
        **{
            "BundleAdjustment.refine_focal_length": 0,
            "BundleAdjustment.refine_principal_point": 0,
            "BundleAdjustment.refine_extra_params": 0,
            "BundleAdjustmentCeres.max_num_iterations": ba_max_iterations,
        },
    )
    colmap("point_filtering", input_path=workdir / "ba_raw", output_path=workdir / "ba", **filters)
    summary = {"filters": filters, "ba_max_iterations": ba_max_iterations}
    summary.update({name: parse_model_analyzer(colmap("model_analyzer", path=workdir / name)) for name in stages})
    (workdir / "triangulate_ba_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def align(colmap: Colmap, workdir: Path, prior_npz: Path, max_error_m: float, trim_fraction: float) -> dict:
    names, prior = _load_prior(prior_npz)
    ba_text = workdir / "ba_txt"
    ba_text.mkdir()
    colmap("model_converter", input_path=workdir / "ba", output_path=ba_text, output_type="TXT")
    ba_poses = read_images_txt(ba_text / "images.txt")
    missing = [n for n in names if n not in ba_poses]
    if missing:
        raise RuntimeError(f"{len(missing)} prior images are missing from the BA model: {missing[:5]}")
    ba = np.stack([ba_poses[n] for n in names])

    # Cross-check: COLMAP's centres-only Sim(3) (degenerate in rotation for a rail, so only its scale is used).
    ref = workdir / "ref_images.txt"
    ref.write_text(
        "".join(f"{n} {c[0]:.10f} {c[1]:.10f} {c[2]:.10f}\n" for n, c in zip(names, prior[:, :3, 3], strict=True))
    )
    (workdir / "aligned_model_aligner").mkdir()
    colmap(
        "model_aligner",
        input_path=workdir / "ba",
        output_path=workdir / "aligned_model_aligner",
        ref_images_path=ref,
        ref_is_gps=0,
        alignment_type="custom",
        alignment_max_error=max_error_m,
        transform_path=workdir / "model_aligner_sim3.txt",
    )
    aligner_scale = float((workdir / "model_aligner_sim3.txt").read_text().split()[0])

    scale, rotation, translation, aligned = align_pose_sim3(ba, prior, trim_fraction=trim_fraction)
    check_poses(aligned, "aligned poses")
    centre_diff = np.linalg.norm(aligned[:, :3, 3] - prior[:, :3, 3], axis=1)
    rotation_diff = rotation_angle_deg(np.einsum("nji,njk->nik", prior[:, :3, :3], aligned[:, :3, :3]))
    data = np.load(prior_npz)
    np.savez_compressed(workdir / "colmap_aligned_poses.npz", frame_stems=data["frame_stems"], camera_to_global=aligned)
    summary = {
        "frame_count": len(names),
        "registered_in_ba_model": len(ba_poses),
        "registration_ratio": len(ba_poses) / len(names),
        "sim3_pose_aware": {
            "scale": scale,
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
            "trim_fraction": trim_fraction,
        },
        "model_aligner_centres_only_scale": aligner_scale,
        "scale_relative_difference": abs(scale - aligner_scale) / aligner_scale,
        "center_diff_to_prior_m": {
            "rms": float(np.sqrt(np.mean(centre_diff**2))),
            "median": float(np.median(centre_diff)),
            "p95": float(np.percentile(centre_diff, 95)),
            "max": float(centre_diff.max()),
        },
        "rotation_diff_to_prior_deg": {
            "median": float(np.median(rotation_diff)),
            "p95": float(np.percentile(rotation_diff, 95)),
            "max": float(rotation_diff.max()),
        },
    }
    (workdir / "colmap_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def depth_scale(workdir: Path, prior_npz: Path, depth_dir: Path, depth_suffix: str, trim_fraction: float) -> dict:
    """Metric scale of the BA model from the mapped depth; rotation/translation from the prior poses."""
    ba_text = workdir / "ba_txt"
    if not (ba_text / "images.txt").is_file():
        raise FileNotFoundError(f"run the align stage first ({ba_text} missing)")
    names, prior = _load_prior(prior_npz)
    images = read_images_txt_with_points(ba_text / "images.txt")
    points = read_points3d_txt(ba_text / "points3D.txt")
    z_model, measured, per_image = [], [], []
    for name in names:
        pose, xy, ids = images[name]
        keep = ids >= 0
        xyz = np.stack([points[i] for i in ids[keep]]) if keep.any() else np.zeros((0, 3))
        w2c = invert(pose)
        z = xyz @ w2c[:3, :3].T[:, 2] + w2c[2, 3]
        depth = np.asarray(Image.open(depth_dir / f"{name.removesuffix(RGB_SUFFIX)}{depth_suffix}"), dtype=np.float64)
        cols = np.clip(xy[keep, 0].astype(int), 0, depth.shape[1] - 1)
        rows = np.clip(xy[keep, 1].astype(int), 0, depth.shape[0] - 1)
        d = depth[rows, cols] * 1e-3
        z_model.append(z)
        measured.append(d)
        valid = (d > 0) & (z > 0)
        if valid.sum() >= 10:
            per_image.append(float(np.median(d[valid] / z[valid])))
    stats = depth_scale_from_observations(np.concatenate(z_model), np.concatenate(measured))
    ba = np.stack([images[n][0] for n in names])
    prior_scale, rotation, _, _ = align_pose_sim3(ba, prior, trim_fraction=trim_fraction)
    aligned = align_with_fixed_scale(ba, prior, stats["scale"], rotation, trim_fraction=trim_fraction)
    check_poses(aligned, "depth-scaled poses")
    data = np.load(prior_npz)
    np.savez_compressed(
        workdir / "colmap_aligned_poses_depthscale.npz", frame_stems=data["frame_stems"], camera_to_global=aligned
    )
    per_image = np.asarray(per_image)
    summary = {
        "depth_dir": depth_dir.name,
        **stats,
        "per_image_scale": {
            "count": int(per_image.size),
            "p5": float(np.percentile(per_image, 5)),
            "median": float(np.median(per_image)),
            "p95": float(np.percentile(per_image, 95)),
        },
        "prior_sim3_scale": prior_scale,
        "depth_scale_over_prior_scale": stats["scale"] / prior_scale,
    }
    (workdir / "colmap_depth_scale.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ("extract-match", "triangulate-ba", "align", "depth-scale"):
        stage = sub.add_parser(name)
        stage.add_argument("--colmap-repo", type=Path, required=True)
        stage.add_argument("--workdir", type=Path, required=True)
    sub.choices["extract-match"].add_argument("--images", type=Path, required=True)
    sub.choices["extract-match"].add_argument("--camera-yaml", type=Path, required=True)
    sub.choices["extract-match"].add_argument("--use-gpu", type=int, choices=(0, 1), required=True)
    sub.choices["extract-match"].add_argument("--overlap", type=int, default=10)
    sub.choices["triangulate-ba"].add_argument("--images", type=Path, required=True)
    sub.choices["triangulate-ba"].add_argument("--prior-npz", type=Path, required=True)
    sub.choices["triangulate-ba"].add_argument("--camera-yaml", type=Path, required=True)
    sub.choices["triangulate-ba"].add_argument("--min-track-len", type=int, default=3)
    sub.choices["triangulate-ba"].add_argument("--max-reproj-error", type=float, default=4.0)
    sub.choices["triangulate-ba"].add_argument("--min-tri-angle", type=float, default=1.5)
    sub.choices["triangulate-ba"].add_argument("--ba-max-iterations", type=int, default=200)
    sub.choices["align"].add_argument("--prior-npz", type=Path, required=True)
    sub.choices["align"].add_argument("--alignment-max-error", type=float, default=0.05)
    sub.choices["align"].add_argument("--trim-fraction", type=float, default=0.1)
    sub.choices["depth-scale"].add_argument("--prior-npz", type=Path, required=True)
    sub.choices["depth-scale"].add_argument("--depth-dir", type=Path, required=True)
    sub.choices["depth-scale"].add_argument("--depth-suffix", default="_depth.png")
    sub.choices["depth-scale"].add_argument("--trim-fraction", type=float, default=0.1)
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    colmap = Colmap(args.colmap_repo, args.workdir / "logs")
    if args.stage == "extract-match":
        result = extract_match(colmap, args.workdir, args.images, args.camera_yaml, args.use_gpu, args.overlap)
    elif args.stage == "triangulate-ba":
        result = triangulate_ba(
            colmap,
            args.workdir,
            args.images,
            args.prior_npz,
            args.camera_yaml,
            min_track_len=args.min_track_len,
            max_reproj_error=args.max_reproj_error,
            min_tri_angle=args.min_tri_angle,
            ba_max_iterations=args.ba_max_iterations,
        )
    elif args.stage == "align":
        result = align(colmap, args.workdir, args.prior_npz, args.alignment_max_error, args.trim_fraction)
    else:
        result = depth_scale(args.workdir, args.prior_npz, args.depth_dir, args.depth_suffix, args.trim_fraction)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
