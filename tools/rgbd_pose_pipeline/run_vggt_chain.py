"""Metric VGGT-Omega RGB-D pose chain with explicitly selected RGB and depth inputs.

This is a thin driver around vggt-omega's ``run_vggt_rgbd_chunk_alignment``
functions (chunk inference, per-chunk metric scale from measured depth,
shared-frame SE(3) chaining).  The upstream entry point hard-codes
``mapped_depth_dense/`` and silently skips frames without depth; here the depth
directory and suffixes are required arguments and a missing depth frame is an
error.

Run inside the vggt-omega environment with the vggt-omega checkout on
``PYTHONPATH`` (it provides ``run_vggt_rgbd_chunk_alignment`` and ``vggt_omega``).
Output: ``vggt_rgbd_global_poses.npz`` (``frame_stems``, ``camera_to_global``
[N,4,4], ``chunk_to_global``, ``chunk_scales``) and ``summary.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def collect_pairs(
    session_dir: Path, rgb_subdir: str, rgb_suffix: str, depth_subdir: str, depth_suffix: str
) -> list[tuple[Path, Path]]:
    rgb_paths = sorted((Path(session_dir) / rgb_subdir).glob(f"*{rgb_suffix}"))
    if not rgb_paths:
        raise FileNotFoundError(f"no *{rgb_suffix} files in {Path(session_dir) / rgb_subdir}")
    pairs = []
    for rgb_path in rgb_paths:
        stem = rgb_path.name.removesuffix(rgb_suffix)
        depth_path = Path(session_dir) / depth_subdir / f"{stem}{depth_suffix}"
        if not depth_path.is_file():
            raise FileNotFoundError(f"missing depth for frame {stem}: {depth_path}")
        pairs.append((rgb_path, depth_path))
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rgb-subdir", required=True)
    parser.add_argument("--rgb-suffix", required=True)
    parser.add_argument("--depth-subdir", required=True)
    parser.add_argument("--depth-suffix", required=True)
    parser.add_argument("--chunk-size", type=int, default=6)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--min-depth-m", type=float, default=0.10)
    parser.add_argument("--max-depth-m", type=float, default=1.30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import numpy as np
    import torch
    from run_vggt_rgbd_chunk_alignment import (
        RGB_SUFFIX,
        AlignmentConfig,
        chunk_start_indices,
        estimate_adjacent_chunk_transform,
        global_frame_poses,
        infer_chunk,
        summarize_edge_residuals,
    )
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_checkpoint_state_dict

    if args.rgb_suffix != RGB_SUFFIX:
        raise ValueError(f"vggt-omega derives frame stems with {RGB_SUFFIX!r}; got --rgb-suffix {args.rgb_suffix!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if args.chunk_size < 2 or not 0 < args.stride < args.chunk_size:
        raise ValueError("require chunk-size >= 2 and 0 < stride < chunk-size")
    if not 0 < args.min_depth_m < args.max_depth_m:
        raise ValueError("require 0 < min-depth-m < max-depth-m")

    pairs = collect_pairs(args.session_dir, args.rgb_subdir, args.rgb_suffix, args.depth_subdir, args.depth_suffix)
    config = AlignmentConfig(
        args.session_dir,
        args.checkpoint,
        args.output_dir,
        args.chunk_size,
        args.stride,
        None,
        args.width,
        args.height,
        args.min_depth_m,
        args.max_depth_m,
        0.005,
    )
    starts = chunk_start_indices(len(pairs), args.chunk_size, args.stride)
    if len(starts) < 2:
        raise RuntimeError("at least two chunks are required")

    model = VGGTOmega().eval().to("cuda")
    model.load_state_dict(load_checkpoint_state_dict(args.checkpoint))
    chunks = []
    for index, start in enumerate(starts):
        chunks.append(infer_chunk(model, pairs[start : start + args.chunk_size], start, config))
        if index % 50 == 0 or index == len(starts) - 1:
            print(f"chunk {index + 1}/{len(starts)} start={start} scale={chunks[-1].scale:.4f}", flush=True)

    chunk_to_global = [np.eye(4, dtype=np.float64)]
    edges = []
    for index in range(1, len(chunks)):
        previous_to_current, metrics = estimate_adjacent_chunk_transform(chunks[index - 1], chunks[index])
        chunk_to_global.append(chunk_to_global[-1] @ np.linalg.inv(previous_to_current))
        edges.append({"source_chunk": index - 1, "target_chunk": index, **metrics})
    stems, camera_to_global, observation_counts = global_frame_poses(chunks, chunk_to_global)
    if len(stems) != len(pairs) or not np.isfinite(camera_to_global).all():
        raise RuntimeError("pose chain did not produce one finite pose per input frame")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "vggt_rgbd_global_poses.npz",
        frame_stems=np.asarray(stems),
        camera_to_global=camera_to_global,
        chunk_to_global=np.stack(chunk_to_global),
        chunk_scales=np.asarray([chunk.scale for chunk in chunks]),
    )
    scales = np.asarray([chunk.scale for chunk in chunks])
    summary = {
        "inputs": {
            "rgb_subdir": args.rgb_subdir,
            "rgb_suffix": args.rgb_suffix,
            "depth_subdir": args.depth_subdir,
            "depth_suffix": args.depth_suffix,
            "checkpoint_name": args.checkpoint.name,
        },
        "frame_count": len(stems),
        "chunk_count": len(chunks),
        "chunk_size": args.chunk_size,
        "stride": args.stride,
        "image_size_wh": [args.width, args.height],
        "metric_depth_range_m": [args.min_depth_m, args.max_depth_m],
        "chunk_scale_stats": {
            "min": float(scales.min()),
            "median": float(np.median(scales)),
            "max": float(scales.max()),
        },
        "min_observations_per_frame": int(min(observation_counts.values())),
        "edge_residual_summary": summarize_edge_residuals(edges),
        "edges": edges,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {k: summary[k] for k in ("frame_count", "chunk_count", "chunk_scale_stats", "edge_residual_summary")},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
