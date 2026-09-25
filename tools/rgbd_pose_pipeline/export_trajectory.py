"""Write camera_to_world poses as the trajectory JSON consumed by the staging exporter.

Contract (vggt-omega ``prepare_colmap_rgbd_training.py``): ``pose_convention``
``camera_to_world``, ``frame_count``, ``frames[{frame_index 0..N-1, image_name,
camera_to_world 4x4}]`` and ``chunk_scales[{chunk_index, global_indices}]``.
Chunks are non-overlapping blocks of ``chunk_size`` frames; training sequences
never cross a chunk boundary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rgbd_pose_pipeline.se3 import check_poses

RGB_SUFFIX = "_rgb.png"


def build_trajectory(stems: list[str], camera_to_world: np.ndarray, chunk_size: int) -> dict:
    poses = check_poses(camera_to_world, "trajectory poses")
    if len(stems) != len(poses):
        raise ValueError(f"{len(stems)} stems but {len(poses)} poses")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    frames = [
        {"frame_index": index, "image_name": f"{stem}{RGB_SUFFIX}", "camera_to_world": pose.tolist()}
        for index, (stem, pose) in enumerate(zip(stems, poses, strict=True))
    ]
    chunks = [
        {"chunk_index": chunk, "global_indices": list(range(start, min(start + chunk_size, len(frames))))}
        for chunk, start in enumerate(range(0, len(frames), chunk_size))
    ]
    return {"pose_convention": "camera_to_world", "frame_count": len(frames), "frames": frames, "chunk_scales": chunks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--poses-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(args.output_json)
    data = np.load(args.poses_npz)
    payload = build_trajectory([str(s) for s in data["frame_stems"]], data["camera_to_global"], args.chunk_size)
    args.output_json.write_text(json.dumps(payload) + "\n")
    print(json.dumps({"frame_count": payload["frame_count"], "chunks": len(payload["chunk_scales"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
