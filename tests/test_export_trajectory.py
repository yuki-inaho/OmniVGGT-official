import json

import numpy as np
import pytest
from rgbd_pose_pipeline.export_trajectory import build_trajectory


def _poses(n):
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 0, 3] = np.arange(n) * 0.01
    return poses


def test_trajectory_contract_and_non_overlapping_chunks():
    stems = [f"{i:08d}" for i in range(1, 2 * 250, 2)]
    payload = build_trajectory(stems, _poses(len(stems)), chunk_size=100)
    assert payload["pose_convention"] == "camera_to_world"
    assert payload["frame_count"] == 250
    assert [f["frame_index"] for f in payload["frames"]] == list(range(250))
    assert payload["frames"][1]["image_name"] == "00000003_rgb.png"
    chunks = [c["global_indices"] for c in payload["chunk_scales"]]
    assert [len(c) for c in chunks] == [100, 100, 50]
    assert sorted(i for c in chunks for i in c) == list(range(250))
    json.dumps(payload)


def test_rejects_invalid_poses():
    poses = _poses(3)
    poses[1, :3, :3] *= 2.0
    with pytest.raises(ValueError):
        build_trajectory(["a", "b", "c"], poses, chunk_size=2)


def test_rejects_length_mismatch():
    with pytest.raises(ValueError):
        build_trajectory(["a", "b"], _poses(3), chunk_size=2)
