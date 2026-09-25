import numpy as np
import pytest
from rgbd_pose_pipeline.depth_consistency import pair_depth_consistency

K = np.array([[100.0, 0, 40], [0, 100.0, 30], [0, 0, 1]])


def _plane_depth(z=1.0, shape=(60, 80)):
    return np.full(shape, z)


def test_consistent_poses_give_full_inliers():
    pose_i, pose_j = np.eye(4), np.eye(4)
    pose_j[0, 3] = 0.05  # camera j moved 5 cm along +x; a fronto-parallel plane stays at z=1
    stats = pair_depth_consistency(_plane_depth(), _plane_depth(), K, pose_i, pose_j, stride=4)
    assert stats["inlier_ratio"] == pytest.approx(1.0)
    assert stats["median_abs_error_m"] == pytest.approx(0.0, abs=1e-9)


def test_wrong_depth_offset_is_detected():
    pose_i, pose_j = np.eye(4), np.eye(4)
    pose_j[2, 3] = 0.05  # j moved 5 cm forward: the plane is at 0.95 in j
    good = pair_depth_consistency(_plane_depth(1.0), _plane_depth(0.95), K, pose_i, pose_j, stride=4)
    bad = pair_depth_consistency(_plane_depth(1.0), _plane_depth(0.95), K, pose_i, np.eye(4), stride=4)
    assert good["inlier_ratio"] > 0.99
    assert bad["inlier_ratio"] < 0.01 and bad["median_abs_error_m"] == pytest.approx(0.05, abs=1e-6)
