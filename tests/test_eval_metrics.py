import numpy as np
import pytest
from eval_colmap_rgbd import depth_metrics, pose_metrics
from scipy.spatial.transform import Rotation


def _w2c(n, seed=0):
    rng = np.random.default_rng(seed)
    out = np.zeros((n, 3, 4))
    out[:, :, :3] = Rotation.random(n, random_state=seed).as_matrix()
    out[:, :, 3] = rng.normal(size=(n, 3))
    return out


def test_pose_metrics_are_zero_for_identical_and_invariant_to_similarity():
    gt = _w2c(6)
    metrics = pose_metrics(gt, gt)
    assert metrics["rot_err_mean_deg"] == pytest.approx(0, abs=1e-5)
    assert metrics["RRA@5"] == 1.0 and metrics["RTA@5"] == 1.0 and metrics["AUC@30"] > 0.99
    # a global similarity of the world does not change relative-pose errors
    s, r, t = 3.0, Rotation.random(random_state=5).as_matrix(), np.array([1.0, 2.0, 3.0])
    moved = gt.copy()
    moved[:, :, :3] = gt[:, :, :3] @ r.T
    moved[:, :, 3] = s * gt[:, :, 3] - (moved[:, :, :3] @ t)
    assert pose_metrics(moved, gt)["trans_dir_err_mean_deg"] == pytest.approx(0, abs=1e-5)


def test_depth_metrics_use_median_scale():
    gt = np.full((2, 4, 4), 2.0)
    mask = np.ones_like(gt, dtype=bool)
    metrics = depth_metrics(gt * 0.5, gt, mask)
    assert metrics["abs_rel"] == pytest.approx(0.0) and metrics["delta<1.25"] == 1.0
