import numpy as np
import pytest
from rgbd_pose_pipeline.lane_refine import fit_rail, refine_to_rail
from scipy.spatial.transform import Rotation


def _poses(centers, seed=0):
    rng = np.random.default_rng(seed)
    poses = np.tile(np.eye(4), (len(centers), 1, 1))
    poses[:, :3, :3] = Rotation.from_rotvec(rng.normal(0, 0.01, (len(centers), 3))).as_matrix()
    poses[:, :3, 3] = centers
    return poses


def _line(n=400, noise=0.003, seed=0):
    rng = np.random.default_rng(seed)
    axis = np.array([0.1, -1.0, 0.05])
    axis /= np.linalg.norm(axis)
    s = np.linspace(0.0, 6.0, n)
    return np.array([0.2, 0.5, -0.1]) + s[:, None] * axis + rng.normal(0, noise, (n, 3)), axis


def test_fit_rail_recovers_axis_and_rejects_outliers():
    centers, axis = _line()
    centers[::50] += np.array([0.3, 0.0, 0.3])
    fit = fit_rail(centers)
    assert np.degrees(np.arccos(abs(fit.axis @ axis))) < 0.5
    assert fit.axis @ (centers[-1] - centers[0]) > 0
    assert not fit.inliers[::50].any()
    assert fit.inliers.mean() > 0.95


def test_refine_scales_cross_rail_residual_and_keeps_rotation():
    centers, _ = _line()
    poses = _poses(centers)
    refined, summary = refine_to_rail(poses, projection_strength=0.9)
    assert np.array_equal(refined[:, :3, :3], poses[:, :3, :3])
    assert summary["cross_rail_rms_after_m"] == pytest.approx(0.1 * summary["cross_rail_rms_before_m"], rel=1e-9)
    assert summary["one_axis_adopted"] is True
    assert set(summary) >= {"rail_axis", "rail_centroid_m", "rail_inlier_count", "linearity_ratio"}


def test_curved_trajectory_is_not_adopted():
    t = np.linspace(0, np.pi, 300)
    centers = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=1) * 3.0
    _, summary = refine_to_rail(_poses(centers), projection_strength=0.9)
    assert summary["one_axis_adopted"] is False


def test_invalid_strength_is_rejected():
    centers, _ = _line()
    with pytest.raises(ValueError):
        refine_to_rail(_poses(centers), projection_strength=1.0)
