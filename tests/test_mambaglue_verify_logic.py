import numpy as np
import pytest
from rgbd_pose_pipeline.mambaglue_verify import (
    EdgeThresholds,
    edge_passes,
    motion_errors,
    pose_relative_motion,
    rail_axis_in_camera,
    scale_intrinsics,
)
from scipy.spatial.transform import Rotation


def _pose(rotvec, centre):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    pose[:3, 3] = centre
    return pose


def test_pose_relative_motion_maps_source_camera_points_to_target_camera():
    pose_i, pose_j = _pose([0.1, 0.2, 0.0], [0, 0, 0]), _pose([0.1, 0.25, 0.01], [0.3, -0.1, 0.05])
    world = np.array([0.4, 0.2, 2.0])
    in_i = np.linalg.inv(pose_i) @ np.append(world, 1)
    in_j = np.linalg.inv(pose_j) @ np.append(world, 1)
    assert np.allclose(pose_relative_motion(pose_i, pose_j) @ in_i, in_j)


def test_motion_errors_zero_for_identical_and_detect_rotation():
    motion = _pose([0.01, 0.02, 0.0], [0.012, 0.0, 0.001])
    rot, trans, norm = motion_errors(motion, motion)
    assert rot == pytest.approx(0.0, abs=1e-6) and trans == pytest.approx(0.0, abs=1e-12)
    assert norm == pytest.approx(np.linalg.norm([0.012, 0.0, 0.001]))
    rotated = motion.copy()
    rotated[:3, :3] = Rotation.from_rotvec([0, 0, np.radians(1.0)]).as_matrix() @ motion[:3, :3]
    assert motion_errors(rotated, motion)[0] == pytest.approx(1.0, abs=1e-6)


def test_edge_pass_thresholds_are_inclusive_and_relative():
    thresholds = EdgeThresholds(
        min_geometric_inliers=30,
        min_support=20,
        max_rotation_deg=2.0,
        min_translation_tolerance_m=0.01,
        relative_translation_tolerance=0.1,
    )
    ok = {
        "status": "metric",
        "geometric_inlier_count": 30,
        "support_count": 20,
        "rotation_error_deg": 2.0,
        "translation_error_m": 0.01,
        "pose_translation_m": 0.05,
    }
    assert edge_passes(ok, thresholds)
    assert not edge_passes({**ok, "geometric_inlier_count": 29}, thresholds)
    assert not edge_passes({**ok, "rotation_error_deg": 2.01}, thresholds)
    assert edge_passes({**ok, "translation_error_m": 0.02, "pose_translation_m": 0.2}, thresholds)
    assert not edge_passes({**ok, "status": "insufficient_depth"}, thresholds)


def test_rail_axis_in_camera_and_intrinsics_scaling():
    pose = _pose([0.0, np.pi / 2, 0.0], [0, 0, 0])
    axis_camera = rail_axis_in_camera(pose, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(pose[:3, :3] @ axis_camera, [1.0, 0.0, 0.0])
    k = np.array([[554.0, 0, 395.0], [0, 561.0, 295.0], [0, 0, 1]])
    assert np.allclose(scale_intrinsics(k, 0.8), [[443.2, 0, 316.0], [0, 448.8, 236.0], [0, 0, 1]])


class _Estimate:
    status, selected_model, geometric_inlier_count, support_count = "metric", "se3", 40, 30
    se3_rms_m, one_axis_rms_m, reason = 0.004, 0.006, None
    transform = np.eye(4)


def test_measure_edge_records_estimator_errors_explicitly():
    from rgbd_pose_pipeline.mambaglue_verify import measure_edge

    def failing(*args, **kwargs):
        raise RuntimeError("degenerate")

    points = np.zeros((10, 2))
    record = measure_edge(failing, points, points, error_types=(RuntimeError,))
    assert record["status"] == "estimator_error" and record["measured_motion"] is None
    assert "degenerate" in record["reason"]
    ok = measure_edge(lambda *a, **k: _Estimate(), points, points, error_types=(RuntimeError,))
    assert ok["status"] == "metric" and ok["measured_motion"] == np.eye(4).tolist()
    few = measure_edge(failing, points[:5], points[:5], error_types=(RuntimeError,))
    assert few["status"] == "too_few_matches"


def test_summary_reports_translation_scale_ratio():
    from rgbd_pose_pipeline.mambaglue_verify import EdgeThresholds, _compare, _summary

    poses = np.tile(np.eye(4), (3, 1, 1))
    poses[:, 0, 3] = [0.0, 0.01, 0.02]
    measured = np.eye(4)
    measured[0, 3] = -0.012  # 1.2x the pose motion
    base = {
        "status": "metric",
        "selected_model": "translation_1d",
        "geometric_inlier_count": 50,
        "support_count": 20,
        "measured_motion": measured.tolist(),
    }
    edges = [
        _compare({**base, "i": 0, "j": 1, "step": 1}, poses, EdgeThresholds()),
        _compare({**base, "i": 1, "j": 2, "step": 1}, poses, EdgeThresholds()),
    ]
    summary = _summary(edges, 3, EdgeThresholds())
    assert summary["translation_scale_ratio"]["translation_1d"]["median"] == pytest.approx(1.2)
