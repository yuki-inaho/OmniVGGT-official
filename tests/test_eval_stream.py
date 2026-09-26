import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def _gt_c2w(frames, step_m=0.014, seed=0):
    """Camera-to-world poses along a rail: constant steps along x with small rotations."""
    rng = np.random.default_rng(seed)
    out = np.tile(np.eye(4), (frames, 1, 1))
    out[:, :3, :3] = Rotation.from_rotvec(np.deg2rad(0.5) * rng.normal(size=(frames, 3))).as_matrix()
    out[:, 0, 3] = step_m * np.arange(frames)
    out[:, 1, 3] = 0.002 * np.sin(np.arange(frames) / 5.0)
    return out


def _similarity(c2w, scale, rotation, translation):
    out = c2w.copy()
    out[:, :3, :3] = np.einsum("ij,njk->nik", rotation, c2w[:, :3, :3])
    out[:, :3, 3] = scale * c2w[:, :3, 3] @ rotation.T + translation
    return out


def _drifted(c2w, per_frame):
    """Poses whose i-th step length is multiplied by (1 + per_frame * i)."""
    relative = np.einsum("nji,njk->nik", c2w[:-1, :3, :3], c2w[1:, :3, :3])
    steps = np.einsum("nji,nj->ni", c2w[:-1, :3, :3], c2w[1:, :3, 3] - c2w[:-1, :3, 3])
    out = [c2w[0]]
    for i in range(len(steps)):
        step = np.eye(4)
        step[:3, :3], step[:3, 3] = relative[i], steps[i] * (1 + per_frame * i)
        out.append(out[-1] @ step)
    return np.array(out)


def _w2c(c2w):
    return np.linalg.inv(c2w)[:, :3]


ROTATION = Rotation.from_euler("xyz", [10, -20, 30], degrees=True).as_matrix()


def test_exact_similarity_gives_zero_trajectory_errors():
    import stream_metrics

    gt = _gt_c2w(40)
    pred = _similarity(gt, 0.37, ROTATION, np.array([1.0, 2.0, 3.0]))
    assert stream_metrics.ate_first_frame(pred, gt, 1 / 0.37) == pytest.approx(np.zeros(40), abs=1e-9)
    assert stream_metrics.ate_sim3(pred, gt) == pytest.approx(np.zeros(40), abs=1e-9)
    for delta in (1, 8, 16):
        assert stream_metrics.rpe_translation(pred, gt, 1 / 0.37, delta) == pytest.approx(0.0, abs=1e-9)
    assert stream_metrics.pose_scale_ratios(pred, gt, 1 / 0.37) == pytest.approx(np.ones(32))


def test_step_length_drift_shows_in_first_frame_gauge_and_pose_scale_ratio():
    import stream_metrics

    gt = _gt_c2w(64)
    pred = _similarity(_drifted(gt, 0.002), 0.37, ROTATION, np.zeros(3))
    first_frame = stream_metrics.ate_first_frame(pred, gt, 1 / 0.37)
    sim3 = stream_metrics.ate_sim3(pred, gt)
    rmse = lambda errors: np.sqrt(np.mean(errors**2))  # noqa: E731
    assert rmse(first_frame) > 2 * rmse(sim3) > 0
    assert first_frame[0] == pytest.approx(0.0, abs=1e-12)  # frame 1 is the gauge
    ratios = stream_metrics.pose_scale_ratios(pred, gt, 1 / 0.37)
    assert len(ratios) == 64 - stream_metrics.DRIFT_DELTA
    assert np.all(np.diff(ratios) > 0) and ratios[0] > 1.0


def test_ate_sim3_uses_every_frame(monkeypatch):
    import stream_metrics

    gt = _gt_c2w(30)
    pred = _similarity(gt, 0.5, ROTATION, np.zeros(3))
    pred[7, :3, 3] += 0.05  # one outlier camera centre
    seen = {}
    original = stream_metrics.align_pose_sim3

    def spy(source, target, trim_fraction=0.1, **kwargs):
        seen["trim_fraction"] = trim_fraction
        return original(source, target, trim_fraction=trim_fraction, **kwargs)

    monkeypatch.setattr(stream_metrics, "align_pose_sim3", spy)
    errors = stream_metrics.ate_sim3(pred, gt)
    assert seen["trim_fraction"] == 0.0
    _, _, _, aligned = original(pred, gt, trim_fraction=0.0)
    assert errors == pytest.approx(np.linalg.norm(aligned[:, :3, 3] - gt[:, :3, 3], axis=1))


@pytest.mark.parametrize(
    "frames,deltas", [(8, (1,)), (9, (1, 8)), (32, (1, 8, 16)), (63, (1, 8, 16)), (64, (1, 8, 16, 32))]
)
def test_rpe_deltas_the_window_can_hold(frames, deltas):
    import stream_metrics

    assert stream_metrics.rpe_deltas(frames) == deltas


def test_undefined_rpe_and_pose_drift_are_rejected_not_nan():
    import stream_metrics

    gt = _gt_c2w(8)
    with pytest.raises(ValueError, match="delta"):
        stream_metrics.rpe_translation(gt, gt, 1.0, 8)
    with pytest.raises(ValueError, match="delta"):
        stream_metrics.pose_scale_ratios(gt, gt, 1.0)
    still = np.tile(np.eye(4), (12, 1, 1))
    with pytest.raises(ValueError, match="displacement"):
        stream_metrics.pose_scale_ratios(still, still, 1.0)


def _depth_window(frames, seed=0):
    rng = np.random.default_rng(seed)
    gt = rng.uniform(0.5, 1.5, size=(frames, 4, 6))
    mask = rng.uniform(size=gt.shape) > 0.3
    mask[:, 0, 0] = True  # every frame has a valid pixel
    return gt, mask


def test_depth_scales_follow_the_first_frame():
    import stream_metrics

    gt, mask = _depth_window(12)
    drift = 1 + 0.01 * np.arange(12)
    pred = gt / (0.37 * drift)[:, None, None]
    assert stream_metrics.depth_scale(pred[0], gt[0], mask[0]) == pytest.approx(0.37)
    ratios = stream_metrics.depth_scales(pred, gt, mask) / stream_metrics.depth_scale(pred[0], gt[0], mask[0])
    assert ratios == pytest.approx(drift)
    mask[3] = False
    with pytest.raises(ValueError, match="valid"):
        stream_metrics.depth_scales(pred, gt, mask)


def test_time_bins_cover_the_window_once():
    import stream_metrics

    assert stream_metrics.time_bins(8) == [("t1-8", slice(0, 8))]
    assert stream_metrics.time_bins(40) == [("t1-8", slice(0, 8)), ("t9-32", slice(8, 32)), ("t33-64", slice(32, 40))]
    assert [label for label, _ in stream_metrics.time_bins(146)] == ["t1-8", "t9-32", "t33-64", "t65+"]
    assert stream_metrics.time_bins(146)[-1][1] == slice(64, 146)


def test_depth_report_by_time_bin_and_scaling():
    import eval_colmap_rgbd
    import stream_metrics

    gt, mask = _depth_window(70)
    error = np.where(np.arange(70) < 8, 0.0, np.where(np.arange(70) < 32, 0.1, 0.2))  # relative error per frame
    pred = gt * (1 + error)[:, None, None] / 2.0
    s1 = stream_metrics.depth_scale(pred[0], gt[0], mask[0])
    assert s1 == pytest.approx(2.0)
    report = stream_metrics.depth_report(pred, gt, mask, s1)
    assert report["abs_rel_s1@t1-8"] == pytest.approx(0.0, abs=1e-12)
    assert report["abs_rel_s1@t9-32"] == pytest.approx(0.1)
    assert report["abs_rel_s1@t33-64"] == pytest.approx(0.2)
    assert report["abs_rel_s1@t65+"] == pytest.approx(0.2)
    assert report["delta<1.25_s1@t33-64"] == 1.0
    pixels = mask.sum(axis=(1, 2))
    pooled = (pixels * error).sum() / pixels.sum()
    assert report["abs_rel_s1"] == pytest.approx(pooled)
    # the window-median scaling is eval_colmap_rgbd.depth_metrics, bit for bit
    legacy = eval_colmap_rgbd.depth_metrics(pred, gt, mask)
    assert report["abs_rel_median"] == legacy["abs_rel"]
    assert report["delta<1.25_median"] == legacy["delta<1.25"]
    assert {"abs_rel_median@t1-8", "abs_rel_median@t65+", "delta<1.25_median@t9-32"} <= set(report)


def test_window_metrics_are_finite_and_omit_undefined_values():
    import stream_metrics

    gt_c2w = _gt_c2w(32)
    pred_c2w = _similarity(gt_c2w, 0.5, ROTATION, np.zeros(3))
    gt_depth, mask = _depth_window(32)
    result = stream_metrics.window_metrics(_w2c(pred_c2w), _w2c(gt_c2w), gt_depth * 0.5, gt_depth, mask)
    metrics = result["metrics"]
    assert metrics["s1"] == pytest.approx(2.0)
    assert metrics["ate_g1_rmse_mm"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["ate_sim3_rmse_mm"] == pytest.approx(0.0, abs=1e-6)
    assert {"rpe_t_mm@1", "rpe_t_mm@8", "rpe_t_mm@16"} <= set(metrics) and "rpe_t_mm@32" not in metrics
    assert metrics["pose_scale_ratio_max_dev"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["depth_scale_ratio_last"] == pytest.approx(1.0)
    assert metrics["AUC@30"] > 0.99 and "abs_rel_median@t33-64" not in metrics
    assert all(np.isfinite(value) for value in metrics.values())
    assert len(result["series"]["ate_g1_err_mm"]) == 32 and len(result["series"]["depth_scale_ratio"]) == 32
    short = stream_metrics.window_metrics(_w2c(pred_c2w[:8]), _w2c(gt_c2w[:8]), gt_depth[:8], gt_depth[:8], mask[:8])
    assert "rpe_t_mm@8" not in short["metrics"] and "pose_scale_ratio_first" not in short["metrics"]
    assert "pose_scale_ratio" not in short["series"]
