"""VGGT-Long style chunk alignment (tools/chunk_align.py): weighted Umeyama, Huber IRLS, overlap correspondences,
chunk splitting and per-frame chunk selection."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

AXIS = np.array([1.0, 2.0, -0.5]) / np.linalg.norm([1.0, 2.0, -0.5])
ROTATION = Rotation.from_rotvec(np.deg2rad(10.0) * AXIS).as_matrix()  # 10 degrees
SCALE, TRANSLATION = 0.8, np.array([0.3, -0.2, 0.05])


def _points(count, seed=0):
    """A cloud in front of a camera (x, y within +-1 m, z in 0.4..1.6 m)."""
    return np.random.default_rng(seed).uniform([-1.0, -0.7, 0.4], [1.0, 0.7, 1.6], size=(count, 3))


def _similar(points):
    return SCALE * points @ ROTATION.T + TRANSLATION


def _errors(estimate, points):
    """(RMS transfer error of ``points``, rotation error [deg]) against the known similarity."""
    from rgbd_pose_pipeline.se3 import rotation_angle_deg

    scale, rotation, translation = estimate
    transfer = scale * points @ rotation.T + translation - _similar(points)
    return np.sqrt(np.mean(np.sum(transfer**2, axis=1))), float(rotation_angle_deg(rotation.T @ ROTATION))


def _assert_known_similarity(estimate, atol=1e-6):
    scale, rotation, translation = estimate
    assert scale == pytest.approx(SCALE, abs=atol)
    np.testing.assert_allclose(rotation, ROTATION, atol=atol, rtol=0)
    np.testing.assert_allclose(translation, TRANSLATION, atol=atol, rtol=0)


# ---------------------------------------------------------------- weighted Umeyama and Huber IRLS
def test_irls_recovers_a_known_similarity():
    import chunk_align

    source = _points(500)
    weights = np.random.default_rng(1).uniform(1.0, 5.0, 500)
    _assert_known_similarity(chunk_align.irls_sim3(source, _similar(source), weights))
    _assert_known_similarity(chunk_align.weighted_umeyama(source, _similar(source), weights))
    assert np.linalg.det(chunk_align.irls_sim3(source, _similar(source), weights)[1]) == pytest.approx(1.0)


def test_umeyama_weights_the_points():
    import chunk_align

    source = _points(300)
    target = _similar(source)
    target[:60] += 0.3  # corrupted points without weight do not move the fit
    weights = np.r_[np.zeros(60), np.full(240, 2.0)]
    _assert_known_similarity(chunk_align.weighted_umeyama(source, target, weights), atol=1e-9)
    fitted = chunk_align.weighted_umeyama(source[60:], target[60:], np.ones(240))
    for got, want in zip(chunk_align.weighted_umeyama(source, target, weights), fitted, strict=True):
        np.testing.assert_allclose(got, want, atol=1e-12)


def test_huber_weights():
    import chunk_align

    residuals = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
    np.testing.assert_allclose(chunk_align.huber_weights(residuals, 1.0), [1.0, 1.0, 1.0, 0.5, 0.25])


def test_irls_reweights_by_confidence_times_huber_from_the_initial_median(monkeypatch):
    import chunk_align

    rng = np.random.default_rng(2)
    source = _points(200, seed=2)
    target = _similar(source) + rng.normal(scale=0.01, size=source.shape)
    confidence = rng.uniform(1.0, 3.0, 200)
    calls, original = [], chunk_align.weighted_umeyama

    def spy(source_points, target_points, weights):
        calls.append(np.array(weights))
        return original(source_points, target_points, weights)

    monkeypatch.setattr(chunk_align, "weighted_umeyama", spy)
    chunk_align.irls_sim3(source, target, confidence)
    assert len(calls) == 1 + chunk_align.IRLS_ITERATIONS == 6
    np.testing.assert_array_equal(calls[0], confidence)
    residuals = []
    for weights in calls:
        scale, rotation, translation = original(source, target, weights)
        residuals.append(np.linalg.norm(scale * source @ rotation.T + translation - target, axis=1))
    delta = np.median(residuals[0])  # fixed at the median of the initial residuals
    for iteration in range(1, len(calls)):
        want = confidence * chunk_align.huber_weights(residuals[iteration - 1], delta)
        np.testing.assert_allclose(calls[iteration], want, rtol=1e-12)


def test_irls_beats_the_non_robust_fit_with_20_percent_outliers():
    import chunk_align

    rng = np.random.default_rng(3)
    source = _points(1000, seed=3)
    target = _similar(source) + rng.normal(scale=0.002, size=source.shape)
    outliers = rng.choice(1000, 200, replace=False)  # 20 %
    target[outliers] += rng.uniform(0.1, 0.4, size=(200, 1)) * np.array([0.6, -0.3, 0.74])
    weights = rng.uniform(1.0, 2.0, 1000)
    robust = _errors(chunk_align.irls_sim3(source, target, weights), source)
    plain = _errors(chunk_align.weighted_umeyama(source, target, weights), source)
    for got, baseline in zip(robust, plain, strict=True):  # about 0.26 x and 0.3 x the non-robust errors
        assert got < 0.5 * baseline


@pytest.mark.parametrize("count", [0, 2])
def test_too_few_points_are_rejected(count):
    import chunk_align

    source = _points(max(count, 1))[:count]
    with pytest.raises(ValueError, match="point"):
        chunk_align.irls_sim3(source, _similar(source), np.ones(count))


def test_degenerate_weights_or_points_are_rejected():
    import chunk_align

    source = _points(10)
    with pytest.raises(ValueError, match="weight"):
        chunk_align.weighted_umeyama(source, _similar(source), np.zeros(10))
    with pytest.raises(ValueError, match="weight"):
        chunk_align.weighted_umeyama(source, _similar(source), -np.ones(10))
    same = np.repeat(source[:1], 10, axis=0)
    with pytest.raises(ValueError, match="spread"):
        chunk_align.weighted_umeyama(same, _similar(same), np.ones(10))
    line = np.linspace(0.0, 1.0, 10)[:, None] * np.array([1.0, 2.0, 3.0])  # the rotation about it is undetermined
    with pytest.raises(ValueError, match="collinear"):
        chunk_align.weighted_umeyama(line, _similar(line), np.ones(10))


# ---------------------------------------------------------------- overlap correspondences
FRAMES, H, W = 2, 13, 18
GRID = (slice(None), slice(None, None, 4), slice(None, None, 4))


def _overlap(seed=4):
    """(target points, target conf, source points, source conf, valid) of FRAMES overlap frames, exactly similar."""
    rng = np.random.default_rng(seed)
    source = _points(FRAMES * H * W, seed=seed).reshape(FRAMES, H, W, 3)
    valid = rng.uniform(size=(FRAMES, H, W)) > 0.2
    return _similar(source), rng.uniform(2.0, 3.0, (FRAMES, H, W)), source, rng.uniform(1.0, 2.0, (FRAMES, H, W)), valid


def test_correspondences_are_every_fourth_valid_pixel_weighted_by_both_confidences():
    import chunk_align

    target, target_conf, source, source_conf, valid = _overlap()
    got_source, got_target, got_weights = chunk_align.overlap_correspondences(
        target, target_conf, source, source_conf, valid
    )
    keep = valid[GRID]  # weights here lie in 2..6, so nothing falls below 0.1 x median
    np.testing.assert_array_equal(got_source, source[GRID][keep])
    np.testing.assert_array_equal(got_target, target[GRID][keep])
    np.testing.assert_array_equal(got_weights, (target_conf * source_conf)[GRID][keep])
    assert chunk_align.PIXEL_STEP == 4 and chunk_align.MIN_WEIGHT_RATIO == 0.1


def test_weights_below_a_tenth_of_the_median_are_dropped():
    import chunk_align

    target, target_conf, source, source_conf, valid = _overlap()
    valid[:] = True
    low = np.zeros((FRAMES, H, W), bool)
    low[0, 4, ::4] = True  # five grid pixels of frame 0
    source_conf[low] = 0.01  # weight <= 0.03, the median weight is >= 2
    target[low] += 5.0  # and wrong: they would move the fit if they were kept
    got_source, _, got_weights = chunk_align.overlap_correspondences(target, target_conf, source, source_conf, valid)
    weights = (target_conf * source_conf)[GRID]
    kept = weights >= 0.1 * np.median(weights)
    assert kept.sum() == weights.size - 5
    np.testing.assert_array_equal(got_weights, weights[kept])
    np.testing.assert_array_equal(got_source, source[GRID][kept])
    _assert_known_similarity(chunk_align.align_overlap(target, target_conf, source, source_conf, valid)[:3])
    assert chunk_align.align_overlap(target, target_conf, source, source_conf, valid)[3] == weights.size - 5


def test_pixels_off_the_grid_or_invalid_are_not_used():
    import chunk_align

    target, target_conf, source, source_conf, valid = _overlap()
    off_grid = np.ones((FRAMES, H, W), bool)
    off_grid[GRID] = False
    target[off_grid] += 5.0  # confident but wrong, off the 4-pixel grid
    target[~valid] -= 5.0  # on the grid but outside valid_mask
    scale, rotation, translation, count = chunk_align.align_overlap(target, target_conf, source, source_conf, valid)
    _assert_known_similarity((scale, rotation, translation))
    assert count == valid[GRID].sum()


def test_no_valid_overlap_point_is_rejected():
    import chunk_align

    target, target_conf, source, source_conf, valid = _overlap()
    with pytest.raises(ValueError, match="overlap"):
        chunk_align.align_overlap(target, target_conf, source, source_conf, np.zeros_like(valid))
    valid[:] = False
    valid[:, 1:4, 1:4] = True  # valid pixels, but none on the 4-pixel grid
    with pytest.raises(ValueError, match="overlap"):
        chunk_align.overlap_correspondences(target, target_conf, source, source_conf, valid)


def test_correspondence_shapes_must_agree():
    import chunk_align

    target, target_conf, source, source_conf, valid = _overlap()
    with pytest.raises(ValueError, match="shape"):
        chunk_align.overlap_correspondences(target[:1], target_conf, source, source_conf, valid)
    with pytest.raises(ValueError, match="shape"):
        chunk_align.overlap_correspondences(target, target_conf, source, source_conf[..., None], valid)


# ---------------------------------------------------------------- chunk splitting and selection
@pytest.mark.parametrize(
    "frames, chunk, overlap, starts",
    [
        (32, 12, 6, [0, 6, 12, 18, 20]),  # V32: the last chunk is moved back to end at frame 32
        (64, 12, 6, [0, 6, 12, 18, 24, 30, 36, 42, 48, 52]),  # CONF
        (64, 24, 12, [0, 12, 24, 36, 40]),
        (36, 12, 6, [0, 6, 12, 18, 24]),  # ends exactly
        (12, 12, 6, [0]),
        (9, 5, 3, [0, 2, 4]),
    ],
)
def test_chunks_step_by_chunk_minus_overlap_and_the_last_one_ends_at_the_window_end(frames, chunk, overlap, starts):
    import chunk_align

    assert chunk_align.chunk_starts(frames, chunk, overlap) == starts
    assert starts[-1] + chunk == frames


def test_each_frame_comes_from_the_chunk_with_the_closest_centre():
    import chunk_align

    starts = chunk_align.chunk_starts(32, 12, 6)  # centres 5.5, 11.5, 17.5, 23.5, 25.5
    assert chunk_align.chunk_owners(starts, 12, 32) == [0] * 9 + [1] * 6 + [2] * 6 + [3] * 4 + [4] * 7


def test_equally_close_centres_give_the_later_chunk():
    import chunk_align

    starts = chunk_align.chunk_starts(9, 5, 3)  # centres 2, 4, 6: frames 3 and 5 are ties
    assert chunk_align.chunk_owners(starts, 5, 9) == [0, 0, 0, 1, 1, 2, 2, 2, 2]


@pytest.mark.parametrize(
    "frames, chunk, overlap, match",
    [
        (32, 12, 12, "overlap"),
        (32, 12, 0, "overlap"),
        (32, 12, 13, "overlap"),
        (32, 1, 0, "chunk"),
        (32, 12.0, 6, "chunk"),
        (32, 12, True, "overlap"),
        (11, 12, 6, "frames"),
    ],
)
def test_invalid_chunking_is_rejected(frames, chunk, overlap, match):
    import chunk_align

    with pytest.raises(ValueError, match=match):
        chunk_align.chunk_starts(frames, chunk, overlap)


# ---------------------------------------------------------------- applying the similarity to a chunk prediction
def test_similarity_moves_cameras_depth_and_points_and_keeps_the_intrinsics():
    import chunk_align
    import torch

    from omnivggt.utils.geometry import unproject_depth_to_world_points_torch
    from omnivggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri

    size_hw = (6, 8)
    c2w = np.tile(np.eye(4), (3, 1, 1))
    c2w[:, :3, :3] = Rotation.from_rotvec(np.deg2rad(3.0) * _points(3, seed=5)).as_matrix()
    c2w[:, :3, 3] = _points(3, seed=6)
    w2c = torch.from_numpy(np.linalg.inv(c2w)[None, :, :3]).float()
    focal = torch.tensor([7.0, 8.0, 9.0])
    k = torch.zeros(1, 3, 3, 3)
    k[..., 0, 0], k[..., 1, 1], k[..., 0, 2], k[..., 1, 2], k[..., 2, 2] = focal, focal + 1, 4.0, 3.0, 1.0
    depth = torch.rand(1, 3, *size_hw, 1, generator=torch.Generator().manual_seed(0)) + 0.5
    pose_enc = extri_intri_to_pose_encoding(w2c, k, size_hw)
    prediction = {
        "pose_enc": pose_enc,
        "depth": depth,
        "depth_conf": torch.full((1, 3, *size_hw), 2.0),
        "world_points": unproject_depth_to_world_points_torch(depth, *pose_encoding_to_extri_intri(pose_enc, size_hw)),
    }
    moved = chunk_align.transform_prediction(prediction, SCALE, ROTATION, TRANSLATION, size_hw)

    want_c2w = c2w.copy()
    want_c2w[:, :3, :3] = ROTATION @ c2w[:, :3, :3]
    want_c2w[:, :3, 3] = SCALE * c2w[:, :3, 3] @ ROTATION.T + TRANSLATION
    got_w2c, got_k = pose_encoding_to_extri_intri(moved["pose_enc"], size_hw)
    np.testing.assert_allclose(got_w2c[0].double().numpy(), np.linalg.inv(want_c2w)[:, :3], atol=1e-5)
    _, k_before = pose_encoding_to_extri_intri(pose_enc, size_hw)
    torch.testing.assert_close(got_k, k_before)  # the chunk's own intrinsics
    torch.testing.assert_close(moved["pose_enc"][..., 7:], pose_enc[..., 7:])
    torch.testing.assert_close(moved["depth"], SCALE * depth)
    assert moved["depth"].dtype == depth.dtype and moved["pose_enc"].dtype == torch.float32
    torch.testing.assert_close(moved["depth_conf"], prediction["depth_conf"])
    want_points = SCALE * prediction["world_points"].double().numpy() @ ROTATION.T + TRANSLATION
    np.testing.assert_allclose(moved["world_points"].numpy(), want_points, atol=1e-12)
    # the moved points are the unprojection of the moved depth by the moved cameras
    replayed = unproject_depth_to_world_points_torch(moved["depth"], *pose_encoding_to_extri_intri(moved["pose_enc"],
                                                                                                    size_hw))
    np.testing.assert_allclose(replayed.numpy(), moved["world_points"].numpy(), atol=2e-5)
