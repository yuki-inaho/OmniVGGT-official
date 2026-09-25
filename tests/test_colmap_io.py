import numpy as np
import pytest
from rgbd_pose_pipeline.colmap_prior_ba import (
    c2w_to_qvec_tvec,
    parse_model_analyzer,
    qvec_tvec_to_c2w,
    read_images_txt,
    umeyama_sim3,
    write_images_txt,
)
from scipy.spatial.transform import Rotation


def _random_c2w(n, seed=0):
    rng = np.random.default_rng(seed)
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, :3, :3] = Rotation.random(n, random_state=seed).as_matrix()
    poses[:, :3, 3] = rng.normal(size=(n, 3))
    return poses


def test_qvec_roundtrip_uses_world_to_camera_convention():
    c2w = _random_c2w(5)
    for pose in c2w:
        qvec, tvec = c2w_to_qvec_tvec(pose)
        w2c_rotation = Rotation.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()
        assert np.allclose(w2c_rotation, pose[:3, :3].T)
        assert np.allclose(tvec, -pose[:3, :3].T @ pose[:3, 3])
        assert np.allclose(qvec_tvec_to_c2w(qvec, tvec), pose)


def test_images_txt_roundtrip(tmp_path):
    c2w = _random_c2w(3, seed=1)
    records = [(i + 1, f"{i:08d}_rgb.png", 1, c2w[i]) for i in range(3)]
    write_images_txt(tmp_path / "images.txt", records)
    loaded = read_images_txt(tmp_path / "images.txt")
    assert sorted(loaded) == [r[1] for r in records]
    for _, name, _, pose in records:
        assert np.allclose(loaded[name], pose, atol=1e-9)


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(3)
    source = rng.normal(size=(50, 3))
    rotation = Rotation.random(random_state=4).as_matrix()
    target = 2.5 * source @ rotation.T + np.array([1.0, -2.0, 0.5])
    scale, rot, trans = umeyama_sim3(source, target)
    assert scale == pytest.approx(2.5)
    assert np.allclose(rot, rotation)
    assert np.allclose(trans, [1.0, -2.0, 0.5])


def test_parse_model_analyzer():
    text = """I20260925 Rigs: 1
I20260925 Cameras: 1
I20260925 Images: 1000
I20260925 Registered images: 998
I20260925 Points: 12345
I20260925 Observations: 67890
I20260925 Mean track length: 5.5
I20260925 Mean observations per image: 67.9
I20260925 Mean reprojection error: 0.61px
"""
    stats = parse_model_analyzer(text)
    assert stats["registered_images"] == 998
    assert stats["points"] == 12345
    assert stats["mean_reprojection_error_px"] == pytest.approx(0.61)


def test_pose_sim3_is_well_posed_for_collinear_centres():
    from rgbd_pose_pipeline.colmap_prior_ba import align_pose_sim3

    n = 200
    prior = np.tile(np.eye(4), (n, 1, 1))
    prior[:, :3, :3] = Rotation.from_euler("y", np.linspace(0, 3, n), degrees=True).as_matrix()
    prior[:, :3, 3] = np.outer(np.linspace(0, 10, n), [0.0, -1.0, 0.1])  # collinear rail
    true_rotation = Rotation.from_rotvec([0.3, -2.0, 0.5]).as_matrix()
    true_scale, true_t = 0.7, np.array([1.0, 2.0, 3.0])
    # ba = inverse similarity applied to prior
    ba = prior.copy()
    ba[:, :3, :3] = true_rotation.T @ prior[:, :3, :3]
    ba[:, :3, 3] = ((prior[:, :3, 3] - true_t) @ true_rotation) / true_scale
    ba[::20, :3, 3] += 5.0  # a few outliers
    scale, rotation, _translation, aligned = align_pose_sim3(ba, prior, trim_fraction=0.2)
    assert scale == pytest.approx(true_scale, rel=1e-6)
    assert np.allclose(rotation, true_rotation, atol=1e-6)
    inliers = np.ones(n, bool)
    inliers[::20] = False
    assert np.allclose(aligned[inliers], prior[inliers], atol=1e-6)


def test_depth_scale_from_observations_is_robust_median():
    from rgbd_pose_pipeline.colmap_prior_ba import depth_scale_from_observations

    rng = np.random.default_rng(0)
    z_model = rng.uniform(0.5, 1.2, 5000)
    measured = 1.18 * z_model * (1 + rng.normal(0, 0.01, 5000))
    measured[:300] = 0.0  # invalid depth
    measured[300:600] *= 3.0  # outliers (e.g. background seen through a hole)
    stats = depth_scale_from_observations(z_model, measured)
    assert stats["scale"] == pytest.approx(1.18, rel=0.01)
    assert stats["valid_observations"] == 4700


def test_align_with_fixed_scale_keeps_rotation_and_scale():
    from rgbd_pose_pipeline.colmap_prior_ba import align_pose_sim3, align_with_fixed_scale

    prior = _random_c2w(50, seed=3)
    rotation = Rotation.random(random_state=6).as_matrix()
    ba = prior.copy()
    ba[:, :3, :3] = rotation.T @ prior[:, :3, :3]
    ba[:, :3, 3] = (prior[:, :3, 3] @ rotation) / 2.0
    _, rot, _, _ = align_pose_sim3(ba, prior)
    aligned = align_with_fixed_scale(ba, prior, scale=2.0, rotation=rot)
    assert np.allclose(aligned, prior, atol=1e-6)
    rescaled = align_with_fixed_scale(ba, prior, scale=2.4, rotation=rot)
    steps = np.linalg.norm(np.diff(rescaled[:, :3, 3], axis=0), axis=1)
    assert np.allclose(steps, 1.2 * np.linalg.norm(np.diff(prior[:, :3, 3], axis=0), axis=1))
