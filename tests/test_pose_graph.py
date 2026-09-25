import numpy as np
from rgbd_pose_pipeline.pose_graph import PoseGraphConfig, optimize_pose_graph
from rgbd_pose_pipeline.se3 import relative, rotation_angle_deg
from scipy.spatial.transform import Rotation

# Weak priors: the optimiser itself must recover the truth from the edges.
WEAK_PRIOR = PoseGraphConfig(prior_rotation_deg=5.0, prior_translation_m=0.5)


def _truth(n=120):
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, :3, :3] = Rotation.from_rotvec(np.column_stack([np.zeros(n), np.full(n, 1.2), np.zeros(n)])).as_matrix()
    poses[:, :3, 3] = np.outer(np.arange(n) * 0.012, [0.05, -1.0, 0.02])
    return poses


def _edges(truth, rng, steps=(1, 2, 4), outlier_every=0):
    edges = []
    for step in steps:
        for i in range(len(truth) - step):
            motion = relative(truth[i], truth[i + step])
            noisy = motion.copy()
            noisy[:3, :3] = Rotation.from_rotvec(rng.normal(0, np.radians(0.02), 3)).as_matrix() @ motion[:3, :3]
            noisy[:3, 3] += rng.normal(0, 0.0005, 3)
            if outlier_every and len(edges) % outlier_every == 0:
                noisy[:3, 3] += 0.08
            edges.append(
                {
                    "i": i,
                    "j": i + step,
                    "status": "metric",
                    "geometric_inlier_count": 100,
                    "support_count": 80,
                    "measured_motion": noisy.tolist(),
                }
            )
    return edges


def _perturb(truth, rng):
    init = truth.copy()
    init[1:, :3, :3] = (
        Rotation.from_rotvec(rng.normal(0, np.radians(0.8), (len(truth) - 1, 3))).as_matrix() @ truth[1:, :3, :3]
    )
    init[1:, :3, 3] += rng.normal(0, 0.02, (len(truth) - 1, 3))
    return init


def _errors(poses, truth):
    rot = rotation_angle_deg(np.einsum("nji,njk->nik", truth[:, :3, :3], poses[:, :3, :3]))
    return np.median(rot), np.median(np.linalg.norm(poses[:, :3, 3] - truth[:, :3, 3], axis=1))


def test_pose_graph_recovers_truth_from_perturbed_initialisation():
    rng = np.random.default_rng(0)
    truth = _truth()
    init = _perturb(truth, rng)
    final, summary = optimize_pose_graph(init, _edges(truth, rng), rail=None, config=WEAK_PRIOR)
    before, after = _errors(init, truth), _errors(final, truth)
    assert after[0] < 0.25 * before[0]
    assert after[1] < 0.5 * before[1]
    assert np.allclose(final[0], init[0])
    assert summary["edges_used"] > 0


def test_pose_graph_is_robust_to_outlier_edges():
    rng = np.random.default_rng(1)
    truth = _truth()
    init = _perturb(truth, rng)
    final, summary = optimize_pose_graph(init, _edges(truth, rng, outlier_every=15), rail=None, config=WEAK_PRIOR)
    assert _errors(final, truth)[1] < 0.5 * _errors(init, truth)[1]
    assert summary["downweighted_edges"] > 0


def test_rail_prior_pulls_centres_to_line_without_along_constraint():
    rng = np.random.default_rng(2)
    truth = _truth()
    init = truth.copy()
    init[1:, :3, 3] += np.outer(np.ones(len(truth) - 1), [0.03, 0.0, 0.0])  # cross-rail offset
    axis = truth[-1, :3, 3] - truth[0, :3, 3]
    rail = {"centroid": truth[:, :3, 3].mean(0), "axis": axis / np.linalg.norm(axis)}
    _final, summary = optimize_pose_graph(init, _edges(truth, rng), rail=rail, config=PoseGraphConfig())
    assert summary["rail_cross_rms_after_m"] < 0.5 * summary["rail_cross_rms_before_m"]
