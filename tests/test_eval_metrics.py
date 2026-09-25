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


def test_model_from_config_dispatch(monkeypatch, tmp_path):
    import eval_colmap_rgbd

    import omnivggt.models.omnivggt as omnivggt_module
    from omnivggt.models.omnivggt_omega import OmniVGGTOmega

    class FakeOmniVGGT:
        pass

    built = {}
    monkeypatch.setattr(omnivggt_module, "OmniVGGT", FakeOmniVGGT)
    monkeypatch.setattr(OmniVGGTOmega, "from_variant", staticmethod(lambda path: built.setdefault("path", path)))
    assert isinstance(eval_colmap_rgbd._build_model(None), FakeOmniVGGT)
    variant = tmp_path / "V9.json"
    variant.write_text('{"name": "V9"}')
    eval_colmap_rgbd._build_model(variant)
    assert built["path"] == variant


def test_model_config_provenance(tmp_path):
    import eval_colmap_rgbd

    assert eval_colmap_rgbd._model_config_record(None) is None
    variant = tmp_path / "V9.json"
    variant.write_text('{"name": "V9", "global_rope": false}')
    record = eval_colmap_rgbd._model_config_record(variant)
    assert record["path"] == str(variant) and len(record["sha256"]) == 64
    assert record["variant"] == {"name": "V9", "global_rope": False}


def test_variant_must_match_training_run(tmp_path):
    import json

    import eval_colmap_rgbd

    run = tmp_path / "run"
    checkpoint = run / "omnivggt-omega-colmap-rgbd" / "final_checkpoint"
    checkpoint.mkdir(parents=True)
    trained, other = tmp_path / "V4.json", tmp_path / "V2.json"
    trained.write_text('{"name": "V4"}')
    other.write_text('{"name": "V2"}')
    (run / "weight_transfer_report.json").write_text(json.dumps({"variant": {"sha256": eval_colmap_rgbd._sha256(trained)}}))
    assert eval_colmap_rgbd._check_variant_provenance(trained, checkpoint) == "verified"
    with pytest.raises(ValueError, match="variant"):
        eval_colmap_rgbd._check_variant_provenance(other, checkpoint)
    assert eval_colmap_rgbd._check_variant_provenance(trained, tmp_path / "init.safetensors") == "unverified"
    assert eval_colmap_rgbd._check_variant_provenance(None, checkpoint) is None
