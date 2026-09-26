import copy
import json
from pathlib import Path

import compare_eval
import pytest

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["rot_err_mean_deg", "trans_dir_err_mean_deg", "RRA@5", "RTA@5", "AUC@30", "abs_rel", "delta<1.25"]
CONDITIONS = ["rgb", "depth", "camera", "depth+camera"]


def _eval(values: dict, anchors=(0, 5)) -> dict:
    samples = [{"anchor": a, **{c: dict(values) for c in CONDITIONS}} for a in anchors]
    return {"anchors": list(anchors), "aggregate": {c: dict(values) for c in CONDITIONS}, "samples": samples}


BASE = {
    "rot_err_mean_deg": 0.2,
    "trans_dir_err_mean_deg": 1.0,
    "RRA@5": 1.0,
    "RTA@5": 1.0,
    "AUC@30": 0.95,
    "abs_rel": 0.05,
    "delta<1.25": 0.95,
}


def _thresholds():
    return json.loads((ROOT / "configs" / "omnivggt_omega" / "equivalence_thresholds.json").read_text())


def _shifted(**deltas):
    values = dict(BASE)
    for key, delta in deltas.items():
        values[key] = BASE[key] + delta
    return _eval(values)


@pytest.mark.parametrize(
    "metric, at_limit, beyond",
    [
        ("AUC@30", -0.010, -0.0100001),
        ("RTA@5", -0.020, -0.0200001),
        ("rot_err_mean_deg", 0.05, 0.0500001),
        ("trans_dir_err_mean_deg", 0.30, 0.3000001),
        ("abs_rel", 0.05 * 0.10 + 0.002, 0.05 * 0.10 + 0.0020001),
        ("delta<1.25", -0.010, -0.0100001),
    ],
)
def test_threshold_boundaries(metric, at_limit, beyond):
    baseline = _eval(BASE)
    ok = compare_eval.compare(baseline, _shifted(**{metric: at_limit}), _thresholds())
    bad = compare_eval.compare(baseline, _shifted(**{metric: beyond}), _thresholds())
    assert ok["all_pass"] and ok["conditions"]["rgb"][metric]["pass"]
    assert not bad["all_pass"] and not bad["conditions"]["depth"][metric]["pass"]


def test_anchor_mismatch_raises():
    with pytest.raises(ValueError, match="anchors"):
        compare_eval.compare(_eval(BASE), _eval(BASE, anchors=(0, 6)), _thresholds())


def test_missing_samples_raises():
    candidate = copy.deepcopy(_eval(BASE))
    del candidate["samples"][0]["camera"]
    with pytest.raises(KeyError):
        compare_eval.compare(_eval(BASE), candidate, _thresholds())


def test_bootstrap_ci_deterministic():
    anchors = (0, 1, 2, 3)
    baseline = _eval(BASE, anchors=anchors)
    candidate = _eval(BASE, anchors=anchors)
    for sample, value in zip(candidate["samples"], (0.946, 0.93, 0.946, 0.946), strict=True):
        sample["rgb"]["AUC@30"] = value
    candidate["aggregate"]["rgb"]["AUC@30"] = sum(s["rgb"]["AUC@30"] for s in candidate["samples"]) / len(anchors)
    first = compare_eval.compare(baseline, candidate, _thresholds())
    second = compare_eval.compare(baseline, candidate, _thresholds())
    result = first["conditions"]["rgb"]["AUC@30"]
    assert result["ci95"] == second["conditions"]["rgb"]["AUC@30"]["ci95"]
    assert result["ci95"][0] <= result["delta"] <= result["ci95"][1]
    assert result["delta"] == pytest.approx(-0.008)


@pytest.mark.parametrize(
    "field, value", [("split", "smoke"), ("frames", 6), ("stride", 2), ("resolution_wh", [518, 392]), ("roots", ["x"])]
)
def test_protocol_mismatch_raises(field, value):
    baseline = {
        **_eval(BASE),
        "split": "val",
        "frames": 8,
        "stride": 3,
        "resolution_wh": [392, 294],
        "roots": ["a", "b"],
    }
    candidate = {**copy.deepcopy(baseline), field: value}
    with pytest.raises(ValueError, match=field):
        compare_eval.compare(baseline, candidate, _thresholds())


def test_samples_must_follow_anchor_order():
    candidate = _eval(BASE)
    candidate["samples"] = candidate["samples"][::-1]
    with pytest.raises(ValueError, match="order"):
        compare_eval.compare(_eval(BASE), candidate, _thresholds())


def test_rgbd_thresholds_match_preregistered():
    """The RGB-D judgement uses the pre-registered numbers, restricted to the conditions with depth input."""
    registered = json.loads((ROOT / "configs/omnivggt_omega/equivalence_thresholds.json").read_text())
    rgbd = json.loads((ROOT / "configs/omnivggt_omega/equivalence_thresholds_rgbd.json").read_text())
    assert rgbd["metrics"] == registered["metrics"]
    assert rgbd["bootstrap"] == registered["bootstrap"]
    assert rgbd["conditions"] == ["depth", "depth+camera"]
    assert set(rgbd["conditions"]) <= set(registered["conditions"])
    result = compare_eval.compare(_eval(BASE), _eval(BASE), rgbd)
    assert set(result["conditions"]) == {"depth", "depth+camera"} and result["all_pass"]
