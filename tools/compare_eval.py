"""Equivalence test of a candidate evaluation against a baseline (``eval_colmap_rgbd`` outputs).

For every condition and metric the candidate aggregate is compared with the baseline aggregate using
the pre-registered rule in the thresholds file; a paired bootstrap 95% CI of the per-sample
difference is reported alongside (the decision uses only the rule). Both evaluations must use the
same anchors.

usage: PYTHONPATH=tools uv run python -m compare_eval --baseline eval_a.json --candidate eval_b.json \
           --thresholds configs/omnivggt_omega/equivalence_thresholds.json --output equivalence.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

EPS = 1e-12  # absorbs float noise exactly at a threshold
PROTOCOL_FIELDS = ("split", "frames", "stride", "resolution_wh", "roots")


def _passes(rule: dict, baseline: float, candidate: float) -> bool:
    kind = rule["type"]
    if kind == "min_delta":
        return candidate - baseline >= rule["value"] - EPS
    if kind == "max_delta":
        return candidate - baseline <= rule["value"] + EPS
    if kind == "max_ratio_plus":
        return candidate <= baseline * rule["ratio"] + rule["plus"] + EPS
    raise ValueError(f"unknown rule type {kind!r}")


def _bootstrap_ci(differences: np.ndarray, resamples: int, seed: int, level: float) -> list[float]:
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(differences), size=(resamples, len(differences)))
    means = differences[index].mean(axis=1)
    tail = (1.0 - level) / 2.0
    return [float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))]


def compare(baseline: dict, candidate: dict, thresholds: dict) -> dict:
    if list(baseline["anchors"]) != list(candidate["anchors"]):
        raise ValueError("baseline and candidate were evaluated on different anchors")
    for field in PROTOCOL_FIELDS:
        if baseline.get(field) != candidate.get(field):
            raise ValueError(
                f"evaluation protocol differs in {field}: {baseline.get(field)!r} != {candidate.get(field)!r}"
            )
    for name, evaluation in (("baseline", baseline), ("candidate", candidate)):
        if [sample["anchor"] for sample in evaluation["samples"]] != list(evaluation["anchors"]):
            raise ValueError(f"{name} samples are not in anchor order")
    boot = thresholds["bootstrap"]
    conditions = {}
    for condition in thresholds["conditions"]:
        rows = {}
        for metric, rule in thresholds["metrics"].items():
            base_value = baseline["aggregate"][condition][metric]
            cand_value = candidate["aggregate"][condition][metric]
            differences = np.array(
                [
                    c[condition][metric] - b[condition][metric]
                    for b, c in zip(baseline["samples"], candidate["samples"], strict=True)
                ]
            )
            rows[metric] = {
                "baseline": base_value,
                "candidate": cand_value,
                "delta": cand_value - base_value,
                "ci95": _bootstrap_ci(differences, boot["resamples"], boot["seed"], boot["level"]),
                "rule": rule,
                "pass": _passes(rule, base_value, cand_value),
            }
        conditions[condition] = rows
    return {"all_pass": all(r["pass"] for rows in conditions.values() for r in rows.values()), "conditions": conditions}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(*(json.loads(p.read_text()) for p in (args.baseline, args.candidate, args.thresholds)))
    result["inputs"] = {
        name: {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for name, path in (("baseline", args.baseline), ("candidate", args.candidate), ("thresholds", args.thresholds))
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    failed = [f"{c}:{m}" for c, rows in result["conditions"].items() for m, r in rows.items() if not r["pass"]]
    print(json.dumps({"all_pass": result["all_pass"], "failed": failed}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
