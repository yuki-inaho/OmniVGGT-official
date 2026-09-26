"""Selection and confirmation tables from ``eval_stream`` results (pre-registered rules).

The primary metric is ATE_g1 RMSE [mm] (lower is better). A result's value per session is the mean over
that session's windows; its overall value is the mean of the session values.

* Selection (``--stage NAME``, SEL windows of the smoke split only): candidates are ordered by KV bytes
  (the largest analytic KV of any stream step; equal KV bytes keep the listed order) and the first one
  is the reference. Another candidate qualifies only if its overall value is at least 5% lower than
  the reference's and it is lower in every session. The qualifying candidate with the lowest overall
  value is selected; if none qualifies, the reference is.
* Confirmation (``--confirm``): one row per result, and for each ``--compare A:B`` the difference rule:
  A and B differ only if their overall values differ by at least 5% of B's and the difference has the
  same sign in every session; otherwise there is no difference.

usage: PYTHONPATH=tools uv run python -m compare_stream (--stage CA | --confirm) \
           --results NAME=RESULT.json [NAME=RESULT.json ...] [--compare A:B ...] --output PREFIX
(writes PREFIX.json and PREFIX.md)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from stream_metrics import mean_over_windows

PRIMARY = "ate_g1_rmse_mm"
SECONDARY = ("ate_sim3_rmse_mm", "abs_rel_s1")
THRESHOLD = 0.05
EPS = 1e-12  # absorbs float noise exactly at the threshold
SELECTION_SPLIT = "smoke"
PROTOCOL_FIELDS = ("split", "roots", "resolution_wh")


def _mean(sessions: dict) -> float:
    return float(np.mean(list(sessions.values())))


def _efficiencies(result: dict, condition: str) -> list[dict]:
    return [window["conditions"][condition]["efficiency"] for window in result["windows"]]


def session_means(result: dict, condition: str = "depth", metric: str = PRIMARY) -> dict:
    """Mean of ``metric`` over the windows of each session."""
    means = mean_over_windows(result["windows"], condition)
    means.pop("all")
    missing = sorted(session for session, values in means.items() if metric not in values)
    if missing:
        raise ValueError(f"{metric} is missing in sessions {missing}")
    return {session: means[session][metric] for session in sorted(means)}


def kv_bytes(result: dict, condition: str = "depth") -> int:
    """The largest analytic KV bytes of any stream step of any window."""
    values = [efficiency.get("kv_bytes_max") for efficiency in _efficiencies(result, condition)]
    if None in values:
        raise ValueError("a result without KV bytes (not a stream result) cannot be ordered by KV bytes")
    return max(values)


def check_protocol(results: dict) -> None:
    """All results must come from the same split, roots, resolution, windows and precision."""
    (first_name, first), *others = results.items()
    for name, result in others:
        for field in PROTOCOL_FIELDS:
            if result[field] != first[field]:
                raise ValueError(f"{name} and {first_name} differ in {field}: {result[field]!r} != {first[field]!r}")
        if [w["spec"] for w in result["windows"]] != [w["spec"] for w in first["windows"]]:
            raise ValueError(f"{name} and {first_name} were evaluated on different windows")
        precision, first_precision = result["provenance"]["precision"]["name"], first["provenance"]["precision"]["name"]
        if precision != first_precision:
            raise ValueError(f"{name} and {first_name} differ in precision: {precision} != {first_precision}")


def select(candidates: list[dict], threshold: float = THRESHOLD) -> dict:
    """The selection rule over candidates ``{"name", "kv_bytes", "sessions": {session: value}}``."""
    names = [candidate["name"] for candidate in candidates]
    if not candidates or len(set(names)) != len(names):
        raise ValueError(f"candidates need distinct names, got {names}")
    sessions = set(candidates[0]["sessions"])
    for candidate in candidates:
        if set(candidate["sessions"]) != sessions:
            raise ValueError(
                f"{candidate['name']} covers sessions {sorted(candidate['sessions'])}, not {sorted(sessions)}"
            )
    ordered = sorted(candidates, key=lambda candidate: candidate["kv_bytes"])
    reference = ordered[0]
    reference_mean = _mean(reference["sessions"])
    rows = []
    for candidate in ordered:
        mean = _mean(candidate["sessions"])
        improvement = (reference_mean - mean) / reference_mean
        same_direction = all(candidate["sessions"][s] < reference["sessions"][s] for s in sessions)
        qualifies = candidate is not reference and same_direction and improvement >= threshold - EPS
        rows.append(
            {
                **candidate,
                "mean": mean,
                "improvement": improvement,
                "same_direction": same_direction,
                "qualifies": qualifies,
            }
        )
    qualified = [row for row in rows if row["qualifies"]]
    selected = min(qualified, key=lambda row: row["mean"]) if qualified else rows[0]
    return {"reference": reference["name"], "selected": selected["name"], "threshold": threshold, "rows": rows}


def differs(a: dict, b: dict, threshold: float = THRESHOLD) -> dict:
    """The difference rule between session values ``a`` and ``b`` (relative to ``b``)."""
    if set(a) != set(b):
        raise ValueError(f"cannot compare sessions {sorted(a)} with {sorted(b)}")
    a_mean, b_mean = _mean(a), _mean(b)
    relative = (a_mean - b_mean) / b_mean
    directions = {(a[s] > b[s]) - (a[s] < b[s]) for s in a}
    same_direction = len(directions) == 1 and 0 not in directions
    verdict = same_direction and abs(relative) >= threshold - EPS
    return {
        "a_mean": a_mean,
        "b_mean": b_mean,
        "relative_difference": relative,
        "same_direction": same_direction,
        "differs": verdict,
        "better": ("a" if relative < 0 else "b") if verdict else None,
    }


def selection_table(results: dict, stage: str, condition: str = "depth") -> dict:
    for name, result in results.items():
        if result["split"] != SELECTION_SPLIT:
            raise ValueError(
                f"selection uses the {SELECTION_SPLIT} split (SEL); {name} is on the {result['split']} split"
            )
    check_protocol(results)
    candidates = [
        {"name": name, "kv_bytes": kv_bytes(result, condition), "sessions": session_means(result, condition)}
        for name, result in results.items()
    ]
    return {"stage": stage, "condition": condition, "metric": PRIMARY, **select(candidates)}


def _confirmation_row(name: str, result: dict, condition: str) -> dict:
    sessions = session_means(result, condition)
    row = {
        "name": name,
        "mode": result["provenance"]["mode"],
        "policy": result["provenance"]["policy"],
        "sessions": sessions,
        "mean": _mean(sessions),
    }
    for metric in SECONDARY:
        row[metric] = _mean(session_means(result, condition, metric))
    row["per_frame_ms"] = float(np.mean([e["per_frame_ms"] for e in _efficiencies(result, condition)]))
    if row["mode"] == "stream":
        row["kv_bytes"] = kv_bytes(result, condition)
    return row


def confirmation_table(results: dict, comparisons: list[tuple[str, str]], condition: str = "depth") -> dict:
    check_protocol(results)
    for a, b in comparisons:
        if a not in results or b not in results:
            raise ValueError(f"comparison {a}:{b} names a result that was not given ({sorted(results)})")
    rows = [_confirmation_row(name, result, condition) for name, result in results.items()]
    sessions = {row["name"]: row["sessions"] for row in rows}
    return {
        "condition": condition,
        "metric": PRIMARY,
        "threshold": THRESHOLD,
        "rows": rows,
        "comparisons": [{"a": a, "b": b, **differs(sessions[a], sessions[b])} for a, b in comparisons],
    }


def _table(header: list[str], align: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"] + [
        "| " + " | ".join(row) + " |" for row in rows
    ]


def _yes(flag: bool) -> str:
    return "yes" if flag else "no"


def selection_markdown(table: dict) -> str:
    sessions = sorted(table["rows"][0]["sessions"])
    rows = [
        [
            row["name"],
            str(row["kv_bytes"]),
            *(f"{row['sessions'][s]:.3f}" for s in sessions),
            f"{row['mean']:.3f}",
            f"{100 * row['improvement']:.1f}%",
            _yes(row["same_direction"]),
            _yes(row["qualifies"]),
        ]
        for row in table["rows"]
    ]
    header = ["candidate", "KV bytes", *sessions, "mean", "improvement", "lower in every session", "qualifies"]
    lines = [
        f"## Selection {table['stage']} ({table['condition']}, {table['metric']})",
        "",
        f"reference: {table['reference']}; selected: {table['selected']}",
        "",
        *_table(header, [":---", "---:", *("---:" for _ in sessions), "---:", "---:", ":---:", ":---:"], rows),
    ]
    return "\n".join(lines) + "\n"


def confirmation_markdown(table: dict) -> str:
    sessions = sorted(table["rows"][0]["sessions"])
    rows = [
        [
            row["name"],
            row["mode"],
            *(f"{row['sessions'][s]:.3f}" for s in sessions),
            f"{row['mean']:.3f}",
            *(f"{row[metric]:.4f}" for metric in SECONDARY),
            f"{row['per_frame_ms']:.1f}",
            str(row["kv_bytes"]) if "kv_bytes" in row else "-",
        ]
        for row in table["rows"]
    ]
    header = ["row", "mode", *sessions, "mean", *SECONDARY, "ms/frame", "KV bytes"]
    align = [":---", ":---", *("---:" for _ in sessions), "---:", *("---:" for _ in SECONDARY), "---:", "---:"]
    comparisons = [
        [
            c["a"],
            c["b"],
            f"{c['a_mean']:.3f}",
            f"{c['b_mean']:.3f}",
            f"{100 * c['relative_difference']:+.1f}%",
            _yes(c["same_direction"]),
            f"difference ({c[c['better']]} lower)" if c["differs"] else "no difference",
        ]
        for c in table["comparisons"]
    ]
    lines = [
        f"## Confirmation ({table['condition']}, {table['metric']})",
        "",
        *_table(header, align, rows),
        "",
        "### Comparisons",
        "",
        *_table(
            ["A", "B", "A mean", "B mean", "A vs B", "same sign in every session", "verdict"],
            [":---", ":---", "---:", "---:", "---:", ":---:", ":---"],
            comparisons,
        ),
    ]
    return "\n".join(lines) + "\n"


def _named_paths(values: list[str]) -> dict:
    named = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"--results takes NAME=PATH, got {value!r}")
        if name in named:
            raise ValueError(f"result name {name!r} is given twice")
        named[name] = Path(path)
    return named


def _pairs(values: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for value in values:
        a, separator, b = value.partition(":")
        if not separator or not a or not b:
            raise ValueError(f"--compare takes A:B, got {value!r}")
        pairs.append((a, b))
    return pairs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    kind = parser.add_mutually_exclusive_group(required=True)
    kind.add_argument("--stage", help="selection stage name (e.g. CA)")
    kind.add_argument("--confirm", action="store_true", help="confirmation table and --compare verdicts")
    parser.add_argument("--results", nargs="+", required=True, metavar="NAME=RESULT.json")
    parser.add_argument("--compare", nargs="+", default=[], metavar="A:B", help="--confirm only")
    parser.add_argument("--condition", choices=["depth", "rgb"], default="depth")
    parser.add_argument("--output", type=Path, required=True, help="writes OUTPUT.json and OUTPUT.md")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    paths = _named_paths(args.results)
    if args.stage is not None and args.compare:
        raise ValueError("--compare applies to --confirm only")
    outputs = [args.output.parent / f"{args.output.name}{suffix}" for suffix in (".json", ".md")]
    for output in outputs:
        if output.exists():
            raise FileExistsError(output)
    results = {name: json.loads(path.read_text()) for name, path in paths.items()}
    if args.stage is not None:
        table = selection_table(results, args.stage, args.condition)
        markdown = selection_markdown(table)
    else:
        table = confirmation_table(results, _pairs(args.compare), args.condition)
        markdown = confirmation_markdown(table)
    table["inputs"] = {
        name: {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for name, path in paths.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(json.dumps(table, indent=1) + "\n")
    outputs[1].write_text(markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
