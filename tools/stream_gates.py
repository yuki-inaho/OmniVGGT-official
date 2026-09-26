"""Correctness gates of the frame-causal OmniVGGTOmega stream on one window of real frames.

(a) ``equivalence``: frame-by-frame ``StreamingOmega`` with a full cache reproduces one batch forward of the same
    frame-causal model (pose_enc, depth).
(b) ``future_invariance``: replacing the last frame (image, depth, mask) by the same frame number of the other
    session leaves the outputs of every earlier frame exactly unchanged.
(c)+(d) ``cache_invariants``: with a bounded policy, every cache satisfies ``LayerKVCache.invariants()`` after
    every step, and long-patch rows that stay in the cache across a step keep bit-identical K/V (quantised
    once, never re-quantised).

The model is built with ``causal=True, depth_norm="first_frame"`` and run in fp32 with TF32 off.

usage: PYTHONPATH=tools uv run python -m stream_gates --model-config VARIANT.json --checkpoint W|DIR \
           --roots R0 R1 --split smoke --window s0:950-965 --policy JSON --output gates.json
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import torch
from eval_colmap_rgbd import _load_model
from eval_stream import WindowLoader, git_state, parse_window, precision_setup, window_inputs

from omnivggt.stream.kv_cache import CachePolicy
from omnivggt.stream.streaming import StreamingOmega, all_caches

POSE_TOLERANCE = 1e-4  # fp32 real weights (§1.1 of the workdoc)
DEPTH_REL_TOLERANCE = 1e-3


def _batch(model, inputs: dict) -> dict:
    frames = inputs["images"].shape[1]
    with torch.no_grad():
        out = model.inference(**inputs, depth_gt_index=list(range(frames)), camera_gt_index=[])
    return {"pose_enc": out["pose_enc"], "depth": out["depth"]}


def _stream(model, inputs: dict, policy: CachePolicy, on_step=None) -> dict:
    streamer = StreamingOmega(model, policy)
    poses, depths = [], []
    for t in range(inputs["images"].shape[1]):
        out = streamer.step(inputs["images"][:, t], inputs["depth"][:, t], inputs["mask"][:, t])
        poses.append(out["pose_enc"])
        depths.append(out["depth"])
        if on_step is not None:
            on_step(streamer)
    return {"pose_enc": torch.cat(poses, 1), "depth": torch.cat(depths, 1)}


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


def _max_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a - b).abs() / b.abs().clamp_min(1e-12)).max())


def equivalence(model, inputs: dict, pose_tol: float = POSE_TOLERANCE, depth_tol: float = DEPTH_REL_TOLERANCE) -> dict:
    batch, stream = _batch(model, inputs), _stream(model, inputs, CachePolicy.full())
    pose, depth = _max_abs(stream["pose_enc"], batch["pose_enc"]), _max_rel(stream["depth"], batch["depth"])
    return {"pose_enc_max_abs": pose, "depth_max_rel": depth, "pose_tol": pose_tol, "depth_rel_tol": depth_tol,
            "ok": pose <= pose_tol and depth <= depth_tol}


def _past_difference(first: dict, second: dict) -> float:
    return max(_max_abs(first[key][:, :-1], second[key][:, :-1]) for key in ("pose_enc", "depth"))


def future_invariance(model, inputs: dict, changed: dict) -> dict:
    """Streams ``inputs`` and ``changed`` (which differ in the last frame only) with a full cache."""
    past = _past_difference(_stream(model, inputs, CachePolicy.full()), _stream(model, changed, CachePolicy.full()))
    return {"past_max_abs": past, "ok": past == 0.0}


def future_invariance_batch(model, inputs: dict, changed: dict) -> dict:
    """The same check on one batch forward (a model without the frame mask fails it)."""
    past = _past_difference(_batch(model, inputs), _batch(model, changed))
    return {"past_max_abs": past, "ok": past == 0.0}


def changed_rows(previous: dict, current: dict) -> tuple[int, int]:
    """(rows present in both snapshots, rows among them whose K or V changed at all)."""
    common = previous.keys() & current.keys()
    changed = sum(
        not (torch.equal(previous[row][0], current[row][0]) and torch.equal(previous[row][1], current[row][1]))
        for row in common
    )
    return len(common), changed


def _long_patch_rows(cache) -> dict:
    """{(frame_id, token_id): (K row, V row)} of the rows outside the anchor/recent frames and the specials."""
    if cache.size == 0 or cache.policy.is_full:
        return {}
    keys, values = cache.read()
    frame_id, token_id = cache.row_ids()
    protected = torch.tensor(cache.policy.protected_frames(cache.last_frame), device=frame_id.device)
    long_patch = ~torch.isin(frame_id, protected) & (token_id >= cache.special_count)
    return {
        (int(frame_id[row]), int(token_id[row])): (keys[:, :, row].clone(), values[:, :, row].clone())
        for row in torch.nonzero(long_patch).flatten().tolist()
    }


def cache_invariants(model, inputs: dict, policy: CachePolicy, quant_layers=(0,)) -> dict:
    """Stream with ``policy``; check every cache after every step and quantize-once on ``quant_layers``."""
    record = {"steps": 0, "invariants_failed": [], "compared_rows": 0, "requantized_rows": 0}
    snapshots = {}

    def on_step(streamer):
        record["steps"] += 1
        for index, cache in enumerate(all_caches(streamer.caches)):
            checks = cache.invariants()
            if not checks["ok"]:
                record["invariants_failed"].append({"t": streamer.t, "cache": index, "checks": checks})
        for layer in quant_layers:
            rows = _long_patch_rows(streamer.caches["global"][layer])
            compared, changed = changed_rows(snapshots.get(layer, {}), rows)
            record["compared_rows"] += compared
            record["requantized_rows"] += changed
            snapshots[layer] = rows

    _stream(model, inputs, policy, on_step)
    record["ok"] = not record["invariants_failed"] and record["requantized_rows"] == 0
    return record


def _swap_last_frame(inputs: dict, other: dict) -> dict:
    changed = {key: value.clone() for key, value in inputs.items()}
    for key in ("images", "depth", "mask"):
        changed[key][:, -1] = other[key][:, -1]
    return changed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--roots", nargs=2, required=True, help="two sessions; the other one supplies the swap")
    parser.add_argument("--split", choices=["val", "smoke"], required=True)
    parser.add_argument("--window", required=True, help="sK:START-END[/STRIDE]")
    parser.add_argument("--policy", required=True, help="bounded CachePolicy JSON for gates (c) and (d)")
    parser.add_argument("--quant-layers", default="0", help="global layer indices checked for quantize-once")
    parser.add_argument("--resolution", type=int, nargs=2, default=(392, 294), metavar=("W", "H"))
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, precision = precision_setup("fp32", device)
    model, weights = _load_model(args.checkpoint, device, args.model_config, causal=True, depth_norm="first_frame")
    policy = CachePolicy(**json.loads(args.policy))
    window = parse_window(args.window)
    other = dataclasses.replace(window, session=1 - window.session)
    loader = WindowLoader(args.roots, args.split, args.resolution)
    inputs = window_inputs(loader.load(window), device)
    changed = _swap_last_frame(inputs, window_inputs(loader.load(other), device))
    quant_layers = tuple(int(x) for x in args.quant_layers.split(","))
    gates = {
        "equivalence": equivalence(model, inputs),
        "future_invariance": future_invariance(model, inputs, changed),
        "cache_invariants": cache_invariants(model, inputs, policy, quant_layers),
    }
    result = {
        "tool": "stream_gates",
        "window": window.spec,
        "swap_from": other.spec,
        "split": args.split,
        "policy": dataclasses.asdict(policy),
        "precision": precision,
        "model_config": {"path": str(args.model_config), "sha256": _sha256(args.model_config)},
        "weights": {"file": f"{Path(weights).parent.name}/{Path(weights).name}", "sha256": _sha256(Path(weights))},
        "git": git_state(),
        "gates": gates,
        "ok": all(gate["ok"] for gate in gates.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({name: gate["ok"] for name, gate in gates.items()} | {"ok": result["ok"]}))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
