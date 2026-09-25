"""Inference latency and peak GPU memory of OmniVGGT / OmniVGGTOmega under one fixed protocol.

The model forward (``inference``) is timed with CUDA events under ``torch.inference_mode`` and bf16
autocast, after warm-up, on fixed synthetic inputs (latency does not depend on image content).
``rgb`` passes no auxiliary input; ``depth+camera`` passes a constant depth map and fixed cameras for
every view. Peak memory is measured after the weights are resident (``peak_allocated_mib_above_weights``).

usage: PYTHONPATH=tools uv run python -m bench_inference [--model-config VARIANT.json] \
           (--checkpoint W.safetensors|ACCELERATE_DIR | --random-weights) --width 392 --height 294 \
           --frames 8 16 32 --condition rgb [--no-points] [--sections] --output bench.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from eval_colmap_rgbd import _build_model, _check_variant_provenance, _model_config_record, _sha256


def summarize(latencies_ms) -> dict:
    values = np.asarray(latencies_ms, dtype=float)
    return {
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "min": float(values.min()),
        "max": float(values.max()),
        "n": int(values.size),
    }


def section_totals(timings) -> dict:
    totals = defaultdict(float)
    for name, milliseconds in timings:
        totals[name] += milliseconds
    return dict(totals)


def check_options(model_config, no_points: bool) -> None:
    if no_points and model_config is None:
        raise ValueError("--no-points needs an OmniVGGTOmega variant: OmniVGGT always runs its point head")


def make_inputs(frames: int, height: int, width: int, condition: str, device) -> dict:
    generator = torch.Generator().manual_seed(0)
    images = torch.rand(1, frames, 3, height, width, generator=generator)
    extrinsics = torch.eye(3, 4).repeat(1, frames, 1, 1)
    extrinsics[0, :, 0, 3] = 0.01 * torch.arange(frames)
    intrinsics = torch.tensor([[width, 0.0, width / 2], [0.0, width, height / 2], [0.0, 0.0, 1.0]]).repeat(
        1, frames, 1, 1
    )
    if condition == "rgb":
        depth, mask, views = torch.zeros(1, frames, height, width, 1), None, []
    elif condition == "depth+camera":
        depth, mask, views = (
            torch.ones(1, frames, height, width, 1),
            torch.ones(1, frames, height, width),
            list(range(frames)),
        )
    else:
        raise ValueError(f"unknown condition {condition!r}")
    move = lambda x: None if x is None else x.to(device)  # noqa: E731
    return {
        "images": move(images),
        "extrinsics": move(extrinsics),
        "intrinsics": move(intrinsics),
        "depth": move(depth),
        "mask": move(mask),
        "depth_gt_index": views,
        "camera_gt_index": list(views),
    }


def _section_modules(model) -> dict:
    aggregator = model.aggregator
    register_layers = getattr(aggregator, "register_attention_layers", frozenset())
    groups = {
        "image_encoder": [aggregator.patch_embed],
        "frame_blocks": list(aggregator.frame_blocks),
        "global_blocks_full": [b for i, b in enumerate(aggregator.global_blocks) if i not in register_layers],
        "global_blocks_register": [b for i, b in enumerate(aggregator.global_blocks) if i in register_layers],
        "camera_head": [model.camera_head],
        "depth_head": [model.depth_head],
    }
    if getattr(model, "point_head", None) is not None:
        groups["point_head"] = [model.point_head]
    if getattr(model, "point_unprojector", None) is not None:
        groups["unprojection"] = [model.point_unprojector]
    return groups


def _time_sections(model, run) -> dict:
    events, pending, handles = [], {}, []
    for name, modules in _section_modules(model).items():
        for module in modules:

            def pre(mod, args, name=name):
                start = torch.cuda.Event(enable_timing=True)
                start.record()
                pending[id(mod)] = start

            def post(mod, args, output, name=name):
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                events.append((name, pending.pop(id(mod)), end))

            handles += [module.register_forward_pre_hook(pre), module.register_forward_hook(post)]
    try:
        run()
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    return section_totals((name, start.elapsed_time(end)) for name, start, end in events)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-config", type=Path, help="OmniVGGTOmega variant JSON (default: OmniVGGT)")
    weights = parser.add_mutually_exclusive_group(required=True)
    weights.add_argument("--checkpoint", type=Path)
    weights.add_argument(
        "--random-weights", action="store_true", help="architecture-only timing (weights do not change latency)"
    )
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--condition", choices=["rgb", "depth+camera"], default="rgb")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--no-points", action="store_true", help="OmniVGGTOmega only: skip the unprojected world points"
    )
    parser.add_argument("--sections", action="store_true", help="also time the modules in one extra forward")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check_options(args.model_config, args.no_points)
    if args.output.exists():
        raise FileExistsError(args.output)

    model = _build_model(args.model_config)
    weights_record = "random"
    variant_provenance = (
        None if args.checkpoint is None else _check_variant_provenance(args.model_config, args.checkpoint)
    )
    if args.checkpoint is not None:
        from safetensors.torch import load_file

        path = args.checkpoint / "model.safetensors" if args.checkpoint.is_dir() else args.checkpoint
        model.load_state_dict(load_file(str(path)), strict=True)
        weights_record = {"path": str(path.resolve()), "sha256": _sha256(path)}
    model = model.to("cuda").eval()
    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated()
    options = {} if args.model_config is None else {"return_points": not args.no_points}

    results = []
    for frames in args.frames:
        inputs = make_inputs(frames, args.height, args.width, args.condition, "cuda")

        def run(inputs=inputs):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                model.inference(**inputs, **options)

        row = {"frames": frames, "oom": False}
        try:
            for _ in range(args.warmup):
                run()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            latencies = []
            for _ in range(args.repeats):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                run()
                end.record()
                torch.cuda.synchronize()
                latencies.append(start.elapsed_time(end))
            row["latency_ms"] = summarize(latencies)
            row["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
            row["peak_allocated_mib_above_weights"] = (torch.cuda.max_memory_allocated() - resident) / 2**20
            row["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
            if args.sections:
                row["sections_ms"] = _time_sections(model, run)
        except torch.cuda.OutOfMemoryError:
            row = {"frames": frames, "oom": True}
        torch.cuda.empty_cache()
        results.append(row)
        print(json.dumps(row), flush=True)

    output = {
        "tool": "bench_inference",
        "model_config": _model_config_record(args.model_config),
        "model": "OmniVGGT" if args.model_config is None else json.loads(args.model_config.read_text())["name"],
        "weights": weights_record,
        "variant_provenance": variant_provenance,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "protocol": {"autocast": "bf16", "inference_mode": True, "warmup": args.warmup, "repeats": args.repeats},
        "resolution_wh": [args.width, args.height],
        "condition": args.condition,
        "points": args.model_config is None or not args.no_points,
        "weights_resident_mib": resident / 2**20,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
