"""Per-step cost of the frame-causal OmniVGGTOmega stream as a function of the stream length t.

Synthetic RGB-D frames (content does not change the cost) are fed one at a time through ``StreamingOmega``; at
the requested steps t the step latency (CUDA events on GPU, perf_counter on CPU), the median of the last five
steps, the analytic KV bytes and, on GPU, memory allocated and the peak so far are recorded. An out-of-memory
step is recorded as such and ends the run.
``OMNIVGGT_VRAM_LIMIT_GB`` (GiB, optional) caps the memory of the process on the GPU before the model is built
(``omnivggt.utils.vram``); the output records the cap and the peak memory of the run.

usage: [OMNIVGGT_VRAM_LIMIT_GB=G] PYTHONPATH=tools uv run python -m bench_stream --model-config VARIANT.json --random-init \
           --policy full|JSON --steps 1,32,128,256,512,1000 [--precision bf16] --output bench.json
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import functools
import json
import statistics
import time
from pathlib import Path

import torch
from eval_stream import git_state

from omnivggt.stream.kv_cache import CachePolicy
from omnivggt.stream.streaming import StreamingOmega
from omnivggt.utils.vram import apply_vram_limit, memory_peaks

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def parse_steps(text: str) -> tuple[int, ...]:
    steps = tuple(int(part) for part in text.split(","))
    if not steps or any(step < 1 for step in steps) or list(steps) != sorted(set(steps)):
        raise ValueError(f"--steps must be increasing positive integers, got {text!r}")
    return steps


def synthetic_frame(t: int, image_hw, device, dtype):
    generator = torch.Generator().manual_seed(t)
    height, width = image_hw
    image = torch.rand(1, 3, height, width, generator=generator, dtype=dtype).to(device)
    depth = (0.5 + torch.rand(1, height, width, 1, generator=generator, dtype=dtype)).to(device)
    return image, depth, torch.ones(1, height, width, dtype=dtype, device=device)


def _timed(call, cuda: bool):
    if not cuda:
        start = time.perf_counter()
        call()
        return 1e3 * (time.perf_counter() - start)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    call()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def run(model, policy: CachePolicy, steps, image_hw, dtype, device, autocast: bool = False) -> list[dict]:
    """Stream up to ``max(steps)`` frames; one row per requested step."""
    cuda = torch.device(device).type == "cuda"
    streamer = StreamingOmega(model, policy, dtype if autocast else None)
    context = torch.autocast("cuda", dtype=torch.bfloat16) if autocast else contextlib.nullcontext()
    input_dtype = torch.float32 if autocast else dtype
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    rows, recent = [], []
    wanted = set(steps)
    for t in range(1, max(steps) + 1):
        image, depth, mask = synthetic_frame(t, image_hw, device, input_dtype)
        try:
            with context:
                milliseconds = _timed(functools.partial(streamer.step, image, depth, mask), cuda)
        except torch.cuda.OutOfMemoryError:
            rows.extend({"t": step, "oom": True} for step in steps if step >= t)
            break
        recent = [*recent, milliseconds][-5:]
        if t in wanted:
            row = {"t": t, "step_ms": milliseconds, "step_ms_median5": statistics.median(recent),
                   "kv_bytes": int(streamer.kv_bytes())}
            if cuda:
                row |= {"allocated_bytes": torch.cuda.memory_allocated(), "peak_bytes": torch.cuda.max_memory_allocated()}
            rows.append(row)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-config", type=Path, required=True)
    weights = parser.add_mutually_exclusive_group(required=True)
    weights.add_argument("--checkpoint", type=Path)
    weights.add_argument("--random-init", action="store_true", help="architecture-only timing")
    parser.add_argument("--policy", required=True, help='"full" or a CachePolicy JSON object')
    parser.add_argument("--steps", type=parse_steps, default=parse_steps("1,32,128,256,512,1000"))
    parser.add_argument("--precision", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--width", type=int, default=392)
    parser.add_argument("--height", type=int, default=294)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    from eval_colmap_rgbd import _load_model

    from omnivggt.models.omnivggt_omega import OmniVGGTOmega

    args = build_parser().parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vram_limit = apply_vram_limit(device)  # OMNIVGGT_VRAM_LIMIT_GB, before the model is built
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    options = {"causal": True, "depth_norm": "first_frame"}
    if args.random_init:
        torch.manual_seed(0)
        model = OmniVGGTOmega.from_variant(args.model_config, **options).to(device).eval()
        weights = "random-init"
    else:
        model, path = _load_model(args.checkpoint, device, args.model_config, **options)
        weights = f"{Path(path).parent.name}/{Path(path).name}"
    policy = CachePolicy.full() if args.policy == "full" else CachePolicy(**json.loads(args.policy))
    autocast = args.precision == "bf16"
    rows = run(model, policy, args.steps, (args.height, args.width), DTYPES[args.precision], device, autocast)
    result = {
        "tool": "bench_stream",
        "model_config": str(args.model_config),
        "weights": weights,
        "policy": "full" if policy.is_full else dataclasses.asdict(policy),
        "precision": {"name": args.precision, "autocast": "bfloat16" if autocast else None, "tf32": False},
        "resolution_wh": [args.width, args.height],
        "gpu": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "torch": torch.__version__,
        "git": git_state(),
        "vram": {"limit": vram_limit, "peaks": memory_peaks(device)},
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    for row in rows:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
