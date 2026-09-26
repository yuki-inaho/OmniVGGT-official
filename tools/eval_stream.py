"""Evaluate OmniVGGTOmega on ordered windows of a colmap_rgbd_v1 split, offline or as a stream.

A window ``sK:START-END[/STRIDE]`` is frames START, START+STRIDE, ..., END (inclusive) of the K-th
``--roots`` entry; every frame must lie inside ``--split``. Windows are read through ``ColmapRgbd``
sequential views exactly like ``eval_colmap_rgbd``; ``--preset L8`` takes eval_colmap_rgbd's default
windows (val, 8 frames, stride 3, 16 evenly spaced anchors).

Modes (the model options each one builds are in ``MODES``):

* ``bidir``: one bidirectional forward over the window (depth input normalised jointly);
* ``bidir_f0``: ``bidir`` with the depth input normalised by the first frame only;
* ``bidir_prefix``: frame t is the last frame of a ``bidir_f0`` forward over frames 1..t (online, O(t^2));
* ``stream``: the frame-causal model fed one frame at a time (``StreamingOmega`` with ``--policy``).

Conditions: ``depth`` gives GT depth for every frame, ``rgb`` none; cameras are never given. Metrics are
``stream_metrics.window_metrics`` per window, plus window means per session. Efficiency: per-step latency
(CUDA events on GPU, perf_counter on CPU; one step = the whole window for ``bidir``/``bidir_f0``), the
analytic KV bytes after every stream step and, on GPU, memory allocated / peak above the window start.
``--precision fp32`` runs without autocast and with TF32 off (accuracy rows); ``bf16`` adds bf16
autocast and a bf16 stream cache (efficiency only).

usage: PYTHONPATH=tools uv run python -m eval_stream --model-config VARIANT.json --checkpoint W|DIR \
           --roots R0 R1 --split {val,smoke} (--windows "s0:950-981,s1:950-981" | --preset L8) \
           --mode {bidir,bidir_f0,bidir_prefix,stream} [--policy full|JSON] [--conditions depth rgb] \
           [--precision fp32|bf16] --output result.json
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from bench_inference import summarize
from eval_colmap_rgbd import (
    _check_variant_provenance,
    _load_model,
    _model_config_record,
    _sha256,
    sequential_anchors,
    sequential_dataset,
)
from stream_metrics import mean_over_windows, window_metrics

from omnivggt.utils.pose_enc import pose_encoding_to_extri_intri

MODES = {
    "bidir": {"causal": False, "depth_norm": "joint"},
    "bidir_f0": {"causal": False, "depth_norm": "first_frame"},
    "bidir_prefix": {"causal": False, "depth_norm": "first_frame"},
    "stream": {"causal": True, "depth_norm": "first_frame"},
}
CONDITIONS = ("depth", "rgb")
PRECISIONS = ("fp32", "bf16")
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
L8 = {"frames": 8, "stride": 3, "num_samples": 16}  # eval_colmap_rgbd defaults
WINDOW_PATTERN = re.compile(r"s(\d+):(\d+)-(\d+)(?:/(\d+))?")
REPO = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Window:
    session: int
    start: int
    end: int
    stride: int = 1

    @property
    def frames(self) -> list[int]:
        return list(range(self.start, self.end + 1, self.stride))

    @property
    def spec(self) -> str:
        suffix = f"/{self.stride}" if self.stride != 1 else ""
        return f"s{self.session}:{self.start}-{self.end}{suffix}"


def parse_window(spec: str) -> Window:
    match = WINDOW_PATTERN.fullmatch(spec.strip())
    if match is None:
        raise ValueError(f"window {spec!r} is not sK:START-END[/STRIDE]")
    session, start, end = (int(group) for group in match.groups()[:3])
    stride = int(match[4]) if match[4] is not None else 1
    if end <= start or stride < 1 or (end - start) % stride:
        raise ValueError(f"window {spec!r} needs START < END with END - START a multiple of STRIDE >= 1")
    return Window(session, start, end, stride)


def parse_windows(text: str) -> list[Window]:
    return [parse_window(spec) for spec in text.split(",")]


def _frame_number(path: str) -> int:
    return int(Path(path).stem.removeprefix("frame_"))


class WindowLoader:
    """Windows of one split, read through one sequential ``ColmapRgbd`` per (root, stride)."""

    def __init__(self, roots, split: str, resolution):
        self.roots, self.split, self.resolution = list(roots), split, tuple(resolution)
        self._datasets = {}

    def dataset(self, session: int, stride: int):
        if not 0 <= session < len(self.roots):
            raise ValueError(f"window session s{session} does not name one of the {len(self.roots)} roots")
        if (session, stride) not in self._datasets:
            dataset = sequential_dataset([self.roots[session]], self.split, self.resolution, stride)
            if len(set(dataset.scene_labels)) != 1:
                raise ValueError(f"window root {self.roots[session]} must hold exactly one scene")
            self._datasets[session, stride] = dataset
        return self._datasets[session, stride]

    def anchor(self, window: Window) -> int:
        """Dataset index of the window's first frame; every window frame must be in the split."""
        frames = [_frame_number(path) for path in self.dataset(window.session, window.stride).rgb_paths]
        if window.start not in frames:
            raise ValueError(f"window {window.spec} starts outside the {self.split} split")
        anchor = frames.index(window.start)
        if frames[anchor : anchor + window.end - window.start + 1 : window.stride] != window.frames:
            raise ValueError(f"window {window.spec} leaves the {self.split} split")
        return anchor

    def load(self, window: Window) -> dict:
        return self.dataset(window.session, window.stride)[(self.anchor(window), 0, len(window.frames))]


def l8_windows(roots, split: str, resolution) -> tuple[list[Window], list[int]]:
    """eval_colmap_rgbd's default windows as ``Window`` s, with its dataset anchors."""
    if split != "val":
        raise ValueError(f"the L8 preset is defined on the val split, not {split!r}")
    sessions = {Path(root).name: index for index, root in enumerate(roots)}
    if len(sessions) != len(roots):
        raise ValueError("the L8 preset needs --roots with distinct directory names")
    dataset = sequential_dataset(roots, split, resolution, L8["stride"])
    anchors = sequential_anchors(dataset, **L8)
    windows = []
    for anchor in anchors:
        start = _frame_number(dataset.rgb_paths[anchor])
        session = sessions[dataset.scene_labels[anchor].split("/")[0]]
        windows.append(Window(session, start, start + (L8["frames"] - 1) * L8["stride"], L8["stride"]))
    return windows, anchors


def check_options(mode: str, policy: str | None):
    """The parsed ``--policy``: required by ``stream`` ("full" or a JSON object), refused by other modes."""
    if mode != "stream":
        if policy is not None:
            raise ValueError(f"--policy applies to --mode stream only, not {mode}")
        return None
    if policy is None:
        raise ValueError("--mode stream needs --policy (full or a CachePolicy JSON object)")
    if policy == "full":
        return "full"
    parsed = json.loads(policy)
    if not isinstance(parsed, dict):
        raise ValueError(f"--policy must be full or a JSON object, got {policy!r}")
    return parsed


def precision_setup(precision: str, device):
    """(context, record): TF32 off for every precision; ``bf16`` adds bf16 autocast."""
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    autocast = precision == "bf16"
    context = torch.autocast(torch.device(device).type, dtype=torch.bfloat16) if autocast else contextlib.nullcontext()
    record = {
        "name": precision,
        "dtype": str(DTYPES[precision]).removeprefix("torch."),
        "autocast": "bfloat16" if autocast else None,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }
    return context, record


def git_state(repo: Path = REPO) -> dict:
    """HEAD commit of ``repo`` and whether tracked files differ from it."""

    def git(*args) -> str:
        return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout

    dirty = git("status", "--porcelain", "--untracked-files=no").strip()
    return {"commit": git("rev-parse", "HEAD").strip(), "dirty": bool(dirty)}


class Meter:
    """Latency of one call (CUDA events on a CUDA device, perf_counter otherwise) and, on CUDA only,
    memory allocated / peak above the last ``start``."""

    def __init__(self, device):
        self.cuda = torch.device(device).type == "cuda"
        self._base = None

    def start(self) -> None:
        if self.cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self._base = torch.cuda.memory_allocated()

    def timed(self, call):
        if not self.cuda:
            start = time.perf_counter()
            result = call()
            return result, 1e3 * (time.perf_counter() - start)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = call()
        end.record()
        torch.cuda.synchronize()
        return result, start.elapsed_time(end)

    def memory(self) -> dict:
        if not self.cuda:
            return {}
        torch.cuda.synchronize()
        return {
            "allocated_above_start_bytes": torch.cuda.memory_allocated() - self._base,
            "peak_above_start_bytes": torch.cuda.max_memory_allocated() - self._base,
        }


def _batched(array, device) -> torch.Tensor:
    return torch.from_numpy(array)[None].float().to(device)


def window_inputs(item: dict, device) -> dict:
    """Batched model inputs of one window, built as in eval_colmap_rgbd."""
    return {
        "images": item["images"][None].to(device),
        "extrinsics": _batched(item["extrinsic"], device),
        "intrinsics": _batched(item["intrinsic"], device),
        "depth": _batched(item["depth"], device),
        "mask": _batched(item["valid_mask"], device),
    }


def _inference(model, inputs: dict, frames: slice, use_depth: bool) -> dict:
    """``model.inference`` on the window ``frames``: GT depth for every view or none, never cameras."""
    views = {key: value[:, frames] for key, value in inputs.items()}
    count = views["images"].shape[1]
    return model.inference(
        images=views["images"],
        extrinsics=views["extrinsics"],
        intrinsics=views["intrinsics"],
        depth=views["depth"],
        mask=views["mask"],
        depth_gt_index=list(range(count)) if use_depth else [],
        camera_gt_index=[],
    )


def _on_cpu(prediction: dict) -> dict:
    return {"pose_enc": prediction["pose_enc"].float().cpu(), "depth": prediction["depth"].float().cpu()}


def predict_batch(model, inputs: dict, use_depth: bool, meter: Meter) -> dict:
    """``bidir`` / ``bidir_f0``: one forward over the whole window."""
    meter.start()
    prediction, milliseconds = meter.timed(functools.partial(_inference, model, inputs, slice(None), use_depth))
    prediction = _on_cpu(prediction)
    return {**prediction, "step_ms": [milliseconds], "memory": meter.memory()}


def predict_prefix(model, inputs: dict, use_depth: bool, meter: Meter) -> dict:
    """``bidir_prefix``: frame t is the last frame of a forward over frames 1..t."""
    meter.start()
    poses, depths, step_ms = [], [], []
    for t in range(1, inputs["images"].shape[1] + 1):
        prediction, milliseconds = meter.timed(functools.partial(_inference, model, inputs, slice(0, t), use_depth))
        prediction = _on_cpu(prediction)
        poses.append(prediction["pose_enc"][:, -1:])
        depths.append(prediction["depth"][:, -1:])
        step_ms.append(milliseconds)
    return {
        "pose_enc": torch.cat(poses, 1),
        "depth": torch.cat(depths, 1),
        "step_ms": step_ms,
        "memory": meter.memory(),
    }


def _one_frame(output: dict) -> dict:
    if output["pose_enc"].shape[1] != 1 or output["depth"].shape[1] != 1:
        raise ValueError(
            f"a stream step must return one frame (pose_enc (B,1,9), depth (B,1,H,W,1)), got "
            f"{tuple(output['pose_enc'].shape)} and {tuple(output['depth'].shape)}"
        )
    return _on_cpu(output)


def predict_stream(streamer, inputs: dict, use_depth: bool, meter: Meter) -> dict:
    """``stream``: frame t from ``streamer.step`` after frames 1..t-1; analytic KV bytes after each step."""
    streamer.reset()
    meter.start()
    poses, depths, step_ms, kv_bytes = [], [], [], []
    for t in range(inputs["images"].shape[1]):
        depth, mask = (inputs["depth"][:, t], inputs["mask"][:, t]) if use_depth else (None, None)
        output, milliseconds = meter.timed(functools.partial(streamer.step, inputs["images"][:, t], depth, mask))
        output = _one_frame(output)
        poses.append(output["pose_enc"])
        depths.append(output["depth"])
        step_ms.append(milliseconds)
        kv_bytes.append(int(streamer.kv_bytes()))
    return {
        "pose_enc": torch.cat(poses, 1),
        "depth": torch.cat(depths, 1),
        "step_ms": step_ms,
        "kv_bytes": kv_bytes,
        "memory": meter.memory(),
    }


def make_streamer(model, policy, precision: str):
    """``StreamingOmega`` over ``model`` with the cache ``policy`` ("full" or ``CachePolicy`` fields)."""
    from omnivggt.stream.kv_cache import CachePolicy
    from omnivggt.stream.streaming import StreamingOmega

    cache_policy = CachePolicy.full() if policy == "full" else CachePolicy(**policy)
    return StreamingOmega(model, cache_policy, DTYPES[precision])


def make_predictor(mode: str, model, policy, precision: str, meter: Meter):
    """``predictor(inputs, use_depth) -> prediction`` of one mode."""
    if mode == "stream":
        return functools.partial(predict_stream, make_streamer(model, policy, precision), meter=meter)
    if mode == "bidir_prefix":
        return functools.partial(predict_prefix, model, meter=meter)
    return functools.partial(predict_batch, model, meter=meter)


def efficiency(prediction: dict, frames: int) -> dict:
    total = float(sum(prediction["step_ms"]))
    record = {"step_ms": summarize(prediction["step_ms"]), "total_ms": total, "per_frame_ms": total / frames}
    if "kv_bytes" in prediction:
        record["kv_bytes_max"] = max(prediction["kv_bytes"])
        record["kv_bytes_last"] = prediction["kv_bytes"][-1]
    return {**record, **prediction["memory"]}


def evaluate_window(predictor, item: dict, conditions, device, precision_context) -> dict:
    """Metrics, per-frame series and efficiency of one window under each condition; only the model runs
    inside ``precision_context``."""
    inputs = window_inputs(item, device)
    records = {}
    for condition in conditions:
        with torch.no_grad(), precision_context:
            prediction = predictor(inputs, condition == "depth")
        pred_w2c, _ = pose_encoding_to_extri_intri(prediction["pose_enc"], inputs["images"].shape[-2:])
        result = window_metrics(
            pred_w2c[0].numpy(),
            item["extrinsic"],
            prediction["depth"][0, ..., 0].numpy(),
            item["depth"][..., 0],
            item["valid_mask"],
        )
        series = {**result["series"], "step_ms": prediction["step_ms"]}
        if "kv_bytes" in prediction:
            series["kv_bytes"] = prediction["kv_bytes"]
        records[condition] = {
            "metrics": result["metrics"],
            "series": series,
            "efficiency": efficiency(prediction, len(item["instance"])),
        }
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-config", type=Path, required=True, help="OmniVGGTOmega variant JSON")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--roots", nargs="+", required=True, help="session roots; sK names the K-th")
    parser.add_argument("--split", choices=["val", "smoke"], required=True)
    windows = parser.add_mutually_exclusive_group(required=True)
    windows.add_argument("--windows", help='comma-separated "sK:START-END[/STRIDE]"')
    windows.add_argument("--preset", choices=["L8"])
    parser.add_argument("--mode", choices=list(MODES), required=True)
    parser.add_argument("--policy", help='stream only: "full" or a CachePolicy JSON object')
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=["depth"])
    parser.add_argument("--precision", choices=PRECISIONS, default="fp32")
    parser.add_argument("--resolution", type=int, nargs=2, default=(392, 294), metavar=("W", "H"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    policy = check_options(args.mode, args.policy)
    if args.output.exists():
        raise FileExistsError(args.output)
    loader = WindowLoader(args.roots, args.split, args.resolution)
    if args.preset == "L8":
        windows, anchors = l8_windows(args.roots, args.split, args.resolution)
    else:
        windows, anchors = parse_windows(args.windows), None
    for window in windows:
        loader.anchor(window)  # every window is checked before the model is loaded
    variant_provenance = _check_variant_provenance(args.model_config, args.checkpoint)
    model, weights = _load_model(args.checkpoint, args.device, args.model_config, **MODES[args.mode])
    context, precision = precision_setup(args.precision, args.device)
    predictor = make_predictor(args.mode, model, policy, args.precision, Meter(args.device))

    records = []
    start = time.time()
    for index, window in enumerate(windows):
        record = {"spec": window.spec, "session": f"s{window.session}", "frames": window.frames}
        if anchors is not None:
            record["anchor"] = anchors[index]
        item = loader.load(window)
        record["conditions"] = evaluate_window(predictor, item, args.conditions, args.device, context)
        records.append(record)
    output = {
        "provenance": {
            "checkpoint": {"path": str(args.checkpoint), "file": weights.name, "sha256": _sha256(weights)},
            "model_config": _model_config_record(args.model_config),
            "variant_provenance": variant_provenance,
            "mode": args.mode,
            "model_options": MODES[args.mode],
            "policy": policy,
            "precision": precision,
            "git": git_state(),
            "device": str(args.device),
            "torch": torch.__version__,
        },
        "split": args.split,
        "roots": [Path(root).name for root in args.roots],
        "resolution_wh": list(args.resolution),
        "preset": args.preset,
        "anchors": anchors,
        "conditions": args.conditions,
        "seconds": time.time() - start,
        "aggregate": {condition: mean_over_windows(records, condition) for condition in args.conditions},
        "windows": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=1) + "\n")
    print(json.dumps({condition: means["all"] for condition, means in output["aggregate"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
