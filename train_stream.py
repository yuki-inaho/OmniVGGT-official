#!/usr/bin/env python3
"""Training on the stream (T-X): the frame-causal OmniVGGTOmega fed one frame at a time.

``StreamingOmega(train=True)`` runs every frame of an ordered window of the train split with the KV caches of
``--policy`` ("full" or a ``CachePolicy`` JSON object, as for ``tools/eval_stream.py``). The gradient of a frame
goes through that frame only: the cached past is detached (one-frame BPTT, XStream-style). The recipe follows the
batch trainer (train_omnivggt.py) wherever it applies:

* windows: ``--window-length`` (48) sequential views of the ``ColmapRgbd`` of the config's ``train_dataset`` (its
  split, strides ``OMNIVGGT_SEQ_STRIDES``, augmentation and sample seed ``OMNIVGGT_DATA_SEED``); a window that would
  leave its scene/split starts earlier. The window anchors are seeded permutations of the train frames
  (``WINDOW_SEED``);
* targets: normalised once per window with the first-frame target scale; the model gets the raw depth of every
  frame and no camera, the loss the normalised targets;
* loss: ``MultitaskLoss`` of every frame with ``progress`` = update / ``--updates``, backward of objective / 12;
* update, every 12 frames (the images of one batch-trainer step): ``clip_grad_norm_`` to ``max_grad_norm``, then
  one optimizer step (AMUSE: warm-up from ``configure_schedule(total_steps=--updates)``), with the encoder frozen as
  configured;
* precision: ``mixed_precision="bf16"`` autocasts every stream step (the heads run in fp32) and stores the caches
  in bf16; the loss runs without autocast. TF32 is left at PyTorch's defaults, as in train_omnivggt.py.

A window spans several updates, so after the first update of a window its cache holds keys and values written with
the weights before the update (stale by up to window/12 - 1 updates); the stream is not recomputed.

The run writes ``<output-dir>/weight_transfer_report.json`` (the strict load of ``--init``) and, under
``<output-dir>/<exp_name>/``, ``checkpoint-u<N>`` and ``final_checkpoint`` (``model.safetensors`` of the evaluation
weights, laid out as train_omnivggt.py's checkpoints), ``stream_training.json`` and TensorBoard logs. There is no
resume: a stopped run starts again, in a new output directory.

k-best checkpoints: every checkpoint is scored with the same evaluation weights by the mean stream loss (the loss
above at progress 1, without gradients, the model in eval mode, every random state restored so that the run trains
as without it) over ``--val-windows`` fixed windows of the smoke split (the val split is left for the confirmation),
read like the training windows without augmentation, at the smallest configured stride, from evenly spaced
anchors. The ``--keep-best`` checkpoints with the lowest score stay (``final_checkpoint`` always stays and counts
towards them; the latest one also stays while the run trains); ``checkpoints.json`` records every score and what
was removed, and TensorBoard ``val/*`` the scores.

usage: uv run --locked python train_stream.py --config configs/train_colmap_rgbd_omega.py --init T-A/final_checkpoint \
           --policy full|JSON [--window-length 48] [--updates 1560] [--keep-best 1] [--val-windows 4] --output-dir OUT
"""

import argparse
import contextlib
import dataclasses
import json
import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from accelerate import PartialState
from accelerate.utils import set_seed
from safetensors.torch import save_file

import omnivggt.datasets
from omnivggt.datasets.base.easy_dataset import ResizedDataset
from omnivggt.datasets.colmap_rgbd import ColmapRgbd
from omnivggt.stream.kv_cache import CachePolicy
from omnivggt.stream.streaming import StreamingOmega
from omnivggt.utils.configs import read_config
from omnivggt.utils.normalization import normalize_camera_extrinsics_and_points_batch
from train_utils import (
    SMOKE_SPLIT,
    VALIDATION_PROGRESS,
    CheckpointKeeper,
    build_loss_criterion,
    build_optimizer,
    configure_schedule,
    evaluation_mode,
    evaluation_weights,
    isolated_rng,
    load_model,
    setup_tensorboard,
    smoke_split_spec,
)

FRAMES_PER_UPDATE = 12  # images per optimizer update: one 12-frame clip of the batch trainer
WINDOW_SEED = 42
WINDOW_TENSORS = ("images", "depth", "extrinsic", "intrinsic", "world_points", "valid_mask")
PRECISIONS = {"no": None, "bf16": torch.bfloat16}  # mixed_precision -> autocast and backbone cache dtype

logger = logging.getLogger("train_stream")


def parse_policy(text: str) -> CachePolicy:
    """``--policy``: "full" or the fields of a ``CachePolicy`` as a JSON object."""
    if text == "full":
        return CachePolicy.full()
    fields = json.loads(text)
    if not isinstance(fields, dict):
        raise ValueError(f"--policy must be full or a JSON object of CachePolicy fields, got {text!r}")
    return CachePolicy(**fields)


def window_anchors(count: int, frames: int, seed: int = WINDOW_SEED) -> np.ndarray:
    """First frames (dataset indices) of ``count`` windows: consecutive permutations of the ``frames`` train frames
    drawn from ``numpy.random.default_rng(seed)``, so every frame anchors a window before any anchors two."""
    if count < 1 or frames < 1:
        raise ValueError(f"need at least one window and one frame, got {count} windows of {frames} frames")
    generator = np.random.default_rng(seed)
    return np.concatenate([generator.permutation(frames) for _ in range(-(-count // frames))])[:count]


def _sequential_dataset(spec: str) -> ColmapRgbd:
    dataset = eval(spec, vars(omnivggt.datasets))  # "N @ ColmapRgbd(...)", as get_data_loader does
    if isinstance(dataset, ResizedDataset):
        dataset = dataset.dataset
    if not isinstance(dataset, ColmapRgbd) or dataset.view_selection != "sequential":
        raise ValueError("training on the stream reads ordered windows of ColmapRgbd: set "
                         "OMNIVGGT_VIEW_SELECTION=sequential")
    return dataset


def smallest_stride(stride) -> int:
    """The smallest of a ``ColmapRgbd.sequential_stride`` (one stride or a list of them)."""
    return min(stride) if isinstance(stride, list) else stride


def window_dataset(cfg) -> ColmapRgbd:
    """The ``ColmapRgbd`` of the config's ``train_dataset``, which must draw sequential views of the train split."""
    dataset = _sequential_dataset(cfg["train_dataset"])
    if dataset.split != "train":
        raise ValueError(f"training reads the train split, got {dataset.split!r}")
    return dataset


def validation_windows(cfg, count: int, length: int) -> List[Dict]:
    """The selection data of a run: ``count`` windows (``load_window``, plus their ``anchor``) of ``length`` sequential
    views of the smoke split, read like the training windows (``smoke_split_spec`` of the config's ``train_dataset``:
    the same roots, resolution and sample seed) without augmentation, at the smallest configured stride (a scene of
    the smoke split is short), from evenly spaced anchors: the same windows for every checkpoint."""
    try:
        dataset = _sequential_dataset(smoke_split_spec(cfg["train_dataset"], 1))
        dataset.sequential_stride = smallest_stride(dataset.sequential_stride)
        anchors = np.linspace(0, len(dataset) - 1, count).round().astype(int).tolist()
        return [{**load_window(dataset, anchor, length), "anchor": anchor} for anchor in anchors]
    except ValueError as err:
        raise ValueError(f"checkpoints are scored on {count} windows of {length} frames of the smoke split, which "
                         f"cannot be read: {err}") from err


def load_window(dataset: ColmapRgbd, anchor: int, length: int) -> Dict:
    """The window of ``length`` sequential views from ``anchor``: ``WINDOW_TENSORS`` as tensors [1, L, ...], and the
    ``instance`` (file) and ``label`` (scene) names of its frames."""
    sample = dataset[(int(anchor), 0, length)]
    window = {key: torch.as_tensor(sample[key])[None] for key in WINDOW_TENSORS}
    return {**window, "instance": list(sample["instance"]), "label": list(sample["label"])}


def split_window(window: Dict) -> Tuple[Dict, Dict]:
    """(model inputs, loss targets) of a window [1, L, ...]. The targets are normalised once over the whole window
    by its first frame (camera and mean valid point distance), as train_omnivggt.py normalises a clip; the inputs
    keep the raw depth."""
    extrinsic, _, world_points, depth = normalize_camera_extrinsics_and_points_batch(
        extrinsics=window["extrinsic"],
        cam_points=None,
        world_points=window["world_points"],
        depths=window["depth"],
        point_masks=window["valid_mask"],
        target_scale="first_frame",
    )
    inputs = {"images": window["images"], "depth": window["depth"], "mask": window["valid_mask"]}
    targets = {"images": window["images"], "extrinsic": extrinsic, "intrinsic": window["intrinsic"], "depth": depth,
               "world_points": world_points, "valid_mask": window["valid_mask"]}
    return inputs, targets


def frame_targets(targets: Dict, frame: int) -> Dict:
    """The targets of the window's frame ``frame`` (0-based), each [1, 1, ...]."""
    return {key: value[:, frame:frame + 1] for key, value in targets.items()}


class StreamTrainer:
    """The training state of a run: ``model`` streamed by ``StreamingOmega(policy, train=True)``, with the
    optimizer and its schedule for ``updates`` updates, the loss and the gradient clipping of ``cfg``. ``update``
    counts the updates done."""

    def __init__(self, model, policy: CachePolicy, cfg, updates: int):
        if updates < 1:
            raise ValueError(f"updates must be positive, got {updates}")
        if cfg["mixed_precision"] not in PRECISIONS:
            raise ValueError(f"mixed_precision must be one of {sorted(PRECISIONS)}, got {cfg['mixed_precision']!r}")
        self.model = model
        self.updates = updates
        self.update = 0
        self.autocast_dtype = PRECISIONS[cfg["mixed_precision"]]
        self.stream = StreamingOmega(model, policy, dtype=self.autocast_dtype, train=True)
        self.optimizer = build_optimizer(model, cfg)  # freezes the encoder as configured
        self.scheduler = configure_schedule(self.optimizer, cfg, updates)  # None for AMUSE (warm-up only)
        self.criterion = build_loss_criterion(cfg)
        self.max_grad_norm = cfg["max_grad_norm"]
        if hasattr(self.optimizer, "train_mode"):  # AMUSE takes the gradients at its training point y
            self.optimizer.train()

    def train_window(self, inputs: Dict, targets: Dict,
                     on_update: Optional[Callable[[int, Dict[str, float]], None]] = None) -> None:
        """Stream one window from t=1 (``split_window``): the loss of every frame goes backward as objective / 12,
        and every 12 frames one update follows. ``on_update(update, record)`` runs after each update, with the mean
        of every loss term over its frames and the gradient norm before clipping."""
        frames = inputs["images"].shape[1]
        if frames % FRAMES_PER_UPDATE:
            raise ValueError(f"a window holds whole updates: its length must be a multiple of {FRAMES_PER_UPDATE}, "
                             f"got {frames}")
        if self.update + frames // FRAMES_PER_UPDATE > self.updates:
            raise ValueError(f"a window of {frames} frames from update {self.update} runs past update {self.updates}")
        device_type = inputs["images"].device.type
        self.stream.reset(max_frames=frames)
        terms = []
        for frame in range(frames):
            with self._autocast(device_type):
                predictions = self.stream.step(inputs["images"][:, frame], inputs["depth"][:, frame],
                                               inputs["mask"][:, frame])
            loss = self.criterion(predictions, frame_targets(targets, frame), progress=self.update / self.updates)
            (loss["objective"] / FRAMES_PER_UPDATE).backward()
            terms.append({key: float(value) for key, value in loss.items()})
            if (frame + 1) % FRAMES_PER_UPDATE == 0:
                record = self._step(terms)
                terms = []
                if on_update is not None:
                    on_update(self.update, record)

    def evaluate_window(self, inputs: Dict, targets: Dict) -> Dict[str, float]:
        """The mean of every loss term over the frames of one window (``split_window``) streamed from t=1 without
        gradients, on a stream of its own (a training window in progress keeps its caches), with the loss at
        ``VALIDATION_PROGRESS`` (the end-of-training objective, the same for every checkpoint); no update."""
        frames = inputs["images"].shape[1]
        device_type = inputs["images"].device.type
        stream = StreamingOmega(self.model, self.stream.policy, dtype=self.autocast_dtype, train=True)
        stream.reset(max_frames=frames)
        terms = []
        with torch.no_grad():
            for frame in range(frames):
                with self._autocast(device_type):
                    predictions = stream.step(inputs["images"][:, frame], inputs["depth"][:, frame],
                                              inputs["mask"][:, frame])
                loss = self.criterion(predictions, frame_targets(targets, frame), progress=VALIDATION_PROGRESS)
                terms.append({key: float(value) for key, value in loss.items()})
        return {key: float(np.mean([term[key] for term in terms])) for key in terms[0]}

    def _autocast(self, device_type: str):
        if self.autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type, dtype=self.autocast_dtype)

    def _step(self, terms) -> Dict[str, float]:
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad()
        self.update += 1
        return {**{key: float(np.mean([term[key] for term in terms])) for key in terms[0]},
                "grad_norm": float(grad_norm)}


class StreamWindows(torch.utils.data.Dataset):
    """The windows of a run in order (``load_window`` of every anchor), for a loader that reads ahead."""

    def __init__(self, dataset: ColmapRgbd, anchors: np.ndarray, length: int):
        self.dataset, self.anchors, self.length = dataset, anchors, length

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, index: int) -> Dict:
        return load_window(self.dataset, self.anchors[index], self.length)


def validation_loss(trainer: StreamTrainer, windows: List[Dict], device) -> Dict[str, float]:
    """The score of the current (evaluation) weights: the mean of every loss term over ``windows``
    (``validation_windows``, each ``StreamTrainer.evaluate_window``), with the model in eval mode and every random
    state restored afterwards, so that the run trains on as without it."""
    terms = []
    with isolated_rng(), evaluation_mode(trainer.model):
        for window in windows:
            terms.append(trainer.evaluate_window(*split_window({key: window[key].to(device) for key in WINDOW_TENSORS})))
    return {key: float(np.mean([term[key] for term in terms])) for key in terms[0]}


def save_checkpoint(model, optimizer, directory: Path) -> None:
    """``directory/model.safetensors``: the evaluation weights (AMUSE: the averaged x), as train_omnivggt.py saves
    them."""
    directory.mkdir(parents=True)
    with evaluation_weights(optimizer):
        save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()},
                  str(directory / "model.safetensors"), metadata={"format": "pt"})
    logger.info(f"saved {directory}")


def configure(cfg, args) -> None:
    """Check that ``cfg`` and ``args`` describe a run on the stream, and point ``cfg`` at the run's initial weights
    and output directory."""
    if args.window_length < FRAMES_PER_UPDATE or args.window_length % FRAMES_PER_UPDATE:
        raise ValueError(f"--window-length must be a positive multiple of {FRAMES_PER_UPDATE}, got {args.window_length}")
    per_window = args.window_length // FRAMES_PER_UPDATE
    if args.updates < 1 or args.updates % per_window:
        raise ValueError(f"--updates must be a positive multiple of the {per_window} updates of a window, "
                         f"got {args.updates}")
    if args.checkpoint_every < 1:
        raise ValueError(f"--checkpoint-every must be positive, got {args.checkpoint_every}")
    if args.keep_best < 0:
        raise ValueError(f"--keep-best must be a non-negative integer, got {args.keep_best}")
    if args.val_windows < 1:
        raise ValueError(f"--val-windows must be positive, got {args.val_windows}")
    if cfg["target_scale"] != "first_frame":
        raise ValueError("training on the stream normalises the targets by the first frame: set "
                         "OMNIVGGT_TARGET_SCALE=first_frame")
    if not cfg["depth_all_views"]:
        raise ValueError("the stream gives every frame its depth: set OMNIVGGT_DEPTH_ALL_VIEWS=1")
    if cfg["wandb"]:
        raise NotImplementedError("train_stream.py logs to TensorBoard only")
    if cfg["init_checkpoint"] not in (None, str(args.init)):
        raise ValueError(f"the initial weights are named once, by --init {args.init}: unset OMNIVGGT_INIT_CHECKPOINT "
                         f"({cfg['init_checkpoint']})")
    cfg.init_checkpoint = str(args.init)
    cfg.output_dir = str(args.output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/train_colmap_rgbd_omega.py")
    parser.add_argument("--init", type=Path, required=True,
                        help="trained OmniVGGTOmega checkpoint (directory or model.safetensors), loaded strictly")
    parser.add_argument("--policy", required=True, help='"full" or a CachePolicy JSON object')
    parser.add_argument("--window-length", type=int, default=48, help="frames per window, a multiple of 12")
    parser.add_argument("--updates", type=int, default=1560, help="optimizer updates (12 frames each)")
    parser.add_argument("--checkpoint-every", type=int, default=390, help="updates between checkpoint-u<N>")
    parser.add_argument("--keep-best", type=int, default=1,
                        help="checkpoints kept by their smoke-split score; final_checkpoint always stays and counts "
                             "towards them (0: final_checkpoint only)")
    parser.add_argument("--val-windows", type=int, default=4,
                        help="fixed smoke-split windows of --window-length frames that score every checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2, help="processes loading the next windows")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = read_config(args.config)
    configure(cfg, args)
    policy = parse_policy(args.policy)
    dataset = window_dataset(cfg)
    windows = validation_windows(cfg, args.val_windows, args.window_length)  # before training: needs the smoke split
    save_dir = Path(cfg["output_dir"]) / cfg["exp_name"]
    if save_dir.exists():
        raise FileExistsError(f"{save_dir} exists: a run starts in a new output directory (there is no resume)")
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
    PartialState()  # train_utils logs through accelerate
    set_seed(cfg["seed"])
    device = torch.device(args.device)
    model, _ = load_model(cfg, device)  # strict load of --init; writes <output-dir>/weight_transfer_report.json
    model.train()
    trainer = StreamTrainer(model, policy, cfg, args.updates)
    per_window = args.window_length // FRAMES_PER_UPDATE
    anchors = window_anchors(args.updates // per_window, len(dataset))
    save_dir.mkdir(parents=True)
    keeper = CheckpointKeeper(save_dir, args.keep_best)
    run = {
        "init": str(args.init), "policy": dataclasses.asdict(policy), "window_length": args.window_length,
        "updates": args.updates, "frames_per_update": FRAMES_PER_UPDATE, "checkpoint_every": args.checkpoint_every,
        "window_seed": WINDOW_SEED, "data_seed": cfg["data_seed"], "seed": cfg["seed"],
        "strides": dataset.sequential_stride, "mixed_precision": cfg["mixed_precision"],
        "max_grad_norm": trainer.max_grad_norm, "optimizer": cfg["optimizer_type"],
        "tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn": torch.backends.cudnn.allow_tf32},
        "keep_best": args.keep_best,
        "validation": {"split": SMOKE_SPLIT, "windows": args.val_windows, "window_length": args.window_length,
                       "stride": smallest_stride(dataset.sequential_stride), "progress": VALIDATION_PROGRESS,
                       "anchors": [window["anchor"] for window in windows]},
        "window_anchors": anchors.tolist(),
    }
    (save_dir / "stream_training.json").write_text(json.dumps(run, indent=1) + "\n")
    logger.info(f"training on the stream: {json.dumps({k: v for k, v in run.items() if k != 'window_anchors'})}")
    writer = setup_tensorboard(cfg, str(save_dir))
    loader = torch.utils.data.DataLoader(StreamWindows(dataset, anchors, args.window_length), batch_size=None,
                                         num_workers=args.num_workers, pin_memory=device.type == "cuda")
    clock = {"start": time.time(), "last": time.time()}

    def checkpoint(name: str, update: int, final: bool = False) -> None:
        with evaluation_weights(trainer.optimizer):  # save and score the same (evaluation) weights
            save_checkpoint(model, trainer.optimizer, save_dir / name)
            score = validation_loss(trainer, windows, device)
        extra = {"split": SMOKE_SPLIT, "windows": len(windows), "progress": VALIDATION_PROGRESS, "components": score}
        if final:
            entry = keeper.finalize(name, score["objective"], update, extra=extra)
        else:
            entry = keeper.record(name, score["objective"], update, extra=extra)
        logger.info(f"{name}: smoke-split objective {score['objective']:.6f}; kept {entry['kept']}, "
                    f"removed {entry['removed']}")
        if writer is not None:
            for key, value in score.items():
                writer.add_scalar(f"val/{key}", value, update)

    def on_update(update: int, record: Dict[str, float]) -> None:
        now = time.time()
        if writer is not None:
            for key, value in record.items():
                writer.add_scalar(f"train/{key}", value, update)
            for index, group in enumerate(trainer.optimizer.param_groups):
                writer.add_scalar(f"lr/group_{index}", group["lr"], update)
            writer.add_scalar("train/seconds_per_update", now - clock["last"], update)
        clock["last"] = now
        if update % cfg["num_save_log"] == 0 or update == args.updates:
            logger.info(f"update {update}/{args.updates} objective {record['objective']:.4f} "
                        f"grad_norm {record['grad_norm']:.3f} elapsed {now - clock['start']:.0f}s")
        if update % args.checkpoint_every == 0 and update < args.updates:
            checkpoint(f"checkpoint-u{update}", update)

    for window in loader:
        inputs, targets = split_window({key: window[key].to(device, non_blocking=True) for key in WINDOW_TENSORS})
        trainer.train_window(inputs, targets, on_update)
    checkpoint("final_checkpoint", args.updates, final=True)
    if writer is not None:
        writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
