"""Evaluate OmniVGGT on a colmap_rgbd_v1 split with and without auxiliary inputs.

For fixed, evenly spaced anchors of the split, ``--frames`` views are taken
sequentially (``--stride`` apart) and inferred under four conditions:
RGB only, +depth (all views), +camera (all views), +depth+camera.
Metrics: pairwise relative rotation / translation-direction errors (RRA@5,
RTA@5, AUC@30) and depth AbsRel / delta<1.25 after one median scale per sample
(OmniVGGT predicts in a normalised scale).

usage: PYTHONPATH=tools uv run python -m eval_colmap_rgbd --roots R1 [R2 ...] --split val \
           --checkpoint W.safetensors|ACCELERATE_DIR [--model-config VARIANT.json] --output eval.json
(``--model-config`` selects an OmniVGGTOmega variant; without it the model is OmniVGGT.)
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np

CONDITIONS = {"rgb": (False, False), "depth": (True, False), "camera": (False, True), "depth+camera": (True, True)}


def _homogeneous(w2c: np.ndarray) -> np.ndarray:
    out = np.tile(np.eye(4), (len(w2c), 1, 1))
    out[:, :3, :] = w2c
    return out


def _angle_deg(rotation: np.ndarray) -> float:
    cos = (np.trace(rotation) - 1.0) / 2.0
    skew = rotation - rotation.T
    sin = 0.5 * np.linalg.norm([skew[2, 1], skew[0, 2], skew[1, 0]])
    return float(np.degrees(np.arctan2(sin, cos)))


def pose_metrics(pred_w2c: np.ndarray, gt_w2c: np.ndarray) -> dict:
    pred, gt = _homogeneous(np.asarray(pred_w2c, float)), _homogeneous(np.asarray(gt_w2c, float))
    rot, tra = [], []
    for i, j in itertools.combinations(range(len(gt)), 2):
        rel_p, rel_g = pred[j] @ np.linalg.inv(pred[i]), gt[j] @ np.linalg.inv(gt[i])
        rot.append(_angle_deg(rel_p[:3, :3].T @ rel_g[:3, :3]))
        tp, tg = rel_p[:3, 3], rel_g[:3, 3]
        cos = tp @ tg / max(np.linalg.norm(tp) * np.linalg.norm(tg), 1e-12)
        tra.append(float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))))
    rot, tra = np.array(rot), np.array(tra)
    worst = np.maximum(rot, tra)
    auc = float(np.mean([np.mean(worst < t) for t in range(1, 31)]))
    return {
        "rot_err_mean_deg": float(rot.mean()),
        "trans_dir_err_mean_deg": float(tra.mean()),
        "RRA@5": float(np.mean(rot < 5)),
        "RTA@5": float(np.mean(tra < 5)),
        "AUC@30": auc,
    }


def depth_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict:
    p, g = np.asarray(pred, float)[mask], np.asarray(gt, float)[mask]
    p = p * (np.median(g) / np.median(p))
    return {"abs_rel": float(np.mean(np.abs(p - g) / g)), "delta<1.25": float(np.mean(np.maximum(p / g, g / p) < 1.25))}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sequential_dataset(roots, split: str, resolution, stride: int):
    """The evaluation dataset: ``ColmapRgbd`` sequential views, ``stride`` frames apart, no augmentation."""
    from omnivggt.datasets.colmap_rgbd import ColmapRgbd
    from omnivggt.datasets.utils.transforms import ImgNorm

    return ColmapRgbd(
        roots=roots,
        split=split,
        resolution=[tuple(resolution)],
        transform=ImgNorm,
        aug_crop=0,
        seed=1,
        view_selection="sequential",
        sequential_stride=stride,
    )


def sequential_anchors(dataset, frames: int, stride: int, num_samples: int) -> list[int]:
    """``num_samples`` evenly spaced anchors whose ``frames`` sequential views stay inside one scene."""
    span = (frames - 1) * stride
    valid = [
        i
        for i in range(len(dataset))
        if i + span < len(dataset) and dataset.scene_labels[i + span] == dataset.scene_labels[i]
    ]
    return [valid[k] for k in np.linspace(0, len(valid) - 1, num_samples).round().astype(int)]


def _build_model(model_config, **model_options):
    """OmniVGGT, or the OmniVGGTOmega variant described by ``model_config`` (a variant JSON) built with
    ``model_options`` (e.g. ``causal``, ``depth_norm``)."""
    if model_config is None:
        if model_options:
            raise ValueError(f"model options {sorted(model_options)} need an OmniVGGTOmega variant")
        from omnivggt.models import omnivggt

        return omnivggt.OmniVGGT()
    from omnivggt.models.omnivggt_omega import OmniVGGTOmega

    return OmniVGGTOmega.from_variant(model_config, **model_options)


def _model_config_record(model_config):
    if model_config is None:
        return None
    path = Path(model_config)
    return {"path": str(path), "sha256": _sha256(path), "variant": json.loads(path.read_text())}


def _check_variant_provenance(model_config, checkpoint: Path):
    """For a training run checkpoint (<output_dir>/<exp>/<checkpoint>/), require that ``model_config`` is the
    variant the run was trained with (recorded in <output_dir>/weight_transfer_report.json)."""
    if model_config is None:
        return None
    report = Path(checkpoint).parent.parent / "weight_transfer_report.json"
    if not Path(checkpoint).is_dir() or not report.is_file():
        return "unverified"
    trained = json.loads(report.read_text())["variant"]["sha256"]
    if trained != _sha256(Path(model_config)):
        raise ValueError(f"variant {model_config} is not the one this run was trained with ({report})")
    return "verified"


def _load_model(checkpoint: Path, device: str, model_config=None, **model_options):
    from safetensors.torch import load_file

    path = checkpoint / "model.safetensors" if checkpoint.is_dir() else checkpoint
    if not path.is_file():
        raise FileNotFoundError(path)
    model = _build_model(model_config, **model_options)
    model.load_state_dict(load_file(str(path)), strict=True)
    return model.to(device).eval(), path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, help="OmniVGGTOmega variant JSON (default: OmniVGGT)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--resolution", type=int, nargs=2, default=(392, 294), metavar=("W", "H"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    import torch

    from omnivggt.utils.pose_enc import pose_encoding_to_extri_intri

    dataset = sequential_dataset(args.roots, args.split, args.resolution, args.stride)
    anchors = sequential_anchors(dataset, args.frames, args.stride, args.num_samples)
    variant_provenance = _check_variant_provenance(args.model_config, args.checkpoint)
    model, weights = _load_model(args.checkpoint, "cuda", args.model_config)

    samples = []
    start = time.time()
    for anchor in anchors:
        item = dataset[(anchor, 0, args.frames)]
        images = item["images"][None].cuda()
        extrinsics = torch.from_numpy(item["extrinsic"])[None].float().cuda()
        intrinsics = torch.from_numpy(item["intrinsic"])[None].float().cuda()
        depth = torch.from_numpy(item["depth"])[None].float().cuda()
        mask = torch.from_numpy(item["valid_mask"])[None].float().cuda()
        record = {"anchor": int(anchor), "label": item["label"][0], "instances": item["instance"]}
        for name, (use_depth, use_camera) in CONDITIONS.items():
            views = list(range(args.frames))
            with torch.no_grad():
                pred = model.inference(
                    images=images,
                    extrinsics=extrinsics,
                    intrinsics=intrinsics,
                    depth=depth,
                    mask=mask,
                    depth_gt_index=views if use_depth else [],
                    camera_gt_index=views if use_camera else [],
                )
            pred_w2c, _ = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
            record[name] = {
                **pose_metrics(pred_w2c[0].cpu().numpy(), item["extrinsic"]),
                **depth_metrics(pred["depth"][0, ..., 0].cpu().numpy(), item["depth"][..., 0], item["valid_mask"]),
            }
        samples.append(record)
    aggregate = {
        name: {key: float(np.mean([s[name][key] for s in samples])) for key in samples[0][name]} for name in CONDITIONS
    }
    output = {
        "weights": weights.name,
        "weights_record": {"path": str(weights.resolve()), "sha256": _sha256(weights)},
        "model_config": _model_config_record(args.model_config),
        "variant_provenance": variant_provenance,
        "split": args.split,
        "roots": [Path(r).name for r in args.roots],
        "frames": args.frames,
        "stride": args.stride,
        "resolution_wh": list(args.resolution),
        "anchors": anchors,
        "seconds": time.time() - start,
        "aggregate": aggregate,
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=1) + "\n")
    print(json.dumps(aggregate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
