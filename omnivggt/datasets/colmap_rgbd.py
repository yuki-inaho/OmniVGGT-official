"""Loader for ``colmap_rgbd_v1`` staging sets (RGB-D sequences with aligned metric poses).

Layout of one root::

    dataset.json                              format colmap_rgbd_v1, depth unit millimeters, invalid 0
    scenes/<scene>/rgb/frame_%06d.png         uint8 RGB
    scenes/<scene>/depth/frame_%06d.png       uint16 depth [mm], 0 = invalid
    scenes/<scene>/cameras.npz                intrinsics (N,3,3), extrinsics_w2c (N,3,4) (OpenCV)
    scenes/<scene>/sequences.npz              sequences (M,L) padded with -1, lengths (M,), split_ids (M,)

A split (``train`` / ``val`` / ``smoke``) is the set of frames referenced by the
sequences of that split; the exporter separates splits with guard frames.  Views
of one sample are the anchor frame plus frames drawn from its ``top_k`` nearest
poses inside the same scene and split, so no view crosses a split boundary.
"""

import json
import os.path as osp
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from omnivggt.datasets.base.base_stereo_view_dataset import (
    BaseStereoViewDataset,
    is_good_type,
    transpose_to_landscape,
    view_name,
)
from omnivggt.datasets.utils.image_ranking import compute_ranking
from omnivggt.utils.geometry import closed_form_inverse_se3, depth_to_world_coords_points

SPLIT_NAMES = ("train", "val", "smoke")
FORMAT = "colmap_rgbd_v1"


def _check_metadata(meta: dict, root: Path) -> float:
    if meta.get("format") != FORMAT:
        raise ValueError(f"{root}: expected format {FORMAT!r}, got {meta.get('format')!r}")
    depth, camera = meta.get("depth", {}), meta.get("camera", {})
    if depth.get("unit") != "millimeters" or depth.get("invalid_value") != 0:
        raise ValueError(f"{root}: depth must be millimeters with invalid_value 0")
    if camera.get("extrinsics") != "opencv_world_to_camera":
        raise ValueError(f"{root}: extrinsics must be opencv_world_to_camera")
    return 1e-3


class ColmapRgbd(BaseStereoViewDataset):
    def __init__(
        self,
        roots,
        split="train",
        top_k=32,
        z_far=2.0,
        view_selection="random_topk",
        sequential_stride=1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if split not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {SPLIT_NAMES}, got {split!r}")
        if view_selection not in ("random_topk", "sequential"):
            raise ValueError(f"unknown view_selection {view_selection!r}")
        self.view_selection, self.sequential_stride = view_selection, sequential_stride
        self.dataset_label = "ColmapRgbd"
        self.split, self.top_k, self.z_far = split, top_k, z_far
        roots = [roots] if isinstance(roots, (str, Path)) else list(roots)
        if not roots:
            raise ValueError("at least one colmap_rgbd_v1 root is required")

        self.rgb_paths, self.depth_paths, self.scene_labels = [], [], []
        self.intrinsics, self.camera_to_world, self.rank = [], [], {}
        self.depth_scale = []
        for root in map(Path, roots):
            depth_scale = _check_metadata(json.loads((root / "dataset.json").read_text()), root)
            for scene in sorted(p for p in (root / "scenes").iterdir() if p.is_dir()):
                self._add_scene(root, scene, depth_scale)
        if not self.rgb_paths:
            raise ValueError(f"split {split!r} contains no frames")
        self.full_idxs = list(range(len(self.rgb_paths)))

    def _add_scene(self, root: Path, scene: Path, depth_scale: float) -> None:
        with np.load(scene / "cameras.npz") as cameras, np.load(scene / "sequences.npz") as sequences:
            intrinsics, w2c = cameras["intrinsics"], cameras["extrinsics_w2c"]
            selected = sequences["split_ids"] == SPLIT_NAMES.index(self.split)
            rows = zip(sequences["sequences"][selected], sequences["lengths"][selected], strict=True)
            frames = sorted({int(f) for seq, n in rows for f in seq[:n]})
        if not frames:
            return
        c2w = closed_form_inverse_se3(np.asarray(w2c, dtype=np.float32)[frames])
        offset = len(self.rgb_paths)
        ranking, _ = compute_ranking(c2w, lambda_t=1.0, normalize=True, batched=True)
        for local, frame in enumerate(frames):
            self.rgb_paths.append(str(scene / "rgb" / f"frame_{frame:06d}.png"))
            self.depth_paths.append(str(scene / "depth" / f"frame_{frame:06d}.png"))
            self.scene_labels.append(f"{root.name}/{scene.name}")
            self.intrinsics.append(np.asarray(intrinsics[frame], dtype=np.float32))
            self.camera_to_world.append(c2w[local].astype(np.float32))
            self.depth_scale.append(depth_scale)
            self.rank[offset + local] = ranking[local] + offset

    def __len__(self):
        return len(self.full_idxs)

    def _get_views(self, index, num, resolution, rng):
        anchor = self.full_idxs[index]
        if self.view_selection == "sequential":
            chosen = [anchor + k * self.sequential_stride for k in range(num)]
            if chosen[-1] >= len(self.rgb_paths) or self.scene_labels[chosen[-1]] != self.scene_labels[anchor]:
                raise IndexError(f"sequential views from {anchor} leave the scene/split")
        elif num > 1:
            candidates = self.rank[anchor][: min(self.top_k, len(self.rank[anchor]))]
            chosen = [anchor, *rng.choice(candidates, size=num - 1, replace=True).tolist()]
        else:
            chosen = [anchor]
        views = []
        for frame in chosen:
            rgb = Image.open(self.rgb_paths[frame]).convert("RGB")
            depth = np.asarray(Image.open(self.depth_paths[frame]), dtype=np.float32) * self.depth_scale[frame]
            rgb, depth, intrinsics = self._crop_resize_if_necessary(
                rgb, depth, self.intrinsics[frame].copy(), resolution, rng, info=self.rgb_paths[frame]
            )
            views.append(
                {
                    "img": rgb,
                    "depthmap": depth.astype(np.float32),
                    "camera_pose": self.camera_to_world[frame].copy(),
                    "camera_intrinsics": intrinsics.astype(np.float32),
                    "dataset": self.dataset_label,
                    "label": self.scene_labels[frame],
                    "instance": osp.basename(self.rgb_paths[frame]),
                }
            )
        return views

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx, ar_idx, *num_args = idx
            num = num_args[0] if num_args else 1
        else:
            assert len(self._resolutions) == 1
            ar_idx, num = 0, 1
        if self.seed:
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, "_rng"):
            self._rng = np.random.default_rng()
        views = self._get_views(idx, num, self._resolutions[ar_idx], self._rng)

        for v, view in enumerate(views):
            assert np.isfinite(view["depthmap"]).all(), f"NaN in depthmap for view {view_name(view)}"
            view["idx"] = (idx, ar_idx, v)
            view["true_shape"] = np.int32(view["img"].size[::-1])
            view["img"] = self.transform(view["img"])
            for key, val in view.items():
                res, err_msg = is_good_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"
            view["camera_pose"] = closed_form_inverse_se3(view["camera_pose"][None])[0]  # -> world_to_camera
            world, _, mask = depth_to_world_coords_points(
                view["depthmap"], view["camera_pose"], view["camera_intrinsics"], z_far=self.z_far
            )
            view["world_coords_points"], view["point_mask"] = world, mask
        for view in views:
            transpose_to_landscape(view)

        return {
            "images": torch.stack([v["img"] for v in views]),
            "depth": np.stack([v["depthmap"][:, :, None] for v in views]),
            "extrinsic": np.stack([v["camera_pose"][:3] for v in views]),
            "intrinsic": np.stack([v["camera_intrinsics"] for v in views]),
            "world_points": np.stack([v["world_coords_points"] for v in views]),
            "true_shape": np.array([v["true_shape"] for v in views]),
            "valid_mask": np.stack([v["point_mask"] for v in views]),
            "label": [v["label"] for v in views],
            "instance": [v["instance"] for v in views],
            "dataset": self.dataset_label,
        }
