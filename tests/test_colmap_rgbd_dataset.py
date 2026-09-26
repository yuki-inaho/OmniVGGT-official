import json

import numpy as np
import pytest
from PIL import Image

from omnivggt.datasets.colmap_rgbd import ColmapRgbd
from omnivggt.datasets.utils.transforms import ImgNorm

W, H = 64, 48
RESOLUTION = (56, 42)


def _write_root(root, frames=40, max_depth_mm=1300):
    scene = root / "scenes" / "scene_000000"
    (scene / "rgb").mkdir(parents=True)
    (scene / "depth").mkdir()
    rng = np.random.default_rng(0)
    k = np.array([[60.0, 0, W / 2], [0, 60.0, H / 2], [0, 0, 1]], dtype=np.float32)
    w2c = np.zeros((frames, 3, 4), dtype=np.float32)
    for index in range(frames):
        w2c[index, :3, :3] = np.eye(3)
        w2c[index, :3, 3] = [0.0, -0.01 * index, 0.0]  # camera centre moves along +y
        Image.fromarray(rng.integers(0, 255, (H, W, 3), dtype=np.uint8)).save(scene / "rgb" / f"frame_{index:06d}.png")
        depth = rng.integers(400, max_depth_mm, (H, W), dtype=np.uint16)
        depth[:4] = 0
        Image.fromarray(depth).save(scene / "depth" / f"frame_{index:06d}.png")
    np.savez_compressed(
        scene / "cameras.npz",
        frame_ids=np.arange(frames),
        intrinsics=np.repeat(k[None], frames, 0),
        extrinsics_w2c=w2c,
        quality_flags=np.ones(frames, bool),
        chunk_ids=np.zeros(frames, np.int64),
    )
    # train: frames 0..23, val: 28..35, smoke: 38..39 (sequences of length 4, padded with -1)
    sequences = np.array(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23],
            [28, 29, 30, 31],
            [32, 33, 34, 35],
            [38, 39, -1, -1],
        ]
    )
    np.savez_compressed(
        scene / "sequences.npz",
        sequences=sequences,
        lengths=np.array([4] * 8 + [2]),
        split_ids=np.array([0] * 6 + [1, 1, 2]),
        chunk_ids=np.zeros(9, np.int64),
    )
    meta = {
        "format": "colmap_rgbd_v1",
        "schema_version": 1,
        "frame_count": frames,
        "depth": {"unit": "millimeters", "invalid_value": 0, "max_depth_mm": max_depth_mm},
        "camera": {"extrinsics": "opencv_world_to_camera", "intrinsics": "pixel_units"},
    }
    (root / "dataset.json").write_text(json.dumps(meta))
    return root


def _dataset(tmp_path, split="train", **kwargs):
    return ColmapRgbd(
        roots=[str(_write_root(tmp_path / "root"))],
        split=split,
        top_k=8,
        z_far=2.0,
        resolution=[RESOLUTION],
        transform=ImgNorm,
        aug_crop=0,
        seed=7,
        **kwargs,
    )


def test_sample_keys_shapes_and_units(tmp_path):
    sample = _dataset(tmp_path)[(3, 0, 4)]
    assert sample["images"].shape == (4, 3, RESOLUTION[1], RESOLUTION[0])
    assert sample["depth"].shape == (4, RESOLUTION[1], RESOLUTION[0], 1)
    assert sample["extrinsic"].shape == (4, 3, 4) and sample["intrinsic"].shape == (4, 3, 3)
    assert sample["world_points"].shape == (4, RESOLUTION[1], RESOLUTION[0], 3)
    assert sample["valid_mask"].dtype == bool and sample["valid_mask"].any()
    valid_depth = sample["depth"][..., 0][sample["valid_mask"]]
    assert 0.39 < valid_depth.min() and valid_depth.max() < 1.3


def test_world_points_reproject_to_their_pixels(tmp_path):
    sample = _dataset(tmp_path)[(5, 0, 3)]
    for view in range(3):
        extrinsic, intrinsic = sample["extrinsic"][view], sample["intrinsic"][view]
        mask = sample["valid_mask"][view]
        points = sample["world_points"][view][mask]
        camera = points @ extrinsic[:3, :3].T + extrinsic[:3, 3]
        pixels = camera @ intrinsic.T
        pixels = pixels[:, :2] / pixels[:, 2:]
        v, u = np.nonzero(mask)
        assert np.allclose(pixels, np.stack([u, v], axis=1), atol=1e-3)


@pytest.mark.parametrize("split,allowed", [("train", set(range(24))), ("val", set(range(28, 36)))])
def test_views_stay_inside_the_split(tmp_path, split, allowed):
    dataset = _dataset(tmp_path, split=split)
    assert len(dataset) == len(allowed)
    for index in range(len(dataset)):
        sample = dataset[(index, 0, 6)]
        frames = {int(name.split("frame_")[1].split(".")[0]) for name in sample["instance"]}
        assert frames <= allowed


def test_unknown_split_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        _dataset(tmp_path, split="test")


def test_resized_dataset_tuple_indices_follow_the_sampler_contract(tmp_path):
    dataset = 8 @ _dataset(tmp_path)
    dataset.set_epoch(0)
    samples = dataset[(0, 1, 0, 4)]
    assert len(samples) == 2 and all(s["images"].shape[0] == 2 for s in samples)


def test_sequential_view_selection_is_deterministic(tmp_path):
    dataset = _dataset(tmp_path, view_selection="sequential", sequential_stride=2)
    sample = dataset[(3, 0, 4)]
    assert sample["instance"] == [f"frame_{i:06d}.png" for i in (3, 5, 7, 9)]
    late = dataset[(20, 0, 4)]  # 20..26 would run past the end of the train split (23): moved back to end there
    assert late["instance"] == [f"frame_{i:06d}.png" for i in (17, 19, 21, 23)]
