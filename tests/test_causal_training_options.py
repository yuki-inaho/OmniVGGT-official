"""Stream-Omega training options: environment overrides of configs/train_colmap_rgbd_omega.py, the frame-0
target scale, ordered clips (ColmapRgbd sequential views) and full clips (AnchorFrameSampler)."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from omnivggt.datasets import get_data_loader
from omnivggt.datasets.base.batched_sampler import AnchorFrameSampler
from omnivggt.datasets.base.easy_dataset import EasyDataset
from omnivggt.datasets.colmap_rgbd import ColmapRgbd
from omnivggt.datasets.utils.transforms import ImgNorm
from omnivggt.utils.configs import read_config
from omnivggt.utils.normalization import normalize_camera_extrinsics_and_points_batch

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "train_colmap_rgbd_omega.py"
OPTIONS = ("OMNIVGGT_CAUSAL", "OMNIVGGT_DEPTH_NORM", "OMNIVGGT_TARGET_SCALE", "OMNIVGGT_VIEW_SELECTION",
           "OMNIVGGT_SEQ_STRIDES", "OMNIVGGT_CAM_DROP_PROB", "OMNIVGGT_FULL_CLIPS")
DEFAULTS = {"causal": False, "depth_norm": "joint", "target_scale": "all", "view_selection": "random_topk",
            "sequential_strides": [1], "cam_drop_prob": 0.1, "full_clips": False}
STAGE_A = {"OMNIVGGT_DEPTH_NORM": "first_frame", "OMNIVGGT_TARGET_SCALE": "first_frame",
           "OMNIVGGT_CAM_DROP_PROB": "1", "OMNIVGGT_VIEW_SELECTION": "sequential", "OMNIVGGT_SEQ_STRIDES": "1,2,3,4",
           "OMNIVGGT_FULL_CLIPS": "1"}


def _train_dataset(views=""):
    return ("1000 @ ColmapRgbd(roots=['/data/a', '/data/b'], split='train', top_k=32, z_far=2, "
            f"aug_crop=16, resolution=[(392, 294)], transform=ColorJitter, seed=985{views})")


# == configs/train_colmap_rgbd_omega.py ==

@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("OMNIVGGT_COLMAP_RGBD_ROOTS", "/data/a,/data/b")
    monkeypatch.setenv("OMNIVGGT_OMEGA_VARIANT", "configs/omnivggt_omega/variants/V5.json")
    for name in (*OPTIONS, "OMNIVGGT_STEPS_PER_EPOCH", "OMNIVGGT_GRAD_ACCUM", "OMNIVGGT_RESOLUTION", "OMNIVGGT_DATA_SEED"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _config(env, overrides=None):
    for name, value in (overrides or {}).items():
        env.setenv(name, value)
    return {k: v for k, v in read_config(str(CONFIG)).to_dict().items() if not k.startswith("_")}


def test_defaults_keep_the_existing_recipe(env):
    cfg = _config(env)
    assert {key: cfg.get(key) for key in DEFAULTS} == DEFAULTS
    assert cfg["train_dataset"] == _train_dataset()


@pytest.mark.parametrize(
    "overrides, changed",
    [
        ({"OMNIVGGT_DEPTH_NORM": "first_frame"}, {"depth_norm": "first_frame"}),
        ({"OMNIVGGT_TARGET_SCALE": "first_frame"}, {"target_scale": "first_frame"}),
        ({"OMNIVGGT_CAM_DROP_PROB": "1"}, {"cam_drop_prob": 1.0}),
        ({"OMNIVGGT_FULL_CLIPS": "1"}, {"full_clips": True}),
        ({"OMNIVGGT_CAUSAL": "1", "OMNIVGGT_CAM_DROP_PROB": "1"}, {"causal": True, "cam_drop_prob": 1.0}),
        ({"OMNIVGGT_VIEW_SELECTION": "sequential"},
         {"view_selection": "sequential",
          "train_dataset": _train_dataset(", view_selection='sequential', sequential_stride=[1]")}),
        ({"OMNIVGGT_VIEW_SELECTION": "sequential", "OMNIVGGT_SEQ_STRIDES": "1,2,3,4"},
         {"view_selection": "sequential", "sequential_strides": [1, 2, 3, 4],
          "train_dataset": _train_dataset(", view_selection='sequential', sequential_stride=[1, 2, 3, 4]")}),
    ],
)
def test_each_override_changes_only_its_own_keys(env, overrides, changed):
    default, cfg = _config(env), _config(env, overrides)
    assert {key for key in cfg if cfg[key] != default.get(key)} == set(changed)
    assert {key: cfg[key] for key in changed} == changed


def test_stage_a_runs_differ_only_in_causal(env):
    bidirectional = _config(env, {**STAGE_A, "OMNIVGGT_CAUSAL": "0"})
    causal = _config(env, {"OMNIVGGT_CAUSAL": "1"})
    assert {key for key in causal if causal[key] != bidirectional[key]} == {"causal"}
    assert causal["causal"] is True and bidirectional["causal"] is False
    assert bidirectional["depth_norm"] == "first_frame" and bidirectional["target_scale"] == "first_frame"
    assert bidirectional["full_clips"] is True and bidirectional["cam_drop_prob"] == 1.0
    assert "view_selection='sequential', sequential_stride=[1, 2, 3, 4]" in bidirectional["train_dataset"]


@pytest.mark.parametrize("cam_drop_prob", [None, "0", "0.5", "0.99"])
def test_causal_training_needs_camera_input_always_dropped(env, cam_drop_prob):
    env.setenv("OMNIVGGT_CAUSAL", "1")
    if cam_drop_prob is not None:
        env.setenv("OMNIVGGT_CAM_DROP_PROB", cam_drop_prob)
    with pytest.raises(ValueError, match="OMNIVGGT_CAM_DROP_PROB"):
        read_config(str(CONFIG))


@pytest.mark.parametrize(
    "overrides, name",
    [
        ({"OMNIVGGT_DEPTH_NORM": "frame0"}, "OMNIVGGT_DEPTH_NORM"),
        ({"OMNIVGGT_TARGET_SCALE": "last"}, "OMNIVGGT_TARGET_SCALE"),
        ({"OMNIVGGT_VIEW_SELECTION": "topk"}, "OMNIVGGT_VIEW_SELECTION"),
        ({"OMNIVGGT_VIEW_SELECTION": "sequential", "OMNIVGGT_SEQ_STRIDES": "0"}, "OMNIVGGT_SEQ_STRIDES"),
        ({"OMNIVGGT_VIEW_SELECTION": "sequential", "OMNIVGGT_SEQ_STRIDES": "1,,2"}, "OMNIVGGT_SEQ_STRIDES"),
        ({"OMNIVGGT_VIEW_SELECTION": "sequential", "OMNIVGGT_SEQ_STRIDES": "1.5"}, "OMNIVGGT_SEQ_STRIDES"),
        ({"OMNIVGGT_SEQ_STRIDES": "1,2"}, "OMNIVGGT_SEQ_STRIDES"),  # strides only apply to sequential clips
        ({"OMNIVGGT_CAM_DROP_PROB": "1.5"}, "OMNIVGGT_CAM_DROP_PROB"),
    ],
)
def test_invalid_values_are_rejected(env, overrides, name):
    with pytest.raises(ValueError, match=name):
        _config(env, overrides)


# == normalize_camera_extrinsics_and_points_batch(target_scale=...) ==

def _targets(seed=0, batch=2, frames=3, height=8, width=10):
    g = torch.Generator().manual_seed(seed)
    rotation = torch.linalg.qr(torch.randn(batch, frames, 3, 3, generator=g)).Q
    extrinsics = torch.cat([rotation, torch.randn(batch, frames, 3, 1, generator=g)], dim=-1)
    world = 2 * torch.randn(batch, frames, height, width, 3, generator=g)
    depth = torch.rand(batch, frames, height, width, 1, generator=g) + 0.5
    mask = torch.rand(batch, frames, height, width, generator=g) > 0.3
    return extrinsics, world, depth, mask


def _normalize(extrinsics, world, depth, mask, **kwargs):
    return normalize_camera_extrinsics_and_points_batch(
        extrinsics=extrinsics, cam_points=None, world_points=world, depths=depth, point_masks=mask, **kwargs
    )


def test_first_frame_scale_is_the_mean_distance_of_frame0_valid_points():
    extrinsics, world, depth, mask = _targets()
    rotation, translation = extrinsics[:, 0, :, :3], extrinsics[:, 0, :, 3]
    in_first_camera = torch.einsum("bij,bhwj->bhwi", rotation, world[:, 0]) + translation[:, None, None]
    expected = torch.stack([in_first_camera[b][mask[b, 0]].norm(dim=-1).mean() for b in range(len(world))])

    _, _, new_world, new_depth = _normalize(extrinsics, world, depth, mask, target_scale="first_frame")
    scale = depth[..., 0] / new_depth  # equals ``expected`` up to the 1e-3 added to the valid-point count
    assert torch.allclose(scale, expected.view(-1, 1, 1, 1).expand_as(scale), rtol=1e-4)
    assert torch.allclose(new_world[:, 0] * expected.view(-1, 1, 1, 1), in_first_camera, rtol=1e-4, atol=1e-5)

    changed = mask.clone()
    changed[:, 1:] = ~changed[:, 1:]  # other frames do not enter the frame-0 scale
    assert torch.equal(_normalize(extrinsics, world, depth, changed, target_scale="first_frame")[3], new_depth)


def test_all_frames_scale_is_the_default():
    inputs = _targets(seed=1)
    for default, explicit in zip(_normalize(*inputs), _normalize(*inputs, target_scale="all"), strict=True):
        assert (default is None and explicit is None) or torch.equal(default, explicit)


def test_first_frame_scale_needs_valid_points_in_frame0():
    extrinsics, world, depth, mask = _targets()
    mask[1, 0] = False
    with pytest.raises(ValueError, match="frame 0"):
        _normalize(extrinsics, world, depth, mask, target_scale="first_frame")


def test_unknown_target_scale_is_rejected():
    with pytest.raises(ValueError, match="target_scale"):
        _normalize(*_targets(), target_scale="last_frame")


# == ColmapRgbd sequential clips ==

W, H = 64, 48
RESOLUTION = (56, 42)
TRAIN, VAL = 50, 12  # frames per scene: train 0..49, guard 50..51, val 52..63
SCENES = ("scene_000000", "scene_000001")


def _write_scene(scene, rng, train_frames=TRAIN, val_frames=VAL):
    (scene / "rgb").mkdir(parents=True)
    (scene / "depth").mkdir()
    frames = train_frames + 2 + val_frames
    k = np.array([[60.0, 0, W / 2], [0, 60.0, H / 2], [0, 0, 1]], dtype=np.float32)
    w2c = np.zeros((frames, 3, 4), dtype=np.float32)
    for index in range(frames):
        w2c[index, :3, :3] = np.eye(3)
        w2c[index, :3, 3] = [0.0, -0.01 * index, 0.0]
        Image.fromarray(rng.integers(0, 255, (H, W, 3), dtype=np.uint8)).save(scene / "rgb" / f"frame_{index:06d}.png")
        depth = rng.integers(400, 1300, (H, W), dtype=np.uint16)
        depth[:4] = 0
        Image.fromarray(depth).save(scene / "depth" / f"frame_{index:06d}.png")
    np.savez_compressed(scene / "cameras.npz", intrinsics=np.repeat(k[None], frames, 0), extrinsics_w2c=w2c)
    train, val = np.arange(train_frames).reshape(-1, 10), np.arange(train_frames + 2, frames).reshape(-1, 6)
    np.savez_compressed(
        scene / "sequences.npz",  # sequences padded with -1 to length 10
        sequences=np.concatenate([train, np.pad(val, ((0, 0), (0, 4)), constant_values=-1)]),
        lengths=np.array([10] * len(train) + [6] * len(val)),
        split_ids=np.array([0] * len(train) + [1] * len(val)),
    )


@pytest.fixture(scope="module")
def staging(tmp_path_factory):
    root = tmp_path_factory.mktemp("staging")
    rng = np.random.default_rng(0)
    for name in SCENES:
        _write_scene(root / "scenes" / name, rng)
    meta = {"format": "colmap_rgbd_v1", "depth": {"unit": "millimeters", "invalid_value": 0},
            "camera": {"extrinsics": "opencv_world_to_camera"}}
    (root / "dataset.json").write_text(json.dumps(meta))
    return root


def _dataset(root, split="train", **kwargs):
    return ColmapRgbd(roots=[str(root)], split=split, top_k=8, z_far=2.0, resolution=[RESOLUTION], transform=ImgNorm,
                      aug_crop=0, seed=7, view_selection="sequential", **kwargs)


def _frames(sample):
    return [int(name.split("frame_")[1].split(".")[0]) for name in sample["instance"]]


def test_every_sequential_window_stays_inside_its_scene_split(staging):
    dataset = _dataset(staging)
    assert len(dataset) == len(SCENES) * TRAIN
    for anchor in range(len(dataset)):
        first = anchor // TRAIN * TRAIN
        last = first + TRAIN - 1
        for num in (2, 3, 6, 12):
            for stride in (1, 2, 3, 4):
                window = dataset.sequential_window(anchor, num, stride)
                start = min(anchor, last - (num - 1) * stride)
                assert window == list(range(start, start + num * stride, stride))
                assert first <= window[0] and window[-1] <= last
                assert {dataset.scene_labels[i] for i in window} == {dataset.scene_labels[anchor]}


def test_windows_past_the_end_of_the_split_are_moved_back(staging):
    sample = _dataset(staging, sequential_stride=4)[(2 * TRAIN - 1, 0, 12)]  # last train frame of the 2nd scene
    assert _frames(sample) == list(range(TRAIN - 1 - 44, TRAIN, 4))
    assert set(sample["label"]) == {f"{staging.name}/{SCENES[1]}"}


def test_stride_list_is_drawn_with_the_sample_rng(staging):
    dataset, again = (_dataset(staging, sequential_stride=[1, 2, 3, 4]) for _ in range(2))
    strides = set()
    for index in range(0, len(dataset), 3):
        frames = _frames(dataset[(index, 0, 3)])
        steps = set(np.diff(frames).tolist())
        assert len(steps) == 1 and min(steps) > 0  # ascending, one stride per clip
        assert _frames(again[(index, 0, 3)]) == frames
        strides |= steps
    assert strides == {1, 2, 3, 4}


def test_window_longer_than_the_split_is_rejected(staging):
    dataset = _dataset(staging, split="val", sequential_stride=4)  # 12 val frames cannot hold 12 views 4 apart
    with pytest.raises(ValueError, match="fit"):
        dataset[(0, 0, 12)]


@pytest.mark.parametrize("stride", [0, -1, [], [1, 0], [2, "3"], "2", 1.5])
def test_invalid_strides_are_rejected(staging, stride):
    with pytest.raises(ValueError, match="sequential_stride"):
        _dataset(staging, sequential_stride=stride)


# == full clips ==

class _Sized:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


class _Frames(EasyDataset):
    _resolutions = ((28, 28),)

    def __len__(self):
        return 50

    def __getitem__(self, idx):
        index, _, views = idx
        return torch.tensor([index, views])


@pytest.mark.parametrize("images", [4, 12, 24])
def test_full_clips_yield_one_clip_of_all_images(images):
    sampler = AnchorFrameSampler(_Sized(200), images, pool_size=1, full_clips=True)
    sampler.set_epoch(0)
    items = list(sampler)
    assert len(items) == 200 and all(len(item) == 3 and item[-2:] == (0, images) for item in items)


def test_data_loader_passes_full_clips_to_the_sampler():
    loader = get_data_loader(24 @ _Frames(), batch_size=12, num_workers=0, pin_mem=False, full_clips=True)
    loader.dataset.set_epoch(0)
    loader.sampler.set_epoch(0)
    batches = list(loader)
    assert len(batches) == 24 and all(len(clips) == 1 and clips[0][0, 1] == 12 for clips in batches)
    default = get_data_loader(24 @ _Frames(), batch_size=12, num_workers=0, pin_mem=False)
    default.dataset.set_epoch(0)
    default.sampler.set_epoch(0)
    assert {len(clips) for clips in default} == {1, 2, 4, 6}


def test_full_clips_need_the_anchor_frame_sampler():
    with pytest.raises(ValueError, match="full_clips"):
        get_data_loader(list(range(8)), batch_size=4, num_workers=0, pin_mem=False, full_clips=True)


def test_training_script_passes_the_options():
    source = (REPO / "train_omnivggt.py").read_text()
    assert 'full_clips=cfg.get("full_clips", False)' in source
    assert 'target_scale = cfg.get("target_scale", "all")' in source
    assert "target_scale=target_scale" in source
