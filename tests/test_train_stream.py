"""Training on the stream (train_stream.py, T-X): ordered windows of the train split, the window targets, the
per-frame loss accumulated over 12-frame updates, the update recipe, the run layout and the config's data seed."""

import contextlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from accelerate import PartialState
from safetensors.torch import load_file, save_file
from test_causal_training_options import SCENES, _write_scene
from test_stream_causal import TINY

import train_stream
import train_utils
from omnivggt.datasets.colmap_rgbd import ColmapRgbd
from omnivggt.models.omnivggt_omega import OmniVGGTOmega
from omnivggt.stream.kv_cache import CachePolicy
from omnivggt.stream.streaming import StreamingOmega
from omnivggt.utils.configs import read_config
from omnivggt.utils.normalization import normalize_camera_extrinsics_and_points_batch

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "train_colmap_rgbd_omega.py"
TRAIN_FRAMES, VAL_FRAMES, SMOKE_FRAMES = 100, 12, 26  # per scene: train 0..99, val 102..113, smoke 116..141
T_X_RECIPE = {  # the environment of temp/stream_omega/train_tx.sh, at the staging image size
    "OMNIVGGT_OMEGA_VARIANT": "configs/omnivggt_omega/variants/V5.json", "OMNIVGGT_OPTIMIZER": "amuse",
    "OMNIVGGT_DEPTH_ALL_VIEWS": "1", "OMNIVGGT_DEPTH_DROP_PROB": "0", "OMNIVGGT_CAM_DROP_PROB": "1",
    "OMNIVGGT_DEPTH_NORM": "first_frame", "OMNIVGGT_TARGET_SCALE": "first_frame",
    "OMNIVGGT_VIEW_SELECTION": "sequential", "OMNIVGGT_SEQ_STRIDES": "1,2", "OMNIVGGT_PATCH_EMBED_FREEZE": "1",
    "OMNIVGGT_CAUSAL": "1", "OMNIVGGT_RESOLUTION": "56x42",
}
UNSET = ("OMNIVGGT_DATA_SEED", "OMNIVGGT_INIT_CHECKPOINT", "OMNIVGGT_OUTPUT_DIR", "OMNIVGGT_FULL_CLIPS",
         "OMNIVGGT_TRAIN_BATCH_IMAGES", "OMNIVGGT_GRAD_ACCUM", "OMNIVGGT_STEPS_PER_EPOCH")
POLICY = CachePolicy(recent=1, long_special=1, long_patch=4, selector="query", quant="int8")
TENSORS = ("images", "depth", "extrinsic", "intrinsic", "world_points", "valid_mask")


@pytest.fixture(autouse=True)
def accelerate_state():
    PartialState()  # train_utils logs through accelerate


@pytest.fixture(scope="module")
def staging(tmp_path_factory):
    root = tmp_path_factory.mktemp("staging")
    rng = np.random.default_rng(0)
    for name in SCENES:
        _write_scene(root / "scenes" / name, rng, train_frames=TRAIN_FRAMES, val_frames=VAL_FRAMES,
                     smoke_frames=SMOKE_FRAMES)
    meta = {"format": "colmap_rgbd_v1", "depth": {"unit": "millimeters", "invalid_value": 0},
            "camera": {"extrinsics": "opencv_world_to_camera"}}
    (root / "dataset.json").write_text(json.dumps(meta))
    return root


@pytest.fixture
def recipe(staging, monkeypatch):
    """``recipe(**env)``: configs/train_colmap_rgbd_omega.py read with the T-X environment (plus ``env``)."""

    def config(**env):
        for name in UNSET:
            monkeypatch.delenv(name, raising=False)
        for name, value in {**T_X_RECIPE, "OMNIVGGT_COLMAP_RGBD_ROOTS": str(staging), **env}.items():
            monkeypatch.setenv(name, value)
        return read_config(str(CONFIG))

    return config


def _fp64(cfg):
    """The recipe without autocast, for fp64 checks on the CPU."""
    return {**cfg.to_dict(), "mixed_precision": "no"}


def _tiny_model(dtype=torch.float64):
    torch.manual_seed(0)
    return OmniVGGTOmega(**TINY, causal=True, depth_norm="first_frame", cam_drop_prob=1.0).to(dtype).train()


def _window(cfg, anchor=3, length=24, dtype=torch.float64):
    window = train_stream.load_window(train_stream.window_dataset(cfg), anchor, length)
    window = {key: value.to(dtype) if key in TENSORS and value.is_floating_point() else value
              for key, value in window.items()}
    return train_stream.split_window(window)


def _frame_numbers(instances):
    return [int(name.split("frame_")[1].split(".")[0]) for name in instances]


# --- windows ---------------------------------------------------------------------------------------------------


def test_window_anchors_are_seeded_permutations_of_the_train_frames():
    anchors = train_stream.window_anchors(250, 100)
    assert train_stream.WINDOW_SEED == 42 and anchors.shape == (250,)
    assert sorted(anchors[:100].tolist()) == sorted(anchors[100:200].tolist()) == list(range(100))
    assert np.array_equal(anchors, train_stream.window_anchors(250, 100, seed=42))
    assert not np.array_equal(anchors, train_stream.window_anchors(250, 100, seed=43))
    assert np.array_equal(train_stream.window_anchors(3, 100), anchors[:3])


def test_windows_are_ordered_frames_of_one_scene_in_the_train_split(recipe):
    dataset = train_stream.window_dataset(recipe())
    assert isinstance(dataset, ColmapRgbd) and dataset.split == "train" and dataset.sequential_stride == [1, 2]
    assert dataset.seed == 985 and dataset.aug_crop == 16  # the configured sample seed and augmentation
    assert len(dataset) == len(SCENES) * TRAIN_FRAMES
    strides = set()
    for anchor in range(0, len(dataset), 9):
        window = train_stream.load_window(dataset, anchor, 48)
        frames = _frame_numbers(window["instance"])
        steps = set(np.diff(frames).tolist())
        assert len(frames) == 48 and len(steps) == 1 and steps <= {1, 2}, frames
        assert 0 <= frames[0] and frames[-1] < TRAIN_FRAMES  # never a guard or val frame
        assert set(window["label"]) == {dataset.scene_labels[anchor]}
        strides |= steps
    assert strides == {1, 2}
    shapes = {key: tuple(value.shape) for key, value in window.items() if key in TENSORS}
    assert shapes == {"images": (1, 48, 3, 42, 56), "depth": (1, 48, 42, 56, 1), "extrinsic": (1, 48, 3, 4),
                      "intrinsic": (1, 48, 3, 3), "world_points": (1, 48, 42, 56, 3), "valid_mask": (1, 48, 42, 56)}
    assert window["valid_mask"].dtype == torch.bool


def test_a_window_past_the_end_of_the_split_ends_on_its_last_frame(recipe):
    dataset = train_stream.window_dataset(recipe())
    frames = _frame_numbers(train_stream.load_window(dataset, 2 * TRAIN_FRAMES - 1, 48)["instance"])
    assert frames[-1] == TRAIN_FRAMES - 1 and frames == sorted(frames) and len(frames) == 48


def test_windows_are_reproducible(recipe):
    dataset = train_stream.window_dataset(recipe())
    windows = []
    for _ in range(2):
        torch.manual_seed(0)  # the colour jitter draws from torch's generator, as in train_omnivggt.py's loader
        windows.append(train_stream.load_window(dataset, 37, 24))
    assert windows[0]["instance"] == windows[1]["instance"]
    for key in TENSORS:
        assert torch.equal(windows[0][key], windows[1][key]), key


def test_window_targets_are_normalised_once_by_the_first_frame(recipe):
    window = train_stream.load_window(train_stream.window_dataset(recipe()), 0, 24)
    inputs, targets = train_stream.split_window(window)
    extrinsic, _, world_points, depth = normalize_camera_extrinsics_and_points_batch(
        extrinsics=window["extrinsic"], cam_points=None, world_points=window["world_points"],
        depths=window["depth"], point_masks=window["valid_mask"], target_scale="first_frame")
    assert set(targets) == {"images", "extrinsic", "intrinsic", "depth", "world_points", "valid_mask"}
    assert torch.equal(targets["extrinsic"], extrinsic) and torch.equal(targets["world_points"], world_points)
    assert torch.equal(targets["depth"], depth) and targets["depth"].shape == (1, 24, 42, 56)
    assert torch.equal(targets["intrinsic"], window["intrinsic"]) and torch.equal(targets["images"], window["images"])
    assert set(inputs) == {"images", "depth", "mask"}
    assert torch.equal(inputs["depth"], window["depth"])  # the model gets the raw depth
    assert torch.equal(inputs["images"], window["images"]) and torch.equal(inputs["mask"], window["valid_mask"])
    assert torch.equal(targets["valid_mask"], window["valid_mask"])
    for frame in (0, 5, 23):
        sliced = train_stream.frame_targets(targets, frame)
        assert sliced.keys() == targets.keys()
        for key, value in sliced.items():
            assert torch.equal(value, targets[key][:, frame:frame + 1]), key
    identity = torch.eye(4, dtype=extrinsic.dtype)[:3]
    assert torch.allclose(targets["extrinsic"][0, 0], identity, atol=1e-6)
    # frame 6 keeps the window's frame-1 gauge and scale (it is not normalised on its own)
    assert not torch.allclose(train_stream.frame_targets(targets, 5)["extrinsic"][0, 0], identity, atol=1e-3)


# --- updates ---------------------------------------------------------------------------------------------------


def test_twelve_per_frame_backwards_accumulate_the_gradient_of_the_mean_loss(recipe, monkeypatch):
    cfg = _fp64(recipe())
    inputs, targets = _window(cfg, length=12)
    model = _tiny_model()
    trainer = train_stream.StreamTrainer(model, POLICY, cfg, updates=1560)  # freezes the encoder
    reference = StreamingOmega(model, POLICY, train=True)
    losses = []
    for frame in range(12):
        predictions = reference.step(inputs["images"][:, frame], inputs["depth"][:, frame], inputs["mask"][:, frame])
        losses.append(trainer.criterion(predictions, train_stream.frame_targets(targets, frame), progress=0.0))
    assert all(loss["loss_camera"] > 0 and loss["loss_point"] > 0 for loss in losses)
    torch.stack([loss["objective"] for loss in losses]).mean().backward()
    want = {name: parameter.grad.clone() for name, parameter in model.named_parameters() if parameter.grad is not None}
    model.zero_grad(set_to_none=True)

    got, clip = {}, torch.nn.utils.clip_grad_norm_

    def recorded(parameters, max_norm, *args, **kwargs):
        got.update({name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None})
        return clip(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", recorded)
    trainer.train_window(inputs, targets)
    assert got.keys() == want.keys() and len(got) > 0
    assert not any(name.startswith("aggregator.patch_embed.") for name in got)  # the frozen encoder
    for name in want:
        torch.testing.assert_close(got[name], want[name], rtol=1e-10, atol=1e-13)


def test_every_update_clips_to_1_and_steps_once_with_warmup_78_and_progress(recipe, monkeypatch):
    cfg = _fp64(recipe())
    assert cfg["max_grad_norm"] == 1.0
    trainer = train_stream.StreamTrainer(_tiny_model(), POLICY, cfg, updates=1560)
    optimizer = trainer.optimizer
    assert trainer.scheduler is None and trainer.max_grad_norm == 1.0  # AMUSE is schedule-free
    assert optimizer.warmup_steps == 78 and all(group["warmup_steps"] == 78 for group in optimizer.param_groups)
    assert optimizer.train_mode

    calls = []
    clip, step, criterion = torch.nn.utils.clip_grad_norm_, optimizer.step, trainer.criterion

    def clipped(parameters, max_norm, *args, **kwargs):
        calls.append(("clip", max_norm))
        return clip(parameters, max_norm, *args, **kwargs)

    def stepped(*args, **kwargs):
        calls.append(("step",))
        return step(*args, **kwargs)

    def loss(predictions, batch, progress=None):
        calls.append(("progress", progress))
        return criterion(predictions, batch, progress=progress)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clipped)
    monkeypatch.setattr(optimizer, "step", stepped)
    trainer.criterion = loss
    trainer.update = 5  # the window starts after 5 updates
    updates = []
    trainer.train_window(*_window(cfg), on_update=lambda update, record: updates.append((update, record)))
    update_calls = [("clip", 1.0), ("step",)]
    assert calls == [("progress", 5 / 1560)] * 12 + update_calls + [("progress", 6 / 1560)] * 12 + update_calls
    assert [update for update, _ in updates] == [6, 7] and trainer.update == 7
    for _, record in updates:
        assert math.isfinite(record["objective"]) and record["grad_norm"] > 0
    assert all(parameter.grad is None for parameter in trainer.model.parameters())  # zeroed after each step


def test_one_window_of_a_tiny_model_has_finite_losses_and_updates_the_parameters(recipe):
    cfg = _fp64(recipe())
    model = _tiny_model()
    trainer = train_stream.StreamTrainer(model, POLICY, cfg, updates=4)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    records = []
    trainer.train_window(*_window(cfg), on_update=lambda update, record: records.append(record))
    assert len(records) == 2 and trainer.update == 2
    assert all(math.isfinite(value) for record in records for value in record.values())
    assert {"objective", "loss_camera", "loss_conf_depth", "loss_point", "grad_norm"} <= set(records[0])
    changed = {name for name, parameter in model.named_parameters() if not torch.equal(parameter, before[name])}
    for name in ("aggregator.global_blocks.0.attn.qkv.weight", "camera_head.trunk.0.attn.qkv.weight",
                 "depth_head.scratch.output_conv2.2.weight"):
        assert name in changed, name
    assert not any(name.startswith("aggregator.patch_embed.") for name in changed)  # the encoder is frozen


def test_bf16_autocast_wraps_the_stream_step_and_not_the_loss(recipe, monkeypatch):
    cfg = recipe()
    assert cfg["mixed_precision"] == "bf16"
    trainer = train_stream.StreamTrainer(_tiny_model(torch.float32), POLICY, cfg.to_dict(), updates=2)
    assert trainer.stream.dtype == torch.bfloat16  # a bf16 cache
    seen = {"step": set(), "loss": set()}
    step, criterion = trainer.stream.step, trainer.criterion

    def recorded_step(*args):
        seen["step"].add((torch.is_autocast_enabled("cpu"), torch.get_autocast_dtype("cpu")))
        with torch.autocast("cpu", enabled=False):  # the tiny CPU model is not run in bf16 (its heads would be)
            return step(*args)

    def loss(predictions, batch, progress=None):
        seen["loss"].add(torch.is_autocast_enabled("cpu"))
        return criterion(predictions, batch, progress=progress)

    monkeypatch.setattr(trainer.stream, "step", recorded_step)
    trainer.criterion = loss
    trainer.train_window(*_window(cfg, length=12, dtype=torch.float32))
    assert seen == {"step": {(True, torch.bfloat16)}, "loss": {False}}


def test_a_window_must_hold_whole_updates_before_the_last_one(recipe):
    cfg = _fp64(recipe())
    trainer = train_stream.StreamTrainer(_tiny_model(), POLICY, cfg, updates=1)
    inputs, targets = _window(cfg, length=24)
    with pytest.raises(ValueError, match="past update 1"):
        trainer.train_window(inputs, targets)
    odd = {key: value[:, :13] for key, value in inputs.items()}
    with pytest.raises(ValueError, match="multiple of 12"):
        trainer.train_window(odd, {key: value[:, :13] for key, value in targets.items()})


@pytest.mark.parametrize("text, policy", [
    ("full", CachePolicy.full()),
    ('{"recent": 1, "long_special": 2, "long_patch": 5, "selector": "query", "quant": "int8"}',
     CachePolicy(recent=1, long_special=2, long_patch=5, selector="query", quant="int8")),
])
def test_policy_argument(text, policy):
    assert train_stream.parse_policy(text) == policy


@pytest.mark.parametrize("text", ["[1, 2]", '"full"', "bounded"])
def test_invalid_policy_arguments_raise(text):
    with pytest.raises(ValueError):
        train_stream.parse_policy(text)


# --- the run: arguments, initial weights, checkpoints -----------------------------------------------------------


def _init_checkpoint(tmp_path):
    """A trained checkpoint of the tiny model, laid out as <run>/<exp>/final_checkpoint/model.safetensors."""
    directory = tmp_path / "stream_TA" / "exp" / "final_checkpoint"
    directory.mkdir(parents=True)
    state = {key: value.contiguous() for key, value in _tiny_model(torch.float32).state_dict().items()}
    save_file(state, directory / "model.safetensors")
    return directory, state


@pytest.fixture
def tiny_build(monkeypatch):
    """build_model builds the tiny model (with the recipe's options) instead of the variant, which trains on the CPU
    without autocast (bf16 caches)."""
    monkeypatch.setattr(OmniVGGTOmega, "from_variant", staticmethod(lambda path, **kwargs: OmniVGGTOmega(**TINY, **kwargs)))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (9, 0))
    monkeypatch.setattr(train_stream.StreamTrainer, "_autocast", lambda self, device_type: contextlib.nullcontext())


def _arguments(init, output, *extra):
    return ["--config", str(CONFIG), "--init", str(init), "--policy", json.dumps({
        "recent": 1, "long_special": 1, "long_patch": 4, "selector": "query", "quant": "int8"}),
        "--window-length", "12", "--updates", "2", "--checkpoint-every", "1", "--output-dir", str(output),
        "--device", "cpu", "--num-workers", "0", *extra]


def test_a_run_starts_from_the_init_checkpoint_and_saves_evaluation_checkpoints(recipe, tiny_build, tmp_path):
    recipe()
    init, state = _init_checkpoint(tmp_path)
    output = tmp_path / "stream_TX"
    assert train_stream.main(_arguments(init, output, "--keep-best", "2")) == 0  # both checkpoints stay
    report = json.loads((output / "weight_transfer_report.json").read_text())
    assert report["init_checkpoint"]["sha256"] == train_utils.weight_transfer.sha256_of(init / "model.safetensors")
    run = output / "omnivggt-omega-colmap-rgbd"
    assert sorted(_checkpoint_names(run)) == ["checkpoint-u1", "final_checkpoint"]
    final = load_file(run / "final_checkpoint" / "model.safetensors")
    model = OmniVGGTOmega(**TINY, causal=True, depth_norm="first_frame")
    model.load_state_dict(final, strict=True)  # loads as an evaluation checkpoint
    assert not torch.equal(final["aggregator.global_blocks.0.attn.qkv.weight"],
                           state["aggregator.global_blocks.0.attn.qkv.weight"])
    assert torch.equal(final["aggregator.patch_embed.proj.weight"], state["aggregator.patch_embed.proj.weight"])
    record = json.loads((run / "stream_training.json").read_text())
    assert record["policy"]["quant"] == "int8" and record["updates"] == 2 and record["window_length"] == 12
    assert record["frames_per_update"] == 12 and record["window_seed"] == 42 and record["data_seed"] == 985
    assert len(record["window_anchors"]) == 2 and (run / "tensorboard").is_dir()
    with pytest.raises(FileExistsError):  # no resume: a run starts in a new output directory
        train_stream.main(_arguments(init, output))


@pytest.mark.parametrize("env, extra, match", [
    ({"OMNIVGGT_TARGET_SCALE": "all"}, (), "OMNIVGGT_TARGET_SCALE"),
    ({"OMNIVGGT_DEPTH_ALL_VIEWS": "0"}, (), "OMNIVGGT_DEPTH_ALL_VIEWS"),
    ({"OMNIVGGT_VIEW_SELECTION": "random_topk", "OMNIVGGT_SEQ_STRIDES": "1"}, (), "sequential"),
    ({"OMNIVGGT_INIT_CHECKPOINT": "/elsewhere/final_checkpoint"}, (), "--init"),
    ({}, ("--updates", "3", "--window-length", "24"), "multiple"),
    ({}, ("--window-length", "18"), "multiple of 12"),
    ({}, ("--keep-best", "-1"), "--keep-best"),
    ({}, ("--val-windows", "0"), "--val-windows"),
])
def test_a_run_refuses_another_recipe(recipe, tiny_build, tmp_path, env, extra, match):
    recipe(**env)
    init, _ = _init_checkpoint(tmp_path)
    with pytest.raises(ValueError, match=match):
        train_stream.main(_arguments(init, tmp_path / "run", *extra))


# --- configs/train_colmap_rgbd_omega.py: the data seed ----------------------------------------------------------


def test_the_data_seed_defaults_to_985_and_sets_the_dataset_seed(recipe):
    default = recipe()
    seeded = recipe(OMNIVGGT_DATA_SEED="986")
    assert default["data_seed"] == 985 and "seed=985" in default["train_dataset"]
    changed = {key for key in default.to_dict() if not key.startswith("_") and seeded[key] != default[key]}
    assert changed == {"data_seed", "train_dataset"}
    assert seeded["train_dataset"] == default["train_dataset"].replace("seed=985", "seed=986")
    datasets = [train_stream.window_dataset(cfg) for cfg in (default, seeded)]
    assert [dataset.seed for dataset in datasets] == [985, 986]
    draws = [[_draw(train_stream.load_window(dataset, anchor, 12)) for anchor in range(0, 200, 25)]
             for dataset in datasets]
    assert draws[0] != draws[1]  # the strides and crops of the samples follow the seed


def _draw(window):
    return window["instance"], window["intrinsic"].tolist()


@pytest.mark.parametrize("seed", ["0", "-1", "1.5", "seed"])
def test_invalid_data_seeds_are_rejected(recipe, seed):
    with pytest.raises(ValueError, match="OMNIVGGT_DATA_SEED"):
        recipe(OMNIVGGT_DATA_SEED=seed)


# --- k-best checkpoints: the smoke-split score -----------------------------------------------------------------------


def _staging_without_smoke(root):
    rng = np.random.default_rng(0)
    for name in SCENES:
        _write_scene(root / "scenes" / name, rng, train_frames=TRAIN_FRAMES, val_frames=VAL_FRAMES)
    (root / "dataset.json").write_text(json.dumps({
        "format": "colmap_rgbd_v1", "depth": {"unit": "millimeters", "invalid_value": 0},
        "camera": {"extrinsics": "opencv_world_to_camera"}}))
    return root


def test_validation_windows_are_fixed_smoke_windows_at_the_smallest_stride(recipe):
    cfg = recipe()  # strides 1 and 2: 24 views 2 apart do not fit in the 26 smoke frames of a scene
    runs = []
    for seed in (0, 1):
        torch.manual_seed(seed)  # no colour jitter: the windows do not depend on torch's generator
        runs.append(train_stream.validation_windows(cfg, 4, 24))
    windows = runs[0]
    assert len(windows) == 4
    for window in windows:
        frames = _frame_numbers(window["instance"])
        assert len(frames) == 24 and set(np.diff(frames).tolist()) == {1}, frames
        assert 116 <= frames[0] and frames[-1] <= 141  # smoke frames only
        assert window["images"].shape == (1, 24, 3, 42, 56)
    assert len({window["label"][0] for window in windows}) == len(SCENES)  # evenly spaced over the scenes
    for first, second in zip(*runs, strict=True):
        assert first["instance"] == second["instance"]
        for key in TENSORS:
            assert torch.equal(first[key], second[key]), key


def test_validation_needs_smoke_windows_of_the_window_length(recipe, tmp_path):
    with pytest.raises(ValueError, match="smoke"):
        train_stream.validation_windows(recipe(), 4, 48)  # longer than the smoke frames of a scene
    cfg = recipe(OMNIVGGT_COLMAP_RGBD_ROOTS=str(_staging_without_smoke(tmp_path / "no_smoke")))
    with pytest.raises(ValueError, match="smoke"):
        train_stream.validation_windows(cfg, 4, 12)


def test_evaluate_window_is_the_mean_stream_loss_without_gradients_or_updates(recipe):
    cfg = _fp64(recipe())
    inputs, targets = _window(cfg, length=12)
    model = _tiny_model()
    trainer = train_stream.StreamTrainer(model, POLICY, cfg, updates=4)
    reference, frames = StreamingOmega(model, POLICY, train=True), []
    with torch.no_grad():
        for frame in range(12):
            predictions = reference.step(inputs["images"][:, frame], inputs["depth"][:, frame], inputs["mask"][:, frame])
            loss = trainer.criterion(predictions, train_stream.frame_targets(targets, frame), progress=1.0)
            frames.append({key: float(value) for key, value in loss.items()})
    trainer.stream.reset(max_frames=24)  # a training window in progress keeps its caches
    trainer.stream.step(inputs["images"][:, 0], inputs["depth"][:, 0], inputs["mask"][:, 0])
    caches = trainer.stream.caches
    weights = {key: value.clone() for key, value in model.state_dict().items()}
    got = trainer.evaluate_window(inputs, targets)
    assert got.keys() == frames[0].keys()
    for key in got:
        assert got[key] == pytest.approx(np.mean([frame[key] for frame in frames]), rel=1e-12, abs=1e-15), key
    assert trainer.stream.t == 1 and trainer.stream.caches is caches and trainer.update == 0
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(value, weights[key]) for key, value in model.state_dict().items())


def test_validation_loss_is_the_window_mean_in_eval_mode_with_the_random_states_restored(recipe, monkeypatch):
    cfg = _fp64(recipe())
    model = _tiny_model(torch.float32)
    trainer = train_stream.StreamTrainer(model, POLICY, cfg, updates=4)
    windows = train_stream.validation_windows(recipe(), 2, 12)
    seen, evaluate = [], trainer.evaluate_window

    def recorded(inputs, targets):
        seen.append((model.training, tuple(inputs["images"].shape)))
        seen.append(evaluate(inputs, targets))
        return seen[-1]

    monkeypatch.setattr(trainer, "evaluate_window", recorded)
    torch.manual_seed(5)
    state = torch.get_rng_state()
    score = train_stream.validation_loss(trainer, windows, torch.device("cpu"))
    assert torch.equal(torch.get_rng_state(), state) and model.training
    assert seen[0] == seen[2] == (False, (1, 12, 3, 42, 56))
    assert score == {key: pytest.approx((seen[1][key] + seen[3][key]) / 2) for key in seen[1]}


def _checkpoint_names(run):
    return {path.name for path in run.iterdir() if path.is_dir() and path.name.startswith(("checkpoint", "final"))}


@pytest.mark.parametrize("keep_best", [0, 1])
def test_a_run_keeps_the_best_checkpoints_by_their_smoke_score(recipe, tiny_build, tmp_path, keep_best):
    recipe()
    init, _ = _init_checkpoint(tmp_path)
    output = tmp_path / "stream_TX"
    assert train_stream.main(_arguments(init, output, "--keep-best", str(keep_best), "--val-windows", "2")) == 0
    run = output / "omnivggt-omega-colmap-rgbd"
    audit = json.loads((run / "checkpoints.json").read_text())
    scores = {entry["name"]: entry["score"] for entry in audit["checkpoints"]}
    assert list(scores) == ["checkpoint-u1", "final_checkpoint"] and audit["keep_best"] == keep_best
    assert all(math.isfinite(score) for score in scores.values())
    best = min(scores, key=scores.get)
    assert _checkpoint_names(run) == {"final_checkpoint"} | ({best} if keep_best else set())
    extra = audit["checkpoints"][0]["extra"]
    assert extra["split"] == "smoke" and extra["windows"] == 2 and extra["components"]["objective"] == scores[
        "checkpoint-u1"]
    record = json.loads((run / "stream_training.json").read_text())
    assert record["keep_best"] == keep_best
    assert record["validation"]["split"] == "smoke" and len(record["validation"]["anchors"]) == 2


def test_scoring_the_checkpoints_does_not_change_the_training_run(recipe, tiny_build, tmp_path, monkeypatch):
    recipe()
    init, _ = _init_checkpoint(tmp_path)
    arguments = ("--window-length", "24", "--keep-best", "2")  # checkpoint-u1 is saved and scored mid-window
    assert train_stream.main(_arguments(init, tmp_path / "scored", *arguments)) == 0
    monkeypatch.setattr(train_stream, "validation_loss", lambda trainer, windows, device: {"objective": 0.0})
    assert train_stream.main(_arguments(init, tmp_path / "unscored", *arguments)) == 0
    for name in ("checkpoint-u1", "final_checkpoint"):
        scored, unscored = (load_file(tmp_path / run / "omnivggt-omega-colmap-rgbd" / name / "model.safetensors")
                            for run in ("scored", "unscored"))
        assert scored.keys() == unscored.keys()
        assert all(torch.equal(scored[key], unscored[key]) for key in scored), name
