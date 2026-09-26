"""k-best checkpoints: the retention rule (train_utils.CheckpointKeeper and its checkpoints.json audit trail), the
score that ranks the checkpoints (the training objective on fixed smoke-split samples, without gradients, in eval
mode and with every random state restored), its configuration and its use by train_omnivggt.py."""

import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from accelerate import PartialState
from test_causal_training_options import SCENES, _write_scene
from test_stream_causal import TINY

import train_utils
from omnivggt.datasets.utils.misc import merge_dicts
from omnivggt.models.omnivggt_omega import OmniVGGTOmega
from omnivggt.utils.configs import read_config
from omnivggt.utils.normalization import normalize_camera_extrinsics_and_points_batch

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "train_colmap_rgbd_omega.py"
CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def accelerate_state():
    PartialState()  # train_utils logs through accelerate


# --- CheckpointKeeper: which checkpoint directories stay -----------------------------------------------------------


def _save(keeper, name, score, step, final=False, extra=None):
    """Write the checkpoint directory ``name``, then record it (or finalize with it)."""
    (keeper.save_dir / name).mkdir()
    (keeper.save_dir / name / "model.safetensors").write_bytes(b"weights")
    return (keeper.finalize if final else keeper.record)(name, score, step, extra=extra)


def _on_disk(directory):
    return sorted(path.name for path in Path(directory).iterdir() if path.is_dir())


def _audit(directory):
    return json.loads((Path(directory) / train_utils.CHECKPOINTS_FILE).read_text())


def test_the_best_k_and_the_latest_checkpoint_are_kept_while_the_run_trains(tmp_path):
    keeper = train_utils.CheckpointKeeper(tmp_path, 2)
    _save(keeper, "a", 3.0, 1)
    assert keeper.kept == ["a"]
    _save(keeper, "b", 1.0, 2)
    assert keeper.kept == ["a", "b"]
    entry = _save(keeper, "c", 2.0, 3)  # the 2 best are b and c
    assert entry["removed"] == ["a"] and entry["kept"] == keeper.kept == ["b", "c"]
    _save(keeper, "d", 5.0, 4)  # kept as the latest: a stopped run resumes from it
    assert keeper.kept == ["b", "c", "d"]
    entry = _save(keeper, "e", 0.5, 5)
    assert entry["removed"] == ["c", "d"] and keeper.kept == ["b", "e"]
    assert _on_disk(tmp_path) == ["b", "e"]


def test_the_latest_checkpoint_is_kept_until_a_later_one_is_recorded(tmp_path):
    keeper = train_utils.CheckpointKeeper(tmp_path, 1)
    _save(keeper, "a", 1.0, 1)
    _save(keeper, "b", 2.0, 2)
    assert keeper.kept == ["a", "b"]
    _save(keeper, "c", 3.0, 3)
    assert keeper.kept == ["a", "c"] and _on_disk(tmp_path) == ["a", "c"]


@pytest.mark.parametrize("final_score, kept", [(0.5, ["final_checkpoint"]), (1.5, ["a", "final_checkpoint"]),
                                                (9.0, ["a", "final_checkpoint"])])
def test_the_final_checkpoint_counts_towards_k_and_is_always_kept(tmp_path, final_score, kept):
    keeper = train_utils.CheckpointKeeper(tmp_path, 1)
    _save(keeper, "a", 1.0, 1)
    _save(keeper, "b", 2.0, 2)
    entry = _save(keeper, "final_checkpoint", final_score, 3, final=True)
    assert keeper.kept == entry["kept"] == kept and _on_disk(tmp_path) == kept


def test_keep_best_0_keeps_the_latest_checkpoint_then_only_the_final_one(tmp_path):
    keeper = train_utils.CheckpointKeeper(tmp_path, 0)
    _save(keeper, "a", 1.0, 1)
    assert keeper.kept == ["a"]
    _save(keeper, "b", 2.0, 2)
    assert keeper.kept == ["b"]
    _save(keeper, "final_checkpoint", 3.0, 3, final=True)
    assert keeper.kept == ["final_checkpoint"] == _on_disk(tmp_path)


def test_ties_keep_the_earlier_step(tmp_path):
    keeper = train_utils.CheckpointKeeper(tmp_path, 1)
    _save(keeper, "a", 1.0, 10)
    _save(keeper, "b", 1.0, 20)
    _save(keeper, "c", 2.0, 30)
    assert keeper.kept == ["a", "c"]
    _save(keeper, "final_checkpoint", 1.0, 40, final=True)
    assert keeper.kept == ["a", "final_checkpoint"]


@pytest.mark.parametrize("score", [math.nan, math.inf])
def test_a_non_finite_score_ranks_below_every_finite_one(tmp_path, score):
    keeper = train_utils.CheckpointKeeper(tmp_path, 1)
    _save(keeper, "a", score, 1)
    _save(keeper, "b", 5.0, 2)
    assert keeper.kept == ["b"]


def test_checkpoints_json_is_the_audit_trail(tmp_path):
    keeper = train_utils.CheckpointKeeper(tmp_path, 1)
    _save(keeper, "a", 2.0, 100, extra={"split": "smoke", "components": {"objective": 2.0}})
    _save(keeper, "b", 1.0, 200)
    _save(keeper, "final_checkpoint", 3.0, 300, final=True)
    audit = _audit(tmp_path)
    assert audit["keep_best"] == 1 and audit["finalized"] is True
    rows = [(e["name"], e["step"], e["score"], e["final"], e["kept"], e["removed"]) for e in audit["checkpoints"]]
    assert rows == [("a", 100, 2.0, False, ["a"], []), ("b", 200, 1.0, False, ["b"], ["a"]),
                    ("final_checkpoint", 300, 3.0, True, ["b", "final_checkpoint"], [])]
    assert audit["checkpoints"][0]["extra"] == {"split": "smoke", "components": {"objective": 2.0}}
    assert audit["checkpoints"][1]["extra"] == {}


def test_a_resumed_run_continues_the_audit_trail(tmp_path):
    first = train_utils.CheckpointKeeper(tmp_path, 1)
    _save(first, "a", 1.0, 1)
    _save(first, "b", 2.0, 2)
    resumed = train_utils.CheckpointKeeper(tmp_path, 1)
    assert resumed.kept == ["a", "b"]
    _save(resumed, "c", 3.0, 3)  # b was recorded before the resume: it goes
    assert resumed.kept == ["a", "c"] and _on_disk(tmp_path) == ["a", "c"]
    _save(resumed, "final_checkpoint", 0.5, 4, final=True)  # so does a, ranked by the score recorded before
    assert _on_disk(tmp_path) == ["final_checkpoint"]
    assert [entry["name"] for entry in _audit(tmp_path)["checkpoints"]] == ["a", "b", "c", "final_checkpoint"]


def test_a_resumed_run_refuses_an_audit_trail_whose_checkpoint_is_gone(tmp_path):
    _save(train_utils.CheckpointKeeper(tmp_path, 1), "a", 1.0, 1)
    shutil.rmtree(tmp_path / "a")
    with pytest.raises(FileNotFoundError, match="'a'"):
        train_utils.CheckpointKeeper(tmp_path, 1)


def test_a_finished_run_records_nothing_more_and_is_not_continued(tmp_path):
    keeper = train_utils.CheckpointKeeper(tmp_path, 1)
    _save(keeper, "final_checkpoint", 1.0, 1, final=True)
    with pytest.raises(RuntimeError, match="final"):
        _save(keeper, "late", 1.0, 2)
    with pytest.raises(ValueError, match="finished"):
        train_utils.CheckpointKeeper(tmp_path, 1)


def test_only_recorded_intermediate_checkpoints_are_deleted(tmp_path):
    (tmp_path / "tensorboard").mkdir()
    (tmp_path / "checkpoint-epoch-9").mkdir()  # written by another run: not recorded
    (tmp_path / "notes.txt").write_text("kept")
    keeper = train_utils.CheckpointKeeper(tmp_path, 0)
    _save(keeper, "a", 1.0, 1)
    _save(keeper, "b", 2.0, 2)
    _save(keeper, "final_checkpoint", 3.0, 3, final=True)
    assert _on_disk(tmp_path) == ["checkpoint-epoch-9", "final_checkpoint", "tensorboard"]
    assert (tmp_path / "notes.txt").read_text() == "kept"


def test_a_checkpoint_missing_at_deletion_is_an_error(tmp_path):
    keeper = train_utils.CheckpointKeeper(tmp_path, 0)
    _save(keeper, "a", 1.0, 1)
    shutil.rmtree(tmp_path / "a")
    with pytest.raises(FileNotFoundError, match="'a'"):
        _save(keeper, "b", 2.0, 2)


def test_a_checkpoint_is_recorded_after_its_directory_is_written(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing"):
        train_utils.CheckpointKeeper(tmp_path, 1).record("missing", 1.0, 1)


@pytest.mark.parametrize("name", ["", ".", "..", "../outside", "sub/dir", "link"])
def test_a_checkpoint_is_one_real_directory_inside_save_dir(tmp_path, name):
    run = tmp_path / "run"
    (run / "sub" / "dir").mkdir(parents=True)
    (tmp_path / "outside").mkdir()
    (run / "link").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(ValueError, match="checkpoint"):
        train_utils.CheckpointKeeper(run, 1).record(name, 1.0, 1)
    assert (tmp_path / "outside").is_dir()


@pytest.mark.parametrize("keep_best", [-1, 1.5, True, "1", None])
def test_keep_best_is_a_non_negative_integer(tmp_path, keep_best):
    with pytest.raises(ValueError, match="keep_best"):
        train_utils.CheckpointKeeper(tmp_path, keep_best)


# --- random states and module modes --------------------------------------------------------------------------------


def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)


def _draws():
    return random.random(), float(np.random.random()), float(torch.rand(()))


def _states():
    python, numpy, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    return python, (numpy[0], numpy[1].tolist(), *numpy[2:]), torch_state.tolist()


def test_isolated_rng_restores_the_python_numpy_and_torch_states():
    _seed_all(0)
    expected = _draws()
    _seed_all(0)
    before = _states()
    with train_utils.isolated_rng():
        inside = [_draws() for _ in range(3)]
    assert _states() == before
    assert inside[0] == expected and _draws() == expected  # the draws after the block are those without it


def test_isolated_rng_restores_the_states_when_the_block_raises():
    _seed_all(0)
    before = _states()
    with pytest.raises(KeyError), train_utils.isolated_rng():
        _draws()
        raise KeyError("validation failed")
    assert _states() == before


def test_isolated_rng_restores_every_cuda_device(monkeypatch):
    devices = [torch.tensor([1, 2], dtype=torch.uint8), torch.tensor([3, 4], dtype=torch.uint8)]
    restored = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [state.clone() for state in devices])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", restored.append)
    with train_utils.isolated_rng():
        devices[0] += 7  # draws on device 0
    assert len(restored) == 1 and [state.tolist() for state in restored[0]] == [[1, 2], [3, 4]]


def test_evaluation_mode_restores_the_mode_of_every_module():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.5), torch.nn.BatchNorm1d(2)).train()
    model[2].eval()  # a module the training script keeps in eval mode stays so
    modes = [True, True, True, False]
    with train_utils.evaluation_mode(model):
        assert not any(module.training for module in model.modules())
    assert [module.training for module in model.modules()] == modes
    with pytest.raises(KeyError), train_utils.evaluation_mode(model):
        raise KeyError("validation failed")
    assert [module.training for module in model.modules()] == modes


# --- configs/train_colmap_rgbd_omega.py: keep_best and val_samples -------------------------------------------------


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("OMNIVGGT_COLMAP_RGBD_ROOTS", "/data/a,/data/b")
    monkeypatch.setenv("OMNIVGGT_OMEGA_VARIANT", "configs/omnivggt_omega/variants/V5.json")
    for name in ("OMNIVGGT_KEEP_BEST", "OMNIVGGT_VAL_SAMPLES", "OMNIVGGT_VIEW_SELECTION", "OMNIVGGT_SEQ_STRIDES",
                 "OMNIVGGT_STEPS_PER_EPOCH", "OMNIVGGT_GRAD_ACCUM", "OMNIVGGT_RESOLUTION", "OMNIVGGT_DATA_SEED"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _config(env, **overrides):
    for name, value in overrides.items():
        env.setenv(name, value)
    return {key: value for key, value in read_config(str(CONFIG)).to_dict().items() if not key.startswith("_")}


def test_keep_best_defaults_to_1_and_val_samples_to_16(env):
    cfg = _config(env)
    assert cfg["keep_best"] == 1 and cfg["val_samples"] == 16


@pytest.mark.parametrize("name, value, changed", [
    ("OMNIVGGT_KEEP_BEST", "0", {"keep_best": 0}),
    ("OMNIVGGT_KEEP_BEST", "3", {"keep_best": 3}),
    ("OMNIVGGT_VAL_SAMPLES", "4", {"val_samples": 4}),
])
def test_keep_best_and_val_samples_change_only_their_own_keys(env, name, value, changed):
    default, cfg = _config(env), _config(env, **{name: value})
    assert {key: cfg[key] for key in cfg if cfg[key] != default[key]} == changed


@pytest.mark.parametrize("name, value", [
    ("OMNIVGGT_KEEP_BEST", "-1"), ("OMNIVGGT_KEEP_BEST", "1.5"), ("OMNIVGGT_KEEP_BEST", "best"),
    ("OMNIVGGT_VAL_SAMPLES", "0"), ("OMNIVGGT_VAL_SAMPLES", "-2"), ("OMNIVGGT_VAL_SAMPLES", "all"),
])
def test_invalid_keep_best_and_val_samples_are_rejected(env, name, value):
    with pytest.raises(ValueError, match=name):
        _config(env, **{name: value})


# --- the smoke split: the selection data ---------------------------------------------------------------------------


@pytest.mark.parametrize("overrides", [{}, {"OMNIVGGT_VIEW_SELECTION": "sequential", "OMNIVGGT_SEQ_STRIDES": "1,2"}])
def test_the_smoke_split_is_drawn_like_the_training_data_without_augmentation(env, overrides):
    train = _config(env, **overrides)["train_dataset"]
    smoke = train_utils.smoke_split_spec(train, 16)
    assert smoke == (train.replace("1000 @ ", "16 @ ").replace("split='train'", "split='smoke'")
                     .replace("aug_crop=16", "aug_crop=0").replace("transform=ColorJitter", "transform=ImgNorm"))


_SPEC = "ColmapRgbd(roots=['r'], split='{split}', aug_crop=16, resolution=[(56, 42)], transform=ColorJitter, seed=1)"


@pytest.mark.parametrize("spec", [
    "22_400 @ ARKitScenesHigh(dset='Training', aug_crop=16, resolution=[(518, 392)], transform=ColorJitter, seed=985)",
    _SPEC.format(split="train"),  # no "N @": the number of samples is not set
    "8 @ " + _SPEC.format(split="val"),
    f"8 @ {_SPEC.format(split='train')} + 8 @ {_SPEC.format(split='train')}",
])
def test_the_smoke_split_is_derived_from_one_colmap_rgbd_train_spec_only(spec):
    with pytest.raises(ValueError, match="smoke"):
        train_utils.smoke_split_spec(spec, 16)


@pytest.mark.parametrize("samples", [0, -1, 1.5])
def test_the_smoke_split_has_a_positive_number_of_samples(samples):
    with pytest.raises(ValueError, match="samples"):
        train_utils.smoke_split_spec("8 @ " + _SPEC.format(split="train"), samples)


TRAIN_FRAMES, VAL_FRAMES, SMOKE_FRAMES = 20, 6, 14  # per scene: train 0..19, val 22..27, smoke 30..43
RECIPE = {  # the T-A environment (batch trainer, ordered full clips) at a small size
    "OMNIVGGT_OMEGA_VARIANT": "configs/omnivggt_omega/variants/V5.json", "OMNIVGGT_OPTIMIZER": "amuse",
    "OMNIVGGT_DEPTH_ALL_VIEWS": "1", "OMNIVGGT_DEPTH_DROP_PROB": "0", "OMNIVGGT_CAM_DROP_PROB": "1",
    "OMNIVGGT_DEPTH_NORM": "first_frame", "OMNIVGGT_TARGET_SCALE": "first_frame", "OMNIVGGT_CAUSAL": "1",
    "OMNIVGGT_VIEW_SELECTION": "sequential", "OMNIVGGT_SEQ_STRIDES": "1,2", "OMNIVGGT_FULL_CLIPS": "1",
    "OMNIVGGT_TRAIN_BATCH_IMAGES": "4", "OMNIVGGT_RESOLUTION": "56x42", "OMNIVGGT_VAL_SAMPLES": "3",
}
UNSET = ("OMNIVGGT_DATA_SEED", "OMNIVGGT_INIT_CHECKPOINT", "OMNIVGGT_OUTPUT_DIR", "OMNIVGGT_GRAD_ACCUM",
         "OMNIVGGT_STEPS_PER_EPOCH", "OMNIVGGT_KEEP_BEST")


def _staging(root, smoke_frames):
    rng = np.random.default_rng(0)
    for name in SCENES:
        _write_scene(root / "scenes" / name, rng, train_frames=TRAIN_FRAMES, val_frames=VAL_FRAMES,
                     smoke_frames=smoke_frames)
    meta = {"format": "colmap_rgbd_v1", "depth": {"unit": "millimeters", "invalid_value": 0},
            "camera": {"extrinsics": "opencv_world_to_camera"}}
    (root / "dataset.json").write_text(json.dumps(meta))
    return root


@pytest.fixture(scope="module")
def staging(tmp_path_factory):
    return _staging(tmp_path_factory.mktemp("staging"), SMOKE_FRAMES)


@pytest.fixture
def recipe(staging, monkeypatch):
    """``recipe(**env)``: configs/train_colmap_rgbd_omega.py read with RECIPE (plus ``env``), loading in-process."""

    def config(**env):
        for name in UNSET:
            monkeypatch.delenv(name, raising=False)
        for name, value in {**RECIPE, "OMNIVGGT_COLMAP_RGBD_ROOTS": str(staging), **env}.items():
            monkeypatch.setenv(name, value)
        cfg = read_config(str(CONFIG))
        cfg.num_workers = 0
        return cfg

    return config


def _frame_numbers(instances):
    names = [name for item in instances for name in (item if isinstance(item, (list, tuple)) else [item])]
    return [int(name.split("frame_")[1].split(".")[0]) for name in names]


def _read(loader):
    return [merge_dicts(batch) for batch in loader]


def test_the_validation_loader_reads_fixed_samples_of_the_smoke_split(recipe):
    cfg = recipe()
    runs = []
    for seed in (0, 1):
        _seed_all(seed)  # nothing is drawn from the global generators: no colour jitter, a fixed order
        runs.append(_read(train_utils.build_validation_loader(cfg)))
    assert len(runs[0]) == 3
    for batch in runs[0]:
        frames = _frame_numbers(batch["instance"])
        assert len(frames) == 4 and all(30 <= frame <= 43 for frame in frames), frames
        assert batch["images"].shape == (1, 4, 3, 42, 56)
    for first, second in zip(*runs, strict=True):
        assert first["instance"] == second["instance"]
        for key in ("images", "depth", "extrinsic", "intrinsic", "world_points", "valid_mask"):
            assert torch.equal(torch.as_tensor(first[key]), torch.as_tensor(second[key])), key


def test_the_validation_loader_needs_the_smoke_split(recipe, tmp_path):
    cfg = recipe(OMNIVGGT_COLMAP_RGBD_ROOTS=str(_staging(tmp_path / "no_smoke", 0)))
    with pytest.raises(ValueError, match="smoke"):
        train_utils.build_validation_loader(cfg)


def test_split_batch_normalises_the_targets_and_keeps_the_raw_inputs(recipe):
    batch = _read(train_utils.build_validation_loader(recipe()))[0]
    raw = {key: torch.as_tensor(batch[key]).clone() for key in ("extrinsic", "depth", "world_points", "valid_mask")}
    inputs, targets = train_utils.split_batch(batch, target_scale="first_frame")
    extrinsic, _, world_points, depth = normalize_camera_extrinsics_and_points_batch(
        extrinsics=raw["extrinsic"], cam_points=None, world_points=raw["world_points"], depths=raw["depth"],
        point_masks=raw["valid_mask"], target_scale="first_frame")
    assert set(inputs) == {"images", "extrinsics", "intrinsics", "depth", "mask"}
    assert torch.equal(inputs["extrinsics"], raw["extrinsic"]) and torch.equal(inputs["depth"], raw["depth"])
    assert torch.equal(inputs["mask"], raw["valid_mask"]) and torch.equal(inputs["images"], batch["images"])
    assert torch.equal(targets["extrinsic"], extrinsic) and torch.equal(targets["world_points"], world_points)
    assert torch.equal(targets["depth"], depth) and torch.equal(targets["intrinsic"], batch["intrinsic"])


def _batch_model(**options):
    torch.manual_seed(0)
    return OmniVGGTOmega(**TINY, **options).train()


def test_the_score_is_the_training_objective_at_the_end_of_training_without_gradients(recipe):
    cfg = recipe()
    model = _batch_model(causal=True, depth_norm="first_frame", cam_drop_prob=1.0, depth_all_views=True)
    criterion, calls = train_utils.build_loss_criterion(cfg), []

    def recorded(predictions, batch, progress=None):
        calls.append((progress, torch.is_grad_enabled(), model.training))
        return criterion(predictions, batch, progress=progress)

    score = train_utils.validation_objective(model, train_utils.build_validation_loader(cfg), recorded,
                                             target_scale="first_frame", seed=42, device=CPU)
    assert calls == [(1.0, False, False)] * 3  # progress 1: every checkpoint is scored on the same objective
    assert {"objective", "loss_camera", "loss_conf_depth", "loss_point"} <= set(score)
    assert all(isinstance(value, float) and math.isfinite(value) for value in score.values())


def test_the_score_changes_nothing_and_is_reproducible(recipe):
    cfg = recipe()
    loader, criterion = train_utils.build_validation_loader(cfg), train_utils.build_loss_criterion(cfg)
    # random camera and depth inputs: the draws come from generators keyed by the sample
    model = _batch_model(cam_drop_prob=0.5, depth_drop_prob=0.5)
    model.aggregator.patch_embed.eval()
    weights = {key: value.clone() for key, value in model.state_dict().items()}
    modes = [module.training for module in model.modules()]
    _seed_all(3)
    before = _states()
    scores = [train_utils.validation_objective(model, loader, criterion, target_scale="all", seed=42, device=CPU)
              for _ in range(2)]
    assert _states() == before
    assert scores[0] == scores[1]
    assert [module.training for module in model.modules()] == modes
    assert all(torch.equal(value, weights[key]) for key, value in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())
    other = train_utils.validation_objective(model, loader, criterion, target_scale="all", seed=43, device=CPU)
    assert other != scores[0]  # another seed draws other auxiliary inputs


# --- train_omnivggt.py -----------------------------------------------------------------------------------------------


def test_the_training_script_scores_and_keeps_its_checkpoints():
    source = (REPO / "train_omnivggt.py").read_text()
    assert "split_batch(batch, target_scale=target_scale)" in source  # one batch path for training and validation
    assert "build_validation_loader(cfg)" in source and "CheckpointKeeper(save_dir, keep_best)" in source
    assert "validation_objective(" in source and "keeper.record(" in source and "keeper.finalize(" in source
    assert 'epoch + 1 < cfg.get("num_train_epochs")' in source  # the last epoch is saved as final_checkpoint only
    assert 'writer.add_scalar("val/objective"' in source
