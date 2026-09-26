import pytest
import torch
from safetensors.torch import save_file

from train_utils import load_initial_weights


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(3, 2)


def test_init_checkpoint_is_loaded_strictly(tmp_path):
    source = Tiny()
    save_file(source.state_dict(), tmp_path / "w.safetensors")
    target = Tiny()
    assert (
        load_initial_weights(target, {"init_checkpoint": str(tmp_path / "w.safetensors")})
        == "init_checkpoint:w.safetensors"
    )
    assert torch.equal(target.layer.weight, source.layer.weight)


def test_missing_init_checkpoint_raises_instead_of_training_from_scratch(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_initial_weights(Tiny(), {"init_checkpoint": str(tmp_path / "missing.safetensors")})


def test_mismatched_keys_raise(tmp_path):
    save_file({"other.weight": torch.zeros(2, 3)}, tmp_path / "bad.safetensors")
    with pytest.raises(RuntimeError):
        load_initial_weights(Tiny(), {"init_checkpoint": str(tmp_path / "bad.safetensors")})


# --- OmniVGGTOmega training path -------------------------------------------------------------

import json  # noqa: E402
from pathlib import Path  # noqa: E402

from accelerate import PartialState  # noqa: E402

import train_utils  # noqa: E402
from omnivggt.models.omnivggt_omega import OmniVGGTOmega  # noqa: E402

TINY_OMEGA = dict(
    img_size=28,
    patch_size=14,
    embed_dim=32,
    num_register_tokens=4,
    register_attention_layers=(1,),
    global_rope=False,
    cached_layers=(0, 1, 2, 3),
    aggregator_kwargs=dict(depth=4, num_heads=2, patch_embed="conv"),
    camera_head_kwargs=dict(trunk_depth=1, num_heads=2),
    depth_head_kwargs=dict(features=16, out_channels=[8, 16, 32, 32], intermediate_layer_idx=[0, 1, 2, 3]),
)


@pytest.fixture
def omega_cfg(tmp_path, monkeypatch):
    PartialState()  # train_utils logs through accelerate
    torch.manual_seed(3)
    source = OmniVGGTOmega(**TINY_OMEGA)
    weights = tmp_path / "source.safetensors"
    save_file({k: v.contiguous() for k, v in source.state_dict().items()}, weights)
    weight_map = tmp_path / "map.json"
    weight_map.write_text(json.dumps({
        "name": "tiny", "sources": {"omni": {"env": "TEST_INIT_OMNI", "format": "safetensors"}},
        "rules": [{"target_prefix": "", "source": "omni", "source_prefix": ""}], "new": [], "drop": {},
    }))
    variant = tmp_path / "variant.json"
    variant.write_text(json.dumps({"name": "tiny", "weight_map": str(weight_map)}))
    monkeypatch.setenv("TEST_INIT_OMNI", str(weights))
    built = {}

    def fake_from_variant(path, **kwargs):
        built.update(kwargs, path=path)
        return OmniVGGTOmega(**{**TINY_OMEGA, **kwargs})

    monkeypatch.setattr(OmniVGGTOmega, "from_variant", staticmethod(fake_from_variant))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))
    cfg = {"model_name": "omnivggt_omega", "omega_variant": str(variant), "cam_drop_prob": 0.1,
           "depth_drop_prob": 0.3, "output_dir": str(tmp_path / "run"), "seed": 42}
    return cfg, source, built


def test_load_model_omega_variant(omega_cfg):
    cfg, source, built = omega_cfg
    model, _ = train_utils.load_model(cfg, torch.device("cpu"))
    assert built["cam_drop_prob"] == 0.1 and built["depth_drop_prob"] == 0.3
    assert model.aggregator.depth_drop_prob == 0.3 and model.aggregator.cam_drop_prob == 0.1
    for key, value in source.state_dict().items():
        assert torch.equal(model.state_dict()[key], value)
    report = json.loads(Path(cfg["output_dir"], "weight_transfer_report.json").read_text())
    assert report["variant"]["seed"] == 42 and len(report["variant"]["sha256"]) == 64
    assert report["targets"]["new"]["keys"] == 0


def _trained_checkpoint(tmp_path, extra=None, drop=None):
    """A checkpoint directory as written by accelerate's save_state: <run>/<exp>/<name>/model.safetensors."""
    torch.manual_seed(11)
    trained = OmniVGGTOmega(**TINY_OMEGA).state_dict()
    state = {k: v.contiguous() for k, v in trained.items() if k != drop}
    state.update(extra or {})
    directory = tmp_path / "prev_run" / "exp" / "final_checkpoint"
    directory.mkdir(parents=True)
    save_file(state, directory / "model.safetensors")
    return directory, trained


@pytest.mark.parametrize("as_file", [False, True])
def test_omega_init_checkpoint_is_loaded_strictly_without_weight_map(omega_cfg, tmp_path, monkeypatch, as_file):
    import eval_colmap_rgbd

    cfg, _, _ = omega_cfg
    monkeypatch.delenv("TEST_INIT_OMNI")  # the weight map (and its sources) must not be used
    directory, trained = _trained_checkpoint(tmp_path)
    checkpoint = directory / "model.safetensors" if as_file else directory
    model, _ = train_utils.load_model({**cfg, "init_checkpoint": str(checkpoint)}, torch.device("cpu"))
    for key, value in trained.items():
        assert torch.equal(model.state_dict()[key], value), key
    report = json.loads(Path(cfg["output_dir"], "weight_transfer_report.json").read_text())
    assert report["init_checkpoint"]["file"] == "final_checkpoint/model.safetensors"
    assert len(report["init_checkpoint"]["sha256"]) == 64 and report["init_checkpoint"]["tensors"] == len(trained)
    assert report["variant"]["sha256"] == train_utils.weight_transfer.sha256_of(Path(cfg["omega_variant"]))
    new_checkpoint = Path(cfg["output_dir"], "exp", "final_checkpoint")
    new_checkpoint.mkdir(parents=True)
    assert eval_colmap_rgbd._check_variant_provenance(Path(cfg["omega_variant"]), new_checkpoint) == "verified"


def _write_source_report(directory, variant_sha256):
    report = directory.parent.parent / "weight_transfer_report.json"
    report.write_text(json.dumps({"variant": {"file": "V5.json", "sha256": variant_sha256}}))


def test_omega_init_checkpoint_records_the_source_variant(omega_cfg, tmp_path):
    cfg, _, _ = omega_cfg
    directory, _ = _trained_checkpoint(tmp_path)
    sha = train_utils.weight_transfer.sha256_of(Path(cfg["omega_variant"]))
    _write_source_report(directory, sha)
    train_utils.load_model({**cfg, "init_checkpoint": str(directory)}, torch.device("cpu"))
    report = json.loads(Path(cfg["output_dir"], "weight_transfer_report.json").read_text())
    assert report["init_checkpoint"]["source_variant"] == {"file": "V5.json", "sha256": sha}


def test_omega_init_checkpoint_from_another_variant_raises(omega_cfg, tmp_path):
    cfg, _, _ = omega_cfg
    directory, _ = _trained_checkpoint(tmp_path)
    _write_source_report(directory, "0" * 64)
    with pytest.raises(ValueError, match="variant"):
        train_utils.load_model({**cfg, "init_checkpoint": str(directory)}, torch.device("cpu"))


def test_omega_init_checkpoint_without_source_report_is_unverified(omega_cfg, tmp_path):
    cfg, _, _ = omega_cfg
    directory, _ = _trained_checkpoint(tmp_path)
    train_utils.load_model({**cfg, "init_checkpoint": str(directory / "model.safetensors")}, torch.device("cpu"))
    report = json.loads(Path(cfg["output_dir"], "weight_transfer_report.json").read_text())
    assert report["init_checkpoint"]["source_variant"] == "unverified"


@pytest.mark.parametrize("change", ["missing", "unexpected"])
def test_omega_init_checkpoint_with_mismatched_keys_raises(omega_cfg, tmp_path, change):
    cfg, _, _ = omega_cfg
    if change == "missing":
        directory, _ = _trained_checkpoint(tmp_path, drop="aggregator.camera_token")
    else:
        directory, _ = _trained_checkpoint(tmp_path, extra={"point_head.extra": torch.zeros(1)})
    with pytest.raises(RuntimeError, match=r"Missing key|Unexpected key"):
        train_utils.load_model({**cfg, "init_checkpoint": str(directory)}, torch.device("cpu"))


def test_omega_missing_init_checkpoint_raises(omega_cfg, tmp_path):
    cfg, _, _ = omega_cfg
    with pytest.raises(FileNotFoundError):
        train_utils.load_model({**cfg, "init_checkpoint": str(tmp_path / "nope")}, torch.device("cpu"))


def test_unknown_model_name_raises(omega_cfg):
    cfg, _, _ = omega_cfg
    with pytest.raises(ValueError, match="model_name"):
        train_utils.load_model({**cfg, "model_name": "nope"}, torch.device("cpu"))


def test_build_loss_criterion_derived_mode():
    PartialState()
    derived = train_utils.build_loss_criterion(
        {"point_loss_mode": "derived", "point_loss_weight": 1.0, "point_intrinsics_warmup_ratio": 0.5}
    )
    assert derived.point == {"mode": "derived", "weight": 1.0, "intrinsics_warmup_ratio": 0.5}
    head = train_utils.build_loss_criterion({})
    assert head.point == {"weight": 1.0, "gradient_loss_fn": "normal", "valid_range": 0.98}


def test_build_optimizer_point_head_mismatch_raises():
    PartialState()
    model = OmniVGGTOmega(**TINY_OMEGA)
    base = {"lr": 1e-5, "enable_camera": True, "enable_depth": True}
    with pytest.raises(ValueError, match="point_head"):
        train_utils.build_optimizer(model, {**base, "enable_point": True})
    optimizer = train_utils.build_optimizer(model, {**base, "enable_point": False})
    grouped = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(grouped) == len(set(grouped)) == sum(1 for _ in model.parameters())


@pytest.mark.parametrize(
    "name, expected",
    [("checkpoint-epoch-1", (1, 1560)), ("checkpoint-epoch-2", (2, 3120)), ("checkpoint-0-500", (0, 500))],
)
def test_resume_position(name, expected):
    assert train_utils.resume_position(name, steps_per_epoch=1560) == expected


def test_resume_position_rejects_unknown_names():
    with pytest.raises(ValueError, match="final_checkpoint"):
        train_utils.resume_position("final_checkpoint", steps_per_epoch=1560)
