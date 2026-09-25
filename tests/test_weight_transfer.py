"""Map-driven non-strict weight transfer: every key is accounted for, undeclared differences fail."""

import json

import pytest
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from omnivggt.utils import weight_transfer as wt


class Tiny(nn.Module):
    def __init__(self, registers=4):
        super().__init__()
        self.frame = nn.Linear(3, 3)
        self.global_blocks = nn.ModuleList([nn.Linear(3, 3), nn.Linear(3, 3)])
        self.register_token = nn.Parameter(torch.full((1, 2, registers, 3), 7.0))
        self.extra = nn.Linear(3, 1)


def _source_a():
    torch.manual_seed(1)
    return {**{k: torch.randn_like(v) for k, v in Tiny().state_dict().items()}, "point_head.w": torch.ones(2)}


def _source_b():
    torch.manual_seed(2)
    return {
        "inter_frame_blocks.0.weight": torch.randn(3, 3),
        "inter_frame_blocks.0.bias": torch.randn(3),
        "inter_frame_blocks.0.bias_mask": torch.tensor([1.0, 0.0, 1.0]),
        "inter_frame_blocks.1.weight": torch.randn(3, 3),
        "inter_frame_blocks.1.bias": torch.randn(3),
        "inter_frame_blocks.1.bias_mask": torch.tensor([0.0, 0.0, 0.0]),
        "camera_head.w": torch.randn(4),
    }


def _map(**overrides):
    base = {
        "name": "test",
        "sources": {"a": {"env": "TEST_SRC_A", "format": "safetensors"}},
        "rules": [{"target_prefix": "", "source": "a", "source_prefix": ""}],
        "new": [],
        "drop": {"a": ["point_head."]},
    }
    base.update(overrides)
    return base


def test_transfer_loads_renames_and_reports():
    model = Tiny()
    sources = {"a": _source_a(), "b": _source_b()}
    weight_map = _map(
        sources={"a": {"env": "A", "format": "safetensors"}, "b": {"env": "B", "format": "torch"}},
        rules=[
            {"target_prefix": "", "source": "a", "source_prefix": ""},
            {"target_prefix": "global_blocks.", "source": "b", "source_prefix": "inter_frame_blocks."},
        ],
        drop={
            "a": ["point_head.", "global_blocks."],
            "b": ["camera_head.", "inter_frame_blocks.0.bias_mask", "inter_frame_blocks.1.bias_mask"],
        },
    )
    report = wt.transfer_weights(model, weight_map, sources)
    assert torch.equal(model.global_blocks[0].weight, sources["b"]["inter_frame_blocks.0.weight"])
    assert torch.equal(model.frame.weight, sources["a"]["frame.weight"])
    assert report["targets"]["renamed"]["keys"] == 4
    assert report["targets"]["loaded"]["keys"] == 5  # frame.{weight,bias}, register_token, extra.{weight,bias}
    assert report["targets"]["new"]["keys"] == 0
    assert report["sources"]["a"]["dropped_keys"] == 5  # point_head.w + 4 global_blocks keys of source a
    assert report["sources"]["b"]["dropped_keys"] == 3


def test_unexpected_key_raises():
    sources = {"a": {**_source_a(), "surprise.w": torch.ones(1)}}
    with pytest.raises(wt.WeightTransferError, match=r"surprise\.w"):
        wt.transfer_weights(Tiny(), _map(), sources)


def test_shape_mismatch_not_declared_raises():
    sources = {"a": {**_source_a(), "extra.weight": torch.ones(2, 3)}}
    with pytest.raises(wt.WeightTransferError, match=r"extra\.weight"):
        wt.transfer_weights(Tiny(), _map(), sources)


def test_missing_key_not_declared_new_raises():
    source = _source_a()
    del source["extra.bias"]
    with pytest.raises(wt.WeightTransferError, match=r"extra\.bias"):
        wt.transfer_weights(Tiny(), _map(), {"a": source})
    model = Tiny()
    report = wt.transfer_weights(model, _map(new=["extra.bias"]), {"a": source})
    assert report["new_keys"] == ["extra.bias"]


def test_slice_rule():
    model = Tiny(registers=16)
    init = model.register_token.detach().clone()
    source = _source_a()
    weight_map = _map(
        rules=[
            {"target_prefix": "", "source": "a", "source_prefix": ""},
            {
                "target_prefix": "register_token",
                "source": "a",
                "source_prefix": "register_token",
                "slice": {"dim": 2, "start": 0, "stop": 4},
            },
        ],
        new=["register_token"],
    )
    report = wt.transfer_weights(model, weight_map, {"a": source})
    assert torch.equal(model.register_token[:, :, :4], source["register_token"])
    assert torch.equal(model.register_token[:, :, 4:], init[:, :, 4:])
    assert report["sliced_keys"] == ["register_token"]
    assert report["sliced_new_elements"] == {"register_token": 2 * 12 * 3}


def test_longest_prefix_rule_wins():
    model = Tiny()
    source_a, source_b = _source_a(), _source_b()
    weight_map = _map(
        sources={"a": {"env": "A", "format": "safetensors"}, "b": {"env": "B", "format": "torch"}},
        rules=[
            {"target_prefix": "", "source": "a", "source_prefix": ""},
            {"target_prefix": "global_blocks.1.", "source": "b", "source_prefix": "inter_frame_blocks.1."},
        ],
        drop={"a": ["point_head.", "global_blocks.1."], "b": ["camera_head.", "inter_frame_blocks."]},
    )
    wt.transfer_weights(model, weight_map, {"a": source_a, "b": source_b})
    assert torch.equal(model.global_blocks[0].weight, source_a["global_blocks.0.weight"])
    assert torch.equal(model.global_blocks[1].weight, source_b["inter_frame_blocks.1.weight"])


def test_multiply_by_mask_rule():
    model = Tiny()
    source_a, source_b = _source_a(), _source_b()
    weight_map = _map(
        sources={"a": {"env": "A", "format": "safetensors"}, "b": {"env": "B", "format": "torch"}},
        rules=[
            {"target_prefix": "", "source": "a", "source_prefix": ""},
            {
                "target_prefix": "global_blocks.",
                "source": "b",
                "source_prefix": "inter_frame_blocks.",
                "multiply_by_suffix": {".bias": ".bias_mask"},
            },
        ],
        drop={"a": ["point_head.", "global_blocks."], "b": ["camera_head."]},
    )
    report = wt.transfer_weights(model, weight_map, {"a": source_a, "b": source_b})
    expected = source_b["inter_frame_blocks.0.bias"] * source_b["inter_frame_blocks.0.bias_mask"]
    assert torch.equal(model.global_blocks[0].bias, expected)
    assert torch.equal(model.global_blocks[1].bias, torch.zeros(3))
    assert report["sources"]["b"]["dropped_keys"] == 1  # only camera_head.w; the masks are used
    assert sorted(report["multiplied_keys"]) == ["global_blocks.0.bias", "global_blocks.1.bias"]


def test_report_counts_parameters(tmp_path):
    path = tmp_path / "source.safetensors"
    save_file(_source_a(), str(path))
    map_path = tmp_path / "map.json"
    map_path.write_text(json.dumps(_map()))
    weight_map = wt.load_map(map_path)
    sources, files = wt.load_sources(weight_map, env={"TEST_SRC_A": str(path)})
    model = Tiny()
    report = wt.transfer_weights(model, weight_map, sources, files=files)
    assert report["total_params"] == sum(p.numel() for p in model.parameters())
    assert report["sources"]["a"]["file"] == "source.safetensors"
    assert len(report["sources"]["a"]["sha256"]) == 64
    assert report["map"]["file"] == "map.json" and len(report["map"]["sha256"]) == 64
    classified = sum(report["targets"][kind]["params"] for kind in ("loaded", "renamed", "sliced", "new"))
    assert classified == report["total_params"]


def test_load_sources_missing_env_raises(tmp_path):
    weight_map = _map()
    with pytest.raises(KeyError, match="TEST_SRC_A"):
        wt.load_sources(weight_map, env={})
    with pytest.raises(FileNotFoundError, match="TEST_SRC_A"):
        wt.load_sources(weight_map, env={"TEST_SRC_A": str(tmp_path / "absent.safetensors")})


def test_cli_writes_report(tmp_path):
    path = tmp_path / "source.safetensors"
    save_file(_source_a(), str(path))
    map_path = tmp_path / "map.json"
    map_path.write_text(json.dumps(_map()))
    report_path, save_path = tmp_path / "report.json", tmp_path / "out.safetensors"
    code = wt.main(
        ["--map", str(map_path), "--report", str(report_path), "--save", str(save_path)],
        model=Tiny(),
        env={"TEST_SRC_A": str(path)},
    )
    assert code == 0
    assert json.loads(report_path.read_text())["targets"]["loaded"]["keys"] > 0
    assert set(load_file(str(save_path))) == set(Tiny().state_dict())


def test_cli_resolves_variant_map_from_repo_root(tmp_path, monkeypatch):
    variant = tmp_path / "variant.json"
    variant.write_text(json.dumps({"name": "x", "weight_map": "configs/omnivggt_omega/weight_maps/omni_reg4.json"}))
    monkeypatch.chdir(tmp_path)  # not the repository root
    assert wt.variant_map_path(variant) == wt.REPO_ROOT / "configs/omnivggt_omega/weight_maps/omni_reg4.json"
    assert wt.variant_map_path(variant).is_file()


def test_cli_seed_defaults_to_training_seed():
    assert wt.DEFAULT_SEED == 42
