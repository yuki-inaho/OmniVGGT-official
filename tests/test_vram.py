"""Opt-in per-process VRAM cap for GPU jobs that share a device (OMNIVGGT_VRAM_LIMIT_GB, in GiB): the helper, and
its use at the startup of the training entry points and the stream tools, which record the cap and the peaks."""

import json
from pathlib import Path

import bench_stream
import pytest
import stream_gates
import torch
from test_stream_causal import TINY

from omnivggt.models.omnivggt_omega import OmniVGGTOmega
from omnivggt.utils import vram

REPO = Path(__file__).resolve().parents[1]
ENV = "OMNIVGGT_VRAM_LIMIT_GB"
GIB = 2**30
POLICY = '{"recent": 1, "long_special": 1, "long_patch": 4, "selector": "query", "quant": "int8"}'


class _Properties:
    total_memory = 32 * GIB


@pytest.fixture
def cuda(monkeypatch):
    """A fake 32 GiB CUDA device; ``cuda["fraction"]`` records set_per_process_memory_fraction."""
    calls = {"fraction": [], "properties": []}

    def properties(device):
        calls["properties"].append(device)
        return _Properties()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction",
                        lambda fraction, device=None: calls["fraction"].append((fraction, device)))
    monkeypatch.delenv(ENV, raising=False)
    return calls


def test_no_limit_without_the_variable(cuda, monkeypatch):
    assert vram.apply_vram_limit("cuda") is None
    monkeypatch.setenv(ENV, "")
    assert vram.apply_vram_limit("cuda") is None
    assert vram.apply_vram_limit("cpu") is None  # nothing to cap: the CPU is fine without a limit
    assert cuda["fraction"] == [] and cuda["properties"] == []


@pytest.mark.parametrize("device, index", [("cuda", 0), ("cuda:1", 1), (torch.device("cuda", 2), 2)])
def test_the_limit_is_the_fraction_of_the_device_memory(cuda, monkeypatch, device, index):
    monkeypatch.setenv(ENV, "8")
    record = vram.apply_vram_limit(device)
    assert cuda["fraction"] == [(0.25, index)] and cuda["properties"] == [index]
    assert record == {"limit_gb": 8.0, "fraction": 0.25, "total_gb": 32.0, "device": f"cuda:{index}"}


@pytest.mark.parametrize("value, fraction", [("12.5", 12.5 / 32), ("32", 1.0), (" 4 ", 0.125)])
def test_any_positive_limit_up_to_the_device_memory(cuda, monkeypatch, value, fraction):
    monkeypatch.setenv(ENV, value)
    assert vram.apply_vram_limit("cuda")["fraction"] == fraction
    assert cuda["fraction"] == [(fraction, 0)]


def test_a_limit_above_the_device_memory_is_rejected(cuda, monkeypatch):
    monkeypatch.setenv(ENV, "32.5")
    with pytest.raises(ValueError, match=r"32\.5.*32\.00 GiB"):
        vram.apply_vram_limit("cuda")
    assert cuda["fraction"] == []


@pytest.mark.parametrize("value", ["abc", "0", "-1", "nan", "inf", "8GB"])
def test_invalid_limits_are_rejected_with_the_value(cuda, monkeypatch, value):
    monkeypatch.setenv(ENV, value)
    with pytest.raises(ValueError, match=ENV) as error:
        vram.apply_vram_limit("cuda")
    assert repr(value) in str(error.value)
    assert cuda["fraction"] == []


@pytest.mark.parametrize("device", ["cpu", torch.device("cpu")])
def test_a_limit_for_a_cpu_device_is_an_error(cuda, monkeypatch, device):
    monkeypatch.setenv(ENV, "8")
    with pytest.raises(ValueError, match=f"{ENV}.*cpu"):
        vram.apply_vram_limit(device)
    assert cuda["fraction"] == []


def test_a_limit_without_cuda_is_an_error(cuda, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv(ENV, "8")
    with pytest.raises(ValueError, match="CUDA"):
        vram.apply_vram_limit("cuda")
    assert cuda["fraction"] == []


def test_memory_peaks_in_mib(monkeypatch):
    assert vram.memory_peaks("cpu") is None
    seen = []
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device=None: seen.append(device) or 3 * 2**20)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device=None: seen.append(device) or 5.5 * 2**20)
    assert vram.memory_peaks("cuda:1") == {"max_allocated_mib": 3.0, "max_reserved_mib": 5.5}
    assert seen == [torch.device("cuda:1")] * 2


# --- the entry points ------------------------------------------------------------------------------------------------


def test_train_omnivggt_applies_the_limit_before_building_the_model_and_logs_the_peaks():
    source = (REPO / "train_omnivggt.py").read_text()
    limit, model = source.index("apply_vram_limit(accelerator.device)"), source.index("load_model(cfg, accelerator.device)")
    assert source.index("accelerate.Accelerator(") < limit < model
    assert "memory_peaks(accelerator.device)" in source


def _tiny_variant(order):
    def build(path, **options):
        order.append("model")
        return OmniVGGTOmega(**TINY, **options)

    return staticmethod(build)


def test_bench_stream_applies_the_limit_first_and_records_it_with_the_peaks(monkeypatch, tmp_path):
    order = []
    monkeypatch.setattr(OmniVGGTOmega, "from_variant", _tiny_variant(order))
    monkeypatch.setattr(bench_stream, "apply_vram_limit", lambda device: order.append("limit") or {"limit_gb": 8.0})
    output = tmp_path / "bench.json"
    argv = ["--model-config", "V.json", "--random-init", "--policy", "full", "--steps", "1,2", "--precision", "fp32",
            "--width", "56", "--height", "42", "--output", str(output)]
    assert bench_stream.main(argv) == 0
    assert order == ["limit", "model"]
    assert json.loads(output.read_text())["vram"] == {"limit": {"limit_gb": 8.0}, "peaks": None}  # no peaks on the CPU


def test_stream_gates_applies_the_limit_first_and_records_it_with_the_peaks(monkeypatch, tmp_path):
    order = []
    weights, variant = tmp_path / "model.safetensors", tmp_path / "V.json"
    weights.write_bytes(b"weights")
    variant.write_text("{}")

    class Loader:
        def __init__(self, *args):
            pass

        def load(self, window):
            return {}

    monkeypatch.setattr(stream_gates, "apply_vram_limit", lambda device: order.append("limit") or {"limit_gb": 8.0})
    monkeypatch.setattr(stream_gates, "_load_model", lambda *args, **kwargs: order.append("model") or (None, weights))
    monkeypatch.setattr(stream_gates, "WindowLoader", Loader)
    monkeypatch.setattr(stream_gates, "window_inputs", lambda item, device: {})
    monkeypatch.setattr(stream_gates, "_swap_last_frame", lambda inputs, other: 0)
    for gate in ("equivalence", "future_invariance", "cache_invariants"):
        monkeypatch.setattr(stream_gates, gate, lambda *args, **kwargs: {"ok": True})
    output = tmp_path / "gates.json"
    argv = ["--model-config", str(variant), "--checkpoint", str(weights), "--roots", "r0", "r1", "--split", "smoke",
            "--window", "s0:0-3", "--policy", POLICY, "--output", str(output)]
    assert stream_gates.main(argv) == 0
    assert order == ["limit", "model"]
    assert json.loads(output.read_text())["vram"] == {"limit": {"limit_gb": 8.0}, "peaks": None}


@pytest.mark.parametrize("tool", ["bench_stream", "stream_gates"])
def test_the_tools_refuse_a_limit_on_the_cpu_before_loading_anything(monkeypatch, tmp_path, tool):
    monkeypatch.setenv(ENV, "8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)  # the tools then run on the CPU
    argv = {
        "bench_stream": ["--model-config", "missing.json", "--random-init", "--policy", "full"],
        "stream_gates": ["--model-config", "missing.json", "--checkpoint", "missing", "--roots", "r0", "r1",
                         "--split", "smoke", "--window", "s0:0-3", "--policy", POLICY],
    }[tool]
    with pytest.raises(ValueError, match=ENV):
        {"bench_stream": bench_stream, "stream_gates": stream_gates}[tool].main([*argv, "--output", str(tmp_path / "o")])
