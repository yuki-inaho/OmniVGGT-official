"""Characterization test: ``ZeroAggregator.inference`` must reproduce the frozen golden outputs exactly."""

from pathlib import Path

import pytest
import torch

from omnivggt.models.omnivggt_aggregator import ZeroAggregator

FIXTURE = Path(__file__).parent / "fixtures" / "zero_aggregator_golden.pt"
RECORDED_ON = ("AVX512", "2.7.0+cu128")  # CPU capability and torch build that wrote the fixture (bitwise there)
CONDITIONS = {
    "rgb": ([], []),
    "depth": ([0, 1, 2], []),
    "camera": ([], [0, 1, 2]),
    "depth+camera": ([0, 1, 2], [0, 1, 2]),
    "partial": ([1], [0, 1]),
}


@pytest.fixture(scope="module")
def golden():
    return torch.load(FIXTURE, weights_only=True)


@pytest.mark.parametrize("condition", list(CONDITIONS))
def test_inference_matches_golden(golden, condition):
    model = ZeroAggregator(**golden["config"])
    model.load_state_dict(golden["state_dict"], strict=True)
    model.eval()
    depth_index, camera_index = CONDITIONS[condition]
    with torch.no_grad():
        layers, patch_start_idx = model.inference(
            **golden["inputs"], depth_gt_index=depth_index, camera_gt_index=camera_index
        )
    assert patch_start_idx == golden["outputs"]["patch_start_idx"]
    expected = golden["outputs"][condition]
    assert len(layers) == len(expected)
    for got, want in zip(layers, expected, strict=True):
        if RECORDED_ON == (torch.backends.cpu.get_cpu_capability(), torch.__version__):
            assert torch.equal(got, want)
        else:  # CPU float32 kernels differ by ISA (about 1 ulp); the refactor is still checked bitwise elsewhere
            torch.testing.assert_close(got, want, rtol=0, atol=1e-6)


def test_golden_exercises_the_geoadapter(golden):
    """The frozen conditions must differ, otherwise the camera/depth injection paths would be untested."""
    outputs = golden["outputs"]
    for condition in ("depth", "camera", "depth+camera", "partial"):
        assert not torch.equal(outputs[condition][-1], outputs["rgb"][-1])
