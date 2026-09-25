"""OmniVGGTOmega: OmniVGGT heads on the VGGT-Omega style aggregator, points derived from depth and camera."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from omnivggt.heads.camera_head import CameraHead
from omnivggt.heads.dpt_head import DPTHead
from omnivggt.models.omnivggt_aggregator import ZeroAggregator
from omnivggt.models.omnivggt_omega import OmniVGGTOmega
from omnivggt.utils.geometry import unproject_depth_to_world_points_torch
from omnivggt.utils.pose_enc import pose_encoding_to_extri_intri

FIXTURE = Path(__file__).parent / "fixtures" / "zero_aggregator_golden.pt"
TINY = dict(
    img_size=28,
    patch_size=14,
    embed_dim=32,
    aggregator_kwargs=dict(depth=4, num_heads=2, patch_embed="conv"),
    camera_head_kwargs=dict(trunk_depth=1, num_heads=2),
    depth_head_kwargs=dict(features=16, out_channels=[8, 16, 32, 32], intermediate_layer_idx=[0, 1, 2, 3]),
)
V0 = dict(num_register_tokens=4, register_attention_layers=(), global_rope=True, cached_layers=(0, 1, 2, 3))


@pytest.fixture(scope="module")
def golden():
    return torch.load(FIXTURE, weights_only=True)


def _tiny(**variant):
    torch.manual_seed(0)
    return OmniVGGTOmega(**TINY, **{**V0, **variant}).eval()


def _infer(model, golden, depth_index=(0, 1, 2), camera_index=(0, 1, 2)):
    with torch.no_grad():
        return model.inference(**golden["inputs"], depth_gt_index=list(depth_index), camera_gt_index=list(camera_index))


def test_output_keys_and_shapes(golden):
    model = _tiny(register_attention_layers=(1,), num_register_tokens=16, global_rope=False)
    out = _infer(model, golden)
    batch, frames, _, height, width = golden["inputs"]["images"].shape
    assert out["pose_enc"].shape == (batch, frames, 9)
    assert len(out["pose_enc_list"]) == 4
    assert out["depth"].shape == (batch, frames, height, width, 1)
    assert out["depth_conf"].shape == (batch, frames, height, width)
    assert out["world_points"].shape == (batch, frames, height, width, 3)
    assert "world_points_conf" not in out
    assert out["images"] is golden["inputs"]["images"]


def test_no_point_head_parameters():
    model = _tiny()
    assert not hasattr(model, "point_head")
    assert not any(name.startswith("point_head.") for name in model.state_dict())


def test_world_points_are_derived_from_depth_and_camera(golden):
    model = _tiny(register_attention_layers=(1,))
    out = _infer(model, golden)
    extrinsics, intrinsics = pose_encoding_to_extri_intri(out["pose_enc"], out["depth"].shape[2:4])
    expected = unproject_depth_to_world_points_torch(out["depth"], extrinsics, intrinsics)
    torch.testing.assert_close(out["world_points"], expected)


def test_return_points_false_skips_points(golden):
    model = _tiny()
    with torch.no_grad():
        out = model.inference(**golden["inputs"], depth_gt_index=[], camera_gt_index=[], return_points=False)
    assert "world_points" not in out


def test_depth_head_layers_must_be_cached():
    with pytest.raises(ValueError):
        OmniVGGTOmega(**TINY, **{**V0, "cached_layers": (1, 3)})


class _Reference(nn.Module):
    """OmniVGGT's head wiring on the unmodified ZeroAggregator (tiny config)."""

    def __init__(self, golden):
        super().__init__()
        self.aggregator = ZeroAggregator(**golden["config"])
        self.camera_head = CameraHead(dim_in=64, **TINY["camera_head_kwargs"])
        self.depth_head = DPTHead(
            dim_in=64, output_dim=2, activation="exp", conf_activation="expp1", **TINY["depth_head_kwargs"]
        )

    def inference(self, images, extrinsics, intrinsics, depth, mask, depth_gt_index, camera_gt_index):
        tokens, patch_start_idx = self.aggregator.inference(
            images, extrinsics, intrinsics, depth, mask, depth_gt_index, camera_gt_index
        )
        pose_enc_list = self.camera_head(tokens)
        pred_depth, depth_conf = self.depth_head(tokens, images=images, patch_start_idx=patch_start_idx)
        return {"pose_enc_list": pose_enc_list, "depth": pred_depth, "depth_conf": depth_conf}


@pytest.mark.parametrize("indices", [((), ()), ((0, 1, 2), (0, 1, 2)), ((1,), (0, 1))])
def test_v0_model_matches_omnivggt_heads(golden, indices):
    torch.manual_seed(0)
    reference = _Reference(golden).eval()
    model = _tiny()
    model.load_state_dict(reference.state_dict(), strict=True)
    depth_index, camera_index = indices
    with torch.no_grad():
        want = reference.inference(
            **golden["inputs"], depth_gt_index=list(depth_index), camera_gt_index=list(camera_index)
        )
    got = _infer(model, golden, depth_index, camera_index)
    for got_stage, want_stage in zip(got["pose_enc_list"], want["pose_enc_list"], strict=True):
        torch.testing.assert_close(got_stage, want_stage, rtol=0, atol=1e-6)
    torch.testing.assert_close(got["depth"], want["depth"], rtol=0, atol=1e-6)
    torch.testing.assert_close(got["depth_conf"], want["depth_conf"], rtol=0, atol=1e-6)
