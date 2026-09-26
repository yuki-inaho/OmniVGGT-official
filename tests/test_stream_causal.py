"""Frame-causal (batch) OmniVGGTOmega: the inter-frame mask, the aggregator, the camera head and the wiring."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from omnivggt.heads.camera_head import CameraHead, modulate
from omnivggt.heads.head_act import activate_pose
from omnivggt.models.omnivggt_omega import OmniVGGTOmega
from omnivggt.stream.masks import frame_causal_mask


def test_frame_causal_mask():
    got = frame_causal_mask(3, 2, torch.device("cpu"))
    want = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert got.dtype == torch.bool and got.device.type == "cpu"
    assert torch.equal(got, want)


def test_frame_causal_mask_of_one_token_per_frame_is_lower_triangular():
    assert torch.equal(frame_causal_mask(4, 1, "cpu"), torch.ones(4, 4, dtype=torch.bool).tril())


@pytest.mark.parametrize("frames, tokens", [(0, 2), (3, 0)])
def test_frame_causal_mask_rejects_empty_sizes(frames, tokens):
    with pytest.raises(ValueError):
        frame_causal_mask(frames, tokens, "cpu")


# --- aggregator: frame-causal inter-frame attention and first-frame depth normalisation -------------------

TINY = dict(
    img_size=28,
    patch_size=14,
    embed_dim=32,
    num_register_tokens=16,
    register_attention_layers=(1,),  # layers 0, 2, 3 are full global attention, layer 1 register attention
    global_rope=False,
    cached_layers=(0, 1, 2, 3),
    aggregator_kwargs=dict(depth=4, num_heads=2, patch_embed="conv"),
    camera_head_kwargs=dict(trunk_depth=1, num_heads=2),
    depth_head_kwargs=dict(features=16, out_channels=[8, 16, 32, 32], intermediate_layer_idx=[0, 1, 2, 3]),
)
FRAMES = 4
PAST = slice(0, FRAMES - 1)
DEPTH_INDEX = {"depth": list(range(FRAMES)), "rgb": []}


def _aggregator_model(causal=False, depth_norm="joint", **kwargs):
    """Tiny fp64 OmniVGGTOmega whose aggregator (only) is configured through aggregator_kwargs."""
    torch.manual_seed(0)
    aggregator_kwargs = {**TINY["aggregator_kwargs"], "causal": causal, "depth_norm": depth_norm}
    return OmniVGGTOmega(**{**TINY, "aggregator_kwargs": aggregator_kwargs}, **kwargs).double().eval()


def _inputs(seed=0):
    generator = torch.Generator().manual_seed(seed)
    shape = (1, FRAMES, 28, 42)
    intrinsics = torch.tensor([[30.0, 0.0, 21.0], [0.0, 30.0, 14.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    return {
        "images": torch.rand(1, FRAMES, 3, 28, 42, generator=generator, dtype=torch.float64),
        "extrinsics": torch.eye(4, dtype=torch.float64)[:3].repeat(1, FRAMES, 1, 1),
        "intrinsics": intrinsics.repeat(1, FRAMES, 1, 1),
        "depth": 0.5 + torch.rand(*shape, 1, generator=generator, dtype=torch.float64),
        "mask": (torch.rand(*shape, generator=generator) > 0.2).double(),
    }


def _with_other_last_frame(inputs):
    other = _inputs(seed=1)
    changed = {key: value.clone() for key, value in inputs.items()}
    for key in ("images", "depth", "mask"):
        changed[key][:, -1] = other[key][:, -1]
    return changed


def _infer_aggregator(model, inputs, depth_index, camera_index=()):
    with torch.no_grad():
        layers, _ = model.aggregator.inference(**inputs, depth_gt_index=list(depth_index),
                                               camera_gt_index=list(camera_index))
        out = model.inference(**inputs, depth_gt_index=list(depth_index), camera_gt_index=list(camera_index))
    return layers, out


@pytest.mark.parametrize("condition", list(DEPTH_INDEX))
def test_causal_aggregator_past_frames_ignore_the_last_frame(condition):
    model = _aggregator_model(causal=True, depth_norm="first_frame")
    inputs = _inputs()
    layers, out = _infer_aggregator(model, inputs, DEPTH_INDEX[condition])
    layers_changed, out_changed = _infer_aggregator(model, _with_other_last_frame(inputs), DEPTH_INDEX[condition])
    for layer, layer_changed in zip(layers, layers_changed, strict=True):
        assert torch.equal(layer[:, PAST], layer_changed[:, PAST])
    assert not torch.equal(layers[-1][:, -1], layers_changed[-1][:, -1])
    for key in ("depth", "depth_conf"):  # the depth head is per frame
        assert torch.equal(out[key][:, PAST], out_changed[key][:, PAST])


@pytest.mark.parametrize("condition", list(DEPTH_INDEX))
def test_bidirectional_aggregator_past_frames_see_the_last_frame(condition):
    model = _aggregator_model()
    inputs = _inputs()
    layers, _ = _infer_aggregator(model, inputs, DEPTH_INDEX[condition])
    layers_changed, _ = _infer_aggregator(model, _with_other_last_frame(inputs), DEPTH_INDEX[condition])
    assert not torch.equal(layers[-1][:, PAST], layers_changed[-1][:, PAST])


def test_first_frame_normalize_depth_divides_every_view_by_the_frame_0_valid_mean():
    aggregator = _aggregator_model(depth_norm="first_frame").aggregator
    inputs = _inputs()
    depth, mask = inputs["depth"], inputs["mask"]
    mean = depth[0, 0, ..., 0][mask[0, 0] > 0].mean()
    expected = (depth[..., 0] / (mean + 1e-8) * mask).unsqueeze(-1)
    assert torch.equal(aggregator.normalize_depth(depth, mask), expected)


@pytest.mark.parametrize("depth_norm, frame_0_unchanged", [("first_frame", True), ("joint", False)])
def test_first_frame_depth_norm_frame_0_ignores_later_depth(depth_norm, frame_0_unchanged):
    model = _aggregator_model(causal=True, depth_norm=depth_norm)
    inputs = _inputs()
    scaled = {**inputs, "depth": inputs["depth"].clone()}
    scaled["depth"][:, 1:] *= 2
    layers, out = _infer_aggregator(model, inputs, DEPTH_INDEX["depth"])
    layers_scaled, out_scaled = _infer_aggregator(model, scaled, DEPTH_INDEX["depth"])
    assert torch.equal(layers[-1][:, 0], layers_scaled[-1][:, 0]) is frame_0_unchanged
    assert torch.equal(out["depth"][:, 0], out_scaled["depth"][:, 0]) is frame_0_unchanged


def test_first_frame_depth_norm_without_valid_frame_0_depth_raises():
    model = _aggregator_model(causal=True, depth_norm="first_frame")
    inputs = _inputs()
    inputs["mask"][:, 0] = 0
    with pytest.raises(ValueError, match="frame 0"):
        _infer_aggregator(model, inputs, DEPTH_INDEX["depth"])


@pytest.mark.parametrize("depth_index", [[1, 2], [2, 0, 1]])
def test_first_frame_depth_norm_needs_frame_0_as_the_first_depth_view(depth_index):
    model = _aggregator_model(depth_norm="first_frame")
    with pytest.raises(ValueError, match="frame 0"):
        _infer_aggregator(model, _inputs(), depth_index)


def test_unknown_depth_norm_raises():
    with pytest.raises(ValueError, match="depth_norm"):
        _aggregator_model(depth_norm="median")


def test_causal_inference_rejects_camera_input():
    model = _aggregator_model(causal=True)
    with pytest.raises(ValueError, match="camera"):
        _infer_aggregator(model, _inputs(), [], camera_index=[0])


def test_causal_training_rejects_camera_input():
    model = _aggregator_model(causal=True, depth_norm="first_frame", depth_all_views=True).train()
    assert model.aggregator.cam_drop_prob < 1
    with pytest.raises(ValueError, match="cam_drop_prob"):
        model(**_inputs(), modality_rng=np.random.default_rng(0))


def test_first_frame_training_needs_depth_on_every_view():
    """A random depth subset may leave out frame 0, so the option must hold for every training forward."""
    model = _aggregator_model(depth_norm="first_frame", cam_drop_prob=1.0).train()
    with pytest.raises(ValueError, match="depth_all_views"):
        model(**_inputs(), modality_rng=np.random.default_rng(0))


def test_causal_training_forward_through_gradient_checkpoints_is_frame_causal():
    model = _aggregator_model(causal=True, depth_norm="first_frame", cam_drop_prob=1.0, depth_all_views=True).train()
    assert model.aggregator.use_checkpoint  # the training path runs every block under torch.utils.checkpoint
    inputs = _inputs()
    layers, _ = model.aggregator(**inputs, modality_rng=np.random.default_rng(0))
    layers_changed, _ = model.aggregator(**_with_other_last_frame(inputs), modality_rng=np.random.default_rng(0))
    for layer, layer_changed in zip(layers, layers_changed, strict=True):
        assert torch.equal(layer[:, PAST], layer_changed[:, PAST])
    sum(layer[:, PAST].sum() for layer in layers).backward()
    gradients = [p.grad for p in model.aggregator.global_blocks.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)


def test_aggregator_stream_context_is_unset_by_default():
    model = _aggregator_model()
    assert model.aggregator._stream is None
    model.aggregator._stream = object()  # the streaming context is not implemented at this layer yet
    with pytest.raises(NotImplementedError):
        _infer_aggregator(model, _inputs(), [])


# --- camera head: frame-causal trunk ---------------------------------------------------------------------


def _camera_head(causal=False):
    torch.manual_seed(0)
    return CameraHead(dim_in=64, trunk_depth=2, num_heads=2, causal=causal).double().eval()


def _camera_tokens(seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(2, FRAMES, 3, 64, generator=generator, dtype=torch.float64)  # [B, S, P, C]


def _pose_list_through_sequential(head, tokens, num_iterations=4):
    """CameraHead.forward as written before its trunk became an explicit loop: ``head.trunk`` called as a whole."""
    pose_tokens = head.token_norm(tokens[:, :, 0])
    batch, frames, _ = pose_tokens.shape
    pred, poses = None, []
    for _ in range(num_iterations):
        pose = head.empty_pose_tokens.expand(batch, frames, -1) if pred is None else pred.detach()
        shift, scale, gate = head.poseLN_modulation(head.embed_pose(pose)).chunk(3, dim=-1)
        modulated = gate * modulate(head.adaln_norm(pose_tokens), shift, scale) + pose_tokens
        delta = head.pose_branch(head.trunk_norm(head.trunk(modulated)))
        pred = delta if pred is None else pred + delta
        poses.append(activate_pose(pred, trans_act=head.trans_act, quat_act=head.quat_act, fl_act=head.fl_act))
    return poses


def test_camera_head_default_output_is_bitwise_the_sequential_trunk():
    head, tokens = _camera_head(), _camera_tokens()
    with torch.no_grad():
        for got, want in zip(head([tokens]), _pose_list_through_sequential(head, tokens), strict=True):
            assert torch.equal(got, want)


@pytest.mark.parametrize("causal", [True, False])
def test_camera_head_past_poses_ignore_the_last_frame_only_when_causal(causal):
    head, tokens = _camera_head(causal), _camera_tokens()
    changed = tokens.clone()
    changed[:, -1] = _camera_tokens(seed=1)[:, -1]
    with torch.no_grad():
        poses, poses_changed = head([tokens]), head([changed])
    for pose, pose_changed in zip(poses, poses_changed, strict=True):
        assert torch.equal(pose[:, PAST], pose_changed[:, PAST]) is causal
        assert not torch.equal(pose[:, -1], pose_changed[:, -1])


def test_causal_camera_head_frame_t_equals_the_head_on_frames_up_to_t():
    head, tokens = _camera_head(causal=True), _camera_tokens()
    with torch.no_grad():
        whole = head([tokens])[-1]
        for frame in range(FRAMES):
            prefix = head([tokens[:, : frame + 1]])[-1]
            torch.testing.assert_close(prefix[:, frame], whole[:, frame], rtol=0, atol=1e-12)


def test_camera_head_keeps_the_sequential_trunk_and_state_dict_keys():
    head, causal = _camera_head(), _camera_head(causal=True)
    assert isinstance(causal.trunk, nn.Sequential) and len(causal.trunk) == 2
    assert list(causal.state_dict()) == list(head.state_dict())
    causal.load_state_dict(head.state_dict(), strict=True)


def test_camera_head_stream_context_is_unset_by_default():
    head = _camera_head()
    assert head._stream is None
    head._stream = object()  # the streaming context is not implemented at this layer yet
    with pytest.raises(NotImplementedError):
        head([_camera_tokens()])
