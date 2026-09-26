"""Frame-causal (batch) OmniVGGTOmega: the inter-frame mask, the aggregator, the camera head and the wiring;
frame visibility (which key frames each query frame attends to)."""

import json
import logging

import numpy as np
import pytest
import torch
import torch.nn as nn
from accelerate import PartialState
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

import omnivggt.heads.camera_head as camera_head_module
import omnivggt.models.omnivggt_aggregator as omnivggt_aggregator
import train_utils
from omnivggt.heads.camera_head import NUM_ITERATIONS, CameraHead, modulate
from omnivggt.heads.head_act import activate_pose
from omnivggt.layers.block import Block
from omnivggt.layers.rope import RotaryPositionEmbedding2D
from omnivggt.models.omnivggt_omega import OmniVGGTOmega
from omnivggt.stream.masks import frame_causal_mask
from omnivggt.stream.visibility import band_visibility, frame_visibility_mask, visible_frames_block


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


def _model(**kwargs):
    """Tiny fp64 OmniVGGTOmega in eval mode."""
    torch.manual_seed(0)
    return OmniVGGTOmega(**TINY, **kwargs).double().eval()


def _inputs(seed=0, frames=FRAMES):
    generator = torch.Generator().manual_seed(seed)
    shape = (1, frames, 28, 42)
    intrinsics = torch.tensor([[30.0, 0.0, 21.0], [0.0, 30.0, 14.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    return {
        "images": torch.rand(1, frames, 3, 28, 42, generator=generator, dtype=torch.float64),
        "extrinsics": torch.eye(4, dtype=torch.float64)[:3].repeat(1, frames, 1, 1),
        "intrinsics": intrinsics.repeat(1, frames, 1, 1),
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
    model = _model(causal=True, depth_norm="first_frame")
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
    model = _model()
    inputs = _inputs()
    layers, _ = _infer_aggregator(model, inputs, DEPTH_INDEX[condition])
    layers_changed, _ = _infer_aggregator(model, _with_other_last_frame(inputs), DEPTH_INDEX[condition])
    assert not torch.equal(layers[-1][:, PAST], layers_changed[-1][:, PAST])


def test_first_frame_normalize_depth_divides_every_view_by_the_frame_0_valid_mean():
    aggregator = _model(depth_norm="first_frame").aggregator
    inputs = _inputs()
    depth, mask = inputs["depth"], inputs["mask"]
    mean = depth[0, 0, ..., 0][mask[0, 0] > 0].mean()
    expected = (depth[..., 0] / (mean + 1e-8) * mask).unsqueeze(-1)
    assert torch.equal(aggregator.normalize_depth(depth, mask), expected)


@pytest.mark.parametrize("depth_norm, frame_0_unchanged", [("first_frame", True), ("joint", False)])
def test_first_frame_depth_norm_frame_0_ignores_later_depth(depth_norm, frame_0_unchanged):
    model = _model(causal=True, depth_norm=depth_norm)
    inputs = _inputs()
    scaled = {**inputs, "depth": inputs["depth"].clone()}
    scaled["depth"][:, 1:] *= 2
    layers, out = _infer_aggregator(model, inputs, DEPTH_INDEX["depth"])
    layers_scaled, out_scaled = _infer_aggregator(model, scaled, DEPTH_INDEX["depth"])
    assert torch.equal(layers[-1][:, 0], layers_scaled[-1][:, 0]) is frame_0_unchanged
    assert torch.equal(out["depth"][:, 0], out_scaled["depth"][:, 0]) is frame_0_unchanged


def test_first_frame_depth_norm_without_valid_frame_0_depth_raises():
    model = _model(causal=True, depth_norm="first_frame")
    inputs = _inputs()
    inputs["mask"][:, 0] = 0
    with pytest.raises(ValueError, match="frame 0"):
        _infer_aggregator(model, inputs, DEPTH_INDEX["depth"])


@pytest.mark.parametrize("depth_index", [[1, 2], [2, 0, 1]])
def test_first_frame_depth_norm_needs_frame_0_as_the_first_depth_view(depth_index):
    model = _model(depth_norm="first_frame")
    with pytest.raises(ValueError, match="frame 0"):
        _infer_aggregator(model, _inputs(), depth_index)


def test_unknown_depth_norm_raises():
    with pytest.raises(ValueError, match="depth_norm"):
        _model(depth_norm="median")


def test_causal_inference_rejects_camera_input():
    model = _model(causal=True)
    with pytest.raises(ValueError, match="camera"):
        _infer_aggregator(model, _inputs(), [], camera_index=[0])


def test_causal_training_rejects_camera_input():
    model = _model(causal=True, depth_norm="first_frame", depth_all_views=True).train()
    assert model.aggregator.cam_drop_prob < 1
    with pytest.raises(ValueError, match="cam_drop_prob"):
        model(**_inputs(), modality_rng=np.random.default_rng(0))


def test_first_frame_training_needs_depth_on_every_view():
    """A random depth subset may leave out frame 0, so the option must hold for every training forward."""
    model = _model(depth_norm="first_frame", cam_drop_prob=1.0).train()
    with pytest.raises(ValueError, match="depth_all_views"):
        model(**_inputs(), modality_rng=np.random.default_rng(0))


def test_causal_training_forward_through_gradient_checkpoints_is_frame_causal():
    model = _model(causal=True, depth_norm="first_frame", cam_drop_prob=1.0, depth_all_views=True).train()
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
    model = _model(causal=True, depth_norm="first_frame")
    assert model.aggregator._stream is None
    model.aggregator._stream = object()  # a streaming context (omnivggt.stream.streaming) runs one frame per call
    with pytest.raises(ValueError, match="one frame"):
        _infer_aggregator(model, _inputs(), [])


def test_aggregator_stream_context_rejects_the_training_forward():
    model = _model(causal=True, depth_norm="first_frame", cam_drop_prob=1.0, depth_all_views=True).train()
    model.aggregator._stream = object()
    with pytest.raises(NotImplementedError, match="inference"):
        model.aggregator(**_inputs(), modality_rng=np.random.default_rng(0))


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
    head = _camera_head(causal=True)
    assert head._stream is None
    head._stream = object()  # a streaming context (omnivggt.stream.streaming) runs one frame per call
    with pytest.raises(ValueError, match="one frame"):
        head([_camera_tokens()])


# --- OmniVGGTOmega and train_utils wiring ----------------------------------------------------------------

OUTPUTS = ("pose_enc", "depth", "depth_conf", "world_points")


def _infer(model, inputs, depth_index):
    with torch.no_grad():
        return model.inference(**inputs, depth_gt_index=list(depth_index), camera_gt_index=[])


def test_model_passes_the_options_to_the_aggregator_and_the_camera_head():
    model = _model(causal=True, depth_norm="first_frame")
    assert model.aggregator.causal is True and model.camera_head.causal is True
    assert model.aggregator.depth_norm == "first_frame"
    default = _model()
    assert default.aggregator.causal is False and default.camera_head.causal is False
    assert default.aggregator.depth_norm == "joint"
    assert list(model.state_dict()) == list(default.state_dict())


@pytest.mark.parametrize("condition", list(DEPTH_INDEX))
def test_causal_model_past_outputs_ignore_the_last_frame(condition):
    model = _model(causal=True, depth_norm="first_frame")
    inputs = _inputs()
    out = _infer(model, inputs, DEPTH_INDEX[condition])
    out_changed = _infer(model, _with_other_last_frame(inputs), DEPTH_INDEX[condition])
    for key in OUTPUTS:
        assert torch.equal(out[key][:, PAST], out_changed[key][:, PAST]), key
    for pose, pose_changed in zip(out["pose_enc_list"], out_changed["pose_enc_list"], strict=True):
        assert torch.equal(pose[:, PAST], pose_changed[:, PAST])
    assert not torch.equal(out["pose_enc"][:, -1], out_changed["pose_enc"][:, -1])


@pytest.mark.parametrize("condition", list(DEPTH_INDEX))
def test_bidirectional_model_past_poses_see_the_last_frame(condition):
    model = _model()
    inputs = _inputs()
    out = _infer(model, inputs, DEPTH_INDEX[condition])
    out_changed = _infer(model, _with_other_last_frame(inputs), DEPTH_INDEX[condition])
    assert not torch.equal(out["pose_enc"][:, PAST], out_changed["pose_enc"][:, PAST])


def test_from_variant_passes_the_options(tmp_path):
    variant_keys = ("num_register_tokens", "register_attention_layers", "global_rope", "cached_layers")
    variant = tmp_path / "tiny.json"
    variant.write_text(json.dumps({key: TINY[key] for key in variant_keys}))
    rest = {key: value for key, value in TINY.items() if key not in variant_keys}
    model = OmniVGGTOmega.from_variant(variant, **rest, causal=True, depth_norm="first_frame")
    assert model.aggregator.causal is True and model.camera_head.causal is True
    assert model.aggregator.depth_norm == "first_frame"


@pytest.fixture
def built_kwargs(monkeypatch):
    PartialState()  # train_utils logs through accelerate
    built = {}

    def fake_from_variant(path, **kwargs):
        built.update(kwargs)
        return OmniVGGTOmega(**TINY, **kwargs)

    monkeypatch.setattr(OmniVGGTOmega, "from_variant", staticmethod(fake_from_variant))
    return built


OMEGA_CFG = {"model_name": "omnivggt_omega", "omega_variant": "configs/omnivggt_omega/variants/V5.json"}


def test_build_model_passes_the_options(built_kwargs):
    model = train_utils.build_model({**OMEGA_CFG, "causal": True, "depth_norm": "first_frame", "cam_drop_prob": 1.0})
    assert built_kwargs["causal"] is True and built_kwargs["depth_norm"] == "first_frame"
    assert model.aggregator.causal is True and model.camera_head.causal is True
    assert model.aggregator.depth_norm == "first_frame"
    model = train_utils.build_model(OMEGA_CFG)
    assert built_kwargs["causal"] is False and built_kwargs["depth_norm"] == "joint"
    assert model.aggregator.causal is False and model.aggregator.depth_norm == "joint"


def test_build_model_rejects_a_causal_value_that_is_not_a_bool(built_kwargs):
    with pytest.raises(ValueError, match="causal"):
        train_utils.build_model({**OMEGA_CFG, "causal": "0"})


@pytest.mark.parametrize("option", [{"causal": True}, {"depth_norm": "first_frame"}])
def test_build_model_rejects_the_options_for_omnivggt(option):
    with pytest.raises(ValueError, match="omnivggt_omega"):
        train_utils.build_model({"model_name": "omnivggt", **option})


def test_load_model_logs_the_options(built_kwargs, monkeypatch, caplog):
    monkeypatch.setattr(train_utils, "load_initial_weights", lambda model, cfg: "none (test)")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))
    cfg = {**OMEGA_CFG, "causal": True, "depth_norm": "first_frame", "cam_drop_prob": 1.0}
    with caplog.at_level(logging.INFO):
        train_utils.load_model(cfg, torch.device("cpu"))
    assert "causal=True" in caplog.text and "depth_norm=first_frame" in caplog.text


# --- frame visibility: which key frames each query frame attends to, without an [S*P, S*P] mask ------------

TOKENS_PER_FRAME = 17 + 2 * 3  # TINY: camera + 16 registers, and 2 x 3 patches of a 28 x 42 frame
LONG_FRAMES = 146  # the longest CONF window


def _depth_index(condition, frames):
    return list(range(frames)) if condition == "depth" else []


def _infer_visible(model, inputs, condition, **options):
    """``model.inference`` with ``options`` (e.g. ``frame_visibility``); no camera input."""
    with torch.no_grad():
        return model.inference(**inputs, depth_gt_index=_depth_index(condition, inputs["images"].shape[1]),
                               camera_gt_index=[], **options)


def _assert_outputs_equal(got, want):
    for key in OUTPUTS:
        assert torch.equal(got[key], want[key]), key
    for pose, reference in zip(got["pose_enc_list"], want["pose_enc_list"], strict=True):
        assert torch.equal(pose, reference)


def _max_relative_error(got, want):
    return ((got - want).abs().max() / want.abs().max()).item()


def _replace_frame(inputs, frame, seed=1):
    """``inputs`` with the image, depth and mask of ``frame`` taken from another random window."""
    other = _inputs(seed=seed, frames=inputs["images"].shape[1])
    changed = {key: value.clone() for key, value in inputs.items()}
    for key in ("images", "depth", "mask"):
        changed[key][:, frame] = other[key][:, frame]
    return changed


def _frame_rows(tokens, frame):
    return tokens[:, frame * TOKENS_PER_FRAME : (frame + 1) * TOKENS_PER_FRAME]


def _reach(visibility, hops):
    """``reach[a, c]``: frame c can change query frame a through ``hops`` inter-frame layers of ``visibility``."""
    reach = torch.eye(len(visibility), dtype=torch.bool)
    for _ in range(hops):
        reach = (visibility.double() @ reach.double()) > 0
    return reach


def test_band_visibility_sees_the_anchor_and_the_frames_within_the_width():
    want = torch.tensor(
        [
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 0, 1, 1, 1],
            [1, 0, 0, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(band_visibility(5, 1), want)
    for width in (0, 1.5, True):
        with pytest.raises(ValueError, match="width"):
            band_visibility(5, width)


def test_frame_visibility_mask_expands_every_frame_to_its_tokens():
    visibility = band_visibility(4, 1)
    got = frame_visibility_mask(visibility, 3)
    assert got.dtype == torch.bool
    assert torch.equal(got, torch.kron(visibility, torch.ones(3, 3, dtype=torch.bool)))
    lower = torch.ones(4, 4, dtype=torch.bool).tril()
    assert torch.equal(frame_visibility_mask(lower, 3), frame_causal_mask(4, 3, "cpu"))


@pytest.mark.parametrize("condition", list(DEPTH_INDEX))
def test_all_visible_frame_visibility_is_the_bidirectional_model(condition):
    model, inputs = _model(depth_norm="first_frame"), _inputs()
    everything = torch.ones(FRAMES, FRAMES, dtype=torch.bool)
    _assert_outputs_equal(_infer_visible(model, inputs, condition, frame_visibility=everything),
                          _infer_visible(model, inputs, condition))


@pytest.mark.parametrize("condition", list(DEPTH_INDEX))
@pytest.mark.parametrize("causal", [False, True])
def test_lower_triangular_frame_visibility_is_the_causal_model(condition, causal):
    inputs = _inputs()
    lower = torch.ones(FRAMES, FRAMES, dtype=torch.bool).tril()
    got = _infer_visible(_model(causal=causal, depth_norm="first_frame"), inputs, condition, frame_visibility=lower)
    _assert_outputs_equal(got, _infer_visible(_model(causal=True, depth_norm="first_frame"), inputs, condition))


def test_band_layer_query_frame_ignores_the_frames_outside_its_band():
    """One inter-frame layer, S=5, w=1: frame 3 is neither frame 0 nor in the band of query frames 0 and 1."""
    frames, replaced = 5, 3
    block = _model().aggregator.global_blocks[0]
    visibility = band_visibility(frames, 1)
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(1, frames * TOKENS_PER_FRAME, 32, generator=generator, dtype=torch.float64)
    changed = x.clone()
    _frame_rows(changed, replaced).copy_(torch.randn(1, TOKENS_PER_FRAME, 32, generator=generator,
                                                     dtype=torch.float64))
    with torch.no_grad():
        out, out_changed = (visible_frames_block(block, tokens, visibility) for tokens in (x, changed))
    unchanged = [torch.equal(_frame_rows(out, frame), _frame_rows(out_changed, frame)) for frame in range(frames)]
    assert unchanged == (~visibility[:, replaced]).tolist() == [True, True, False, False, False]


@pytest.mark.parametrize("condition", list(DEPTH_INDEX))
def test_band_model_frames_ignore_the_frames_they_cannot_reach(condition):
    """The band applies in every inter-frame layer, so through L layers a frame reaches the frames L widths away
    and, through the anchor, the first L - 1 widths: each output must ignore every frame outside that reach.

    (A frame inside the reach but many layers away may still leave an output bit-identical: its influence shrinks
    by the attention weight and LayerScale at every layer, below fp64 resolution here. So only the band
    neighbours of the replaced frame are required to change.)"""
    frames, replaced = 20, 9
    model, inputs = _model(depth_norm="first_frame"), _inputs(frames=frames)
    visibility = band_visibility(frames, 1)
    depth_hops = model.aggregator.depth  # every global block, register ones included, is one inter-frame layer
    pose_hops = depth_hops + model.camera_head.trunk_depth * NUM_ITERATIONS
    depth_reach, pose_reach = _reach(visibility, depth_hops)[:, replaced], _reach(visibility, pose_hops)[:, replaced]
    assert (~pose_reach).nonzero().flatten().tolist() == [0, 18, 19]
    assert (~depth_reach).nonzero().flatten().tolist() == [0, 1, 2, 3, 4, 14, 15, 16, 17, 18, 19]
    out = _infer_visible(model, inputs, condition, frame_visibility=visibility)
    changed = _infer_visible(model, _replace_frame(inputs, replaced), condition, frame_visibility=visibility)
    for frame in range(frames):
        for key in ("depth", "depth_conf"):
            assert depth_reach[frame] or torch.equal(out[key][:, frame], changed[key][:, frame]), (key, frame)
        for key in ("pose_enc", "world_points"):
            assert pose_reach[frame] or torch.equal(out[key][:, frame], changed[key][:, frame]), (key, frame)
    for frame in (replaced - 1, replaced, replaced + 1):
        for key in OUTPUTS:
            assert not torch.equal(out[key][:, frame], changed[key][:, frame]), (key, frame)


@pytest.mark.parametrize("rope", [False, True])
def test_gathered_visible_frames_equal_the_dense_frame_mask(rope):
    frames, dim = 6, 32
    torch.manual_seed(0)
    block = Block(dim, 2, init_values=0.01, qk_norm=True,
                  rope=RotaryPositionEmbedding2D(100) if rope else None).double().eval()
    generator = torch.Generator().manual_seed(1)
    visibility = (torch.rand(frames, frames, generator=generator) > 0.5) | torch.eye(frames, dtype=torch.bool)
    x = torch.randn(2, frames * TOKENS_PER_FRAME, dim, generator=generator, dtype=torch.float64)
    pos = torch.randint(0, 5, (2, frames * TOKENS_PER_FRAME, 2), generator=generator)
    with torch.no_grad():
        gathered = visible_frames_block(block, x, visibility, pos=pos)
        dense = block(x, pos=pos, attn_mask=frame_visibility_mask(visibility, TOKENS_PER_FRAME))
    torch.testing.assert_close(gathered, dense, rtol=0, atol=1e-10)


def _dense_visible_frames_block(block, x, visibility, pos=None):
    """The reference of ``visible_frames_block``: the block under the dense [S*P, S*P] mask of ``visibility``."""
    return block(x, pos=pos, attn_mask=frame_visibility_mask(visibility, x.shape[1] // len(visibility)))


@pytest.mark.parametrize("condition", list(DEPTH_INDEX))
def test_band_model_equals_the_dense_frame_mask(condition, monkeypatch):
    frames = 7
    model, inputs = _model(depth_norm="first_frame"), _inputs(frames=frames)
    visibility = band_visibility(frames, 1)
    gathered = _infer_visible(model, inputs, condition, frame_visibility=visibility)
    monkeypatch.setattr(omnivggt_aggregator, "visible_frames_block", _dense_visible_frames_block)
    dense = _infer_visible(model, inputs, condition, frame_visibility=visibility)
    for key in OUTPUTS:
        assert _max_relative_error(gathered[key], dense[key]) <= 1e-10, key


def test_causal_model_rejects_a_visibility_that_is_not_lower_triangular():
    band = band_visibility(FRAMES, 1)
    with pytest.raises(ValueError, match="lower-triangular"):
        _infer_visible(_model(causal=True, depth_norm="first_frame"), _inputs(), "depth", frame_visibility=band)
    with pytest.raises(ValueError, match="lower-triangular"):
        _camera_head(causal=True)([_camera_tokens()], frame_visibility=band)


@pytest.mark.parametrize("condition", list(DEPTH_INDEX))
def test_causal_model_takes_a_lower_triangular_band(condition):
    visibility = band_visibility(FRAMES, 1).tril()
    assert not torch.equal(visibility, torch.ones(FRAMES, FRAMES, dtype=torch.bool).tril())  # the gathered path
    model, inputs = _model(causal=True, depth_norm="first_frame"), _inputs()
    out = _infer_visible(model, inputs, condition, frame_visibility=visibility)
    changed = _infer_visible(model, _with_other_last_frame(inputs), condition, frame_visibility=visibility)
    for key in OUTPUTS:
        assert torch.equal(out[key][:, PAST], changed[key][:, PAST]), key


@pytest.mark.parametrize(
    "visibility, match",
    [
        (torch.ones(FRAMES + 1, FRAMES + 1, dtype=torch.bool), "shape"),
        (torch.ones(FRAMES, FRAMES), "bool"),
        (torch.ones(FRAMES, FRAMES, dtype=torch.bool).fill_diagonal_(False), "itself"),
    ],
)
def test_frame_visibility_is_checked(visibility, match):
    with pytest.raises(ValueError, match=match):
        _infer_visible(_model(depth_norm="first_frame"), _inputs(), "depth", frame_visibility=visibility)
    with pytest.raises(ValueError, match=match):
        _camera_head()([_camera_tokens()], frame_visibility=visibility)


def test_stream_context_takes_no_frame_visibility():
    one = torch.ones(1, 1, dtype=torch.bool)
    model = _model(causal=True, depth_norm="first_frame")
    model.aggregator._stream = object()  # a streaming context runs one frame per call against its caches
    with pytest.raises(ValueError, match="stream"):
        _infer_visible(model, _inputs(frames=1), "rgb", frame_visibility=one)
    head = _camera_head(causal=True)
    head._stream = object()
    with pytest.raises(ValueError, match="stream"):
        head([_camera_tokens()[:, :1]], frame_visibility=one)


class _TensorShapes(TorchDispatchMode):
    """Records the shape of every tensor an operator returns."""

    def __init__(self):
        super().__init__()
        self.shapes = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        self.shapes.update(tuple(value.shape) for value in tree_leaves(out) if isinstance(value, torch.Tensor))
        return out


def _token_square_shapes(shapes, tokens):
    """Shapes with two dimensions of size ``tokens`` (S*P): an [S*P, S*P] mask, bias or score matrix."""
    return sorted(shape for shape in shapes if list(shape).count(tokens) >= 2)


def test_tensor_shape_watch_sees_the_dense_frame_mask():
    with _TensorShapes() as watch:
        _infer_visible(_model(causal=True, depth_norm="first_frame"), _inputs(), "depth")
    assert (FRAMES * TOKENS_PER_FRAME,) * 2 in _token_square_shapes(watch.shapes, FRAMES * TOKENS_PER_FRAME)


def test_band_over_a_long_window_builds_no_token_mask(monkeypatch):
    def no_dense_mask(*args, **kwargs):
        raise AssertionError("a dense [S*P, S*P] frame mask was requested")

    monkeypatch.setattr(omnivggt_aggregator, "frame_causal_mask", no_dense_mask)
    monkeypatch.setattr(camera_head_module, "frame_causal_mask", no_dense_mask)
    model = _model(depth_norm="first_frame")
    with _TensorShapes() as watch:
        out = _infer_visible(model, _inputs(frames=LONG_FRAMES), "depth",
                             frame_visibility=band_visibility(LONG_FRAMES, 8))
    assert _token_square_shapes(watch.shapes, LONG_FRAMES * TOKENS_PER_FRAME) == []
    special = model.aggregator.patch_start_idx
    assert (LONG_FRAMES * special,) * 2 in watch.shapes  # the register layer's [S*m, S*m] mask: the watch saw the run
    assert out["pose_enc"].shape == (1, LONG_FRAMES, 9)
    assert all(torch.isfinite(out[key]).all() for key in OUTPUTS)
