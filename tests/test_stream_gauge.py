"""First-frame output gauge (omnivggt.stream.gauge): R_1 = I, c_1 = 0, median valid depth of frame 1 = 1."""

import math

import pytest
import torch

from omnivggt.stream.gauge import FirstFrameGauge, apply_first_frame_gauge

FRAMES, HEIGHT, WIDTH = 4, 5, 6


def _rotation(angles):
    (a, b, c) = angles
    rx = [[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]]
    ry = [[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]]
    rz = [[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]]
    rx, ry, rz = (torch.tensor(r, dtype=torch.float64) for r in (rx, ry, rz))
    return rz @ ry @ rx


def _outputs(seed=0):
    """Camera-from-world extrinsics [S, 3, 4], depth [S, H, W, 1] and valid [S, H, W] in an arbitrary gauge."""
    generator = torch.Generator().manual_seed(seed)
    angles = torch.rand(FRAMES, 3, generator=generator, dtype=torch.float64) - 0.5
    rotations = torch.stack([_rotation(a.tolist()) for a in angles])
    translations = torch.randn(FRAMES, 3, 1, generator=generator, dtype=torch.float64)
    depth = 0.5 + 3 * torch.rand(FRAMES, HEIGHT, WIDTH, 1, generator=generator, dtype=torch.float64)
    valid = torch.rand(FRAMES, HEIGHT, WIDTH, generator=generator) > 0.3
    return torch.cat([rotations, translations], dim=-1), depth, valid


def _homogeneous(extrinsics):
    bottom = torch.tensor([0, 0, 0, 1], dtype=extrinsics.dtype).expand(*extrinsics.shape[:-2], 1, 4)
    return torch.cat([extrinsics, bottom], dim=-2)


def _world_points(extrinsics, camera_points):
    """World points of camera-frame points [S, P, 3]: R^T (X - t)."""
    rotation, translation = extrinsics[..., :3], extrinsics[..., 3]
    return torch.einsum("sji,spj->spi", rotation, camera_points - translation[:, None])


def test_gauge_fixes_frame_1():
    extrinsics, depth, valid = _outputs()
    gauged_extrinsics, gauged_depth = apply_first_frame_gauge(extrinsics, depth, valid)
    identity = torch.cat([torch.eye(3, dtype=torch.float64), torch.zeros(3, 1, dtype=torch.float64)], dim=1)
    torch.testing.assert_close(gauged_extrinsics[0], identity, rtol=0, atol=1e-12)
    assert gauged_depth[0, ..., 0][valid[0]].median() == 1.0
    assert gauged_depth.shape == depth.shape and gauged_extrinsics.shape == extrinsics.shape


def test_gauge_keeps_relative_poses_and_depth_ratios():
    extrinsics, depth, valid = _outputs()
    scale = depth[0, ..., 0][valid[0]].median()
    gauged_extrinsics, gauged_depth = apply_first_frame_gauge(extrinsics, depth, valid)
    torch.testing.assert_close(gauged_depth, depth / scale, rtol=1e-15, atol=0)
    relative = _homogeneous(extrinsics)[:, None] @ torch.linalg.inv(_homogeneous(extrinsics))[None]
    gauged_relative = _homogeneous(gauged_extrinsics)[:, None] @ torch.linalg.inv(_homogeneous(gauged_extrinsics))[None]
    torch.testing.assert_close(gauged_relative[..., :3, :3], relative[..., :3, :3], rtol=0, atol=1e-12)
    torch.testing.assert_close(gauged_relative[..., :3, 3], relative[..., :3, 3] / scale, rtol=0, atol=1e-12)


def test_gauged_world_points_are_frame_1_coordinates_over_its_scale():
    """P'_t = R_1 (P_t - c_1) / s_1 (design doc §7.2), for the points of the depth of every frame."""
    extrinsics, depth, valid = _outputs()
    scale = depth[0, ..., 0][valid[0]].median()
    rays = torch.randn(FRAMES, HEIGHT * WIDTH, 3, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    gauged_extrinsics, gauged_depth = apply_first_frame_gauge(extrinsics, depth, valid)
    points = _world_points(extrinsics, rays * depth.reshape(FRAMES, -1, 1))
    gauged_points = _world_points(gauged_extrinsics, rays * gauged_depth.reshape(FRAMES, -1, 1))
    rotation_1, center_1 = extrinsics[0, :, :3], -extrinsics[0, :, :3].T @ extrinsics[0, :, 3]
    expected = torch.einsum("ij,spj->spi", rotation_1, points - center_1) / scale
    torch.testing.assert_close(gauged_points, expected, rtol=0, atol=1e-12)


def test_streamed_gauge_equals_the_batch_gauge():
    extrinsics, depth, valid = _outputs()
    batch_extrinsics, batch_depth = apply_first_frame_gauge(extrinsics, depth, valid)
    gauge = FirstFrameGauge()
    assert not gauge.initialized
    for frame in range(FRAMES):
        rows = slice(frame, frame + 1)
        valid_rows = valid[rows] if frame == 0 else None  # the valid mask sets the gauge, at frame 1 only
        streamed_extrinsics, streamed_depth = gauge(extrinsics[rows], depth[rows], valid_rows)
        assert torch.equal(streamed_extrinsics, batch_extrinsics[rows])
        assert torch.equal(streamed_depth, batch_depth[rows])
    assert gauge.initialized
    gauge.reset()
    assert not gauge.initialized


def test_gauge_takes_depth_with_or_without_the_channel_axis():
    extrinsics, depth, valid = _outputs()
    with_axis = apply_first_frame_gauge(extrinsics, depth, valid)
    without_axis = apply_first_frame_gauge(extrinsics, depth[..., 0], valid)
    assert torch.equal(with_axis[0], without_axis[0]) and torch.equal(with_axis[1][..., 0], without_axis[1])


@pytest.mark.parametrize("spoil", ["no_valid_pixel", "no_positive_depth", "no_finite_depth"])
def test_gauge_without_a_valid_positive_finite_depth_in_frame_1_raises(spoil):
    extrinsics, depth, valid = _outputs()
    if spoil == "no_valid_pixel":
        valid[0] = False
    elif spoil == "no_positive_depth":
        depth[0] = -1.0
    else:
        depth[0] = float("nan")
    with pytest.raises(ValueError, match="frame 1"):
        apply_first_frame_gauge(extrinsics, depth, valid)


def test_the_first_call_needs_the_valid_mask():
    extrinsics, depth, _ = _outputs()
    with pytest.raises(ValueError, match="valid"):
        FirstFrameGauge()(extrinsics, depth)


@pytest.mark.parametrize("shapes", [((FRAMES, 4, 4), (FRAMES, HEIGHT, WIDTH)), ((FRAMES, 3, 4), (HEIGHT, WIDTH))])
def test_gauge_rejects_unexpected_shapes(shapes):
    extrinsic_shape, depth_shape = shapes
    with pytest.raises(ValueError, match="expected"):
        apply_first_frame_gauge(torch.zeros(extrinsic_shape, dtype=torch.float64),
                                torch.ones(depth_shape, dtype=torch.float64),
                                torch.ones(FRAMES, HEIGHT, WIDTH, dtype=torch.bool))
