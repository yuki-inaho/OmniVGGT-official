import numpy as np
import torch

from omnivggt.utils.geometry import unproject_depth_map_to_point_map, unproject_depth_to_world_points_torch
from omnivggt.utils.normalization import normalize_camera_extrinsics_and_points_batch


def _rotation(yaw: float, pitch: float) -> torch.Tensor:
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    r_yaw = torch.tensor([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    r_pitch = torch.tensor([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    return (r_yaw @ r_pitch).float()


def _scene(height: int = 8, width: int = 12):
    extrinsics = torch.stack(
        [
            torch.cat([_rotation(0.0, 0.0), torch.tensor([[0.0], [0.0], [0.0]])], dim=1),
            torch.cat([_rotation(0.2, -0.1), torch.tensor([[0.3], [-0.05], [0.1]])], dim=1),
        ]
    )[None]
    intrinsics = torch.tensor([[10.0, 0.0, 5.5], [0.0, 11.0, 3.5], [0.0, 0.0, 1.0]]).expand(1, 2, 3, 3).clone()
    depth = 1.0 + torch.rand(1, 2, height, width, 1, generator=torch.Generator().manual_seed(0))
    return depth, extrinsics, intrinsics


def test_unproject_matches_numpy_reference():
    depth, extrinsics, intrinsics = _scene()
    got = unproject_depth_to_world_points_torch(depth, extrinsics, intrinsics)
    want = unproject_depth_map_to_point_map(depth[0].numpy(), extrinsics[0].numpy(), intrinsics[0].numpy())
    assert got.shape == (1, 2, 8, 12, 3)
    np.testing.assert_allclose(got[0].numpy(), want, atol=1e-5)


def test_unproject_gradients_reach_depth_and_pose():
    depth, extrinsics, intrinsics = _scene()
    depth.requires_grad_(True)
    extrinsics.requires_grad_(True)
    intrinsics.requires_grad_(True)
    unproject_depth_to_world_points_torch(depth, extrinsics, intrinsics).square().sum().backward()
    for tensor in (depth, extrinsics, intrinsics):
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0


def test_gt_depth_and_extrinsics_reproduce_normalized_world_points():
    depth, extrinsics, intrinsics = _scene()
    # world frame deliberately different from the first camera, as in real data
    world_from_first = torch.cat([_rotation(0.5, 0.3), torch.tensor([[1.0], [2.0], [-0.5]])], dim=1)
    homogeneous = torch.cat([world_from_first, torch.tensor([[0.0, 0.0, 0.0, 1.0]])])
    extrinsics = extrinsics @ torch.linalg.inv(homogeneous)
    world_points = unproject_depth_to_world_points_torch(depth, extrinsics, intrinsics)
    masks = torch.ones(depth.shape[:-1], dtype=torch.bool)
    new_extrinsics, _, new_world_points, new_depths = normalize_camera_extrinsics_and_points_batch(
        extrinsics=extrinsics, world_points=world_points, depths=depth, point_masks=masks
    )
    rebuilt = unproject_depth_to_world_points_torch(new_depths[..., None], new_extrinsics, intrinsics)
    torch.testing.assert_close(rebuilt, new_world_points, rtol=1e-4, atol=1e-5)
