"""Point loss on points unprojected from the predicted depth and camera (no point head)."""

import numpy as np
import pytest
import torch

from omnivggt.loss import MultitaskLoss, compute_derived_point_loss, relative_depth_weights
from omnivggt.utils.geometry import unproject_depth_to_world_points_torch
from omnivggt.utils.normalization import normalize_camera_extrinsics_and_points_batch
from omnivggt.utils.pose_enc import extri_intri_to_pose_encoding

HEIGHT, WIDTH = 16, 24


def _rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float32)


def _batch(cx=WIDTH / 2, cy=HEIGHT / 2):
    extrinsics = torch.stack(
        [
            torch.cat([_rotation(0.0), torch.zeros(3, 1)], 1),
            torch.cat([_rotation(0.1), torch.tensor([[0.2], [0.0], [0.05]])], 1),
        ]
    )[None]
    intrinsics = torch.tensor([[20.0, 0.0, cx], [0.0, 20.0, cy], [0.0, 0.0, 1.0]]).expand(1, 2, 3, 3).clone()
    depth = 1.0 + torch.rand(1, 2, HEIGHT, WIDTH, 1, generator=torch.Generator().manual_seed(0))
    world = unproject_depth_to_world_points_torch(depth, extrinsics, intrinsics)
    mask = torch.ones(1, 2, HEIGHT, WIDTH, dtype=torch.bool)
    new_extrinsics, _, new_world, new_depth = normalize_camera_extrinsics_and_points_batch(
        extrinsics=extrinsics, world_points=world, depths=depth, point_masks=mask
    )
    return {
        "extrinsic": new_extrinsics,
        "intrinsic": intrinsics,
        "depth": new_depth,
        "world_points": new_world,
        "valid_mask": mask,
    }


def _predictions(batch, focal_scale=1.0):
    intrinsics = batch["intrinsic"].clone()
    intrinsics[..., 0, 0] *= focal_scale
    intrinsics[..., 1, 1] *= focal_scale
    pose_enc = extri_intri_to_pose_encoding(batch["extrinsic"], intrinsics, image_size_hw=(HEIGHT, WIDTH))
    return {"depth": batch["depth"][..., None].clone(), "pose_enc_list": [pose_enc]}


def test_zero_loss_for_perfect_prediction():
    batch = _batch()
    loss = compute_derived_point_loss(_predictions(batch), batch, progress=0.0, min_valid_pts=10)
    assert loss["loss_point"].item() < 1e-4


def test_intrinsics_warmup_switch():
    batch = _batch(
        cx=WIDTH / 2 + 2.0, cy=HEIGHT / 2 - 1.0
    )  # principal point off-centre (a pose encoding cannot carry it)
    wrong_focal = _predictions(batch, focal_scale=1.2)
    early = compute_derived_point_loss(wrong_focal, batch, progress=0.2, min_valid_pts=10)["loss_point"]
    late = compute_derived_point_loss(wrong_focal, batch, progress=0.8, min_valid_pts=10)["loss_point"]
    assert early.item() < 1e-4  # ground-truth K during warm-up
    assert late.item() > 1e-2  # predicted focal length afterwards
    right_focal = compute_derived_point_loss(_predictions(batch), batch, progress=0.8, min_valid_pts=10)["loss_point"]
    assert right_focal.item() < 1e-4  # the ground-truth principal point is kept


def test_relative_weights_clamped():
    depth = torch.tensor([[[[1e-4, 1.0, 1e3, 5.0]]]])
    mask = torch.ones_like(depth, dtype=torch.bool)
    weights = relative_depth_weights(depth, mask)
    assert weights.min() >= 0.1 and weights.max() <= 10.0
    assert weights[..., 2] == pytest.approx(0.1)


def test_gradients_reach_depth_and_camera():
    batch = _batch()
    predictions = _predictions(batch, focal_scale=1.1)
    predictions["depth"] = (predictions["depth"] * 1.05).requires_grad_(True)
    predictions["pose_enc_list"][0].requires_grad_(True)
    compute_derived_point_loss(predictions, batch, progress=0.9, min_valid_pts=10)["loss_point"].backward()
    assert predictions["depth"].grad.abs().sum() > 0
    assert predictions["pose_enc_list"][0].grad.abs().sum() > 0


def test_multitask_loss_derived_mode_without_point_conf():
    batch = _batch()
    predictions = _predictions(batch, focal_scale=1.1)
    criterion = MultitaskLoss(
        point={"mode": "derived", "weight": 1.0, "intrinsics_warmup_ratio": 0.5, "min_valid_pts": 10}
    )
    losses = criterion(predictions, batch, progress=0.9)
    assert "loss_point" in losses and losses["objective"].item() > 0
    with pytest.raises(ValueError):
        criterion(predictions, batch)  # derived mode needs the training progress
    with pytest.raises(KeyError):
        criterion({"pose_enc_list": predictions["pose_enc_list"]}, batch, progress=0.9)
    head = MultitaskLoss(point={"weight": 1.0, "gradient_loss_fn": "normal", "valid_range": 0.98})
    with pytest.raises(KeyError):
        head(
            {**predictions, "world_points": batch["world_points"]}, batch, progress=0.9
        )  # head mode needs world_points_conf


def test_relative_weights_floor_and_upper_clip():
    mask = torch.ones(1, 4, dtype=torch.bool)
    floor = relative_depth_weights(torch.tensor([[0.05, 1.0, 2.0, 20.0]]), mask)
    torch.testing.assert_close(floor, torch.tensor([[1.0 / 0.57625, 1.0, 0.5, 0.1]]), rtol=1e-4, atol=1e-5)
    upper = relative_depth_weights(torch.tensor([[0.01, 0.2, 0.105]]), torch.ones(1, 3, dtype=torch.bool))
    torch.testing.assert_close(upper, torch.tensor([[10.0, 5.0, 1.0 / 0.105]]), rtol=1e-4, atol=1e-5)


def test_uses_the_final_camera_stage():
    batch = _batch()
    good = _predictions(batch)
    wrong = _predictions(batch, focal_scale=1.3)["pose_enc_list"][0].clone()
    wrong[..., :3] += 0.5
    predictions = {"depth": good["depth"], "pose_enc_list": [wrong, good["pose_enc_list"][0]]}
    assert compute_derived_point_loss(predictions, batch, progress=0.9, min_valid_pts=10)["loss_point"].item() < 1e-4


def test_weights_come_from_ground_truth_depth_and_pixels_are_capped():
    batch = _batch()
    predictions = _predictions(batch)
    predictions["depth"] = predictions["depth"] * 2.0  # every point is off; weights must still use the GT depth
    loss = compute_derived_point_loss(predictions, batch, progress=0.0, min_valid_pts=10)["loss_point"]
    gt_extrinsics = batch["extrinsic"]
    points = unproject_depth_to_world_points_torch(predictions["depth"], gt_extrinsics, batch["intrinsic"])
    error = (points - batch["world_points"]).abs()
    weights = relative_depth_weights(batch["depth"], batch["valid_mask"])
    torch.testing.assert_close(loss, (error * weights[..., None]).mean(), rtol=1e-3, atol=1e-6)
    capped = compute_derived_point_loss(predictions, batch, progress=0.0, min_valid_pts=10, max_pixel_loss=1e-3)[
        "loss_point"
    ]
    expected_capped = (error * weights[..., None]).clamp(max=1e-3).mean()
    torch.testing.assert_close(capped, expected_capped, rtol=1e-3, atol=1e-8)
    assert capped < loss  # the cap is active
