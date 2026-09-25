"""Small SE(3) helpers shared by the pose pipeline (camera_to_world 4x4, metres)."""

from __future__ import annotations

import numpy as np


def as_homogeneous(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape[-2:] == (4, 4):
        return poses
    if poses.shape[-2:] != (3, 4):
        raise ValueError(f"expected (...,3,4) or (...,4,4) poses, got {poses.shape}")
    out = np.broadcast_to(np.eye(4), (*poses.shape[:-2], 4, 4)).copy()
    out[..., :3, :] = poses
    return out


def invert(poses: np.ndarray) -> np.ndarray:
    poses = as_homogeneous(poses)
    rotation_t = np.swapaxes(poses[..., :3, :3], -1, -2)
    out = np.broadcast_to(np.eye(4), poses.shape).copy()
    out[..., :3, :3] = rotation_t
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", rotation_t, poses[..., :3, 3])
    return out


def rotation_angle_deg(rotation: np.ndarray) -> np.ndarray:
    """Rotation angle via atan2(|axis part|, cos part): accurate for tiny and large angles."""
    rotation = np.asarray(rotation, dtype=np.float64)
    cos = (np.trace(rotation, axis1=-2, axis2=-1) - 1.0) / 2.0
    skew = rotation - np.swapaxes(rotation, -1, -2)
    sin = 0.5 * np.linalg.norm(np.stack([skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], axis=-1), axis=-1)
    return np.degrees(np.arctan2(sin, cos))


def relative(pose_i: np.ndarray, pose_j: np.ndarray) -> np.ndarray:
    """camera_j <- camera_i transform for camera_to_world poses (T_j^-1 T_i)."""
    return invert(pose_j) @ as_homogeneous(pose_i)


def chordal_mean_rotation(rotations: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(rotations, dtype=np.float64).sum(axis=0))
    mean = u @ vt
    if np.linalg.det(mean) < 0:
        u[:, -1] *= -1
        mean = u @ vt
    return mean


def check_poses(poses: np.ndarray, name: str = "poses", atol: float = 1e-6) -> np.ndarray:
    poses = as_homogeneous(poses)
    if not np.isfinite(poses).all():
        raise ValueError(f"{name} contain non-finite values")
    if not np.allclose(poses[..., 3, :], [0.0, 0.0, 0.0, 1.0], atol=atol):
        raise ValueError(f"{name} last row must be (0,0,0,1)")
    rotation = poses[..., :3, :3]
    identity = np.einsum("...ji,...jk->...ik", rotation, rotation)
    if not np.allclose(identity, np.eye(3), atol=atol) or not np.allclose(np.linalg.det(rotation), 1.0, atol=atol):
        raise ValueError(f"{name} rotations must be orthonormal with det +1")
    return poses
