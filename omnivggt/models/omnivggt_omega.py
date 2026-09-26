"""OmniVGGT with VGGT-Omega style inter-frame attention and depth-derived point maps.

Same inputs, GeoAdapter, camera head and depth head as ``OmniVGGT``. Differences:
the aggregator is ``OmegaStyleAggregator`` (register attention, only head layers cached) and there
is no point head: ``world_points`` are unprojected from the predicted depth and camera, as in
VGGT-Omega's single-dense-head design.

``causal=True`` makes the inter-frame attention of the aggregator and the camera-head trunk frame-causal (no
camera input then); ``depth_norm="first_frame"`` normalises the auxiliary depth by frame 0 (see
``ZeroAggregator``). Together they make the outputs of frame t independent of the frames after t.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from omnivggt.heads.camera_head import CameraHead
from omnivggt.heads.dpt_head import DPTHead
from omnivggt.models.omega_aggregator import OmegaStyleAggregator
from omnivggt.utils.geometry import unproject_depth_to_world_points_torch
from omnivggt.utils.pose_enc import pose_encoding_to_extri_intri

VARIANT_KEYS = ("num_register_tokens", "register_attention_layers", "global_rope", "cached_layers")


class PointUnprojector(nn.Module):
    """World points from the final pose encoding and depth (no parameters; a module so it can be hooked)."""

    def forward(self, pose_enc, depth, image_size_hw):
        extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, image_size_hw)
        return unproject_depth_to_world_points_torch(depth, extrinsics, intrinsics)


class OmniVGGTOmega(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        cam_drop_prob=0.1,
        depth_drop_prob=0.1,
        depth_all_views=False,
        num_register_tokens=16,
        register_attention_layers=(2, 6, 9, 14, 20),
        global_rope=False,
        cached_layers=(4, 11, 17, 23),
        causal=False,
        depth_norm="joint",
        aggregator_kwargs=None,
        camera_head_kwargs=None,
        depth_head_kwargs=None,
    ):
        super().__init__()
        self.aggregator = OmegaStyleAggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            pose_hidden_dim=9,
            cam_drop_prob=cam_drop_prob,
            depth_drop_prob=depth_drop_prob,
            depth_all_views=depth_all_views,
            num_register_tokens=num_register_tokens,
            register_attention_layers=tuple(register_attention_layers),
            global_rope=global_rope,
            cached_layers=tuple(cached_layers),
            causal=causal,
            depth_norm=depth_norm,
            **(aggregator_kwargs or {}),
        )
        self.camera_head = CameraHead(dim_in=2 * embed_dim, causal=causal, **(camera_head_kwargs or {}))
        self.depth_head = DPTHead(
            dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1", **(depth_head_kwargs or {})
        )
        self.point_unprojector = PointUnprojector()
        missing = set(self.depth_head.intermediate_layer_idx) - set(cached_layers)
        if missing:
            raise ValueError(f"depth head reads layers that are not cached: {sorted(missing)}")

    @classmethod
    def from_variant(cls, path, **kwargs):
        """Build the model described by a variant JSON (``configs/omnivggt_omega/variants/*.json``)."""
        variant = json.loads(Path(path).read_text())
        return cls(**{key: variant[key] for key in VARIANT_KEYS}, **kwargs)

    def forward(self, images, extrinsics=None, intrinsics=None, depth=None, mask=None, return_points=True,
                modality_rng=None):
        if images.ndim == 4:
            images = images.unsqueeze(0)
        tokens, patch_start_idx = self.aggregator(
            images=images, extrinsics=extrinsics, intrinsics=intrinsics, depth=depth, mask=mask,
            modality_rng=modality_rng,
        )
        return self._predict(tokens, patch_start_idx, images, return_points)

    def inference(
        self,
        images,
        extrinsics=None,
        intrinsics=None,
        depth=None,
        mask=None,
        depth_gt_index=None,
        camera_gt_index=None,
        return_points=True,
    ):
        if images.ndim == 4:
            images = images.unsqueeze(0)
        tokens, patch_start_idx = self.aggregator.inference(
            images=images,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            depth=depth,
            mask=mask,
            depth_gt_index=depth_gt_index or [],
            camera_gt_index=camera_gt_index or [],
        )
        return self._predict(tokens, patch_start_idx, images, return_points)

    def _predict(self, tokens, patch_start_idx, images, return_points):
        predictions = {}
        with torch.amp.autocast("cuda", enabled=False):
            pose_enc_list = self.camera_head(tokens)
            predictions["pose_enc"] = pose_enc_list[-1]
            predictions["pose_enc_list"] = pose_enc_list
            depth, depth_conf = self.depth_head(tokens, images=images, patch_start_idx=patch_start_idx)
            predictions["depth"] = depth
            predictions["depth_conf"] = depth_conf
            if return_points:
                predictions["world_points"] = self.point_unprojector(predictions["pose_enc"], depth, images.shape[-2:])
        predictions["images"] = images
        return predictions
