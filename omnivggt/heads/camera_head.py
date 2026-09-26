# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from omnivggt.layers import Mlp
from omnivggt.layers.block import Block
from omnivggt.heads.head_act import activate_pose
from omnivggt.stream.masks import frame_causal_mask
from omnivggt.stream.visibility import check_frame_visibility

NUM_ITERATIONS = 4  # refinement iterations of CameraHead.forward (the models use the default)


class CameraHead(nn.Module):
    """
    CameraHead predicts camera parameters from token representations using iterative refinement.

    It applies a series of transformer blocks (the "trunk") to dedicated camera tokens, one per frame.
    With ``causal=True`` the trunk attention is frame-causal (lower-triangular): frame a sees frames b <= a.
    ``_stream`` is set only while ``omnivggt.stream.streaming.StreamingOmega`` runs one frame: trunk block j of
    refinement iteration i then attends to its own KV cache (i, j).
    """

    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
        causal: bool = False,
    ):
        super().__init__()

        if pose_encoding_type == "absT_quaR_FoV":
            self.target_dim = 9
        else:
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")

        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.trunk_depth = trunk_depth
        self.causal = causal
        self._stream = None  # streaming context (omnivggt.stream); None runs whole batches

        # Build the trunk using a sequence of transformer blocks.
        self.trunk = nn.Sequential(
            *[
                Block(
                    dim=dim_in,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                )
                for _ in range(trunk_depth)
            ]
        )

        # Normalizations for camera token and trunk output.
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)

        # Learnable empty camera pose token.
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)

        # Module for producing modulation parameters: shift, scale, and a gate.
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))

        # Adaptive layer normalization without affine parameters.
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        self.pose_branch = Mlp(
            in_features=dim_in,
            hidden_features=dim_in // 2,
            out_features=self.target_dim,
            drop=0,
        )

    def forward(self, aggregated_tokens_list: list, num_iterations: int = NUM_ITERATIONS,
                frame_visibility: Optional[torch.Tensor] = None) -> list:
        """
        Forward pass to predict camera parameters.

        Args:
            aggregated_tokens_list (list): List of token tensors from the network;
                the last tensor is used for prediction.
            num_iterations (int, optional): Number of iterative refinement steps. Defaults to 4.
            frame_visibility (torch.Tensor, optional): [S, S] bool; the trunk attention of frame a sees only the
                frames b with ``frame_visibility[a, b]`` (see ``omnivggt.stream.visibility``).

        Returns:
            list: A list of predicted camera encodings (post-activation) from each iteration.
        """
        # Use tokens from the last block for camera prediction.
        tokens = aggregated_tokens_list[-1]

        # Extract the camera tokens
        pose_tokens = tokens[:, :, 0]
        pose_tokens = self.token_norm(pose_tokens)

        pred_pose_enc_list = self.trunk_fn(pose_tokens, num_iterations, frame_visibility)
        return pred_pose_enc_list

    def trunk_fn(self, pose_tokens: torch.Tensor, num_iterations: int,
                 frame_visibility: Optional[torch.Tensor] = None) -> list:
        """
        Iteratively refine camera pose predictions.

        Args:
            pose_tokens (torch.Tensor): Normalized camera tokens with shape [B, 1, C].
            num_iterations (int): Number of refinement iterations.
            frame_visibility (torch.Tensor, optional): see ``forward``.

        Returns:
            list: List of activated camera encodings from each iteration.
        """
        B, S, C = pose_tokens.shape  # S is expected to be 1.
        pred_pose_enc = None
        pred_pose_enc_list = []
        if self._stream is not None and S != 1:
            raise ValueError(f"a streaming context runs one frame per call, got {S} frames")
        attn_mask = self._trunk_mask(S, pose_tokens.device, frame_visibility)
        if self._stream is not None and num_iterations != self._stream.camera_iterations:
            raise ValueError(f"the stream has camera caches for {self._stream.camera_iterations} iterations, "
                             f"got num_iterations={num_iterations}")

        for iteration in range(num_iterations):
            # Use a learned empty pose for the first iteration.
            if pred_pose_enc is None:
                module_input = self.embed_pose(self.empty_pose_tokens.expand(B, S, -1))
            else:
                # Detach the previous prediction to avoid backprop through time.
                pred_pose_enc = pred_pose_enc.detach()
                module_input = self.embed_pose(pred_pose_enc)

            # Generate modulation parameters and split them into shift, scale, and gate components.
            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)

            # Adaptive layer normalization and modulation.
            pose_tokens_modulated = gate_msa * modulate(self.adaln_norm(pose_tokens), shift_msa, scale_msa)
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens

            # an explicit loop over the Sequential, so that every block takes the mask or its stream cache
            for index, block in enumerate(self.trunk):
                if self._stream is None:
                    pose_tokens_modulated = block(pose_tokens_modulated, attn_mask=attn_mask)
                else:
                    pose_tokens_modulated = self._stream.run_camera(iteration, index, block, pose_tokens_modulated)
            # Compute the delta update for the pose encoding.
            pred_pose_enc_delta = self.pose_branch(self.trunk_norm(pose_tokens_modulated))

            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta

            # Apply final activation functions for translation, quaternion, and field-of-view.
            activated_pose = activate_pose(
                pred_pose_enc,
                trans_act=self.trans_act,
                quat_act=self.quat_act,
                fl_act=self.fl_act,
            )
            pred_pose_enc_list.append(activated_pose)

        return pred_pose_enc_list

    def _trunk_mask(self, S: int, device, frame_visibility: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Attention mask of the trunk. One token per frame: the frame-causal mask is lower-triangular (a stream
        step needs none), and a frame visibility is its own mask (all-visible runs unmasked)."""
        if frame_visibility is None:
            return frame_causal_mask(S, 1, device) if self.causal and self._stream is None else None
        if self._stream is not None:
            raise ValueError("a streaming context attends to its KV caches and takes no frame_visibility")
        check_frame_visibility(frame_visibility, S, self.causal)
        return None if frame_visibility.all() else frame_visibility.to(device)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor using scaling and shifting parameters.
    """
    # modified from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
    return x * (1 + scale) + shift
