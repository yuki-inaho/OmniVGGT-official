import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional, Iterable

from omnivggt.layers import PatchEmbed
from omnivggt.layers.block import Block
from omnivggt.utils.pose_enc import extri_intri_to_pose_encoding
from torch.utils.checkpoint import checkpoint
from omnivggt.utils.geometry import closed_form_inverse_se3
from omnivggt.models.aggregator import Aggregator, slice_expand_and_flatten
from omnivggt.stream.masks import frame_causal_mask
from omnivggt.stream.visibility import check_frame_visibility, visible_frames_block

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]
DEPTH_NORMS = ("joint", "first_frame")

class ZeroAggregator(Aggregator):
    """OmniVGGT aggregator: alternating attention with the GeoAdapter camera/depth injection.

    Frame-causal options (independent of each other):

    * ``causal``: in the inter-frame (global) blocks a token of frame a attends only to frames b <= a
      (all tokens of its own frame included). Camera input is rejected, since its normalisation uses every
      selected camera; with ``depth_norm="joint"`` the auxiliary depth still is normalised over all views.
    * ``depth_norm``: the auxiliary depth is divided by the mean valid depth of all selected views
      (``"joint"``) or of frame 0 only (``"first_frame"``), which must then be the first depth view.

    ``_stream`` is set only while ``omnivggt.stream.streaming.StreamingOmega`` runs one frame through
    ``inference``: the reference special tokens are used at stream time t=1 only, the depth is divided by the
    stream's first-frame scale, and every inter-frame block attends to its KV cache (``_StreamContext``).

    ``inference`` also takes a frame visibility and frame-only global layers (see its docstring).
    """

    register_attention_layers = frozenset()  # global layers over the special tokens only (OmegaStyleAggregator)

    def __init__(self, img_size=518,
                 patch_size=14, 
                 embed_dim=1024, 
                 depth=24, 
                 num_heads=16, 
                 mlp_ratio=4, 
                 num_register_tokens=4, 
                 block_fn=Block, 
                 pose_hidden_dim=9,
                 cam_drop_prob=0.1,
                 depth_drop_prob=0.1,
                 qkv_bias=True, 
                 proj_bias=True, 
                 ffn_bias=True, 
                 patch_embed="dinov2_vitl14_reg", 
                 aa_order=["frame", "global"], 
                 aa_block_size=1, 
                 qk_norm=True, 
                 rope_freq=100, 
                 init_values=0.01,
                 enable_checkpoint=True,
                 depth_all_views=False,
                 causal=False,
                 depth_norm="joint"):
        if depth_norm not in DEPTH_NORMS:
            raise ValueError(f"depth_norm must be one of {DEPTH_NORMS}, got {depth_norm!r}")
        super().__init__(img_size, 
                         patch_size, 
                         embed_dim, 
                         depth, 
                         num_heads, 
                         mlp_ratio, 
                         num_register_tokens, 
                         block_fn,
                         qkv_bias, 
                         proj_bias, 
                         ffn_bias, 
                         patch_embed, 
                         aa_order, 
                         aa_block_size, 
                         qk_norm, 
                         rope_freq, 
                         init_values)
        
        
        self.cam_drop_prob = cam_drop_prob
        self.depth_drop_prob = depth_drop_prob
        self.depth_all_views = depth_all_views  # training: auxiliary depth on every view (RGB-D-only use)
        self.causal = causal
        self.depth_norm = depth_norm
        self._stream = None  # streaming context (omnivggt.stream); None runs whole batches
        self.patch_start_idx = 1 + num_register_tokens
        self.depth_placeholder = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.use_checkpoint = enable_checkpoint
        self.num_groups = self.aa_block_num + 1
        self.pose_embeddings   = nn.ModuleList()
        self.camera_adapters   = nn.ModuleList()

        for _ in range(self.num_groups):
            # pose_embedding
            pose_emb = nn.Linear(pose_hidden_dim, embed_dim)
            
            # camera adapter (zero init)
            cam_adapt = nn.Linear(embed_dim, embed_dim, bias=True)
            nn.init.zeros_(cam_adapt.weight)
            nn.init.zeros_(cam_adapt.bias)

            self.pose_embeddings.append(pose_emb)
            self.camera_adapters.append(cam_adapt)
            
        self.depth_patch_embed = PatchEmbed(img_size=img_size,
                                            patch_size=patch_size,
                                            in_chans=2,
                                            embed_dim=embed_dim)
        
    def _collect_layer(self, layer_idx, frame_out, global_out):
        """Output of one alternating-attention layer for the heads: concat frame and global tokens, [B x S x P x 2C]."""
        return torch.cat([frame_out, global_out], dim=-1)

    def _match_dtype(self, x, reference):
        return x.to(dtype=reference.dtype, device=reference.device)
    
    def normalize_extrinsics(self, extrinsics):
        B, S, _, _ = extrinsics.shape
        device = extrinsics.device
        extrinsics_homog = torch.cat(
            [
                extrinsics,
                torch.zeros((B, S, 1, 4), device=device),
            ],
            dim=-2,
        )
        extrinsics_homog[:, :, -1, -1] = 1.0
        first_cam_extrinsic_inv = closed_form_inverse_se3(extrinsics_homog[:, 0])
        new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv.unsqueeze(1))  # (B,N,4,4)
        
        if S > 1:
            cam_centers = new_extrinsics[:, :, :3, 3]  # (B, S, 3)
            ref_cam = cam_centers[:, 0:1, :]  # (B,1,3)
            rel_distances = torch.norm(cam_centers - ref_cam, dim=-1)[:,1:]  # (B, S)
            scale = rel_distances.mean(dim=1, keepdim=True).clamp(min=1e-6)  # (B, 1)
            new_extrinsics[:, :, :3, 3] /= scale.unsqueeze(-1)
        return new_extrinsics[:, :, :3]
    
    def normalize_depth(self, depth, mask, eps=1e-8):
        """
        depth: [B, V, H, W, 1]
        mask:  [B, V, H, W]

        Divides the views of each sample by the mean valid depth of all of them (``depth_norm="joint"``) or of
        the first one (``"first_frame"``; ``_check_inputs`` makes it frame 0), and zeroes the invalid pixels.
        A streaming context divides by its depth scale instead, set at stream time t=1.
        """
        assert depth.shape[:4] == mask.shape, "mask and depth must have the same first four dimensions"

        B, V, H, W, _ = depth.shape
        depth_squeezed = depth.squeeze(-1)
        norm = torch.zeros_like(depth_squeezed)
        reference_views = V if self.depth_norm == "joint" else 1
        if self._stream is not None and self._stream.depth_scale is None:
            raise ValueError("the stream has no depth scale: its frame 1 had no depth input")

        for b in range(B):
            if self._stream is not None:
                mean = self._stream.depth_scale[b]  # gamma, fixed at stream time t=1
            else:
                valid = depth_squeezed[b, :reference_views][mask[b, :reference_views] > 0]
                if valid.numel() == 0:
                    if self.depth_norm == "first_frame":
                        raise ValueError("depth_norm='first_frame': frame 0 has no valid auxiliary depth pixel")
                    continue
                mean = valid.mean()
            norm_b = depth_squeezed[b] / (mean + eps)

            norm[b] = norm_b * mask[b]

        return norm.unsqueeze(-1)
    
    def select_camera_gt(self, S, cam_drop_prob=0.1, rng=None):
        rng = rng or np.random.default_rng()

        if rng.random() < cam_drop_prob:
            return []

        k = rng.integers(0, S + 1)
        if k == 0:
            return []

        # 按顺序从 0 开始选取 k 个
        idx = list(range(k))

        return idx
    
    def select_depth_gt(self, S, depth_drop_prob=0.1, rng=None):
        rng = rng or np.random.default_rng()

        if rng.random() < depth_drop_prob:
            return []

        k = rng.integers(0, S + 1)
        if k == 0:
            return []

        idx = rng.choice(S, size=k, replace=False)

        return sorted(idx.tolist())

    def training_depth_gt_index(self, S, rng=None):
        """Views that receive the auxiliary depth in a training forward."""
        if self.depth_all_views:
            return list(range(S))
        return self.select_depth_gt(S, self.depth_drop_prob, rng=rng)
    
    def _check_training_options(self):
        """Training draws the auxiliary inputs at random: reject options whose draws the mode cannot use."""
        if self.causal and self.cam_drop_prob < 1:
            raise ValueError(f"causal=True takes no camera input: training needs cam_drop_prob=1, "
                             f"got {self.cam_drop_prob}")
        if self.depth_norm == "first_frame" and not self.depth_all_views:
            raise ValueError("depth_norm='first_frame' needs the depth of frame 0 in every training forward: "
                             "set depth_all_views=True (a random depth subset may leave frame 0 out)")

    def _check_inputs(self, S, depth_gt_index, camera_gt_index):
        """Reject inputs the configured mode cannot use (there is no fallback)."""
        if self._stream is not None and S != 1:
            raise ValueError(f"a streaming context runs one frame per call, got {S} frames")
        if self.causal and len(camera_gt_index) != 0:
            raise ValueError("causal=True takes no camera input (its normalisation uses every selected camera), "
                             f"got camera_gt_index={list(camera_gt_index)}")
        if self.depth_norm == "first_frame" and len(depth_gt_index) != 0 and depth_gt_index[0] != 0:
            raise ValueError("depth_norm='first_frame' needs the depth of frame 0 as the first depth view, "
                             f"got depth_gt_index={list(depth_gt_index)}")

    def forward(self, images: torch.Tensor, 
                extrinsics: torch.Tensor, 
                intrinsics: torch.Tensor,
                depth: torch.Tensor,
                mask: torch.Tensor,
                modality_rng=None) -> Tuple[List[torch.Tensor], int]:
        B, S, C_in, H, W = images.shape
        
        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")
        if self._stream is not None:
            raise NotImplementedError("a streaming context runs through inference() only, one frame per call")
        self._check_training_options()

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)
            
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        K, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # which views get the auxiliary camera / depth (modality_rng: keyed generator from the training loop)
        camera_gt_index = self.select_camera_gt(S, self.cam_drop_prob, rng=modality_rng)
        depth_gt_index  = self.training_depth_gt_index(S, rng=modality_rng)
        self._check_inputs(S, depth_gt_index, camera_gt_index)
        
        if len(camera_gt_index) != 0:
            camera_gt_length = len(camera_gt_index)
            camera_idx_tensor = torch.tensor(camera_gt_index, device=depth.device)

            extrinsics_selected = torch.index_select(extrinsics, dim=1, index=camera_idx_tensor)
            intrinsics_selected = torch.index_select(intrinsics, dim=1, index=camera_idx_tensor)

            extrinsics_gt_normalized = self.normalize_extrinsics(extrinsics_selected)
            pose_encoding = extri_intri_to_pose_encoding(
                        extrinsics=extrinsics_gt_normalized,
                        intrinsics=intrinsics_selected,
                        image_size_hw=(H, W),
                        pose_encoding_type="absT_quaR_FoV",
            )
            gt_camera_token = self.pose_embeddings[0](pose_encoding).view(B * camera_gt_length, C).unsqueeze(1)
            
            device = depth.device
            camera_full = torch.zeros(K, 1, C, device=device, dtype=camera_token.dtype)

            camera_rows = (torch.arange(B, device=device).unsqueeze(1) * S + camera_idx_tensor.unsqueeze(0)).reshape(-1)
            camera_full[camera_rows] = gt_camera_token.to(dtype=camera_token.dtype)
            gt_camera_token = camera_full
        else:
            pose_encoding = None
            gt_camera_token = torch.zeros(K, 1, C, device=depth.device, dtype=camera_token.dtype)


        if len(depth_gt_index) != 0:
            depth_gt_length = len(depth_gt_index)
            idx_tensor = torch.tensor(depth_gt_index, device=depth.device)

            depth_selected = torch.index_select(depth, dim=1, index=idx_tensor)   # [B, gt_len, H, W]
            mask_selected  = torch.index_select(mask,  dim=1, index=idx_tensor)   # [B, gt_len, H, W]

            depth_gt_normalized = self.normalize_depth(depth_selected, mask_selected)

            depth_gt_normalized = depth_gt_normalized.view(B * depth_gt_length, 1, H, W)
            mask_selected = mask_selected.view(B * depth_gt_length, 1, H, W)

            depthmaps = torch.cat([depth_gt_normalized, mask_selected], dim=1)
            depthmaps = self._match_dtype(depthmaps, self.depth_patch_embed.proj.weight)
            gt_depth_token = self.depth_patch_embed(depthmaps)

            device = depth.device
            depth_full = self.depth_placeholder.expand(K, P, C).clone()

            rows = (torch.arange(B, device=device).unsqueeze(1) * S + idx_tensor.unsqueeze(0)).reshape(-1)
            depth_full[rows] = gt_depth_token.to(dtype=patch_tokens.dtype)  # [B*gt_len, P, C]
            gt_depth_token = depth_full                                   # [B*S, P, C]
        else:
            gt_depth_token = self.depth_placeholder.expand(K, P, C)


        camera_token = camera_token + self.camera_adapters[0](gt_camera_token)
        patch_tokens = patch_tokens + gt_depth_token
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        P_old = P
        _, P, C = tokens.shape
        attn_mask = frame_causal_mask(S, P, tokens.device) if self.causal else None  # of the inter-frame blocks

        frame_idx = 0
        global_idx = 0
        output_list = []

        for index in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos, index = index + 1, camera_gt_index = camera_gt_index,
                        pose_encoding=pose_encoding, register_shape = register_token.shape, P_old = P_old
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos, attn_mask=attn_mask
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                output_list.append(self._collect_layer(len(output_list), frame_intermediates[i], global_intermediates[i]))

        del frame_intermediates
        del global_intermediates
        return output_list, self.patch_start_idx


    def inference(self, images: torch.Tensor, 
                extrinsics: torch.Tensor, 
                intrinsics: torch.Tensor,
                depth: torch.Tensor,
                mask: torch.Tensor,
                depth_gt_index: List[int],
                camera_gt_index: List[int],
                frame_visibility: Optional[torch.Tensor] = None,
                frame_only_layers: Iterable[int] = ()) -> Tuple[List[torch.Tensor], int]:
        """``frame_visibility`` ([S, S] bool, optional): query frame a attends only to the key frames b with
        ``frame_visibility[a, b]`` in every inter-frame block (see ``_visibility_masks``); None keeps the mode's
        attention (bidirectional, or frame-causal with ``causal=True``).

        ``frame_only_layers``: global layers (dense ones, not register attention layers) that attend within each
        frame only (AVGGT's global-to-frame): the frames run through the block as a batch, [B*S, P, C], unmasked.
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")
        self._check_inputs(S, depth_gt_index, camera_gt_index)
        self._check_frame_visibility(S, frame_visibility)
        frame_only_layers = self._check_frame_only_layers(frame_only_layers)

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)
            
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        K, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        camera_token, register_token = self._special_tokens(B, S)

        if len(camera_gt_index) != 0:
            camera_gt_length = len(camera_gt_index)
            camera_idx_tensor = torch.tensor(camera_gt_index, device=depth.device)

            extrinsics_selected = torch.index_select(extrinsics, dim=1, index=camera_idx_tensor)
            intrinsics_selected = torch.index_select(intrinsics, dim=1, index=camera_idx_tensor)

            extrinsics_gt_normalized = self.normalize_extrinsics(extrinsics_selected)
            pose_encoding = extri_intri_to_pose_encoding(
                        extrinsics=extrinsics_gt_normalized,
                        intrinsics=intrinsics_selected,
                        image_size_hw=(H, W),
                        pose_encoding_type="absT_quaR_FoV",
            )
            gt_camera_token = self.pose_embeddings[0](pose_encoding).view(B * camera_gt_length, C).unsqueeze(1)

            device = depth.device
            camera_full = torch.zeros(K, 1, C, device=device, dtype=camera_token.dtype)

            camera_rows = (torch.arange(B, device=device).unsqueeze(1) * S + camera_idx_tensor.unsqueeze(0)).reshape(-1)
            camera_full[camera_rows] = gt_camera_token.to(dtype=camera_token.dtype)
            gt_camera_token = camera_full
        else:
            pose_encoding = None
            gt_camera_token = torch.zeros(K, 1, C, device=depth.device, dtype=camera_token.dtype)


        if len(depth_gt_index) != 0:
            depth_gt_length = len(depth_gt_index)
            idx_tensor = torch.tensor(depth_gt_index, device=depth.device)

            depth_selected = torch.index_select(depth, dim=1, index=idx_tensor)
            mask_selected = torch.index_select(mask, dim=1, index=idx_tensor)
            
            depth_gt_normalized = self.normalize_depth(depth_selected, mask_selected)
            
            depth_gt_normalized = depth_gt_normalized.view(B * depth_gt_length, 1, H, W)
            mask_selected = mask_selected.view(B * depth_gt_length, 1, H, W)
            
            depthmaps = torch.cat([depth_gt_normalized, mask_selected], dim=1)
            depthmaps = self._match_dtype(depthmaps, self.depth_patch_embed.proj.weight)
            gt_depth_token = self.depth_patch_embed(depthmaps)
            
            device = depth.device
            depth_full  = self.depth_placeholder.expand(K, P, C).clone()

            rows = (torch.arange(B, device=device).unsqueeze(1) * S + idx_tensor.unsqueeze(0)).reshape(-1)
            depth_full[rows]  = gt_depth_token.to(dtype = patch_tokens.dtype)     
            gt_depth_token  = depth_full                       
        else:
            gt_depth_token = self.depth_placeholder.expand(K, P, C)


        camera_token = camera_token + self.camera_adapters[0](gt_camera_token)
        patch_tokens = patch_tokens + gt_depth_token
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        P_old = P
        _, P, C = tokens.shape
        if frame_visibility is None:
            # of the inter-frame blocks; a stream step needs none: its caches hold only the past and the current frame
            attn_mask = frame_causal_mask(S, P, tokens.device) if self.causal and self._stream is None else None
            visibility = None
        else:
            attn_mask, visibility = self._visibility_masks(frame_visibility, S, P, tokens.device)

        frame_idx = 0
        global_idx = 0
        output_list = []

        for index in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos, index = index + 1, camera_gt_index = camera_gt_index,
                        pose_encoding=pose_encoding, register_shape = register_token.shape, P_old = P_old
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos, attn_mask=attn_mask, visibility=visibility,
                        frame_only_layers=frame_only_layers,
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                output_list.append(self._collect_layer(len(output_list), frame_intermediates[i], global_intermediates[i]))

        del frame_intermediates
        del global_intermediates
        return output_list, self.patch_start_idx

    def _check_frame_visibility(self, S, frame_visibility):
        if frame_visibility is None:
            return
        if self._stream is not None:
            raise ValueError("a streaming context attends to its KV caches and takes no frame_visibility")
        check_frame_visibility(frame_visibility, S, self.causal)

    def _check_frame_only_layers(self, frame_only_layers) -> frozenset:
        """The checked ``frame_only_layers`` as a set: dense global layer indices."""
        layers = frozenset(frame_only_layers)
        invalid = [i for i in layers if not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < self.depth]
        if invalid:
            raise ValueError(f"frame_only_layers must be global layer indices (ints in [0, {self.depth})), "
                             f"got {sorted(invalid, key=repr)}")
        register = sorted(layers & self.register_attention_layers)
        if register:
            raise ValueError(f"frame_only_layers {register} are register attention layers (special tokens of all "
                             "frames): only dense global layers can attend within the frame")
        if layers and self._stream is not None:
            raise ValueError("a streaming context attends to its KV caches and takes no frame_only_layers")
        return layers

    @staticmethod
    def _visibility_masks(frame_visibility, S, P, device):
        """(dense mask, gathered visibility) of the inter-frame blocks for a checked ``frame_visibility``.

        All-visible runs unmasked and the full lower triangle is the frame-causal mask: exactly the bidirectional
        and the causal paths. Any other visibility is gathered per query frame, never as an [S*P, S*P] mask.
        """
        if frame_visibility.all():
            return None, None
        if torch.equal(frame_visibility, torch.ones_like(frame_visibility).tril()):
            return frame_causal_mask(S, P, device), None
        return None, frame_visibility.to(device)

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None, pose_encoding=None,
                                  depth_encoding=None, attn_mask=None, visibility=None, frame_only_layers=frozenset()):
        """Global blocks as in ``Aggregator``, except that

        * a block in ``frame_only_layers`` attends within each frame: the frames are a batch, [B*S, P, C];
        * with ``visibility`` ([S, S], from ``_visibility_masks``) the queries of frame a attend only to the tokens
          of the frames b with ``visibility[a, b]`` (``visible_frames_block``).
        """
        if visibility is None and not frame_only_layers:
            return super()._process_global_attention(tokens, B, S, P, C, global_idx, pos=pos, attn_mask=attn_mask)
        if attn_mask is not None and visibility is not None:
            raise ValueError("pass a dense attn_mask or a gathered visibility, not both")
        tokens = tokens.reshape(B, S * P, C)
        pos = None if pos is None else pos.reshape(B, S * P, 2)
        intermediates = []
        for _ in range(self.aa_block_size):
            if global_idx in frame_only_layers:
                frame_pos = None if pos is None else pos.reshape(B * S, P, 2)
                tokens = self._run_global_block(global_idx, tokens.reshape(B * S, P, C), frame_pos)
                tokens = tokens.reshape(B, S * P, C)
            elif visibility is not None:
                tokens = visible_frames_block(self.global_blocks[global_idx], tokens, visibility, pos=pos)
            else:
                tokens = self._run_global_block(global_idx, tokens, pos, attn_mask)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))
        return tokens, global_idx, intermediates

    def _special_tokens(self, B, S):
        """Camera and register tokens, [B*S, X, C]: the reference ones (index 0) for the first frame only.

        The first frame is batch position 0, or stream time t=1 in a streaming context.
        """
        if self._stream is None:
            return slice_expand_and_flatten(self.camera_token, B, S), slice_expand_and_flatten(self.register_token, B, S)
        index = 0 if self._stream.is_first else 1
        return tuple(token[:, index].expand(B, *token.shape[2:]) for token in (self.camera_token, self.register_token))

    def _run_global_block(self, layer_idx, tokens, pos=None, attn_mask=None):
        if self._stream is None:
            return super()._run_global_block(layer_idx, tokens, pos, attn_mask)
        if attn_mask is not None:
            raise ValueError("a stream step attends to its KV cache and takes no attention mask")
        return self._stream.run_global(layer_idx, self.global_blocks[layer_idx], tokens, pos)

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None, index=None, camera_gt_index=None,
                         pose_encoding=None, register_shape = None, P_old = None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        K, _, _ = tokens.shape
        register_token = torch.zeros(register_shape, device=tokens.device, dtype=tokens.dtype).expand(B * S, -1, -1)
        if len(camera_gt_index) != 0:
            camera_gt_length = len(camera_gt_index)
            camera_idx_tensor = torch.tensor(camera_gt_index, device=tokens.device)
            gt_camera_token = self.pose_embeddings[index](pose_encoding).view(B * camera_gt_length, C).unsqueeze(1)
            camera_full = torch.zeros(K, 1, C, device=tokens.device, dtype=gt_camera_token.dtype)
            camera_rows = (torch.arange(B, device=tokens.device).unsqueeze(1) * S + camera_idx_tensor.unsqueeze(0)).reshape(-1)
            camera_full[camera_rows] = gt_camera_token.to(dtype=camera_full.dtype)
        else:
            camera_full = torch.zeros(K, 1, C, device=tokens.device, dtype=tokens.dtype)

        depth_injection= torch.zeros(K, P_old, C, device=tokens.device, dtype=tokens.dtype)

        camera_injection = self.camera_adapters[index](camera_full)
        injection_tokens = torch.cat([camera_injection, register_token, depth_injection], dim=1)

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            blk = self.frame_blocks[frame_idx]
            if self.use_checkpoint and self.training:
                tokens = checkpoint(
                    lambda inp, p: blk(inp, pos=p,),
                    tokens,
                    pos,
                    use_reentrant=False
                )
            else:
                tokens = blk(tokens, pos=pos,)
            tokens = tokens + injection_tokens
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates