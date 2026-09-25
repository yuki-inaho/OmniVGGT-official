# RGB-D pose pipeline → `colmap_rgbd_v1` → OmniVGGT fine-tuning

Turns a standardized RGB-D sequence (e.g. a camera on a straight rail) into metric,
verified camera poses and a `colmap_rgbd_v1` staging set that
`omnivggt.datasets.ColmapRgbd` loads for fine-tuning.

```text
standardized bundle (rgb/<stem>_rgb.jpg, depth/, mapped_depth/ 3x3, camera_parameters/)
  └─ make_half_view      every k-th frame, RGB → PNG, mapped depth hard-linked, provenance README
  └─ run_vggt_chain      VGGT-Omega RGB-D pose chain (per-chunk scale from measured depth)
  └─ lane_refine         robust rail line; one-axis hypothesis adopted only for rail-like tracks
  └─ colmap_prior_ba     SIFT + sequential matching → triangulate from the prior poses → BA
       align             pose-aware Sim(3) to the prior (rotation from camera orientations)
       depth-scale       metric scale of the BA model from measured depth at its 3D points
  └─ mambaglue_verify    RGB-D MambaGlue matches + MAGSAC + rigid RANSAC (rgbd_tracking_lab)
                         → measured metric motion per edge, compared with the poses
  └─ pose_graph          optional candidate: poses + measured edges (+ rail prior), robust IRLS
  └─ depth_consistency   independent check: reproject measured depth between frames
  └─ export_trajectory   trajectory JSON for vggt-omega `prepare_colmap_rgbd_training.py`
```

Each tool runs in the environment of the code it wraps; paths are always arguments.

| Tool | Environment |
| --- | --- |
| `make_half_view`, `lane_refine`, `colmap_prior_ba`, `pose_graph`, `depth_consistency`, `export_trajectory` | this repo: `PYTHONPATH=tools uv run python -m rgbd_pose_pipeline.<tool>` (`colmap_prior_ba` calls `pixi run --manifest-path <colmap>/pixi.toml colmap ...`) |
| `run_vggt_chain` | vggt-omega `pixi run`, with vggt-omega and this repo's `tools` on `PYTHONPATH` |
| `mambaglue_verify` | mambaglue_rgbd_onnx `pixi run`, with mambaglue_rgbd_onnx, rgbd_tracking_lab and `tools` on `PYTHONPATH` |

## Contracts and lessons

- **Mapped depth**: RGB field of view, nearest-Z splatting, **3x3 nearest-depth dilation**
  (every pixel takes the nearest valid depth in its 3x3 window, so valid pixels can change
  too), uint16 millimetres. `make_half_view` requires a verification report
  (`all_passed`, `dilation_kernel_size == 3`) and writes the provenance README checked by the
  staging exporter.
- **Poses**: `camera_to_world` 4x4, OpenCV, metres. COLMAP text models store world-to-camera.
- **Rail degeneracy**: camera centres on a rail are nearly collinear, so a centres-only Sim(3)
  (COLMAP `model_aligner`) leaves the rotation about the rail undetermined. `align` estimates
  the rotation from camera orientations and uses `model_aligner` only as a scale cross-check.
- **Metric scale**: a feed-forward pose chain on small baselines can under-estimate
  translation. `depth-scale` compares the depth of the BA model's 3D points with the measured
  depth (robust median over all observations) and sets the model scale from it; the
  translation ratio of one-axis MambaGlue edges (measured / pose) is an independent check
  and should be close to 1.
- **Choosing the final poses**: scoring a pose graph with the same edges it was fitted to is
  circular. Compare candidates with `depth_consistency` (inlier ratio of reprojected depth at
  several frame gaps) and keep the best; in rail sequences the COLMAP BA with depth scale was
  better than the edge-fused pose graph because sparse RGB-D rigid fits are noisier than BA.
- **MambaGlue edges**: the ONNX matcher uses 256 keypoints; with ~50 % valid depth the rigid
  support is typically 10–20, hence `min_support = 10`. Estimator failures are recorded as
  `status = estimator_error`, not dropped. `--feature-cache` avoids re-extracting features and
  `--measurements-from` re-scores stored measurements against other poses.

## Fine-tuning OmniVGGT

```bash
export OMNIVGGT_COLMAP_RGBD_ROOTS=/path/staging_a,/path/staging_b
export OMNIVGGT_INIT_CHECKPOINT=checkpoints/OmniVGGT.safetensors
export OMNIVGGT_OUTPUT_DIR=/path/runs
export OMNIVGGT_TRAIN_BATCH_IMAGES=12 OMNIVGGT_PATCH_EMBED_FREEZE=1 OMNIVGGT_STEPS_PER_EPOCH=1560
uv run accelerate launch --num_processes 1 --mixed_precision bf16 \
  train_omnivggt.py --config configs/train_colmap_rgbd.py
PYTHONPATH=tools uv run python -m eval_colmap_rgbd --roots /path/staging_a /path/staging_b --split val \
  --checkpoint /path/runs/omnivggt-colmap-rgbd/final_checkpoint --output eval_finetuned.json
```

On a 32 GB GPU, 12 images per step fit with the DINOv2 patch embedding frozen (the full
model left < 2 GB headroom). Staging images are 384x288 (4:3); the model input is 392x294,
the closest size with the same aspect ratio whose sides are multiples of the 14-pixel patch.
