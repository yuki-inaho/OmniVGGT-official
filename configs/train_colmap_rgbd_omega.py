# ======================================================
# OmniVGGTOmega fine-tuning on colmap_rgbd_v1 staging sets
# ======================================================
# Same recipe as configs/train_colmap_rgbd.py (OmniVGGT) except:
#   * model_name = "omnivggt_omega": VGGT-Omega style inter-frame attention (see the variant JSON)
#   * no point head: the point loss is computed on points unprojected from depth and camera
#   * initial weights come from the variant's weight map (audited non-strict load), not init_checkpoint
# Paths and run-size knobs come from environment variables (mmengine substitutes {{$VAR:default}}):
#   OMNIVGGT_COLMAP_RGBD_ROOTS    comma-separated colmap_rgbd_v1 roots (required)
#   OMNIVGGT_OMEGA_VARIANT        variant JSON, e.g. configs/omnivggt_omega/variants/V4.json (required)
#   OMNIVGGT_INIT_OMNI            OmniVGGT .safetensors read by the weight map (required by every variant)
#   OMNIVGGT_INIT_OMEGA           VGGT-Omega checkpoint (.pt), only for variants whose map uses it
#   OMNIVGGT_OUTPUT_DIR           output directory (default: outputs)
#   OMNIVGGT_TRAIN_BATCH_IMAGES   images per step, one of 24/18/16/12/4 (default: 12)
#   OMNIVGGT_STEPS_PER_EPOCH      samples per epoch drawn from the train split (default: 1000)
#   OMNIVGGT_PATCH_EMBED_FREEZE   1 freezes the DINOv2 patch embedding (default: 0)
#   OMNIVGGT_OPTIMIZER            adamw (default) or amuse
#
# The image encoder is OmniVGGT's DINOv2 (14-pixel patches), so the 384x288 staging images are
# trained at 392x294 exactly as for OmniVGGT (same 4:3 aspect ratio, 28x21 patches).

_roots = "{{$OMNIVGGT_COLMAP_RGBD_ROOTS:}}"
if not _roots:
    raise ValueError("set OMNIVGGT_COLMAP_RGBD_ROOTS to one or more colmap_rgbd_v1 roots")
colmap_rgbd_roots = _roots.split(",")
omega_variant = "{{$OMNIVGGT_OMEGA_VARIANT:}}"
if not omega_variant:
    raise ValueError("set OMNIVGGT_OMEGA_VARIANT to a variant JSON (configs/omnivggt_omega/variants/*.json)")
model_name = "omnivggt_omega"

# == Common Configuration ==
output_dir = "{{$OMNIVGGT_OUTPUT_DIR:outputs}}"
exp_name = "omnivggt-omega-colmap-rgbd"
logging_dir = "logs"

# == Logging Configuration ==
wandb = False
tensorboard = True
report_to = "tensorboard"
num_save_log = 10
num_save_visual = 1000000
checkpointing_steps = 1000000  # only end-of-epoch and final checkpoints

# == Model Configuration ==
model_url = None
enable_point = False  # no point head; see point_loss_mode
enable_depth = True
enable_camera = True

# == Training Configuration ==
mixed_precision = "bf16"
seed = 42
num_train_epochs = 2
gradient_accumulation_steps = 1
max_grad_norm = 1.0
cam_drop_prob = 0.1
depth_drop_prob = 0.3
save_each_epoch = True
patch_embed_freeze = bool(int("{{$OMNIVGGT_PATCH_EMBED_FREEZE:0}}"))

# == Dataset Configuration ==
train_batch_images = int("{{$OMNIVGGT_TRAIN_BATCH_IMAGES:12}}")
num_workers = 8
steps_per_epoch = int("{{$OMNIVGGT_STEPS_PER_EPOCH:1000}}")

# == Optimizer Configuration ==
optimizer_type = "{{$OMNIVGGT_OPTIMIZER:adamw}}"  # "adamw" (cosine schedule below) or "amuse" (schedule-free)
# AMUSE (https://github.com/kjeiun/amuse, vendored in omnivggt/optim): Muon for hidden-layer weight
# matrices, AdamW-style updates for the rest; no external LR schedule, only a warm-up.
amuse_muon_lr = 1e-4
amuse_aux_lr = 1e-5
amuse_beta1 = 0.4
amuse_beta2 = 0.999
amuse_momentum = 0.95
amuse_rho = 0.3
amuse_r = 0.0
amuse_weight_lr_power = 2.0
amuse_warmup_ratio = 0.05
amuse_weight_decay = 0.01
amuse_weight_decay_at_y = 0.0
amuse_patch_embed_lr_scale = 0.5  # image encoder, when trainable: same ratio as lr_patch_embed / lr below
adam_beta1 = 0.9
adam_beta2 = 0.95
adam_epsilon = 1e-8
adam_weight_decay = 0.01

# == Learning Rate Configuration (fine-tuning from OmniVGGT) ==
lr = 1e-5
lr_patch_embed = 5e-6
lr_camera_head = 1e-5
lr_depth_head = 1e-5

# == Learning Rate Scheduler Configuration ==
lr_scheduler_type = "cosine_with_warmup"
warmup_steps = 100
eta_min_factor = 0.1

# == Loss Configuration ==
camera_loss_weight = 5.0
camera_loss_type = "l1"
depth_loss_weight = 1.0
depth_gradient_loss_fn = "grad"
depth_valid_range = 0.98
point_loss_mode = "derived"  # points unprojected from predicted depth and camera
point_loss_weight = 1.0
point_intrinsics_warmup_ratio = 0.5

# == Visualization Configuration ==
save_glb_visualization = False

# == Resume Configuration ==
resume_model_path = None

resolution = [(392, 294)]
train_dataset = (
    f"{steps_per_epoch} @ ColmapRgbd(roots={colmap_rgbd_roots!r}, split='train', top_k=32, z_far=2, "
    f"aug_crop=16, resolution={resolution}, transform=ColorJitter, seed=985)"
)
