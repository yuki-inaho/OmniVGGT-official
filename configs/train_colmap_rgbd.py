# ======================================================
# OmniVGGT fine-tuning on colmap_rgbd_v1 staging sets
# ======================================================
# Paths and run-size knobs come from environment variables so that no local path
# is stored in the repository (mmengine substitutes {{$VAR:default}}):
#   OMNIVGGT_COLMAP_RGBD_ROOTS    comma-separated colmap_rgbd_v1 roots (required)
#   OMNIVGGT_INIT_CHECKPOINT      OmniVGGT .safetensors to start from (required)
#   OMNIVGGT_OUTPUT_DIR           output directory (default: outputs)
#   OMNIVGGT_TRAIN_BATCH_IMAGES   images per step, one of 24/18/16/12/4 (default: 12)
#   OMNIVGGT_STEPS_PER_EPOCH      samples per epoch drawn from the train split (default: 1000)
#   OMNIVGGT_PATCH_EMBED_FREEZE   1 freezes the DINOv2 patch embedding (default: 0)
#
# Images of the staging sets are 384x288 (4:3).  OmniVGGT's DINOv2 patch embedding
# requires multiples of 14, so the training input is 392x294: the closest size with
# exactly the same 4:3 aspect ratio (28x21 patches).

_roots = "{{$OMNIVGGT_COLMAP_RGBD_ROOTS:}}"
if not _roots:
    raise ValueError("set OMNIVGGT_COLMAP_RGBD_ROOTS to one or more colmap_rgbd_v1 roots")
colmap_rgbd_roots = _roots.split(",")
init_checkpoint = "{{$OMNIVGGT_INIT_CHECKPOINT:}}"
if not init_checkpoint:
    raise ValueError("set OMNIVGGT_INIT_CHECKPOINT to the OmniVGGT .safetensors file")

# == Common Configuration ==
output_dir = "{{$OMNIVGGT_OUTPUT_DIR:outputs}}"
exp_name = "omnivggt-colmap-rgbd"
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
enable_point = True
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
optimizer_type = "adamw"
adam_beta1 = 0.9
adam_beta2 = 0.95
adam_epsilon = 1e-8
adam_weight_decay = 0.01

# == Learning Rate Configuration (fine-tuning from OmniVGGT) ==
lr = 1e-5
lr_patch_embed = 5e-6
lr_camera_head = 1e-5
lr_depth_head = 1e-5
lr_point_head = 1e-5

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
point_loss_weight = 1.0
point_gradient_loss_fn = "normal"
point_valid_range = 0.98

# == Visualization Configuration ==
save_glb_visualization = False

# == Resume Configuration ==
resume_model_path = None

resolution = [(392, 294)]
train_dataset = (
    f"{steps_per_epoch} @ ColmapRgbd(roots={colmap_rgbd_roots!r}, split='train', top_k=32, z_far=2, "
    f"aug_crop=16, resolution={resolution}, transform=ColorJitter, seed=985)"
)
