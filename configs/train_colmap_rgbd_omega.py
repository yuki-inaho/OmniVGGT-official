# ======================================================
# OmniVGGTOmega fine-tuning on colmap_rgbd_v1 staging sets
# ======================================================
# Same recipe as configs/train_colmap_rgbd.py (OmniVGGT) except:
#   * model_name = "omnivggt_omega": VGGT-Omega style inter-frame attention (see the variant JSON)
#   * no point head: the point loss is computed on points unprojected from depth and camera
#   * initial weights come from the variant's weight map (audited non-strict load), or strictly from a
#     trained OmniVGGTOmega checkpoint of the same variant (OMNIVGGT_INIT_CHECKPOINT)
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
#   OMNIVGGT_RESOLUTION           training resolution WxH, multiples of 14 (default: 392x294)
#   OMNIVGGT_DEPTH_DROP_PROB      probability of hiding the auxiliary depth from a whole sample (default: 0.3);
#                                 otherwise it goes to a random subset of the views
#   OMNIVGGT_DEPTH_ALL_VIEWS      1 gives the auxiliary depth to every view in training (RGB-D-only use; default: 0)
#   OMNIVGGT_GRAD_ACCUM           micro-batches per optimizer step (default: 1)
#   OMNIVGGT_INIT_CHECKPOINT      trained OmniVGGTOmega checkpoint of the same variant to start from (default: none)
#   OMNIVGGT_CAM_DROP_PROB        probability of hiding the camera input from a whole sample (default: 0.1)
# Stream-Omega (frame-causal) training; each option is independent of the others:
#   OMNIVGGT_CAUSAL               1 trains frame-causal inter-frame attention, needs CAM_DROP_PROB=1 (default: 0)
#   OMNIVGGT_DEPTH_NORM           joint (all views with depth) or first_frame depth-input normalization (default: joint)
#   OMNIVGGT_TARGET_SCALE         all (every frame's points) or first_frame target scale (default: all)
#   OMNIVGGT_VIEW_SELECTION       random_topk (nearest poses) or sequential (ordered clips) views (default: random_topk)
#   OMNIVGGT_SEQ_STRIDES          sequential only: comma-separated frame strides, one drawn per clip (default: 1)
#   OMNIVGGT_FULL_CLIPS           1 makes every step one clip of all its images (default: 0)
#   OMNIVGGT_DATA_SEED            ColmapRgbd sample seed (views, strides, crops; seed + sample index), a positive
#                                 integer (default: 985); another value draws other clips from the same anchors
#
# The image encoder is OmniVGGT's DINOv2 (14-pixel patches), so the 384x288 staging images are
# trained at 392x294 exactly as for OmniVGGT (same 4:3 aspect ratio, 28x21 patches); 640x480 staging
# images are trained at 644x476 (46x34 patches, the nearest multiples of 14).

_roots = "{{$OMNIVGGT_COLMAP_RGBD_ROOTS:}}"
if not _roots:
    raise ValueError("set OMNIVGGT_COLMAP_RGBD_ROOTS to one or more colmap_rgbd_v1 roots")
colmap_rgbd_roots = _roots.split(",")
omega_variant = "{{$OMNIVGGT_OMEGA_VARIANT:}}"
if not omega_variant:
    raise ValueError("set OMNIVGGT_OMEGA_VARIANT to a variant JSON (configs/omnivggt_omega/variants/*.json)")
model_name = "omnivggt_omega"
_init_checkpoint = "{{$OMNIVGGT_INIT_CHECKPOINT:none}}"  # mmengine treats an empty default as "required"
init_checkpoint = None if _init_checkpoint == "none" else _init_checkpoint

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
gradient_accumulation_steps = int("{{$OMNIVGGT_GRAD_ACCUM:1}}")
max_grad_norm = 1.0
cam_drop_prob = float("{{$OMNIVGGT_CAM_DROP_PROB:0.1}}")
if not 0 <= cam_drop_prob <= 1:
    raise ValueError(f"OMNIVGGT_CAM_DROP_PROB must be in [0, 1], got {cam_drop_prob}")
depth_drop_prob = float("{{$OMNIVGGT_DEPTH_DROP_PROB:0.3}}")
depth_all_views = bool(int("{{$OMNIVGGT_DEPTH_ALL_VIEWS:0}}"))
save_each_epoch = True
patch_embed_freeze = bool(int("{{$OMNIVGGT_PATCH_EMBED_FREEZE:0}}"))

# == Stream-Omega (frame-causal) Configuration ==
causal = bool(int("{{$OMNIVGGT_CAUSAL:0}}"))
if causal and cam_drop_prob < 1:
    raise ValueError("OMNIVGGT_CAUSAL=1 needs OMNIVGGT_CAM_DROP_PROB=1 (the camera input normalization is not causal)")
depth_norm = "{{$OMNIVGGT_DEPTH_NORM:joint}}"
if depth_norm not in ("joint", "first_frame"):
    raise ValueError(f"OMNIVGGT_DEPTH_NORM must be joint or first_frame, got {depth_norm!r}")
target_scale = "{{$OMNIVGGT_TARGET_SCALE:all}}"
if target_scale not in ("all", "first_frame"):
    raise ValueError(f"OMNIVGGT_TARGET_SCALE must be all or first_frame, got {target_scale!r}")

# == Dataset Configuration ==
view_selection = "{{$OMNIVGGT_VIEW_SELECTION:random_topk}}"
if view_selection not in ("random_topk", "sequential"):
    raise ValueError(f"OMNIVGGT_VIEW_SELECTION must be random_topk or sequential, got {view_selection!r}")
_strides = "{{$OMNIVGGT_SEQ_STRIDES:1}}"
if not all(s.isdigit() and int(s) > 0 for s in _strides.split(",")):
    raise ValueError(f"OMNIVGGT_SEQ_STRIDES must be comma-separated positive integers, got {_strides!r}")
sequential_strides = [int(s) for s in _strides.split(",")]
if view_selection != "sequential" and sequential_strides != [1]:
    raise ValueError("OMNIVGGT_SEQ_STRIDES applies only to OMNIVGGT_VIEW_SELECTION=sequential")
full_clips = bool(int("{{$OMNIVGGT_FULL_CLIPS:0}}"))
_data_seed = "{{$OMNIVGGT_DATA_SEED:985}}"
if not (_data_seed.isdigit() and int(_data_seed) > 0):  # ColmapRgbd draws unseeded samples for a seed of 0
    raise ValueError(f"OMNIVGGT_DATA_SEED must be a positive integer, got {_data_seed!r}")
data_seed = int(_data_seed)
train_batch_images = int("{{$OMNIVGGT_TRAIN_BATCH_IMAGES:12}}")
num_workers = 8
steps_per_epoch = int("{{$OMNIVGGT_STEPS_PER_EPOCH:1000}}")
if steps_per_epoch % gradient_accumulation_steps:
    raise ValueError("OMNIVGGT_STEPS_PER_EPOCH (micro-batches per epoch) must be a multiple of OMNIVGGT_GRAD_ACCUM")

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

_resolution = "{{$OMNIVGGT_RESOLUTION:392x294}}".split("x")
if len(_resolution) != 2 or not all(v.isdigit() and int(v) % 14 == 0 for v in _resolution):
    raise ValueError(f"OMNIVGGT_RESOLUTION must be WxH in multiples of 14, got {'x'.join(_resolution)!r}")
resolution = [(int(_resolution[0]), int(_resolution[1]))]
_views = "" if view_selection == "random_topk" else (
    f", view_selection={view_selection!r}, sequential_stride={sequential_strides}"
)
train_dataset = (
    f"{steps_per_epoch} @ ColmapRgbd(roots={colmap_rgbd_roots!r}, split='train', top_k=32, z_far=2, "
    f"aug_crop=16, resolution={resolution}, transform=ColorJitter, seed={data_seed}{_views})"
)
