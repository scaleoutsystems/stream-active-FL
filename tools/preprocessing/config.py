"""
Shared configuration for the ZOD preprocessing pipeline.

Pipeline steps (run in order):
  1. crop_images.py      — Crop originals to remove hood + fisheye edges
  2. generate_annotations.py — Run teacher model on cropped images
  3. resize_images.py    — Resize cropped images to training resolution
  4. scale_annotations.py — Scale annotations from crop to training resolution

All paths and parameters are defined here so every script stays consistent.
"""

from pathlib import Path

# =============================================================================
# Paths
# =============================================================================

# Original full-resolution ZOD dataset (read-only)
ORIGINAL_ZOD_ROOT = Path("/mnt/ZOD_clone_2018_scaleout_zenseact")

# Mount point for processed outputs (shared with team)
MOUNT_ROOT = Path("/mnt/pr_2018_scaleout_workdir")

# Parent directory for all processed ZOD variants (will be renamed to ZODResizes)
RESIZES_DIR = MOUNT_ROOT / "ZOD256"

# Cropped images (intermediate — needed for teacher, can be deleted after)
CROPPED_DIR = RESIZES_DIR / "ZODCropped_2840x1600"

# Final training images (resized from cropped)
RESIZED_DIR = RESIZES_DIR / "ZOD_512x288"

# Project-local annotation directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ANNOTATIONS_CROPPED_DIR = PROJECT_ROOT / "data" / "annotations_cropped_2840x1600"
ANNOTATIONS_RESIZED_DIR = PROJECT_ROOT / "data" / "annotations_512x288"

# =============================================================================
# Crop parameters
# =============================================================================

# Original ZOD image size: 3848 x 2168
# Crop: bottom-only (remove ego hood) + side trim (remove fisheye distortion)
# Result: 2840 x 1600, aspect ratio ~16:9 (1.775:1)
CROP_PARAMS = {
    "top": 0,
    "left": 504,
    "height": 1600,
    "width": 2840,
}

CROP_WIDTH = CROP_PARAMS["width"]    # 2840
CROP_HEIGHT = CROP_PARAMS["height"]  # 1600

# =============================================================================
# Resize parameters
# =============================================================================

# Target training resolution — 16:9, both dims multiples of 32
RESIZE_WIDTH = 512
RESIZE_HEIGHT = 288
RESIZE_TARGET = (RESIZE_WIDTH, RESIZE_HEIGHT)  # (width, height) for PIL

# =============================================================================
# Teacher model
# =============================================================================

# Detectron2 Faster R-CNN config
TEACHER_MODEL_CONFIG = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
TEACHER_SCORE_THRESHOLD = 0.4
TEACHER_DEVICE = "cuda"

# COCO class ID -> custom category ID mapping
COCO_TO_CUSTOM = {
    0: 0,   # person -> person
    2: 1,   # car -> car
    9: 2,   # traffic light -> traffic_light
}

CATEGORIES = [
    {"id": 0, "name": "person"},
    {"id": 1, "name": "car"},
    {"id": 2, "name": "traffic_light"},
]

