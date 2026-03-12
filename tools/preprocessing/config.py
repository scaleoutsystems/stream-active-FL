"""
Shared configuration for the preprocessing pipeline.

Pipeline: prepare_data.py performs all steps in one pass:
  1. Read images from ZOD Frames via the zod SDK
  2. Crop (remove ego hood + fisheye edges)
  3. Resize to training resolution
  4. Extract native object_detection annotations, scale boxes, save per-frame JSON
  5. Write a manifest file listing all frames in chronological order
"""

from pathlib import Path
from typing import Dict

# =============================================================================
# Paths
# =============================================================================

ORIGINAL_ZOD_ROOT = Path("/mnt/ZOD_clone_2018_scaleout_zenseact")

OUTPUT_BASE = Path("/mnt/pr_2018_scaleout_workdir/ZOD256")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# =============================================================================
# Crop parameters
# =============================================================================

# Original ZOD image size: 3848 x 2168
# Wide crop: trim 4 px on each side and crop vertically to 1152 px.
# Result: 3840 x 1152 (10:3 aspect ratio).
CROP_PARAMS = {
    "top": 428,
    "left": 4,
    "height": 1152,
    "width": 3840,
}

CROP_WIDTH = CROP_PARAMS["width"]    # 3840
CROP_HEIGHT = CROP_PARAMS["height"]  # 1152

# =============================================================================
# Resize parameters
# =============================================================================

RESIZE_WIDTH = 1600
RESIZE_HEIGHT = 480

RESIZE_TARGET = (RESIZE_WIDTH, RESIZE_HEIGHT)  # (width, height) for PIL

# Output directory (named by resolution so multiple sizes can coexist)
OUTPUT_DIR = OUTPUT_BASE / f"Frames_{RESIZE_WIDTH}x{RESIZE_HEIGHT}"
IMAGES_DIR = OUTPUT_DIR / "images"
ANNOTATIONS_DIR = OUTPUT_DIR / "annotations"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

# =============================================================================
# ZOD category mapping (top-level classes)
# =============================================================================

CATEGORY_NAME_TO_ID: Dict[str, int] = {
    "Vehicle": 0,
    "VulnerableVehicle": 1,
    "Pedestrian": 2,
    "Animal": 3,
    "PoleObject": 4,
    "TrafficSign": 5,
    "TrafficSignal": 6,
    "TrafficGuide": 7,
    "TrafficBeacon": 8,
    "DynamicBarrier": 9,
}

CATEGORY_ID_TO_NAME: Dict[int, str] = {v: k for k, v in CATEGORY_NAME_TO_ID.items()}

NUM_CLASSES = len(CATEGORY_NAME_TO_ID) + 1  # +1 for background (label 0 in torchvision)

# =============================================================================
# Filtering
# =============================================================================

# Keep all annotations at preprocessing time; filter at training time if needed.
MIN_BOX_AREA = 0
