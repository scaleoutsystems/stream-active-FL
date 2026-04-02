"""
Shared configuration for the preprocessing pipeline.

Used by:
  prepare_data.py    - crop, resize, extract annotations, write manifest
  build_manifests.py - (re)generate manifest.json and ordering variants
"""

from pathlib import Path
from typing import Dict
import os

# =============================================================================
# Paths
# =============================================================================

# You can override these per machine through environment variables.
DATA_ROOT = Path(os.environ.get("STREAM_ACTIVE_FL_DATA_ROOT", "data"))

ORIGINAL_ZOD_ROOT = Path(
    os.environ.get(
        "STREAM_ACTIVE_FL_ZOD_ROOT",
        "/path/to/zod",
    )
)

OUTPUT_BASE = Path(
    os.environ.get(
        "STREAM_ACTIVE_FL_PREPROCESSED_ROOT",
        str(DATA_ROOT / "ZOD_frames_preprocessed"),
    )
)

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

# Output directory (named by resolution so multiple sizes can coexist)
OUTPUT_DIR = OUTPUT_BASE / f"Frames_{RESIZE_WIDTH}x{RESIZE_HEIGHT}"

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
