#!/usr/bin/env python3
"""
Preprocess ZOD Frames: crop, resize, extract annotations, write manifest.

Reads the ZOD Frames dataset via the zod SDK, and for each frame:
  1. Reads the camera image
  2. Crops to remove ego hood and fisheye edges
  3. Resizes to training resolution
  4. Extracts native object_detection annotations, scales bounding boxes
  5. Saves the processed image and annotation JSON to disk

Finally writes a manifest JSON listing all frames in chronological order
(sorted by frame_id).

Usage:
    python tools/preprocessing/prepare_data.py \\
        --zod-root /path/to/zod \\
        --version full \\
        [--output-dir /path/to/output]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    ANNOTATIONS_DIR,
    CATEGORY_NAME_TO_ID,
    CROP_HEIGHT,
    CROP_PARAMS,
    CROP_WIDTH,
    IMAGES_DIR,
    MANIFEST_PATH,
    MIN_BOX_AREA,
    ORIGINAL_ZOD_ROOT,
    OUTPUT_DIR,
    RESIZE_HEIGHT,
    RESIZE_WIDTH,
)


def crop_image(img: np.ndarray) -> np.ndarray:
    """Crop a full-resolution ZOD image using CROP_PARAMS."""
    top = CROP_PARAMS["top"]
    left = CROP_PARAMS["left"]
    h = CROP_PARAMS["height"]
    w = CROP_PARAMS["width"]
    return img[top : top + h, left : left + w]


def scale_box(xyxy: np.ndarray) -> list[float]:
    """Scale a bounding box from original image coords to training resolution.

    Steps: shift by crop offset, then scale by resize ratio.
    """
    x1, y1, x2, y2 = xyxy.tolist()

    x1 -= CROP_PARAMS["left"]
    x2 -= CROP_PARAMS["left"]
    y1 -= CROP_PARAMS["top"]
    y2 -= CROP_PARAMS["top"]

    # Clamp to crop region
    x1 = max(0.0, min(x1, CROP_WIDTH))
    x2 = max(0.0, min(x2, CROP_WIDTH))
    y1 = max(0.0, min(y1, CROP_HEIGHT))
    y2 = max(0.0, min(y2, CROP_HEIGHT))

    sx = RESIZE_WIDTH / CROP_WIDTH
    sy = RESIZE_HEIGHT / CROP_HEIGHT

    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]


def process_frame(zod_frame, frame_id: str, images_dir: Path, annotations_dir: Path) -> dict | None:
    """Process a single ZOD frame: crop, resize, extract annotations."""
    from zod.constants import Anonymization, AnnotationProject

    # Read image
    try:
        img_np = zod_frame.get_image(Anonymization.BLUR)
    except Exception as e:
        print(f"Warning: could not read image for frame {frame_id}: {e}")
        return None

    if img_np is None:
        return None

    # Crop and resize
    cropped = crop_image(img_np)
    pil_img = Image.fromarray(cropped)
    resized = pil_img.resize((RESIZE_WIDTH, RESIZE_HEIGHT), Image.LANCZOS)

    # Save image
    img_path = images_dir / f"{frame_id}.jpg"
    resized.save(img_path, quality=95)

    # Extract annotations
    annotations = []
    categories_present: list[str] = []

    if zod_frame.is_annotated(AnnotationProject.OBJECT_DETECTION):
        for obj in zod_frame.get_annotation(AnnotationProject.OBJECT_DETECTION):
            if obj.unclear:
                continue
            class_name = obj.name
            if class_name not in CATEGORY_NAME_TO_ID:
                continue

            scaled = scale_box(obj.box2d.xyxy)
            x1, y1, x2, y2 = scaled
            w = x2 - x1
            h = y2 - y1
            if w * h < MIN_BOX_AREA:
                continue
            if w <= 0 or h <= 0:
                continue

            annotations.append({
                "bbox_xyxy": scaled,
                "category_id": CATEGORY_NAME_TO_ID[class_name],
                "category_name": class_name,
            })

            if class_name not in categories_present:
                categories_present.append(class_name)

    ann_data = {
        "frame_id": frame_id,
        "annotations": annotations,
        "categories_present": categories_present,
        "num_objects": len(annotations),
    }

    ann_path = annotations_dir / f"{frame_id}.json"
    with ann_path.open("w") as f:
        json.dump(ann_data, f, indent=2)

    # Try to extract timestamp from metadata
    timestamp = None
    try:
        metadata = zod_frame.metadata
        if hasattr(metadata, "timestamp"):
            timestamp = metadata.timestamp
        elif hasattr(metadata, "collection_time"):
            timestamp = metadata.collection_time
    except Exception:
        pass

    return {
        "frame_id": frame_id,
        "image_path": str(img_path.relative_to(images_dir.parent)),
        "annotation_path": str(ann_path.relative_to(annotations_dir.parent)),
        "num_objects": len(annotations),
        "categories_present": categories_present,
        "timestamp": timestamp,
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess ZOD Frames")
    parser.add_argument("--zod-root", type=str, default=str(ORIGINAL_ZOD_ROOT))
    parser.add_argument("--version", type=str, default="full", choices=["full", "mini"])
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    annotations_dir = output_dir / "annotations"
    manifest_path = output_dir / "manifest.json"

    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    print(f"ZOD root:   {args.zod_root}")
    print(f"Version:    {args.version}")
    print(f"Output dir: {output_dir}")
    print(f"Resolution: {RESIZE_WIDTH}x{RESIZE_HEIGHT}")
    print(f"Crop:       {CROP_WIDTH}x{CROP_HEIGHT}")
    print()

    from zod import ZodFrames
    from zod.constants import TRAIN, VAL

    zod_frames = ZodFrames(args.zod_root, args.version)

    train_ids = zod_frames.get_split(TRAIN)
    val_ids = zod_frames.get_split(VAL)
    all_ids = sorted(train_ids | val_ids)

    print(f"Total frames: {len(all_ids)} (train: {len(train_ids)}, val: {len(val_ids)})")

    manifest_entries = []
    failed = 0

    for frame_id in tqdm(all_ids, desc="Processing frames"):
        zod_frame = zod_frames[frame_id]
        entry = process_frame(zod_frame, frame_id, images_dir, annotations_dir)
        if entry is None:
            failed += 1
            continue
        entry["split"] = "train" if frame_id in train_ids else "val"
        manifest_entries.append(entry)

    # Sort by frame_id (chronological proxy)
    manifest_entries.sort(key=lambda e: e["frame_id"])

    manifest = {
        "version": args.version,
        "resize_width": RESIZE_WIDTH,
        "resize_height": RESIZE_HEIGHT,
        "crop_params": CROP_PARAMS,
        "num_frames": len(manifest_entries),
        "num_failed": failed,
        "category_name_to_id": CATEGORY_NAME_TO_ID,
        "frames": manifest_entries,
    }

    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone! Processed {len(manifest_entries)} frames ({failed} failed)")
    print(f"Manifest: {manifest_path}")

    # Print category statistics
    from collections import Counter
    cat_counts: Counter = Counter()
    for entry in manifest_entries:
        for cat in entry["categories_present"]:
            cat_counts[cat] += 1
    print("\nCategory distribution (frames containing each):")
    for cat, count in cat_counts.most_common():
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
