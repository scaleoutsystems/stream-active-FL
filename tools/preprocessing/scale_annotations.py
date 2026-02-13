"""
Step 4: Scale annotations from crop resolution to training resolution.

Reads annotations generated at the crop resolution (2840x1600) and scales
bounding boxes to the target training resolution (512x288).

Usage:
    python tools/preprocessing/scale_annotations.py [--dry-run]
"""

import argparse
import json
import os

from tqdm import tqdm

from config import (
    ANNOTATIONS_CROPPED_DIR,
    ANNOTATIONS_RESIZED_DIR,
    CROP_HEIGHT,
    CROP_WIDTH,
    RESIZE_HEIGHT,
    RESIZE_WIDTH,
)


def scale_bbox(bbox: list, scale_x: float, scale_y: float) -> list:
    """Scale a [x, y, w, h] bounding box."""
    x, y, w, h = bbox
    return [x * scale_x, y * scale_y, w * scale_x, h * scale_y]


def process_annotations(src_dir: str, dest_dir: str, *, dry_run: bool = False) -> dict:
    """Scale all annotation files from crop to target resolution."""
    scale_x = RESIZE_WIDTH / CROP_WIDTH
    scale_y = RESIZE_HEIGHT / CROP_HEIGHT

    stats = {"files": 0, "annotations": 0}

    if not os.path.exists(dest_dir) and not dry_run:
        os.makedirs(dest_dir)

    json_files = sorted([f for f in os.listdir(src_dir) if f.endswith(".json")])

    for filename in tqdm(json_files, desc="Scaling annotations"):
        src_path = os.path.join(src_dir, filename)
        dest_path = os.path.join(dest_dir, filename)

        with open(src_path, "r") as f:
            data = json.load(f)

        # Scale annotations
        new_annotations = []
        for ann in data.get("annotations", []):
            ann = dict(ann)  # don't mutate original
            scaled = scale_bbox(ann["bbox"], scale_x, scale_y)
            ann["bbox"] = scaled
            ann["area"] = scaled[2] * scaled[3]
            new_annotations.append(ann)

        data["annotations"] = new_annotations
        stats["annotations"] += len(new_annotations)

        # Update image dimensions
        for img in data.get("images", []):
            img["width"] = RESIZE_WIDTH
            img["height"] = RESIZE_HEIGHT

        if not dry_run:
            with open(dest_path, "w") as f:
                json.dump(data, f, indent=4)

        stats["files"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Step 4: Scale annotations")
    parser.add_argument("--dry-run", action="store_true",
                        help="Process files without writing output")
    args = parser.parse_args()

    src = str(ANNOTATIONS_CROPPED_DIR)
    dest = str(ANNOTATIONS_RESIZED_DIR)

    print(f"Scale: {CROP_WIDTH}x{CROP_HEIGHT} -> {RESIZE_WIDTH}x{RESIZE_HEIGHT}")
    print(f"  scale_x = {RESIZE_WIDTH / CROP_WIDTH:.6f}")
    print(f"  scale_y = {RESIZE_HEIGHT / CROP_HEIGHT:.6f}")
    print(f"Source:      {src}")
    print(f"Destination: {dest}")
    if args.dry_run:
        print("(DRY RUN)\n")
    else:
        print()

    if not os.path.exists(src):
        print(f"ERROR: Source annotations not found: {src}")
        print("Run generate_annotations.py first.")
        return

    stats = process_annotations(src, dest, dry_run=args.dry_run)

    print(f"\nDone!")
    print(f"  Files processed: {stats['files']}")
    print(f"  Annotations:     {stats['annotations']}")


if __name__ == "__main__":
    main()
