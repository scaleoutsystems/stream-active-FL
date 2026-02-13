"""
Step 1: Crop original ZOD images.

Crops the original 3848x2168 images to 2840x1600 (~16:9), removing:
  - Bottom ~568px (ego car hood)
  - Side margins (~504px each side, fisheye edge distortion)

Only processes the sequences/ subdirectory. Cropped images are saved to
CROPPED_DIR on the shared mount. Already-existing files are skipped.

Usage:
    python tools/preprocessing/crop_images.py [--dry-run]
"""

import argparse
import os
import shutil

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from config import CROP_PARAMS, CROPPED_DIR, ORIGINAL_ZOD_ROOT


def crop_image(img: Image.Image) -> Image.Image:
    """Crop a PIL image using the configured crop parameters."""
    top = CROP_PARAMS["top"]
    left = CROP_PARAMS["left"]
    height = CROP_PARAMS["height"]
    width = CROP_PARAMS["width"]
    return img.crop((left, top, left + width, top + height))


def collect_jpg_pairs(src_dir: str, dest_dir: str) -> list[tuple[str, str]]:
    """Recursively collect (src_path, dest_path) for every .jpg file."""
    pairs = []
    for item in sorted(os.listdir(src_dir)):
        src_path = os.path.join(src_dir, item)
        dest_path = os.path.join(dest_dir, item)
        if os.path.isdir(src_path):
            pairs.extend(collect_jpg_pairs(src_path, dest_path))
        elif item.lower().endswith(".jpg"):
            pairs.append((src_path, dest_path))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Step 1: Crop ZOD images")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count files without writing anything")
    args = parser.parse_args()

    # Only process sequences/ (single_frames/ has 100K dirs we don't need)
    src = str(ORIGINAL_ZOD_ROOT / "sequences")
    dest = str(CROPPED_DIR / "sequences")

    crop = CROP_PARAMS
    print(f"Crop parameters: {crop['width']}x{crop['height']} "
          f"from (left={crop['left']}, top={crop['top']})")
    print(f"Source:      {src}")
    print(f"Destination: {dest}")
    if args.dry_run:
        print("(DRY RUN)\n")
    else:
        print()

    if not os.path.exists(src):
        print(f"ERROR: Source directory does not exist: {src}")
        return

    # Copy top-level metadata JSONs that ZOD library needs
    if not args.dry_run:
        dest_root = str(CROPPED_DIR)
        os.makedirs(dest_root, exist_ok=True)
        for f in sorted(os.listdir(str(ORIGINAL_ZOD_ROOT))):
            if f.endswith(".json"):
                src_json = os.path.join(str(ORIGINAL_ZOD_ROOT), f)
                dest_json = os.path.join(dest_root, f)
                if not os.path.exists(dest_json):
                    shutil.copy2(src_json, dest_json)
                    print(f"  Copied {f}")

    # Scan for images
    print("Scanning for images...")
    pairs = collect_jpg_pairs(src, dest)
    print(f"Found {len(pairs)} images.\n")

    # Process images
    stats = {"cropped": 0, "skipped": 0, "errors": 0}

    for src_path, dest_path in tqdm(pairs, desc="Cropping", unit="img"):
        if os.path.exists(dest_path):
            stats["skipped"] += 1
            continue
        if args.dry_run:
            stats["cropped"] += 1
            continue
        try:
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with Image.open(src_path) as img:
                cropped = crop_image(img)
                cropped.save(dest_path)
            stats["cropped"] += 1
        except (UnidentifiedImageError, OSError) as e:
            tqdm.write(f"Error: {src_path}: {e}")
            stats["errors"] += 1

    print(f"\nDone!")
    print(f"  Cropped: {stats['cropped']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Errors:  {stats['errors']}")


if __name__ == "__main__":
    main()
