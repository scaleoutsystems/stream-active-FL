"""
Step 3: Resize cropped images to training resolution.

Resizes the cropped 2840x1600 images to the target training resolution
(default 512x288). Saves to RESIZED_DIR on the shared mount.

Only processes the sequences/ subdirectory. Top-level metadata JSONs are
copied separately. Already-existing files are skipped (safe to resume).

Usage:
    python tools/preprocessing/resize_images.py [--dry-run]
"""

import argparse
import os
import shutil

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from config import CROPPED_DIR, RESIZED_DIR, RESIZE_TARGET


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
    parser = argparse.ArgumentParser(description="Step 3: Resize cropped images")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count files without writing anything")
    args = parser.parse_args()

    # Only process sequences/ (mirrors crop_images.py structure)
    src = str(CROPPED_DIR / "sequences")
    dest = str(RESIZED_DIR / "sequences")
    w, h = RESIZE_TARGET

    print(f"Resize target: {w}x{h}")
    print(f"Source:      {src}")
    print(f"Destination: {dest}")
    if args.dry_run:
        print("(DRY RUN)\n")
    else:
        print()

    if not os.path.exists(src):
        print(f"ERROR: Source directory does not exist: {src}")
        print("Run crop_images.py first.")
        return

    # Copy top-level metadata JSONs that ZOD library needs
    if not args.dry_run:
        dest_root = str(RESIZED_DIR)
        os.makedirs(dest_root, exist_ok=True)
        for f in sorted(os.listdir(str(CROPPED_DIR))):
            src_json = os.path.join(str(CROPPED_DIR), f)
            dest_json = os.path.join(dest_root, f)
            if f.endswith(".json") and not os.path.exists(dest_json):
                shutil.copy2(src_json, dest_json)
                print(f"  Copied {f}")

    # Scan for images
    print("Scanning for images...")
    pairs = collect_jpg_pairs(src, dest)
    print(f"Found {len(pairs)} images.\n")

    # Process images
    stats = {"resized": 0, "skipped": 0, "errors": 0}

    for src_path, dest_path in tqdm(pairs, desc="Resizing", unit="img"):
        if os.path.exists(dest_path):
            stats["skipped"] += 1
            continue
        if args.dry_run:
            stats["resized"] += 1
            continue
        try:
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with Image.open(src_path) as img:
                resized = img.resize(RESIZE_TARGET, Image.LANCZOS)
                resized.save(dest_path)
            stats["resized"] += 1
        except (UnidentifiedImageError, OSError) as e:
            tqdm.write(f"Error: {src_path}: {e}")
            stats["errors"] += 1

    print(f"\nDone!")
    print(f"  Resized: {stats['resized']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Errors:  {stats['errors']}")


if __name__ == "__main__":
    main()
