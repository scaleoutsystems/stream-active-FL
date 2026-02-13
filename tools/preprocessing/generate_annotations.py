"""
Step 2: Generate teacher annotations on cropped images.

Runs Detectron2 Faster R-CNN on the cropped images (2840x1600) and saves
per-sequence annotation JSONs. Since the teacher sees cropped images (no hood),
ego-hood "car" labels are avoided without any spatial filtering.

Sequences that already have annotation files are skipped (safe to resume).

Usage:
    python tools/preprocessing/generate_annotations.py [--dry-run]

Requirements:
    - Detectron2 installed
    - Cropped images available at CROPPED_DIR (run crop_images.py first)
"""

import argparse
import json
import os

import cv2
from tqdm import tqdm

from config import (
    ANNOTATIONS_CROPPED_DIR,
    CATEGORIES,
    COCO_TO_CUSTOM,
    CROPPED_DIR,
    TEACHER_DEVICE,
    TEACHER_MODEL_CONFIG,
    TEACHER_SCORE_THRESHOLD,
)


def create_predictor():
    """Initialize Detectron2 predictor."""
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(TEACHER_MODEL_CONFIG))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(TEACHER_MODEL_CONFIG)
    cfg.MODEL.DEVICE = TEACHER_DEVICE
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = TEACHER_SCORE_THRESHOLD
    return DefaultPredictor(cfg)


def annotate_sequence(seq_dir: str, predictor) -> dict:
    """Run teacher on all frames in a sequence directory, return annotation dict."""
    annotations = {
        "images": [],
        "annotations": [],
        "categories": CATEGORIES,
    }

    # Find camera frame images (ZOD uses camera_front_blur/)
    camera_dir = os.path.join(seq_dir, "camera_front_blur")
    if not os.path.isdir(camera_dir):
        # Fallback: look for any subdirectory containing .jpg files
        for sub in sorted(os.listdir(seq_dir)):
            sub_path = os.path.join(seq_dir, sub)
            if os.path.isdir(sub_path) and any(f.lower().endswith(".jpg") for f in os.listdir(sub_path)):
                camera_dir = sub_path
                break
    jpg_files = sorted([f for f in os.listdir(camera_dir) if f.lower().endswith(".jpg")])

    if not jpg_files:
        return annotations

    img_id = 0
    ann_id = 0

    for frame_idx, filename in enumerate(jpg_files):
        filepath = os.path.join(camera_dir, filename)
        img = cv2.imread(filepath)
        if img is None:
            print(f"  Warning: could not read {filepath}")
            continue

        h, w = img.shape[:2]

        outputs = predictor(img)

        boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        scores = outputs["instances"].scores.cpu().numpy()
        classes = outputs["instances"].pred_classes.cpu().numpy()

        annotations["images"].append({
            "id": img_id,
            "file_name": filename,
            "frame_idx": frame_idx,
            "width": w,
            "height": h,
        })

        for box, cls, score in zip(boxes, classes, scores):
            cls_int = int(cls)
            if cls_int not in COCO_TO_CUSTOM:
                continue

            custom_cls = COCO_TO_CUSTOM[cls_int]
            x1, y1, x2, y2 = box
            bw = float(x2 - x1)
            bh = float(y2 - y1)
            area = bw * bh

            annotations["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "frame_idx": frame_idx,
                "category_id": custom_cls,
                "bbox": [float(x1), float(y1), bw, bh],
                "score": float(score),
                "area": area,
                "iscrowd": 0,
            })
            ann_id += 1

        img_id += 1

    return annotations


def main():
    parser = argparse.ArgumentParser(description="Step 2: Generate teacher annotations")
    parser.add_argument("--dry-run", action="store_true",
                        help="List sequences without running the model")
    args = parser.parse_args()

    cropped_root = str(CROPPED_DIR)
    output_dir = str(ANNOTATIONS_CROPPED_DIR)

    print(f"Cropped images: {cropped_root}")
    print(f"Output dir:     {output_dir}")
    if args.dry_run:
        print("(DRY RUN)\n")

    if not os.path.exists(cropped_root):
        print(f"ERROR: Cropped images not found: {cropped_root}")
        print("Run crop_images.py first.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Find sequence directories
    seq_base = os.path.join(cropped_root, "sequences")
    if not os.path.isdir(seq_base):
        print(f"ERROR: No sequences directory found at {seq_base}")
        return

    seq_ids = sorted([d for d in os.listdir(seq_base) if os.path.isdir(os.path.join(seq_base, d))])
    print(f"Found {len(seq_ids)} sequences\n")

    if args.dry_run:
        already = sum(1 for s in seq_ids if os.path.exists(os.path.join(output_dir, f"{s}.json")))
        print(f"  Already annotated: {already}")
        print(f"  Remaining:         {len(seq_ids) - already}")
        return

    predictor = create_predictor()

    total_images = 0
    total_anns = 0

    for seq_id in tqdm(seq_ids, desc="Annotating sequences"):
        output_file = os.path.join(output_dir, f"{seq_id}.json")
        if os.path.exists(output_file):
            continue

        seq_dir = os.path.join(seq_base, seq_id)
        annotations = annotate_sequence(seq_dir, predictor)

        n_images = len(annotations["images"])
        n_anns = len(annotations["annotations"])
        total_images += n_images
        total_anns += n_anns

        with open(output_file, "w") as f:
            json.dump(annotations, f, indent=4)

    print(f"\nDone!")
    print(f"  Total images processed:  {total_images}")
    print(f"  Total annotations saved: {total_anns}")


if __name__ == "__main__":
    main()
