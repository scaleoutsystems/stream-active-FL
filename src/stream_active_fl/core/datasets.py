"""
Dataset implementations for offline and streaming learning on ZOD.

Offline datasets (for PyTorch DataLoader, shuffled, multi-epoch):
    ZODClassificationDataset         Binary classification per frame
    ZODDetectionDataset     Multi-class object detection per frame

Streaming dataset (iterator-based, strict temporal order):
    StreamingDataset        Supports both classification and detection tasks

Also provides annotation loaders, transforms, augmentations, and collate
functions used by the experiment scripts.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from zod import ZodSequences
import zod.constants as constants

from .items import StreamItem


# =============================================================================
# Category definitions (must match annotate.py)
# =============================================================================

CATEGORY_ID_TO_NAME: Dict[int, str] = {
    0: "person",
    1: "car",
    2: "traffic_light",
}

CATEGORY_NAME_TO_ID: Dict[str, int] = {v: k for k, v in CATEGORY_ID_TO_NAME.items()}


# =============================================================================
# Shared annotation helpers
# =============================================================================


def _read_annotation_json(ann_path: Path) -> Optional[List[Dict[str, Any]]]:
    """
    Read a per-sequence annotation JSON and return its annotation list.

    Returns None (instead of raising) when the file is missing or corrupt,
    so callers can fall back to treating the sequence as unannotated.
    """
    if not ann_path.exists():
        return None
    try:
        with ann_path.open("r") as f:
            data = json.load(f)
        return data.get("annotations", [])
    except Exception as e:
        print(f"Warning: could not read {ann_path}: {e}")
        return None


def load_classification_frame_info(
    ann_path: Path,
    target_category: int,
    min_score: float,
) -> Dict[int, Dict[str, Any]]:
    """
    Load per-frame binary classification info for a single category.

    Used by ZODClassificationDataset and StreamingDataset (classification mode).

    Args:
        ann_path: Path to per-sequence annotation JSON.
        target_category: Category ID to treat as the positive class.
        min_score: Minimum detection score to count as a valid detection.

    Returns:
        Dict mapping frame_idx -> {
            "has_target": bool,
            "max_score": float (max detection score, 0 if none),
        }
        Only frames with qualifying detections are included; absent frame
        indices should be treated as negative (has_target=False, max_score=0).
    """
    annotations = _read_annotation_json(ann_path)
    if annotations is None:
        return {}

    frame_detections: Dict[int, List[float]] = defaultdict(list)
    for ann in annotations:
        if ann.get("category_id") != target_category:
            continue
        score = float(ann.get("score", 1.0))
        if score < min_score:
            continue
        frame_detections[ann["frame_idx"]].append(score)

    return {
        frame_idx: {"has_target": True, "max_score": max(scores)}
        for frame_idx, scores in frame_detections.items()
    }


def load_detection_frame_info(
    ann_path: Path,
    min_score: float,
    min_box_area: float = 0.0,
) -> Dict[int, Dict[str, Any]]:
    """
    Load per-frame detection annotations for all categories.

    Used by ZODDetectionDataset and StreamingDataset (detection mode).

    Args:
        ann_path: Path to per-sequence annotation JSON.
        min_score: Minimum detection score to include.
        min_box_area: Minimum bounding box area (width * height) to include.
            Boxes smaller than this are silently dropped. Useful for filtering
            tiny pseudo-labels that are below the detector's effective resolution.

    Returns:
        Dict mapping frame_idx -> {
            "has_target": True,
            "max_score": float,
            "annotations": list of {"bbox": [x,y,w,h], "category_id": int, "score": float},
        }
        Only frames with qualifying detections are included.
    """
    annotations = _read_annotation_json(ann_path)
    if annotations is None:
        return {}

    frame_annotations: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        score = float(ann.get("score", 1.0))
        if score < min_score:
            continue
        x, y, w, h = ann["bbox"]
        if w * h < min_box_area:
            continue
        frame_annotations[ann["frame_idx"]].append({
            "bbox": ann["bbox"],
            "category_id": ann["category_id"],
            "score": score,
        })

    return {
        frame_idx: {
            "has_target": True,
            "max_score": max(a["score"] for a in anns),
            "annotations": anns,
        }
        for frame_idx, anns in frame_annotations.items()
    }


def format_detection_annotations(
    raw_anns: List[Dict[str, Any]],
) -> Dict[str, torch.Tensor]:
    """
    Convert raw annotation list to torchvision-format detection target.

    Converts bboxes from [x, y, w, h] to [x1, y1, x2, y2] and shifts
    category IDs by +1 (torchvision reserves 0 for background).

    Returns:
        {"boxes": FloatTensor[N, 4], "labels": Int64Tensor[N]}
    """
    if not raw_anns:
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }

    boxes = []
    labels = []
    for ann in raw_anns:
        x, y, w, h = ann["bbox"]
        boxes.append([x, y, x + w, y + h])
        labels.append(ann["category_id"] + 1)

    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


# =============================================================================
# Default image transforms
# =============================================================================

def get_classification_transforms(
    image_size: Tuple[int, int] = (224, 224),
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns (train_transform, val_transform) for classification.

    Both are identical and deterministic (no augmentation). Kept separate
    so train-only augmentation can be added later without changing callers.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, val_transform


def get_detection_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns (train_transform, val_transform) for detection.

    Detection models (e.g. FCOS) handle normalization and resizing internally
    via GeneralizedRCNNTransform, so we only convert PIL images to float
    tensors in [0, 1] range.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform, transform


# =============================================================================
# Detection augmentation
# =============================================================================


class DetectionAugmentation:
    """
    Data augmentation for object detection that handles both image and targets.

    Applies augmentations that are safe for detection training:
    - Random horizontal flip (spatial: modifies both image and bounding boxes)
    - Color jitter (photometric: modifies image only)

    Operates on (PIL Image, target dict) pairs, where the target dict
    contains "boxes" (FloatTensor[N, 4] in xyxy format) and "labels".
    Must be applied BEFORE ToTensor conversion.

    Args:
        hflip_prob: Probability of applying horizontal flip.
        color_jitter: Whether to apply random color jitter.
        brightness: Max brightness jitter factor.
        contrast: Max contrast jitter factor.
        saturation: Max saturation jitter factor.
        hue: Max hue jitter factor.
    """

    def __init__(
        self,
        hflip_prob: float = 0.5,
        color_jitter: bool = True,
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.2,
        hue: float = 0.1,
    ):
        self.hflip_prob = hflip_prob
        self.color_jitter_transform = (
            transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            )
            if color_jitter
            else None
        )

    def __call__(
        self,
        image: Image.Image,
        target: Dict[str, torch.Tensor],
    ) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
        """
        Apply augmentations to an image and its detection target.

        Args:
            image: PIL Image.
            target: Dict with "boxes" (FloatTensor[N, 4] xyxy) and "labels".

        Returns:
            Augmented (image, target) pair.
        """
        # Photometric: color jitter (image only, doesn't affect boxes)
        if self.color_jitter_transform is not None:
            image = self.color_jitter_transform(image)

        # Spatial: random horizontal flip (affects both image and boxes)
        if random.random() < self.hflip_prob:
            image = F.hflip(image)
            boxes = target["boxes"]
            if len(boxes) > 0:
                width = image.width
                # [x1, y1, x2, y2] -> flip x coordinates
                new_boxes = boxes.clone()
                new_boxes[:, 0] = width - boxes[:, 2]
                new_boxes[:, 2] = width - boxes[:, 0]
                target = {**target, "boxes": new_boxes}

        return image, target


def get_detection_augmentation(
    hflip_prob: float = 0.5,
    color_jitter: bool = True,
) -> DetectionAugmentation:
    """Create a DetectionAugmentation instance with the given parameters."""
    return DetectionAugmentation(
        hflip_prob=hflip_prob,
        color_jitter=color_jitter,
    )


# =============================================================================
# ZODClassificationDataset (Offline, Classification)
# =============================================================================

class ZODClassificationDataset(Dataset):
    """
    PyTorch Dataset for frame-level binary classification on ZOD sequences.

    Each sample is a single frame. The binary target indicates whether the frame
    contains at least one detection of `target_category` (with score >= min_score).

    Args:
        dataset_root: Path to ZOD dataset (e.g. ZOD256/ZOD_640x360 or ZODCropped_2840x1600).
        annotations_dir: Path to directory with per-sequence annotation JSONs.
        version: ZOD version, "full" or "mini".
        split: "train", "val", or None (use all sequences).
        transform: Torchvision transform to apply to images.
        target_category: Category ID (0=person, 1=car, 2=traffic_light) to use
            for the binary target. Frame is positive if it has >= 1 detection
            of this category.
        min_score: Minimum detection score to count as a valid detection.
            Detections below this threshold are ignored when computing the target.
        subsample_steps: Use every Nth frame (1 = all frames, 5 = every 5th).
        verbose: Print dataset statistics after loading.

    Item format:
        {
            "image": Tensor (C, H, W) after transform,
            "target": Tensor scalar (0.0 or 1.0),
            "score": float, max detection score for target_category in this frame
                     (0.0 if no detections),
            "metadata": {
                "seq_idx": int,
                "seq_id": str,
                "frame_idx": int,
            },
        }
    """

    def __init__(
        self,
        dataset_root: str | Path,
        annotations_dir: str | Path,
        version: Literal["full", "mini"] = "full",
        split: Optional[Literal["train", "val"]] = None,
        transform: Optional[Callable] = None,
        target_category: int = 0,
        min_score: float = 0.0,
        subsample_steps: int = 1,
        verbose: bool = True,
    ):
        self.dataset_root = Path(dataset_root)
        self.annotations_dir = Path(annotations_dir)
        self.version = version
        self.split = split
        self.transform = transform
        self.target_category = target_category
        self.min_score = min_score
        self.subsample_steps = subsample_steps

        # Load ZOD sequences
        zod_sequences = ZodSequences(str(self.dataset_root), version)

        # Filter by split if requested
        if split == "train":
            sequence_ids = zod_sequences.get_split(constants.TRAIN)
            self.sequences = [zod_sequences[seq_id] for seq_id in sequence_ids]
        elif split == "val":
            sequence_ids = zod_sequences.get_split(constants.VAL)
            self.sequences = [zod_sequences[seq_id] for seq_id in sequence_ids]
        else:
            self.sequences = list(zod_sequences)

        # Build sample index
        # samples: list of (seq_idx, frame_idx, target, max_score)
        # sequence_map: seq_idx -> (seq_id, frames_list)
        self.samples: List[Tuple[int, int, float, float]] = []
        self.sequence_map: Dict[int, Tuple[str, Any]] = {}

        for seq_idx, sequence in enumerate(self.sequences):
            seq_id = sequence.info.id
            frames = sequence.info.get_camera_frames()

            self.sequence_map[seq_idx] = (seq_id, frames)

            # Load annotation JSON for this sequence
            ann_path = self.annotations_dir / f"{seq_id}.json"
            frame_info = self._load_frame_info(ann_path)

            # Create samples for each (subsampled) frame
            for frame_idx in range(0, len(frames), self.subsample_steps):
                info = frame_info.get(frame_idx, {"has_target": False, "max_score": 0.0})
                target = 1.0 if info["has_target"] else 0.0
                max_score = info["max_score"]
                self.samples.append((seq_idx, frame_idx, target, max_score))

        if verbose:
            self._print_summary()

    def _load_frame_info(self, ann_path: Path) -> Dict[int, Dict[str, Any]]:
        """Load per-frame classification info (delegates to module-level helper)."""
        return load_classification_frame_info(ann_path, self.target_category, self.min_score)

    def _print_summary(self) -> None:
        """Print dataset statistics."""
        num_positive = sum(1 for _, _, t, _ in self.samples if t == 1.0)
        num_negative = len(self.samples) - num_positive
        total = len(self.samples)

        cat_name = CATEGORY_ID_TO_NAME.get(self.target_category, str(self.target_category))

        print()
        print("=" * 60)
        print("ZODClassificationDataset Summary")
        print("=" * 60)
        print(f"  Dataset root     : {self.dataset_root}")
        print(f"  Annotations dir  : {self.annotations_dir}")
        print(f"  Split            : {self.split or 'all'}")
        print(f"  Target category  : {self.target_category} ({cat_name})")
        print(f"  Min score filter : {self.min_score}")
        print(f"  Subsample steps  : {self.subsample_steps}")
        print("-" * 60)
        print(f"  Total sequences  : {len(self.sequence_map)}")
        print(f"  Total frames     : {total}")
        print(f"  Positive frames  : {num_positive} ({100 * num_positive / max(total, 1):.1f}%)")
        print(f"  Negative frames  : {num_negative} ({100 * num_negative / max(total, 1):.1f}%)")
        print(f"  Class ratio -/+  : {num_negative / max(num_positive, 1):.2f}")
        print("=" * 60)
        print()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Optional[Dict[str, Any]]:
        seq_idx, frame_idx, target, max_score = self.samples[index]
        seq_id, frames = self.sequence_map[seq_idx]

        # Read image
        try:
            img_np = frames[frame_idx].read()
            if img_np is None:
                return None
        except Exception:
            return None

        img = Image.fromarray(img_np)

        if self.transform is not None:
            img = self.transform(img)

        return {
            "image": img,
            "target": torch.tensor(target, dtype=torch.float32),
            "score": max_score,
            "metadata": {
                "seq_idx": seq_idx,
                "seq_id": seq_id,
                "frame_idx": frame_idx,
            },
        }

    def get_sequence_info(self, seq_idx: int) -> Dict[str, Any]:
        """Return metadata about a sequence (for debugging/inspection)."""
        seq_id, frames = self.sequence_map[seq_idx]
        return {
            "seq_idx": seq_idx,
            "seq_id": seq_id,
            "num_frames": len(frames),
        }


# =============================================================================
# ZODDetectionDataset (Offline, Detection)
# =============================================================================


class ZODDetectionDataset(Dataset):
    """
    PyTorch Dataset for offline multi-class object detection on ZOD sequences.

    Each sample is a single frame with bounding boxes and class labels for all
    detected categories. Used with DataLoader for shuffled, multi-epoch training
    to establish an upper-bound detection baseline.

    Args:
        dataset_root: Path to ZOD dataset.
        annotations_dir: Path to directory with per-sequence annotation JSONs.
        version: ZOD version, "full" or "mini".
        split: "train", "val", or None.
        transform: Torchvision transform to apply to images (e.g. ToTensor).
        min_score: Minimum detection score to include.
        min_box_area: Minimum box area (w*h) to include. Drops tiny boxes
            that are below the detector's effective resolution.
        subsample_steps: Use every Nth frame (1 = all frames).
        augmentation: Optional DetectionAugmentation for training.
            Applied to (PIL image, target) before the image transform.
        verbose: Print dataset statistics after loading.

    Item format:
        (image, target) where:
        - image: Tensor (3, H, W) in [0, 1] after transform
        - target: {"boxes": FloatTensor[N, 4] xyxy, "labels": Int64Tensor[N]}
    """

    def __init__(
        self,
        dataset_root: str | Path,
        annotations_dir: str | Path,
        version: Literal["full", "mini"] = "full",
        split: Optional[Literal["train", "val"]] = None,
        transform: Optional[Callable] = None,
        min_score: float = 0.0,
        min_box_area: float = 0.0,
        subsample_steps: int = 1,
        augmentation: Optional[DetectionAugmentation] = None,
        verbose: bool = True,
    ):
        self.dataset_root = Path(dataset_root)
        self.annotations_dir = Path(annotations_dir)
        self.version = version
        self.split = split
        self.transform = transform
        self.min_score = min_score
        self.min_box_area = min_box_area
        self.subsample_steps = subsample_steps
        self.augmentation = augmentation

        # Load ZOD sequences
        zod_sequences = ZodSequences(str(self.dataset_root), version)

        if split == "train":
            sequence_ids = zod_sequences.get_split(constants.TRAIN)
            self.sequences = [zod_sequences[seq_id] for seq_id in sequence_ids]
        elif split == "val":
            sequence_ids = zod_sequences.get_split(constants.VAL)
            self.sequences = [zod_sequences[seq_id] for seq_id in sequence_ids]
        else:
            self.sequences = list(zod_sequences)

        # Build sample index and load annotations
        # samples: list of (seq_idx, frame_idx)
        # sequence_map: seq_idx -> (seq_id, frames_list)
        # frame_info_map: seq_idx -> {frame_idx -> annotation info}
        self.samples: List[Tuple[int, int]] = []
        self.sequence_map: Dict[int, Tuple[str, Any]] = {}
        self.frame_info_map: Dict[int, Dict[int, Dict[str, Any]]] = {}

        total_annotations = 0
        frames_with_objects = 0

        for seq_idx, sequence in enumerate(self.sequences):
            seq_id = sequence.info.id
            frames = sequence.info.get_camera_frames()

            self.sequence_map[seq_idx] = (seq_id, frames)

            # Load detection annotations for this sequence
            ann_path = self.annotations_dir / f"{seq_id}.json"
            frame_info = load_detection_frame_info(
                ann_path, self.min_score, self.min_box_area,
            )
            self.frame_info_map[seq_idx] = frame_info

            for frame_idx in range(0, len(frames), self.subsample_steps):
                self.samples.append((seq_idx, frame_idx))
                info = frame_info.get(frame_idx, {})
                if info.get("has_target", False):
                    frames_with_objects += 1
                    total_annotations += len(info.get("annotations", []))

        self._frames_with_objects = frames_with_objects
        self._total_annotations = total_annotations

        if verbose:
            self._print_summary()

    def _print_summary(self) -> None:
        """Print dataset statistics."""
        total = len(self.samples)
        empty = total - self._frames_with_objects
        cats = ", ".join(f"{v} ({k})" for k, v in CATEGORY_ID_TO_NAME.items())

        print()
        print("=" * 60)
        print("ZODDetectionDataset Summary")
        print("=" * 60)
        print(f"  Dataset root       : {self.dataset_root}")
        print(f"  Annotations dir    : {self.annotations_dir}")
        print(f"  Split              : {self.split or 'all'}")
        print(f"  Categories         : {cats}")
        print(f"  Min score filter   : {self.min_score}")
        print(f"  Min box area       : {self.min_box_area}")
        print(f"  Subsample steps    : {self.subsample_steps}")
        print(f"  Augmentation       : {'yes' if self.augmentation else 'no'}")
        print("-" * 60)
        print(f"  Total sequences    : {len(self.sequence_map)}")
        print(f"  Total frames       : {total}")
        print(f"  Frames with objects: {self._frames_with_objects} ({100 * self._frames_with_objects / max(total, 1):.1f}%)")
        print(f"  Empty frames       : {empty} ({100 * empty / max(total, 1):.1f}%)")
        print(f"  Total annotations  : {self._total_annotations}")
        print(f"  Avg objects/frame  : {self._total_annotations / max(total, 1):.1f}")
        print("=" * 60)
        print()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Optional[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        seq_idx, frame_idx = self.samples[index]
        seq_id, frames = self.sequence_map[seq_idx]
        frame_info = self.frame_info_map[seq_idx]

        # Read image
        try:
            img_np = frames[frame_idx].read()
            if img_np is None:
                return None
        except Exception:
            return None

        img = Image.fromarray(img_np)

        # Get annotations (empty target if no annotations for this frame)
        info = frame_info.get(frame_idx, {})
        raw_anns = info.get("annotations", [])
        target = format_detection_annotations(raw_anns)

        # Apply detection augmentation (on PIL image + tensor target)
        if self.augmentation is not None:
            img, target = self.augmentation(img, target)

        # Apply image transform (e.g. ToTensor)
        if self.transform is not None:
            img = self.transform(img)

        return img, target


def detection_collate(
    batch: List[Optional[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]],
) -> Optional[Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]]:
    """
    Collate function for detection datasets.

    Filters out None samples and returns (images, targets) as lists,
    which is the format expected by torchvision detection models.

    Returns None if all samples failed.
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return images, targets


# =============================================================================
# StreamingDataset (Online)
# =============================================================================

class StreamingDataset:
    """
    Streaming dataset that processes ZOD sequences in strict temporal order.

    Unlike the offline datasets (ZODClassificationDataset / ZODDetectionDataset) which
    shuffle frames and use a DataLoader, this dataset yields one frame at a
    time in temporal order. The learner (or filter policy) decides for each
    frame whether to train, store, or skip.

    Supports two tasks:
    - ``"classification"``: Binary labels (has target category or not).
    - ``"detection"``: Bounding boxes and class labels for all categories.

    Args:
        dataset_root: Path to ZOD dataset root.
        annotations_dir: Path to directory with per-sequence annotation JSONs.
        version: ZOD version, ``"full"`` or ``"mini"``.
        split: ``"train"`` or ``"val"``.
        transform: Image transform applied after augmentation (e.g. ToTensor).
        target_category: Category ID (0=person, 1=car, 2=traffic_light).
            Only used for classification; ignored for detection.
        min_score: Minimum pseudo-label score to include a detection.
        min_box_area: Minimum box area (w*h) to include. Only used for
            detection; ignored for classification.
        subsample_steps: Use every Nth frame (1 = all frames).
        task: ``"classification"`` or ``"detection"``.
        augmentation: Optional :class:`DetectionAugmentation` applied to
            (PIL image, target dict) pairs before ``transform``.
            Only used when ``task="detection"``.
        verbose: Print dataset statistics after loading.

    Usage::

        dataset = StreamingDataset(...)
        for stream_item in dataset:
            action = policy.select_action(stream_item)
            if action == "train":
                update(model, stream_item)
    """

    def __init__(
        self,
        dataset_root: str | Path,
        annotations_dir: str | Path,
        version: Literal["full", "mini"] = "full",
        split: Literal["train", "val"] = "train",
        transform: Optional[Callable] = None,
        target_category: int = 0,
        min_score: float = 0.0,
        min_box_area: float = 0.0,
        subsample_steps: int = 1,
        task: Literal["classification", "detection"] = "classification",
        augmentation: Optional[DetectionAugmentation] = None,
        verbose: bool = True,
    ):
        self.dataset_root = Path(dataset_root)
        self.annotations_dir = Path(annotations_dir)
        self.version = version
        self.split = split
        self.transform = transform
        self.target_category = target_category
        self.min_score = min_score
        self.min_box_area = min_box_area
        self.subsample_steps = subsample_steps
        self.task = task
        self.augmentation = augmentation

        # Load ZOD sequences
        zod_sequences = ZodSequences(str(self.dataset_root), version)

        # Get sequences for this split
        if split == "train":
            sequence_ids = zod_sequences.get_split(constants.TRAIN)
        else:  # val
            sequence_ids = zod_sequences.get_split(constants.VAL)

        self.sequences = [zod_sequences[seq_id] for seq_id in sequence_ids]

        # Build sequence metadata
        self.sequence_metadata: List[Dict[str, Any]] = []
        self.total_frames = 0

        for seq_idx, sequence in enumerate(self.sequences):
            seq_id = sequence.info.id
            frames = sequence.info.get_camera_frames()

            # Load annotations (dispatches based on self.task)
            ann_path = self.annotations_dir / f"{seq_id}.json"
            frame_info = self._load_frame_info(ann_path)

            # Count subsampled frames
            num_frames = len(range(0, len(frames), self.subsample_steps))
            self.total_frames += num_frames

            self.sequence_metadata.append({
                "seq_idx": seq_idx,
                "seq_id": seq_id,
                "frames": frames,
                "frame_info": frame_info,
                "num_frames": len(frames),
                "num_subsampled_frames": num_frames,
            })

        if verbose:
            self._print_summary()

    def _load_frame_info(self, ann_path: Path) -> Dict[int, Dict[str, Any]]:
        """Load per-frame info (dispatches based on task)."""
        if self.task == "detection":
            return load_detection_frame_info(
                ann_path, self.min_score, self.min_box_area,
            )
        return load_classification_frame_info(ann_path, self.target_category, self.min_score)

    def _read_frame_image(self, frames: Any, frame_idx: int) -> Optional[Image.Image]:
        """Read a single frame and return as PIL Image, or None on failure."""
        try:
            img_np = frames[frame_idx].read()
            if img_np is None:
                return None
        except Exception:
            return None
        return Image.fromarray(img_np)

    def _build_stream_item(
        self,
        img: Image.Image,
        info: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> StreamItem:
        """
        Build a StreamItem from a PIL image and its annotation info.

        Handles detection annotation formatting, augmentation, and the
        image transform so that all frame-processing logic lives in one
        place (used by both ``__iter__`` and ``get_sequence_iterator``).
        """
        target = 1.0 if info.get("has_target", False) else 0.0
        teacher_score = info.get("max_score", 0.0)

        annotations = None
        if self.task == "detection":
            raw_anns = info.get("annotations", [])
            annotations = format_detection_annotations(raw_anns)
            if self.augmentation is not None:
                img, annotations = self.augmentation(img, annotations)

        if self.transform is not None:
            img = self.transform(img)

        return StreamItem(
            image=img,
            target=target,
            teacher_score=teacher_score,
            metadata=metadata,
            annotations=annotations,
        )

    def _print_summary(self) -> None:
        """Print streaming dataset statistics."""
        num_positive = 0
        total_annotations = 0

        for seq_meta in self.sequence_metadata:
            frame_info = seq_meta["frame_info"]
            for frame_idx in range(0, seq_meta["num_frames"], self.subsample_steps):
                info = frame_info.get(frame_idx, {"has_target": False})
                if info["has_target"]:
                    num_positive += 1
                if self.task == "detection":
                    total_annotations += len(info.get("annotations", []))

        num_negative = self.total_frames - num_positive

        print()
        print("=" * 60)
        print("StreamingDataset Summary")
        print("=" * 60)
        print(f"  Dataset root     : {self.dataset_root}")
        print(f"  Annotations dir  : {self.annotations_dir}")
        print(f"  Split            : {self.split}")
        print(f"  Task             : {self.task}")
        if self.task == "classification":
            cat_name = CATEGORY_ID_TO_NAME.get(self.target_category, str(self.target_category))
            print(f"  Target category  : {self.target_category} ({cat_name})")
        else:
            cats = ", ".join(f"{v} ({k})" for k, v in CATEGORY_ID_TO_NAME.items())
            print(f"  Categories       : {cats}")
        print(f"  Min score filter : {self.min_score}")
        if self.task == "detection" and self.min_box_area > 0:
            print(f"  Min box area     : {self.min_box_area}")
        print(f"  Subsample steps  : {self.subsample_steps}")
        print("-" * 60)
        print(f"  Total sequences  : {len(self.sequence_metadata)}")
        print(f"  Total frames     : {self.total_frames}")
        if self.task == "detection":
            print(f"  Frames with objects : {num_positive} ({100 * num_positive / max(self.total_frames, 1):.1f}%)")
            print(f"  Empty frames       : {num_negative} ({100 * num_negative / max(self.total_frames, 1):.1f}%)")
            print(f"  Total annotations  : {total_annotations}")
            print(f"  Avg objects/frame  : {total_annotations / max(self.total_frames, 1):.1f}")
        else:
            print(f"  Positive frames  : {num_positive} ({100 * num_positive / max(self.total_frames, 1):.1f}%)")
            print(f"  Negative frames  : {num_negative} ({100 * num_negative / max(self.total_frames, 1):.1f}%)")
        print(f"  Stream order     : temporal (sequence-by-sequence)")
        print("=" * 60)
        print()

    def __len__(self) -> int:
        """Return total number of stream items."""
        return self.total_frames

    def __iter__(self) -> Iterator[StreamItem]:
        """
        Iterate over all stream items in strict temporal order.

        Yields one :class:`StreamItem` per (subsampled) frame, processing
        sequences one after another. Frames that fail to load are silently
        skipped (the global index still increments to keep provenance
        consistent).
        """
        global_idx = 0

        for seq_meta in self.sequence_metadata:
            seq_id = seq_meta["seq_id"]
            seq_idx = seq_meta["seq_idx"]
            frames = seq_meta["frames"]
            frame_info = seq_meta["frame_info"]

            for frame_idx in range(0, seq_meta["num_frames"], self.subsample_steps):
                info = frame_info.get(frame_idx, {"has_target": False, "max_score": 0.0})

                img = self._read_frame_image(frames, frame_idx)
                if img is None:
                    global_idx += 1
                    continue

                metadata = {
                    "global_idx": global_idx,
                    "seq_idx": seq_idx,
                    "seq_id": seq_id,
                    "frame_idx": frame_idx,
                }

                yield self._build_stream_item(img, info, metadata)
                global_idx += 1

    def get_sequence_iterator(self, seq_idx: int) -> Iterator[StreamItem]:
        """
        Iterate over a single sequence (useful for federated simulation).

        Each simulated client receives its own subset of sequences and
        streams through them independently via this method.

        Args:
            seq_idx: Index of the sequence to iterate over.

        Yields:
            :class:`StreamItem` for each (subsampled) frame in the sequence.

        Raises:
            ValueError: If *seq_idx* is out of range.
        """
        if seq_idx < 0 or seq_idx >= len(self.sequence_metadata):
            raise ValueError(
                f"seq_idx {seq_idx} out of range for dataset with "
                f"{len(self.sequence_metadata)} sequences"
            )

        seq_meta = self.sequence_metadata[seq_idx]
        seq_id = seq_meta["seq_id"]
        frames = seq_meta["frames"]
        frame_info = seq_meta["frame_info"]

        for frame_idx in range(0, seq_meta["num_frames"], self.subsample_steps):
            info = frame_info.get(frame_idx, {"has_target": False, "max_score": 0.0})

            img = self._read_frame_image(frames, frame_idx)
            if img is None:
                continue

            metadata = {
                "seq_idx": seq_idx,
                "seq_id": seq_id,
                "frame_idx": frame_idx,
            }

            yield self._build_stream_item(img, info, metadata)


# =============================================================================
# Collate function that drops None samples
# =============================================================================

def classification_collate(batch: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """
    Collate function that filters out None samples (failed reads).

    Returns None if all samples are None.
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    return {
        "image": torch.stack([b["image"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "score": torch.tensor([b["score"] for b in batch], dtype=torch.float32),
        "metadata": [b["metadata"] for b in batch],
    }
