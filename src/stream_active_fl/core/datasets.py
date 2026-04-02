"""
Dataset implementations for offline and streaming object detection.

Offline dataset (for PyTorch DataLoader, shuffled, multi-epoch):
    DetectionDataset    Multi-class object detection per frame

Streaming dataset (iterator-based, strict chronological order):
    DetectionStream     Yields StreamItem objects one at a time

Both read preprocessed images and per-frame annotation JSONs produced by
tools/preprocessing/prepare_data.py, indexed by a manifest JSON file.

Also provides the ZOD category mapping, transforms, augmentations, and
collate functions used by the experiment scripts.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Set, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F

from .items import StreamItem


# =============================================================================
# Category definitions (ZOD top-level classes)
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

NUM_CLASSES = len(CATEGORY_NAME_TO_ID) + 1  # +1 for background


# =============================================================================
# Class mapping (for training/evaluating on a subset of classes)
# =============================================================================


@dataclass(frozen=True)
class ClassMapping:
    """Contiguous ID mapping for a (sub)set of object classes.

    When all classes are used, IDs match the original CATEGORY_NAME_TO_ID.
    When a subset is selected, IDs are remapped to 0..N-1 (model labels 1..N).
    """

    names: Tuple[str, ...]
    name_to_id: Dict[str, int]
    id_to_name: Dict[int, str]
    label_to_name: Dict[int, str]
    num_classes: int


def build_class_mapping(target_classes: Optional[List[str]] = None) -> ClassMapping:
    """Build a ClassMapping from a list of class names.

    When target_classes is None, uses all ZOD classes in original order.
    """
    if target_classes is None:
        names = tuple(CATEGORY_NAME_TO_ID.keys())
    else:
        for name in target_classes:
            if name not in CATEGORY_NAME_TO_ID:
                raise ValueError(
                    f"Unknown class: {name!r}. Valid: {list(CATEGORY_NAME_TO_ID)}"
                )
        names = tuple(target_classes)

    name_to_id = {name: i for i, name in enumerate(names)}
    id_to_name = {i: name for i, name in enumerate(names)}
    label_to_name = {i + 1: name for i, name in enumerate(names)}

    return ClassMapping(
        names=names,
        name_to_id=name_to_id,
        id_to_name=id_to_name,
        label_to_name=label_to_name,
        num_classes=len(names) + 1,
    )


# =============================================================================
# Manifest helpers
# =============================================================================


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Load the preprocessing manifest JSON."""
    with manifest_path.open("r") as f:
        return json.load(f)


def _load_annotation(ann_path: Path) -> Dict[str, Any]:
    """Load a per-frame annotation JSON."""
    with ann_path.open("r") as f:
        return json.load(f)


def format_detection_annotations(
    raw_anns: List[Dict[str, Any]],
    class_mapping: Optional[ClassMapping] = None,
) -> Dict[str, torch.Tensor]:
    """
    Convert raw annotation list to torchvision-format detection target.

    Bounding boxes are already in [x1, y1, x2, y2] format from preprocessing.
    Labels are shifted by +1 (torchvision reserves 0 for background).

    When class_mapping is provided, only annotations for target classes are
    kept and labels are remapped to contiguous IDs.
    """
    empty = {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.int64),
    }
    if not raw_anns:
        return empty

    boxes = []
    labels = []
    for ann in raw_anns:
        if class_mapping is not None:
            cat_name = ann.get("category_name", "")
            if cat_name not in class_mapping.name_to_id:
                continue
            label = class_mapping.name_to_id[cat_name] + 1
        else:
            label = ann["category_id"] + 1
        boxes.append(ann["bbox_xyxy"])
        labels.append(label)

    if not boxes:
        return empty

    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


def filter_small_boxes(
    target: Dict[str, torch.Tensor],
    min_box_area: float,
) -> Dict[str, torch.Tensor]:
    """Drop boxes smaller than min_box_area from a detection target."""
    if min_box_area <= 0:
        return target

    boxes = target["boxes"]
    labels = target["labels"]
    if boxes.numel() == 0:
        return target

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    areas = widths * heights

    keep = (widths > 0) & (heights > 0) & (areas >= min_box_area)
    return {
        "boxes": boxes[keep],
        "labels": labels[keep],
    }


# =============================================================================
# Default image transforms
# =============================================================================


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
        if self.color_jitter_transform is not None:
            image = self.color_jitter_transform(image)

        if random.random() < self.hflip_prob:
            image = F.hflip(image)  # type: ignore[arg-type]  # accepts PIL at runtime
            boxes = target["boxes"]
            if len(boxes) > 0:
                width = image.width
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
# DetectionDataset (Offline — bootstrap training and validation)
# =============================================================================


class DetectionDataset(Dataset):
    """
    PyTorch Dataset for offline multi-class object detection.

    Reads preprocessed images and annotation JSONs from disk, indexed by
    a manifest file produced by prepare_data.py.  Suitable for DataLoader
    with shuffle for multi-epoch bootstrap training or validation.

    Args:
        manifest_path: Path to the preprocessing manifest JSON.
        split: "train" or "val".
        transform: Torchvision transform applied to images (e.g. ToTensor).
        augmentation: Optional DetectionAugmentation applied to
            (PIL image, target dict) before the image transform.
        min_box_area: Minimum box area to keep (post-scaling filter already
            applied during preprocessing; set 0 to disable further filtering).
        frame_range: Optional (start, end) index range into the
            chronologically-sorted manifest to select a subset of frames
            (e.g. first 1000 for bootstrap).
        target_classes: Optional list of class names to train on. When set,
            only annotations for these classes are kept and labels are remapped
            to contiguous IDs. None means all classes.
        verbose: Print dataset statistics after loading.

    Item format:
        (image, target) where:
        - image: Tensor (3, H, W) in [0, 1]
        - target: {"boxes": FloatTensor[N, 4] xyxy, "labels": Int64Tensor[N]}
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: Literal["train", "val"] = "train",
        transform: Optional[Callable] = None,
        augmentation: Optional[DetectionAugmentation] = None,
        min_box_area: float = 0.0,
        frame_range: Optional[Tuple[int, int]] = None,
        target_classes: Optional[List[str]] = None,
        verbose: bool = True,
    ):
        self.manifest_path = Path(manifest_path)
        self.base_dir = self.manifest_path.parent
        self.split = split
        self.transform = transform
        self.augmentation = augmentation
        self.min_box_area = min_box_area
        self.class_mapping = build_class_mapping(target_classes)

        manifest = load_manifest(self.manifest_path)
        all_frames = manifest["frames"]

        split_frames = [f for f in all_frames if f["split"] == split]

        if frame_range is not None:
            start, end = frame_range
            split_frames = split_frames[start:end]

        self.frames = split_frames

        if verbose:
            self._print_summary()

    def _print_summary(self) -> None:
        total = len(self.frames)
        with_objects = sum(1 for f in self.frames if f["num_objects"] > 0)
        total_anns = sum(f["num_objects"] for f in self.frames)
        classes_str = ", ".join(self.class_mapping.names)

        print()
        print("=" * 60)
        print("DetectionDataset Summary")
        print("=" * 60)
        print(f"  Manifest         : {self.manifest_path}")
        print(f"  Split            : {self.split}")
        print(f"  Classes ({len(self.class_mapping.names)}): {classes_str}")
        print("-" * 60)
        print(f"  Total frames     : {total}")
        print(f"  Frames with objs : {with_objects} ({100 * with_objects / max(total, 1):.1f}%)")
        print(f"  Total annotations: {total_anns} (before class filter)")
        print(f"  Avg objects/frame: {total_anns / max(total, 1):.1f} (before class filter)")
        print("=" * 60)
        print()

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> Optional[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        frame_entry = self.frames[index]

        # Read image
        img_path = self.base_dir / frame_entry["image_path"]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            return None

        # Load annotation
        ann_path = self.base_dir / frame_entry["annotation_path"]
        try:
            ann_data = _load_annotation(ann_path)
        except Exception:
            ann_data = {"annotations": []}

        target = format_detection_annotations(
            ann_data.get("annotations", []), self.class_mapping,
        )

        if self.augmentation is not None:
            img, target = self.augmentation(img, target)

        # Optional training-time filtering for very small boxes
        target = filter_small_boxes(target, self.min_box_area)

        # Apply image transform (PIL to Tensor)
        if self.transform is not None:
            img = self.transform(img)

        assert isinstance(img, torch.Tensor)
        return img, target

    def get_frame_entry(self, index: int) -> Dict[str, Any]:
        """Return the raw manifest entry for a frame (for debugging)."""
        return self.frames[index]


# =============================================================================
# DetectionStream (Streaming — chronological single-pass)
# =============================================================================


class DetectionStream:
    """
    Streaming dataset that yields frames in strict chronological order.

    Unlike DetectionDataset which supports shuffled DataLoader access,
    this class is an iterator that produces one StreamItem at a time in the
    order they appear in the manifest (sorted by frame_id / timestamp).

    Args:
        manifest_path: Path to the preprocessing manifest JSON.
        split: "train" or "val".
        transform: Image transform (e.g. ToTensor).
        augmentation: Optional DetectionAugmentation.
        min_box_area: Minimum box area to keep. Set 0 to disable.
        frame_range: Optional (start, end) index range to select a subset.
        target_classes: Optional list of class names. None means all classes.
        verbose: Print dataset statistics after loading.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: Literal["train", "val"] = "train",
        transform: Optional[Callable] = None,
        augmentation: Optional[DetectionAugmentation] = None,
        min_box_area: float = 0.0,
        frame_range: Optional[Tuple[int, int]] = None,
        target_classes: Optional[List[str]] = None,
        verbose: bool = True,
    ):
        self.manifest_path = Path(manifest_path)
        self.base_dir = self.manifest_path.parent
        self.split = split
        self.transform = transform
        self.augmentation = augmentation
        self.min_box_area = min_box_area
        self.class_mapping = build_class_mapping(target_classes)

        manifest = load_manifest(self.manifest_path)
        all_frames = manifest["frames"]

        split_frames = [f for f in all_frames if f["split"] == split]

        if frame_range is not None:
            start, end = frame_range
            split_frames = split_frames[start:end]

        self.frames = split_frames

        if verbose:
            self._print_summary()

    def _print_summary(self) -> None:
        total = len(self.frames)
        with_objects = sum(1 for f in self.frames if f["num_objects"] > 0)
        classes_str = ", ".join(self.class_mapping.names)

        print()
        print("=" * 60)
        print("DetectionStream Summary")
        print("=" * 60)
        print(f"  Manifest     : {self.manifest_path}")
        print(f"  Split        : {self.split}")
        print(f"  Classes ({len(self.class_mapping.names)}): {classes_str}")
        print(f"  Total frames : {total}")
        print(f"  With objects : {with_objects} ({100 * with_objects / max(total, 1):.1f}%)")
        print(f"  Stream order : chronological (by frame_id)")
        print("=" * 60)
        print()

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[StreamItem]:
        for global_idx, frame_entry in enumerate(self.frames):
            item = self._load_item(frame_entry, global_idx)
            if item is not None:
                yield item

    def _load_item(self, frame_entry: Dict[str, Any], global_idx: int) -> Optional[StreamItem]:
        """Load a single frame as a StreamItem."""
        img_path = self.base_dir / frame_entry["image_path"]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            return None

        ann_path = self.base_dir / frame_entry["annotation_path"]
        try:
            ann_data = _load_annotation(ann_path)
        except Exception:
            ann_data = {"annotations": [], "categories_present": []}

        annotations = format_detection_annotations(
            ann_data.get("annotations", []), self.class_mapping,
        )
        annotations = filter_small_boxes(annotations, self.min_box_area)
        label_map = self.class_mapping.label_to_name
        categories: Set[str] = {
            label_map[int(label.item())]
            for label in annotations["labels"]
            if int(label.item()) in label_map
        }

        if self.augmentation is not None:
            img, annotations = self.augmentation(img, annotations)

        if self.transform is not None:
            img = self.transform(img)

        assert isinstance(img, torch.Tensor)

        metadata = {
            "global_idx": global_idx,
            "frame_id": frame_entry["frame_id"],
        }

        return StreamItem(
            image=img,
            annotations=annotations,
            categories=categories,
            metadata=metadata,
        )


# =============================================================================
# Collate function for detection datasets
# =============================================================================


def detection_collate(
    batch: List[Optional[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]],
) -> Optional[Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]]:
    """
    Collate function for detection datasets.

    Filters out None samples and returns (images, targets) as lists,
    which is the format expected by torchvision detection models.
    """
    valid: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]] = [
        b for b in batch if b is not None
    ]
    if len(valid) == 0:
        return None

    images = [b[0] for b in valid]
    targets = [b[1] for b in valid]
    return images, targets
