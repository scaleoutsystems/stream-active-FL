"""
Dataset implementations for offline and streaming learning.

Provides two dataset types:
- ZODFrameDataset: Offline frame-level dataset with batch sampling (for PyTorch DataLoader)
- StreamingDataset: Temporal sequence dataset for online learning (iterator-based)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
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
# Default image transforms
# =============================================================================

def get_default_transforms(
    image_size: Tuple[int, int] = (224, 224),
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns (train_transform, val_transform).
    
    Both transforms are deterministic (no augmentation).
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


# =============================================================================
# ZODFrameDataset (Offline)
# =============================================================================

class ZODFrameDataset(Dataset):
    """
    PyTorch Dataset for frame-level binary classification on ZOD sequences.

    Each sample is a single frame. The binary target indicates whether the frame
    contains at least one detection of `target_category` (with score >= min_score).

    Args:
        dataset_root: Path to ZOD dataset (original or ZODCropped).
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

    def _load_frame_info(
        self, ann_path: Path
    ) -> Dict[int, Dict[str, Any]]:
        """
        Load annotation JSON and extract per-frame info for the target category.

        Returns:
            Dict mapping frame_idx -> {
                "has_target": bool,
                "max_score": float (max detection score, 0 if none),
            }
        """
        result: Dict[int, Dict[str, Any]] = {}

        if not ann_path.exists():
            return result

        try:
            with ann_path.open("r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: could not read {ann_path}: {e}")
            return result

        # Group detections by frame_idx
        frame_detections: Dict[int, List[float]] = defaultdict(list)
        for ann in data.get("annotations", []):
            if ann.get("category_id") != self.target_category:
                continue
            score = float(ann.get("score", 1.0))
            if score < self.min_score:
                continue
            frame_detections[ann["frame_idx"]].append(score)

        # Build result dict
        for frame_idx, scores in frame_detections.items():
            result[frame_idx] = {
                "has_target": True,
                "max_score": max(scores),
            }

        return result

    def _print_summary(self) -> None:
        """Print dataset statistics."""
        num_positive = sum(1 for _, _, t, _ in self.samples if t == 1.0)
        num_negative = len(self.samples) - num_positive
        total = len(self.samples)

        cat_name = CATEGORY_ID_TO_NAME.get(self.target_category, str(self.target_category))

        print()
        print("=" * 60)
        print("ZODFrameDataset Summary")
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
# StreamingDataset (Online)
# =============================================================================

class StreamingDataset:
    """
    Streaming dataset that processes ZOD sequences in temporal order.

    Unlike the standard ZODFrameDataset which shuffles all frames and uses batch DataLoader,
    this dataset is designed for streaming: sequences are processed strictly in temporal order,
    and the learner decides for each frame whether to train, store, or skip.

    Args:
        dataset_root: Path to ZOD dataset (original or ZODCropped).
        annotations_dir: Path to directory with per-sequence annotation JSONs.
        version: ZOD version, "full" or "mini".
        split: "train" or "val". Unlike offline training, we don't mix splits.
        transform: Torchvision transform to apply to images.
        target_category: Category ID (0=person, 1=car, 2=traffic_light).
        min_score: Minimum detection score to count as a valid detection.
        subsample_steps: Use every Nth frame (1 = all frames).
        verbose: Print dataset statistics.

    Usage:
        dataset = StreamingDataset(...)
        for stream_item in dataset:
            # Decide: train, store, or skip
            action = policy.select_action(stream_item)
            if action == "train":
                model.update(stream_item)
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

            # Load annotations
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
        """Load per-frame annotation info for target category."""
        result: Dict[int, Dict[str, Any]] = {}

        if not ann_path.exists():
            return result

        try:
            with ann_path.open("r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: could not read {ann_path}: {e}")
            return result

        # Group detections by frame_idx
        frame_detections: Dict[int, List[float]] = defaultdict(list)
        for ann in data.get("annotations", []):
            if ann.get("category_id") != self.target_category:
                continue
            score = float(ann.get("score", 1.0))
            if score < self.min_score:
                continue
            frame_detections[ann["frame_idx"]].append(score)

        # Build result dict
        for frame_idx, scores in frame_detections.items():
            result[frame_idx] = {
                "has_target": True,
                "max_score": max(scores),
            }

        return result

    def _print_summary(self) -> None:
        """Print streaming dataset statistics."""
        cat_name = CATEGORY_ID_TO_NAME.get(self.target_category, str(self.target_category))

        # Count positive frames
        num_positive = 0
        for seq_meta in self.sequence_metadata:
            frame_info = seq_meta["frame_info"]
            for frame_idx in range(0, seq_meta["num_frames"], self.subsample_steps):
                info = frame_info.get(frame_idx, {"has_target": False})
                if info["has_target"]:
                    num_positive += 1

        num_negative = self.total_frames - num_positive

        print()
        print("=" * 60)
        print("StreamingDataset Summary")
        print("=" * 60)
        print(f"  Dataset root     : {self.dataset_root}")
        print(f"  Annotations dir  : {self.annotations_dir}")
        print(f"  Split            : {self.split}")
        print(f"  Target category  : {self.target_category} ({cat_name})")
        print(f"  Min score filter : {self.min_score}")
        print(f"  Subsample steps  : {self.subsample_steps}")
        print("-" * 60)
        print(f"  Total sequences  : {len(self.sequence_metadata)}")
        print(f"  Total frames     : {self.total_frames}")
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
        Iterate over stream items in strict temporal order.

        Yields:
            StreamItem with image, target, teacher_score, and metadata.
        """
        global_idx = 0

        for seq_meta in self.sequence_metadata:
            seq_id = seq_meta["seq_id"]
            seq_idx = seq_meta["seq_idx"]
            frames = seq_meta["frames"]
            frame_info = seq_meta["frame_info"]

            for frame_idx in range(0, seq_meta["num_frames"], self.subsample_steps):
                # Get frame info
                info = frame_info.get(frame_idx, {"has_target": False, "max_score": 0.0})
                target = 1.0 if info["has_target"] else 0.0
                teacher_score = info["max_score"]

                # Read image
                try:
                    img_np = frames[frame_idx].read()
                    if img_np is None:
                        global_idx += 1
                        continue
                except Exception:
                    global_idx += 1
                    continue

                img = Image.fromarray(img_np)

                # Apply transform
                if self.transform is not None:
                    img = self.transform(img)

                # Create stream item
                metadata = {
                    "global_idx": global_idx,
                    "seq_idx": seq_idx,
                    "seq_id": seq_id,
                    "frame_idx": frame_idx,
                }

                yield StreamItem(
                    image=img,
                    target=target,
                    teacher_score=teacher_score,
                    metadata=metadata,
                )

                global_idx += 1

    def get_sequence_iterator(self, seq_idx: int) -> Iterator[StreamItem]:
        """
        Get an iterator for a specific sequence (useful for federated setting).

        Args:
            seq_idx: Index of sequence to iterate over.

        Yields:
            StreamItem for each frame in the sequence.
        """
        if seq_idx < 0 or seq_idx >= len(self.sequence_metadata):
            raise ValueError(f"Invalid seq_idx {seq_idx}, dataset has {len(self.sequence_metadata)} sequences")

        seq_meta = self.sequence_metadata[seq_idx]
        seq_id = seq_meta["seq_id"]
        frames = seq_meta["frames"]
        frame_info = seq_meta["frame_info"]

        for frame_idx in range(0, seq_meta["num_frames"], self.subsample_steps):
            info = frame_info.get(frame_idx, {"has_target": False, "max_score": 0.0})
            target = 1.0 if info["has_target"] else 0.0
            teacher_score = info["max_score"]

            try:
                img_np = frames[frame_idx].read()
                if img_np is None:
                    continue
            except Exception:
                continue

            img = Image.fromarray(img_np)

            if self.transform is not None:
                img = self.transform(img)

            metadata = {
                "seq_idx": seq_idx,
                "seq_id": seq_id,
                "frame_idx": frame_idx,
            }

            yield StreamItem(
                image=img,
                target=target,
                teacher_score=teacher_score,
                metadata=metadata,
            )


# =============================================================================
# Collate function that drops None samples
# =============================================================================

def collate_drop_none(batch: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
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
