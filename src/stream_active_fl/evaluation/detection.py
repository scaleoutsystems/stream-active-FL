"""
Evaluation for streaming 2D object detection.

Computes COCO-style mean Average Precision (mAP) at multiple IoU thresholds
using torchvision's box_iou and PASCAL VOC-style all-point interpolation.

Primary metrics reported:
- mAP:    COCO primary -- averaged over IoU thresholds [0.5 : 0.05 : 0.95]
- mAP_50: AP at IoU = 0.5
- mAP_75: AP at IoU = 0.75
- Per-class AP (averaged over all IoU thresholds)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torchvision.ops import box_iou

from ..core import CATEGORY_ID_TO_NAME, ClassMapping, DetectionStream


# Default label-to-name for all ZOD classes (model labels are +1 from annotation IDs)
DETECTION_LABEL_TO_NAME: Dict[int, str] = {
    k + 1: v for k, v in CATEGORY_ID_TO_NAME.items()
}

COCO_IOU_THRESHOLDS: List[float] = [
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
]


def evaluate_detection(
    model: nn.Module,
    val_stream: DetectionStream,
    device: torch.device,
    score_threshold: float = 0.3,
    iou_thresholds: Sequence[float] = COCO_IOU_THRESHOLDS,
    class_mapping: Optional[ClassMapping] = None,
) -> Dict[str, float]:
    """
    Evaluate detection model on a validation stream with COCO-style metrics.

    Args:
        model: Detection model (returns predictions in eval mode).
        val_stream: DetectionStream with validation data.
        device: Device to run evaluation on.
        score_threshold: Minimum score to consider a prediction.
        iou_thresholds: IoU thresholds for AP computation.
        class_mapping: Optional ClassMapping for per-class AP reporting.
            When None, uses the stream's class_mapping if available,
            otherwise falls back to all ZOD classes.

    Returns:
        Dict with mAP, mAP_50, mAP_75, per-class APs, and counts.
    """
    model.eval()

    all_pred_boxes: List[torch.Tensor] = []
    all_pred_scores: List[torch.Tensor] = []
    all_pred_labels: List[torch.Tensor] = []
    all_gt_boxes: List[torch.Tensor] = []
    all_gt_labels: List[torch.Tensor] = []

    num_items = 0

    with torch.no_grad():
        for stream_item in val_stream:
            image = stream_item.image.to(device)
            predictions = model([image])
            pred = predictions[0]

            keep = pred["scores"] >= score_threshold
            all_pred_boxes.append(pred["boxes"][keep].cpu())
            all_pred_scores.append(pred["scores"][keep].cpu())
            all_pred_labels.append(pred["labels"][keep].cpu())

            gt_boxes = stream_item.annotations["boxes"]
            gt_labels = stream_item.annotations["labels"]
            all_gt_boxes.append(gt_boxes)
            all_gt_labels.append(gt_labels)

            num_items += 1

    gt_classes = set()
    for labels in all_gt_labels:
        gt_classes.update(labels.tolist())

    ap_matrix: Dict[float, Dict[int, float]] = {}
    for iou_thresh in iou_thresholds:
        ap_matrix[iou_thresh] = _compute_per_class_ap(
            all_pred_boxes, all_pred_scores, all_pred_labels,
            all_gt_boxes, all_gt_labels,
            iou_threshold=iou_thresh,
        )

    iou_set = set(iou_thresholds)

    if gt_classes:
        all_aps = [
            ap_matrix[t].get(c, 0.0)
            for t in iou_thresholds
            for c in gt_classes
        ]
        mAP = sum(all_aps) / len(all_aps)
        mAP_50 = (
            sum(ap_matrix[0.5].get(c, 0.0) for c in gt_classes) / len(gt_classes)
            if 0.5 in iou_set else 0.0
        )
        mAP_75 = (
            sum(ap_matrix[0.75].get(c, 0.0) for c in gt_classes) / len(gt_classes)
            if 0.75 in iou_set else 0.0
        )
    else:
        mAP = mAP_50 = mAP_75 = 0.0

    total_preds = sum(len(p) for p in all_pred_boxes)
    total_gt = sum(len(g) for g in all_gt_boxes)

    metrics: Dict[str, float] = {
        "mAP": mAP,
        "mAP_50": mAP_50,
        "mAP_75": mAP_75,
        "num_items": float(num_items),
        "total_predictions": float(total_preds),
        "total_ground_truth": float(total_gt),
    }

    label_map = DETECTION_LABEL_TO_NAME
    if class_mapping is not None:
        label_map = class_mapping.label_to_name
    elif hasattr(val_stream, "class_mapping"):
        label_map = val_stream.class_mapping.label_to_name

    for label, name in label_map.items():
        class_aps = [ap_matrix[t].get(label, 0.0) for t in iou_thresholds]
        metrics[f"AP_{name}"] = sum(class_aps) / len(class_aps)

    return metrics


def _compute_per_class_ap(
    pred_boxes_list: List[torch.Tensor],
    pred_scores_list: List[torch.Tensor],
    pred_labels_list: List[torch.Tensor],
    gt_boxes_list: List[torch.Tensor],
    gt_labels_list: List[torch.Tensor],
    iou_threshold: float = 0.5,
) -> Dict[int, float]:
    """Compute per-class Average Precision at the given IoU threshold."""
    gt_classes: set = set()
    for labels in gt_labels_list:
        gt_classes.update(labels.tolist())

    if not gt_classes:
        return {}

    per_class_ap: Dict[int, float] = {}

    for cls in sorted(gt_classes):
        all_scores: List[float] = []
        all_tp: List[int] = []
        total_gt = 0

        for i in range(len(pred_boxes_list)):
            pred_mask = pred_labels_list[i] == cls
            p_boxes = pred_boxes_list[i][pred_mask]
            p_scores = pred_scores_list[i][pred_mask]

            gt_mask = gt_labels_list[i] == cls
            g_boxes = gt_boxes_list[i][gt_mask]

            total_gt += len(g_boxes)

            if len(p_boxes) == 0:
                continue

            sorted_indices = torch.argsort(p_scores, descending=True)
            p_boxes = p_boxes[sorted_indices]
            p_scores = p_scores[sorted_indices]

            if len(g_boxes) == 0:
                all_scores.extend(p_scores.tolist())
                all_tp.extend([0] * len(p_scores))
                continue

            ious = box_iou(p_boxes, g_boxes)
            matched_gt: set = set()
            for j in range(len(p_boxes)):
                max_iou, max_idx = ious[j].max(dim=0)
                if max_iou.item() >= iou_threshold and max_idx.item() not in matched_gt:
                    all_tp.append(1)
                    matched_gt.add(max_idx.item())
                else:
                    all_tp.append(0)
                all_scores.append(p_scores[j].item())

        if total_gt == 0 or not all_scores:
            per_class_ap[cls] = 0.0
            continue

        sorted_indices = sorted(
            range(len(all_scores)), key=lambda k: all_scores[k], reverse=True
        )
        sorted_tp = [all_tp[i] for i in sorted_indices]

        tp_cumsum = torch.cumsum(torch.tensor(sorted_tp, dtype=torch.float32), dim=0)
        fp_cumsum = torch.cumsum(
            torch.tensor([1 - t for t in sorted_tp], dtype=torch.float32), dim=0
        )

        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / total_gt

        recall = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
        precision = torch.cat([torch.tensor([1.0]), precision, torch.tensor([0.0])])

        for k in range(len(precision) - 2, -1, -1):
            precision[k] = max(precision[k].item(), precision[k + 1].item())

        recall_diff = recall[1:] - recall[:-1]
        ap = (recall_diff * precision[1:]).sum().item()
        per_class_ap[cls] = ap

    return per_class_ap
