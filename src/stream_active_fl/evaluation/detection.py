"""
Evaluation for streaming 2D object detection.

Computes COCO-style mean Average Precision (mAP) at multiple IoU thresholds
using torchvision's box_iou and PASCAL VOC-style all-point interpolation.
No external dependencies beyond torchvision.

Primary metrics reported:
- mAP:    COCO primary — averaged over IoU thresholds [0.5 : 0.05 : 0.95]
- mAP_50: AP at IoU = 0.5
- mAP_75: AP at IoU = 0.75
- Per-class AP (averaged over all IoU thresholds)
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
from torchvision.ops import box_iou

from ..core import StreamingDataset


# Category labels used by the model (shifted by +1 from annotation IDs)
DETECTION_LABEL_TO_NAME: Dict[int, str] = {
    1: "person",
    2: "car",
    3: "traffic_light",
}

# Standard COCO IoU thresholds: 0.50, 0.55, 0.60, ..., 0.95
COCO_IOU_THRESHOLDS: List[float] = [
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
]


def evaluate_detection(
    model: nn.Module,
    val_stream: StreamingDataset,
    device: torch.device,
    score_threshold: float = 0.3,
    iou_thresholds: Sequence[float] = COCO_IOU_THRESHOLDS,
) -> Dict[str, float]:
    """
    Evaluate detection model on validation stream with COCO-style metrics.

    Collects predictions and ground truth over the full validation stream,
    then computes per-class AP at each IoU threshold.  Aggregates into the
    standard COCO metric triplet (mAP, mAP_50, mAP_75).

    Args:
        model: Detection model (returns predictions in eval mode).
        val_stream: Streaming dataset with task="detection".
        device: Device to run evaluation on.
        score_threshold: Minimum score to consider a prediction.
        iou_thresholds: IoU thresholds for AP computation.
            Defaults to COCO standard [0.5 : 0.05 : 0.95].

    Returns:
        Dict with COCO-style mAP, mAP_50, mAP_75, per-class APs (averaged
        over all thresholds), and annotation/prediction counts.
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

            # Filter low-confidence predictions
            keep = pred["scores"] >= score_threshold
            pred_boxes = pred["boxes"][keep].cpu()
            pred_scores = pred["scores"][keep].cpu()
            pred_labels = pred["labels"][keep].cpu()

            # Ground truth
            if stream_item.annotations is not None:
                gt_boxes = stream_item.annotations["boxes"]
                gt_labels = stream_item.annotations["labels"]
            else:
                gt_boxes = torch.zeros((0, 4))
                gt_labels = torch.zeros((0,), dtype=torch.int64)

            all_pred_boxes.append(pred_boxes)
            all_pred_scores.append(pred_scores)
            all_pred_labels.append(pred_labels)
            all_gt_boxes.append(gt_boxes)
            all_gt_labels.append(gt_labels)

            num_items += 1

    # Determine which classes appear in ground truth
    gt_classes = set()
    for labels in all_gt_labels:
        gt_classes.update(labels.tolist())

    # --- Compute per-class AP at each IoU threshold -------------------------
    # ap_matrix[iou_thresh][class_label] = AP value
    ap_matrix: Dict[float, Dict[int, float]] = {}
    for iou_thresh in iou_thresholds:
        ap_matrix[iou_thresh] = _compute_per_class_ap(
            all_pred_boxes,
            all_pred_scores,
            all_pred_labels,
            all_gt_boxes,
            all_gt_labels,
            iou_threshold=iou_thresh,
        )

    # --- Aggregate into COCO-style metrics ----------------------------------
    iou_set = set(iou_thresholds)

    if gt_classes:
        # mAP (COCO primary): average over all (class, threshold) pairs
        all_aps = [
            ap_matrix[t].get(c, 0.0)
            for t in iou_thresholds
            for c in gt_classes
        ]
        mAP = sum(all_aps) / len(all_aps)

        # mAP_50: average over classes at IoU = 0.5 (if threshold was evaluated)
        if 0.5 in iou_set:
            mAP_50 = sum(ap_matrix[0.5].get(c, 0.0) for c in gt_classes) / len(gt_classes)
        else:
            mAP_50 = 0.0

        # mAP_75: average over classes at IoU = 0.75 (if threshold was evaluated)
        if 0.75 in iou_set:
            mAP_75 = sum(ap_matrix[0.75].get(c, 0.0) for c in gt_classes) / len(gt_classes)
        else:
            mAP_75 = 0.0
    else:
        mAP = mAP_50 = mAP_75 = 0.0

    # Count total predictions and ground truths
    total_preds = sum(len(p) for p in all_pred_boxes)
    total_gt = sum(len(g) for g in all_gt_boxes)

    # Build result dict
    metrics: Dict[str, float] = {
        "mAP": mAP,
        "mAP_50": mAP_50,
        "mAP_75": mAP_75,
        "num_items": float(num_items),
        "total_predictions": float(total_preds),
        "total_ground_truth": float(total_gt),
    }

    # Per-class AP averaged over all IoU thresholds (COCO-style)
    for label, name in DETECTION_LABEL_TO_NAME.items():
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
    """
    Compute per-class Average Precision at the given IoU threshold.

    Uses PASCAL VOC all-point interpolation method.

    Returns:
        Dict mapping class label -> AP value.
    """
    # Collect all unique class labels from ground truth
    all_gt_labels: set = set()
    for labels in gt_labels_list:
        all_gt_labels.update(labels.tolist())

    if not all_gt_labels:
        return {}

    per_class_ap: Dict[int, float] = {}

    for cls in sorted(all_gt_labels):
        # Gather all predictions and GTs for this class across all images
        all_scores: List[float] = []
        all_tp: List[int] = []
        total_gt = 0

        for i in range(len(pred_boxes_list)):
            # Predictions for this class in this image
            pred_mask = pred_labels_list[i] == cls
            p_boxes = pred_boxes_list[i][pred_mask]
            p_scores = pred_scores_list[i][pred_mask]

            # Ground truths for this class in this image
            gt_mask = gt_labels_list[i] == cls
            g_boxes = gt_boxes_list[i][gt_mask]

            total_gt += len(g_boxes)

            if len(p_boxes) == 0:
                continue

            # Sort predictions by descending score
            sorted_indices = torch.argsort(p_scores, descending=True)
            p_boxes = p_boxes[sorted_indices]
            p_scores = p_scores[sorted_indices]

            if len(g_boxes) == 0:
                # All predictions are false positives
                all_scores.extend(p_scores.tolist())
                all_tp.extend([0] * len(p_scores))
                continue

            # Compute IoU between predictions and ground truths
            ious = box_iou(p_boxes, g_boxes)  # [num_pred, num_gt]

            # Greedy matching: each GT can only be matched once
            matched_gt: set = set()
            for j in range(len(p_boxes)):
                max_iou, max_idx = ious[j].max(dim=0)
                if max_iou.item() >= iou_threshold and max_idx.item() not in matched_gt:
                    all_tp.append(1)
                    matched_gt.add(max_idx.item())
                else:
                    all_tp.append(0)
                all_scores.append(p_scores[j].item())

        if total_gt == 0:
            per_class_ap[cls] = 0.0
            continue

        if not all_scores:
            per_class_ap[cls] = 0.0
            continue

        # Sort all predictions globally by score (descending)
        sorted_indices = sorted(
            range(len(all_scores)), key=lambda k: all_scores[k], reverse=True
        )
        sorted_tp = [all_tp[i] for i in sorted_indices]

        # Compute precision-recall curve
        tp_cumsum = torch.cumsum(torch.tensor(sorted_tp, dtype=torch.float32), dim=0)
        fp_cumsum = torch.cumsum(
            torch.tensor([1 - t for t in sorted_tp], dtype=torch.float32), dim=0
        )

        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / total_gt

        # Add sentinel values for all-point interpolation
        recall = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
        precision = torch.cat([torch.tensor([1.0]), precision, torch.tensor([0.0])])

        # Make precision monotonically decreasing
        for k in range(len(precision) - 2, -1, -1):
            precision[k] = max(precision[k].item(), precision[k + 1].item())

        # Compute area under PR curve
        recall_diff = recall[1:] - recall[:-1]
        ap = (recall_diff * precision[1:]).sum().item()

        per_class_ap[cls] = ap

    return per_class_ap
