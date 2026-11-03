"""Second-stage NMS variants for torchvision R-CNN RoI heads (Faster R-CNN, Lite R-CNN v2/v3).

``hard``    torchvision ``batched_nms``: the stock ``RoIHeads.postprocess_detections``.
``soft``    exact greedy Gaussian Soft-NMS per class (Bodla et al., ICCV 2017). Every class takes
            its next box in the same step, so the Python loop runs once per rank within a class
            instead of once per kept box, and it stops once the top ``detections_per_img`` are final.
``matrix``  Matrix NMS (Wang et al., SOLOv2, NeurIPS 2020) with the same Gaussian kernel: one
            vectorized decay of the top-scoring candidates, no Python loop. A fast approximation
            of ``soft``: a box decays by its single worst overlap instead of by every better box,
            so more mid-score duplicates survive.

Soft and Matrix NMS run on the candidates above the head's RoI score floor (``score_thresh``) and
keep the boxes, lowering the scores of overlapping same-class boxes. Soft-NMS drops a box once its
score decays to ``min_score`` (0.001) or below, so decayed duplicates under the RoI floor stay
ranked; Matrix NMS drops them at the RoI floor itself (SOLOv2 also filters after the decay at 0.05).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import boxes as box_ops

NMS_METHODS = ("hard", "soft", "matrix")
# Soft-NMS drop floor for decayed scores: the default of Bodla et al.'s code and mmcv's soft_nms,
# and the setting the investigation measured (pre-filter 0.05, prune at or below 0.001).
SOFT_NMS_MIN_SCORE = 0.001
# Candidates entering Matrix NMS's K x K decay matrix (SOLOv2 keeps 500, RetinaNet's nms_pre 1000).
MATRIX_NMS_TOP_K = 1000


def _empty(scores: Tensor) -> tuple[Tensor, Tensor]:
    return torch.empty(0, dtype=torch.int64, device=scores.device), scores.new_empty(0)


def _best_first(index: Tensor, scores: Tensor, max_detections: int) -> tuple[Tensor, Tensor]:
    scores, order = scores.sort(descending=True, stable=True)
    return index[order][:max_detections], scores[:max_detections]


def _iou_against(best: Tensor, best_area: Tensor, boxes: Tensor, area: Tensor) -> Tensor:
    """IoU of best[c] with every boxes[c, j], in box_ops.box_iou's exact arithmetic order."""
    lt = torch.max(best[..., None, :2], boxes[..., :2])
    rb = torch.min(best[..., None, 2:], boxes[..., 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (best_area[..., None] + area - inter)


def soft_nms(boxes: Tensor, scores: Tensor, labels: Tensor, sigma: float = 0.5,
             min_score: float = SOFT_NMS_MIN_SCORE,
             max_detections: int = 100) -> tuple[Tensor, Tensor]:
    """Exact greedy Gaussian Soft-NMS within each label.

    Per label: keep the highest score, multiply the other scores by exp(-IoU^2 / sigma), drop the
    ones at or below ``min_score``, repeat. Candidates are expected above the RoI score floor
    already (PostprocessDetections keeps torchvision's pre-filter); one at or below ``min_score``
    is never kept. Returns the kept indices and their decayed scores, best first, at most
    ``max_detections``.
    """
    if scores.numel() == 0 or max_detections <= 0:
        return _empty(scores)
    device = scores.device
    _, group, counts = torch.unique(labels, return_inverse=True, return_counts=True)
    # [classes, M] grid of candidate indices, input order within a class (argmax tie order).
    order = torch.argsort(group, stable=True)
    starts = torch.cumsum(counts, 0) - counts
    rank = torch.arange(order.numel(), device=device) - starts[group[order]]
    index = torch.full((counts.numel(), int(counts.max())), -1, dtype=torch.int64, device=device)
    index[group[order], rank] = order
    safe = index.clamp(min=0)
    grid = boxes[safe]
    area = (grid[..., 2] - grid[..., 0]) * (grid[..., 3] - grid[..., 1])
    current = scores[safe].masked_fill(index < 0, float("-inf"))
    rows = torch.arange(index.shape[0], device=device)
    picked_index, picked_score = [], []
    kept = 0
    while True:
        best_score, best = current.max(dim=1)
        active = best_score > min_score
        count = int(active.sum())
        if count == 0:
            break
        picked_index.append(index[rows, best])
        picked_score.append(best_score.masked_fill(~active, float("-inf")))
        iou = _iou_against(grid[rows, best], area[rows, best], grid, area)
        current = current * torch.exp(-(iou * iou) / sigma)
        current[rows, best] = float("-inf")
        current = current.masked_fill(~(current > min_score), float("-inf"))
        kept += count
        # Later picks score at most the best remaining score, so the top list is final once it
        # holds max_detections picks above that.
        if kept >= max_detections:
            remaining = current.max()
            if int((torch.cat(picked_score) > remaining).sum()) >= max_detections:
                break
    picked_score = torch.cat(picked_score)
    valid = picked_score > float("-inf")
    return _best_first(torch.cat(picked_index)[valid], picked_score[valid], max_detections)


def matrix_nms(boxes: Tensor, scores: Tensor, labels: Tensor, sigma: float = 0.5,
               score_thresh: float = 0.05, max_detections: int = 100,
               top_k: int = MATRIX_NMS_TOP_K) -> tuple[Tensor, Tensor]:
    """Matrix NMS with the Gaussian kernel exp(-IoU^2 / sigma) over the top ``top_k`` scores.

    Each box is decayed by its worst overlap with a higher-scoring box of the same label, with
    that box's own decay compensated (a box that was itself suppressed suppresses less). The top
    box keeps its score. Returns the kept indices and decayed scores, best first.
    """
    if scores.numel() == 0 or max_detections <= 0:
        return _empty(scores)
    scores, order = scores.sort(descending=True, stable=True)
    scores, order = scores[:top_k], order[:top_k]
    boxes, labels = boxes[order], labels[order]
    same_label = labels[:, None] == labels[None, :]
    iou = box_ops.box_iou(boxes, boxes).triu_(diagonal=1) * same_label
    compensate = iou.max(dim=0).values  # each box's largest IoU with a better same-label box
    decay = torch.exp((compensate[:, None] ** 2 - iou ** 2) / sigma).min(dim=0).values
    scores = scores * decay
    keep = scores > score_thresh
    return _best_first(order[keep], scores[keep], max_detections)


class PostprocessDetections:
    """``RoIHeads.postprocess_detections`` for one head with the given NMS method.

    Everything before the NMS call is torchvision's own code (unchanged from 0.22 to 0.29), so
    ``hard`` reproduces the stock method exactly. Thresholds are read from the head at call time,
    so later ``set_detection_limits`` calls still apply. The head's ``score_thresh`` stays the RoI
    pre-filter for every method; Soft-NMS then drops decayed scores at ``min_score``, Matrix NMS
    at ``score_thresh``. A class rather than a closure, so a deep copy of the model rebinds it to
    the copied head.
    """

    def __init__(self, roi_heads: torch.nn.Module, method: str = "hard", sigma: float = 0.5,
                 min_score: float = SOFT_NMS_MIN_SCORE):
        if method not in NMS_METHODS:
            raise ValueError(f"unknown NMS method {method!r}; choose from {', '.join(NMS_METHODS)}")
        if not sigma > 0:
            raise ValueError(f"Soft/Matrix NMS sigma must be positive, got {sigma}")
        if not 0 <= min_score <= 1:
            raise ValueError(f"Soft-NMS min_score must be within [0, 1], got {min_score}")
        self.roi_heads, self.method, self.sigma = roi_heads, method, sigma
        self.min_score = float(min_score)

    def __call__(self, class_logits, box_regression, proposals, image_shapes):
        heads = self.roi_heads
        device = class_logits.device
        num_classes = class_logits.shape[-1]
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = heads.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)
        all_boxes, all_scores, all_labels = [], [], []
        for boxes, scores, image_shape in zip(pred_boxes.split(boxes_per_image, 0),
                                              pred_scores.split(boxes_per_image, 0), image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            labels = torch.arange(num_classes, device=device).view(1, -1).expand_as(scores)
            boxes, scores, labels = boxes[:, 1:], scores[:, 1:], labels[:, 1:]
            boxes, scores, labels = boxes.reshape(-1, 4), scores.reshape(-1), labels.reshape(-1)
            inds = torch.where(scores > heads.score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            if self.method == "hard":
                keep = box_ops.batched_nms(boxes, scores, labels, heads.nms_thresh)
                keep = keep[: heads.detections_per_img]
                scores = scores[keep]
            elif self.method == "soft":
                keep, scores = soft_nms(boxes, scores, labels, self.sigma, self.min_score,
                                        heads.detections_per_img)
            else:
                keep, scores = matrix_nms(boxes, scores, labels, self.sigma, heads.score_thresh,
                                          heads.detections_per_img)
            all_boxes.append(boxes[keep])
            all_scores.append(scores)
            all_labels.append(labels[keep])
        return all_boxes, all_scores, all_labels


def install_postprocess(roi_heads: torch.nn.Module, method: str = "hard", sigma: float = 0.5,
                        min_score: float = SOFT_NMS_MIN_SCORE) -> None:
    """Swap the NMS of this RoI head instance; ``hard`` restores torchvision's own method.

    ``min_score`` is Soft-NMS's drop floor for decayed scores; Matrix NMS ignores it.
    """
    if method == "hard":
        vars(roi_heads).pop("postprocess_detections", None)
        return
    roi_heads.postprocess_detections = PostprocessDetections(roi_heads, method, sigma, min_score)
