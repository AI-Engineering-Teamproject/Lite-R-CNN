"""COCO bbox evaluation shared by every model so all AP numbers come from one pycocotools path."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

STAT_NAMES = (
    "AP_50_95", "AP_50", "AP_75", "AP_small", "AP_medium", "AP_large",
    "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large",
)


def summarize_detections(coco_gt: COCO, detections: list[dict[str, Any]],
                         image_ids: list[int]) -> dict[str, Any]:
    """Score COCO-format detections on ``image_ids``; returns the 12 COCO stats and per-class AP.

    An empty detection list is valid: pycocotools' loadRes cannot load it, so an empty result
    set is built directly and scored the same way. Stats with no ground truth in the evaluated
    images (pycocotools -1) are reported as None. ``coco_max_dets`` is the per-image cap that
    pycocotools actually scored (COCO AP is fixed at 100).
    """
    if detections:
        coco_dt = coco_gt.loadRes(detections)
    else:
        coco_dt = COCO()
        coco_dt.dataset = {"images": list(coco_gt.dataset["images"]),
                           "categories": copy.deepcopy(coco_gt.dataset["categories"]),
                           "annotations": []}
        coco_dt.createIndex()
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = list(image_ids)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    stats = {name: float(value) if value >= 0 else None
             for name, value in zip(STAT_NAMES, evaluator.stats)}
    # precision: [IoU thresholds, recall points, categories, area ranges, max detections]
    precision = evaluator.eval["precision"][:, :, :, 0, -1]
    per_category = {}
    for index, category_id in enumerate(evaluator.params.catIds):
        values = precision[:, :, index]
        values = values[values > -1]
        if values.size:  # categories without ground truth in the evaluated images stay out
            per_category[coco_gt.cats[category_id]["name"]] = float(np.mean(values))
    return {**stats, "per_category_ap": per_category, "detections": len(detections),
            "coco_max_dets": int(evaluator.params.maxDets[-1])}
