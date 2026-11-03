"""Shared official COCO validation, including detectors with no predictions."""

import hashlib
import time
from copy import deepcopy

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import box_iou
from tqdm import tqdm

from light_rcnn.coco_eval import summarize_detections


def evaluation_subset(ids, limit=0, seed=0):
    """Positions in ``ids`` to validate: all of them, or a seeded random subset of ``limit``.

    Training validation and both COCO evaluation scripts share this rule, so a limited run scores
    the same images for every model as long as ``ids`` keeps CocoDetection's sorted order.
    """
    if limit > 0 and limit < len(ids):
        generator = torch.Generator().manual_seed(seed)
        return torch.randperm(len(ids), generator=generator)[:limit].tolist()
    return list(range(len(ids)))


def subset_record(image_ids, images_total, seed):
    """Report fields naming the images an evaluation scored, so two reports can be matched."""
    ordered = sorted(int(image_id) for image_id in image_ids)
    partial = len(ordered) < images_total
    record = {
        "images_evaluated": len(ordered),
        "images_total": images_total,
        "is_partial": partial,
        "subset_selection": "seeded_random" if partial else "full",
        "subset_seed": seed if partial else None,
        "image_ids_sha256": hashlib.sha256(",".join(map(str, ordered)).encode()).hexdigest(),
    }
    if partial:
        record["image_ids"] = ordered
    return record


@torch.inference_mode()
def evaluate_coco(model, dataset, device, limit=0, subset_seed=0, batch_size=1, detailed=False):
    """Official COCO bbox AP on ``dataset`` (or its seeded subset).

    ``detailed=True`` scores through ``coco_eval.summarize_detections``, the scorer the YOLO26
    evaluator uses, and adds the AR stats, per-class AP and the evaluated image ids.
    """
    if batch_size < 1:
        raise ValueError("Evaluation batch_size must be at least 1")
    was_training = model.training
    model.eval()
    indices = evaluation_subset(dataset.ids, limit, subset_seed)
    image_ids = [dataset.ids[index] for index in indices]
    if not image_ids:
        raise ValueError("Validation dataset must contain images")
    results = []
    images_with_predictions = 0
    max_score = 0.0
    best_iou_any_class = []
    best_iou_same_class = []
    try:
        start = time.perf_counter()
        for offset in tqdm(range(0, len(indices), batch_size), desc="COCO validation"):
            batch_indices = indices[offset : offset + batch_size]
            samples = [dataset[index] for index in batch_indices]
            images = [image.to(device, non_blocking=True) for image, _ in samples]
            predictions = model(images)
            batch_image_ids = image_ids[offset : offset + len(samples)]
            for image_id, (_, target), prediction in zip(batch_image_ids, samples, predictions):
                if prediction["scores"].numel():
                    images_with_predictions += 1
                    max_score = max(max_score, float(prediction["scores"].max().cpu()))
                pred_boxes = prediction["boxes"].detach().cpu()
                pred_labels = prediction["labels"].detach().cpu()
                gt_boxes = target["boxes"]
                gt_labels = target["labels"]
                if len(pred_boxes) and len(gt_boxes):
                    overlaps = box_iou(pred_boxes, gt_boxes)
                    best_iou_any_class.append(float(overlaps.max()))
                    same_class = pred_labels[:, None] == gt_labels[None, :]
                    best_iou_same_class.append(float(overlaps.masked_fill(~same_class, 0).max()))
                else:
                    best_iou_any_class.append(0.0)
                    best_iou_same_class.append(0.0)
                for box, score, label in zip(
                    prediction["boxes"].cpu().tolist(),
                    prediction["scores"].cpu().tolist(),
                    prediction["labels"].cpu().tolist(),
                ):
                    x1, y1, x2, y2 = box
                    results.append(
                        {
                            "image_id": int(image_id),
                            "category_id": int(label),
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": float(score),
                        }
                    )
        inference_seconds = time.perf_counter() - start  # before scoring, like the YOLO26 evaluator
        if detailed:
            scores = {
                **summarize_detections(dataset.coco, results, image_ids),
                "image_ids": [int(image_id) for image_id in image_ids],
                "inference_seconds": inference_seconds,
            }
        else:
            scores = _official_ap(dataset.coco, results, image_ids)
        return {
            "images_evaluated": len(image_ids),
            "evaluation_batch_size": batch_size,
            "detections": len(results),
            "images_with_detections": images_with_predictions,
            "max_detection_score": max_score,
            "mean_best_iou_any_class": sum(best_iou_any_class) / len(best_iou_any_class),
            "mean_best_iou_same_class": sum(best_iou_same_class) / len(best_iou_same_class),
            "images_with_iou50_any_class": sum(iou >= 0.5 for iou in best_iou_any_class),
            "images_with_iou50_same_class": sum(iou >= 0.5 for iou in best_iou_same_class),
            "evaluation_scope": "subset" if len(image_ids) < len(dataset) else "full",
            "evaluation_subset_seed": subset_seed if len(image_ids) < len(dataset) else None,
            **scores,
        }
    finally:
        model.train(was_training)


def _official_ap(coco, results, image_ids):
    if results:
        detections = coco.loadRes(results)
    else:
        # COCO.loadRes([]) indexes the first result. An empty detection COCO
        # gives valid zero AP for a newly initialized detector instead.
        detections = COCO()
        detections.dataset = {
            "images": deepcopy(coco.dataset["images"]),
            "categories": deepcopy(coco.dataset["categories"]),
            "annotations": [],
        }
        detections.createIndex()
    evaluator = COCOeval(coco, detections, "bbox")
    evaluator.params.imgIds = list(image_ids)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    keys = ("AP_50_95", "AP_50", "AP_75", "AP_small", "AP_medium", "AP_large")
    return {key: float(value) for key, value in zip(keys, evaluator.stats[:6])}
