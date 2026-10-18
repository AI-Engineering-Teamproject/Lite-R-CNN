"""Shared official COCO validation, including detectors with no predictions."""

from copy import deepcopy

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import box_iou
from tqdm import tqdm


@torch.inference_mode()
def evaluate_coco(model, dataset, device, limit=0, subset_seed=0, batch_size=1):
    if batch_size < 1:
        raise ValueError("Evaluation batch_size must be at least 1")
    was_training = model.training
    model.eval()
    if limit > 0 and limit < len(dataset):
        generator = torch.Generator().manual_seed(subset_seed)
        indices = torch.randperm(len(dataset), generator=generator)[:limit].tolist()
    else:
        indices = list(range(len(dataset)))
    image_ids = [dataset.ids[index] for index in indices]
    if not image_ids:
        raise ValueError("Validation dataset must contain images")
    results = []
    images_with_predictions = 0
    max_score = 0.0
    best_iou_any_class = []
    best_iou_same_class = []
    try:
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
        if results:
            detections = dataset.coco.loadRes(results)
        else:
            # COCO.loadRes([]) indexes the first result. An empty detection COCO
            # gives valid zero AP for a newly initialized detector instead.
            detections = COCO()
            detections.dataset = {
                "images": deepcopy(dataset.coco.dataset["images"]),
                "categories": deepcopy(dataset.coco.dataset["categories"]),
                "annotations": [],
            }
            detections.createIndex()
        evaluator = COCOeval(dataset.coco, detections, "bbox")
        evaluator.params.imgIds = list(image_ids)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        keys = ("AP_50_95", "AP_50", "AP_75", "AP_small", "AP_medium", "AP_large")
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
            **{key: float(value) for key, value in zip(keys, evaluator.stats[:6])},
        }
    finally:
        model.train(was_training)
