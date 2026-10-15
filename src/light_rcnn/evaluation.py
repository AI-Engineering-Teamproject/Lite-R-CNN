"""Shared official COCO validation, including detectors with no predictions."""

from copy import deepcopy

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm


@torch.inference_mode()
def evaluate_coco(model, dataset, device, limit=0):
    was_training = model.training
    model.eval()
    image_ids = dataset.ids[:limit] if limit > 0 else dataset.ids
    if not image_ids:
        raise ValueError("Validation dataset must contain images")
    results = []
    try:
        for index, image_id in enumerate(tqdm(image_ids, desc="COCO validation")):
            image, _ = dataset[index]
            prediction = model([image.to(device)])[0]
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
            "evaluation_scope": "subset" if len(image_ids) < len(dataset) else "full",
            **{key: float(value) for key, value in zip(keys, evaluator.stats[:6])},
        }
    finally:
        model.train(was_training)
