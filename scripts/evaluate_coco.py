"""Evaluate a Faster R-CNN model on COCO val2017 with COCO AP metrics."""

import argparse
import json
from pathlib import Path

import torch
import yaml
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from light_rcnn.data import CocoTensorDataset
from light_rcnn.model import build_detector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/coco.yaml")
    parser.add_argument("--model", choices=("light_rcnn", "faster_rcnn"), required=True)
    parser.add_argument("--weights", help="Required for trained Light R-CNN; optional for COCO-pretrained baseline")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="Optional image limit; 0 evaluates full val set")
    parser.add_argument("--output")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    model_cfg = config["model"]
    if args.model == "light_rcnn":
        model = build_detector(
            args.model,
            pretrained_backbone=args.weights is None,
            num_classes=config["num_classes"],
            bifpn_channels=model_cfg["bifpn_channels"],
            roi_channels=model_cfg["roi_channels"],
            roi_size=model_cfg["roi_size"],
            rpn_pre_nms_topk_test=model_cfg["rpn_pre_nms_topk_test"],
            rpn_post_nms_topk_test=model_cfg["rpn_post_nms_topk_test"],
            min_size=config["image_size"],
            max_size=config["image_size"],
        )
    else:
        model = build_detector(
            args.model,
            pretrained=args.weights is None,
            num_classes=config["num_classes"],
            min_size=config["image_size"],
            max_size=config["image_size"],
        )
    if args.weights:
        checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint.get("model", checkpoint))

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model.to(device).eval()
    data = config["data"]
    dataset = CocoTensorDataset(data["val_images"], data["val_annotations"])
    image_ids = dataset.ids[:args.limit] if args.limit > 0 else dataset.ids
    results = []
    with torch.inference_mode():
        for index, image_id in enumerate(tqdm(image_ids, desc=f"COCO val: {args.model}")):
            image, _ = dataset[index]
            prediction = model([image.to(device)])[0]
            boxes = prediction["boxes"].cpu().tolist()
            scores = prediction["scores"].cpu().tolist()
            labels = prediction["labels"].cpu().tolist()
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                results.append({
                    "image_id": int(image_id),
                    "category_id": int(label),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                })
    if not results:
        raise RuntimeError("The detector returned no predictions; COCO AP cannot be computed.")

    coco_gt = dataset.coco
    coco_dt = coco_gt.loadRes(results)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = list(image_ids)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    report = {
        "model": args.model,
        "weights": args.weights or "torchvision_default_coco_weights",
        "images_evaluated": len(image_ids),
        "AP_50_95": float(evaluator.stats[0]),
        "AP_50": float(evaluator.stats[1]),
        "AP_75": float(evaluator.stats[2]),
        "AP_small": float(evaluator.stats[3]),
        "AP_medium": float(evaluator.stats[4]),
        "AP_large": float(evaluator.stats[5]),
    }
    output = Path(args.output or f"outputs/{args.model}_coco_metrics.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
