"""Evaluate a Faster R-CNN model on COCO val2017 with official COCO AP."""

import argparse
import json
from pathlib import Path

import torch
import yaml

from light_rcnn.data import CocoTensorDataset
from light_rcnn.evaluation import evaluate_coco
from light_rcnn.model import build_detector
from light_rcnn.model.detector import light_rcnn_from_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Dataset config; defaults to saved checkpoint config")
    parser.add_argument("--model", choices=("light_rcnn", "faster_rcnn"), required=True)
    parser.add_argument("--weights")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates the full validation set")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.model == "light_rcnn" and not args.weights:
        parser.error("--weights is required to evaluate trained Light R-CNN accuracy")
    checkpoint = (
        torch.load(args.weights, map_location="cpu", weights_only=True) if args.weights else {}
    )
    if args.config or "config" not in checkpoint:
        with open(args.config or "configs/coco.yaml", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    else:
        config = checkpoint["config"]
    if args.model == "light_rcnn":
        model = light_rcnn_from_config(checkpoint.get("config", config))
    else:
        model = build_detector(
            args.model,
            pretrained=not args.weights,
            num_classes=config["num_classes"],
            min_size=config["image_size"],
            max_size=config["image_size"],
        )
    if args.weights:
        model.load_state_dict(checkpoint.get("model", checkpoint))
    device = torch.device(args.device)
    model.to(device).eval()
    data = config["data"]
    dataset = CocoTensorDataset(data["val_images"], data["val_annotations"])
    report = {
        "model": args.model,
        "weights": args.weights or "torchvision_coco_v1",
        "image_size": model.transform.max_size,
        **evaluate_coco(model, dataset, device, args.limit),
    }
    output = Path(args.output or f"outputs/{args.model}_coco_metrics.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
