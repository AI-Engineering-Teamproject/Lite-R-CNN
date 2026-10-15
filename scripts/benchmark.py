"""Measure warmed-up end-to-end model latency on the selected device."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from light_rcnn.model import build_detector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("light_rcnn", "faster_rcnn"), default="light_rcnn")
    parser.add_argument("--weights")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--source", required=True, help="COCO image directory used by all compared models")
    parser.add_argument("--image-count", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output")
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.model == "light_rcnn":
        model = build_detector(
            args.model, pretrained_backbone=args.weights is None,
            min_size=args.size, max_size=args.size,
        )
    else:
        model = build_detector(
            args.model, pretrained=args.weights is None, num_classes=91,
            min_size=args.size, max_size=args.size,
        )
    model = model.to(device).eval()
    if args.weights:
        checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint.get("model", checkpoint))
    source = Path(args.source)
    candidates = [source] if source.is_file() else source.rglob("*")
    paths = sorted(path for path in candidates
                   if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})[:args.image_count]
    if not paths:
        raise SystemExit(f"No benchmark images found under {args.source}")
    images = [to_tensor(Image.open(path).convert("RGB")).to(device) for path in paths]
    with torch.inference_mode():
        for index in range(args.warmup):
            model([images[index % len(images)]])
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings = []
        for index in range(args.iterations):
            start = time.perf_counter()
            model([images[index % len(images)]])
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)
    ordered = sorted(timings)
    result = {
        "model": args.model,
        "model_name": "Light R-CNN" if args.model == "light_rcnn" else "Vanilla Faster R-CNN",
        "weights": args.weights or "torchvision_coco_v1",
        "device": str(device),
        "precision": "fp32",
        "batch_size": 1,
        "image_size": args.size,
        "image_count": len(paths),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "timed_scope": "torchvision transforms + backbone + RPN + RoI heads + NMS; image decode excluded",
        "p50_ms": statistics.median(timings),
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
    }
    output = Path(args.output or f"outputs/{args.model}_latency.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
