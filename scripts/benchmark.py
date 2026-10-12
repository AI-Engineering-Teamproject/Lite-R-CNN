"""Measure warmed-up end-to-end model latency on the selected device."""

import argparse
import statistics
import time

import torch

from light_rcnn.model import build_detector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("light_rcnn", "faster_rcnn"), default="light_rcnn")
    parser.add_argument("--weights")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
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
    image = torch.rand(1, 3, args.size, args.size, device=device)
    with torch.inference_mode():
        for _ in range(args.warmup):
            model([image[0]])
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            model([image[0]])
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)
    print(f"median={statistics.median(timings):.2f} ms p95={sorted(timings)[int(.95*len(timings))-1]:.2f} ms")
    print("Includes torchvision resize/normalize, RPN, RoIAlign, and NMS; excludes image decode.")


if __name__ == "__main__":
    main()
