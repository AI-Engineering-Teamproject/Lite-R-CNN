"""Benchmark the Ultralytics YOLO26 control on a folder of images."""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

# Keep Ultralytics settings inside the project so read-only user profiles work.
_ultralytics_config = Path("outputs/.ultralytics").resolve()
_ultralytics_config.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_ultralytics_config))

import torch
import cv2
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sync(device: str) -> None:
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="yolo26n.pt")
    parser.add_argument("--source", required=True, help="Directory containing benchmark images")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--image-count", type=int, default=64, help="Maximum preloaded images to cycle through")
    parser.add_argument("--output", default="outputs/benchmark_yolo26.json")
    args = parser.parse_args()

    source = Path(args.source)
    candidates = [source] if source.is_file() else source.rglob("*")
    paths = sorted(path for path in candidates if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise SystemExit(f"No image files found under {args.source}")
    paths = paths[:args.image_count]
    images = [cv2.imread(str(path)) for path in paths]
    if any(image is None for image in images):
        raise RuntimeError("Failed to decode one or more benchmark images")
    model = YOLO(args.weights)
    for index in range(args.warmup):
        model.predict(source=images[index % len(images)], imgsz=args.imgsz, device=args.device,
                      half=args.half, verbose=False)
    sync(args.device)

    timings = []
    for index in range(args.iterations):
        image = images[index % len(images)]
        sync(args.device)
        start = time.perf_counter()
        model.predict(source=image, imgsz=args.imgsz, device=args.device,
                      half=args.half, verbose=False)
        sync(args.device)
        timings.append((time.perf_counter() - start) * 1000)

    ordered = sorted(timings)
    result = {
        "model_family": "YOLO26",
        "weights": args.weights,
        "imgsz": args.imgsz,
        "device": args.device,
        "precision": "fp16" if args.half else "fp32",
        "batch_size": 1,
        "timed_scope": "Ultralytics predict API; image decode excluded, model transforms/postprocess included",
        "warmup": args.warmup,
        "iterations": args.iterations,
        "p50_ms": statistics.median(timings),
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "images": [str(path) for path in paths[:min(len(paths), 20)]],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "images"}, indent=2))


if __name__ == "__main__":
    main()
