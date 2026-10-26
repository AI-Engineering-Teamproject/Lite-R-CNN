"""Measure R-CNN latency from decoded host images to final detections (schema-v2 report)."""

import argparse
import json
import time
from urllib.error import URLError

import torch
import yaml
from torch.utils.flop_counter import FlopCounterMode

from light_rcnn.bench import (
    DEFAULT_WEIGHTS,
    comparison_setting,
    load_comparison_model,
    load_images,
    resolve_device,
    set_detection_limits,
)
from light_rcnn.model.deploy import prepare_for_inference
from light_rcnn.reporting import (
    COLD_START_FIRST_FORWARD,
    SCOPE_HOST_TO_DETECTIONS,
    SCOPE_LABELS,
    build_report,
    collect_environment,
    default_report_path,
    display_path,
    file_sha256,
    latency_summary,
    write_report,
)


def timed_call(model: torch.nn.Module, image: torch.Tensor, device: torch.device
               ) -> tuple[float, list[dict]]:
    start = time.perf_counter()
    x = image.to(device)
    x = x.float().div_(255)
    outputs = model([x])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1000, outputs


def count_flops(model: torch.nn.Module, image: torch.Tensor, device: torch.device,
                image_size: int) -> tuple[float | None, list[int] | None, str]:
    """Count FLOPs on a square image_size input, the same basis YOLO's get_flops uses."""
    try:
        x = image.to(device).float().div_(255)
        # Square input, so min_size == max_size == image_size keeps it unscaled in the transform.
        x = torch.nn.functional.interpolate(x[None], size=(image_size, image_size),
                                            mode="bilinear", align_corners=False,
                                            antialias=False)[0]
        model_input = [int(side) for side in model.transform([x])[0].image_sizes[0]]
        counter = FlopCounterMode(display=False)
        with counter:
            model([x])
        total = counter.get_total_flops()
    except Exception as exc:  # noqa: BLE001 - FLOPs are context only, never fatal
        return None, None, f"FLOP count unavailable: {type(exc).__name__}: {exc}"
    height, width = model_input
    return total / 1e9, model_input, (
        f"torch FlopCounterMode on the first benchmark image resized to {image_size}x{image_size} "
        f"(model input {height}x{width}): conv/matmul FLOPs (multiply-add = 2), "
        "RoIAlign/NMS not counted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/coco.yaml",
                        help="Comparison settings; a Light R-CNN checkpoint's own config still "
                             "decides its architecture")
    parser.add_argument("--model", choices=("light_rcnn", "faster_rcnn"), default="light_rcnn")
    parser.add_argument("--weights", help="Checkpoint; omit for torchvision COCO/ImageNet weights")
    parser.add_argument("--device", help="PyTorch device; defaults to the config device")
    parser.add_argument("--size", "--imgsz", dest="size", type=int,
                        help="Override config image_size")
    parser.add_argument("--max-dets", "--max-det", type=int,
                        help="Override comparison.max_detections")
    parser.add_argument("--source", help="Image file, or image folder shared by all compared "
                                         "models; defaults to the config val_images")
    parser.add_argument("--image-count", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--score-thresh", type=float,
                        help="Override comparison.latency_score_threshold")
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction,
                        help="Override comparison.channels_last (shared by every benchmark; "
                             "unset = Ultralytics auto rule)")
    parser.add_argument("--output", help="Defaults to outputs/reports/<model>_latency.json")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    comparison = config.get("comparison") or {}
    image_size = int(args.size if args.size is not None else config["image_size"])
    warmup = args.warmup if args.warmup is not None else comparison_setting(config, "warmup")
    iterations = (args.iterations if args.iterations is not None
                  else comparison_setting(config, "iterations"))
    image_count = (args.image_count if args.image_count is not None
                   else comparison_setting(config, "image_count"))
    score_threshold = (args.score_thresh if args.score_thresh is not None
                       else comparison_setting(config, "latency_score_threshold"))
    max_detections = (args.max_dets if args.max_dets is not None
                      else comparison_setting(config, "max_detections"))
    channels_last = (args.channels_last if args.channels_last is not None
                     else comparison.get("channels_last"))
    if warmup < 0 or iterations < 1 or image_count < 1 or max_detections < 1 or image_size < 1:
        parser.error("--warmup must be >= 0; --iterations, --image-count, --max-dets and --size "
                     "must be >= 1")

    # COCO images vary in shape, so cuDNN autotuning would re-tune per shape inside the timings.
    torch.backends.cudnn.benchmark = False
    try:
        device = resolve_device(args.device or config.get("device", "cuda"), "benchmarking")
    except ValueError as exc:
        parser.error(f"--device: {exc}")
    paths, images = load_images(args.source or config["data"]["val_images"], image_count)
    try:
        model = load_comparison_model(args.model, config, args.weights, image_size)
    except URLError as exc:
        raise SystemExit(f"Could not download default {args.model} weights ({exc}). "
                         "Pass --weights <checkpoint>.") from exc
    set_detection_limits(model, score_threshold, max_detections)
    model.to(device).eval()
    parameters_m = sum(param.numel() for param in model.parameters()) / 1e6
    layout = prepare_for_inference(model, device, channels_last)

    with torch.inference_mode():
        cold_start_ms, _ = timed_call(model, images[0], device)
        for index in range(warmup):
            timed_call(model, images[index % len(images)], device)
        samples, detections = [], []
        for index in range(iterations):
            elapsed, outputs = timed_call(model, images[index % len(images)], device)
            samples.append(elapsed)
            detections.append(len(outputs[0]["boxes"]))
        gflops, flops_input, flops_method = count_flops(model, images[0], device, image_size)

    report = build_report(
        "latency",
        args.model,
        weights=display_path(args.weights) if args.weights else DEFAULT_WEIGHTS[args.model],
        weights_sha256=file_sha256(args.weights),
        image_size=image_size,
        batch_size=1,
        precision="fp32",
        environment=collect_environment(str(device)),
        latency={
            "scope_id": SCOPE_HOST_TO_DETECTIONS,
            "timed_scope": SCOPE_LABELS[SCOPE_HOST_TO_DETECTIONS],
            "warmup": warmup,
            "iterations": iterations,
            "image_count": len(paths),
            "score_threshold": score_threshold,
            "max_detections": max_detections,
            **layout,
            "images": [display_path(path) for path in paths[:20]],
            "mean_detections": sum(detections) / len(detections),
            **latency_summary(samples, cold_start_ms=cold_start_ms),
            "cold_start_scope": COLD_START_FIRST_FORWARD,
        },
        model_stats={
            "parameters_m": parameters_m,
            "deployed_parameters_m": sum(param.numel() for param in model.parameters()) / 1e6,
            "gflops": gflops,
            "flops_input": flops_input,
            "flops_method": flops_method,
        },
    )
    output = write_report(report, args.output or default_report_path("latency", args.model))
    hidden = {"samples_ms", "images", "timed_scope"}
    print(json.dumps({
        "output": str(output),
        "model": report["model"],
        "weights": report["weights"],
        "device": str(device),
        "image_size": report["image_size"],
        "precision": report["precision"],
        "latency": {key: value for key, value in report["latency"].items() if key not in hidden},
        "model_stats": report["model_stats"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
