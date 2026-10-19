"""CLI orchestration for the independent four-model comparison."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from light_rcnn.comparison.adapters import MODEL_NAMES, build_adapter
from light_rcnn.comparison.measurement import benchmark, evaluate
from light_rcnn.comparison.protocol import CocoImages, file_hash, select_ids, write_json
from light_rcnn.comparison.reporting import save_comparison
from light_rcnn.utils import seed_everything


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Independent four-model comparison with common CPU input and official COCOeval."
    )
    result.add_argument("--config", default="configs/my_comparison.yaml")
    result.add_argument("--models", nargs="+", choices=list(MODEL_NAMES))
    result.add_argument("--device", help="cpu, cuda, or cuda:N; FP32 only")
    result.add_argument("--mode", choices=("full", "accuracy", "latency"))
    result.add_argument("--geometry", choices=("square", "native"))
    result.add_argument("--size", type=int)
    result.add_argument("--images")
    result.add_argument("--annotations")
    result.add_argument("--light-weights")
    result.add_argument("--yolo-weights")
    result.add_argument(
        "--limit", type=int, help="0 = full validation; shared random subset otherwise"
    )
    result.add_argument("--seed", type=int)
    result.add_argument("--image-ids", help="JSON ID list or a previous manifest.json")
    result.add_argument("--image-count", type=int)
    result.add_argument("--warmup", type=int)
    result.add_argument("--iterations", type=int)
    result.add_argument("--cpu-threads", type=int)
    result.add_argument(
        "--output-dir", help="New directory; existing results are never overwritten"
    )
    return result


def load_config(args, root: Path) -> dict:
    config = yaml.safe_load((root / args.config).read_text("utf-8"))
    for field, option in (
        ("device", "device"),
        ("mode", "mode"),
        ("geometry", "geometry"),
        ("image_size", "size"),
        ("evaluation_limit", "limit"),
        ("seed", "seed"),
        ("cpu_threads", "cpu_threads"),
    ):
        value = getattr(args, option)
        if value is not None:
            config[field] = value
    for field in ("image_count", "warmup", "iterations"):
        if getattr(args, field) is not None:
            config["benchmark"][field] = getattr(args, field)
    for field in ("images", "annotations"):
        if getattr(args, field) is not None:
            config["data"][field] = getattr(args, field)
    if args.light_weights:
        config["models"]["lite_rcnn"]["weights"] = args.light_weights
    if args.yolo_weights:
        config["models"]["yolo26"]["weights"] = args.yolo_weights
    config["selected_models"] = args.models or [
        name for name in MODEL_NAMES if name in config["models"]
    ]
    if len(set(config["selected_models"])) != len(config["selected_models"]):
        raise ValueError("Select each model at most once")
    if not config["selected_models"] or any(
        name not in MODEL_NAMES for name in config["selected_models"]
    ):
        raise ValueError("Unknown or empty model selection")
    if config["image_size"] < 32 or config["image_size"] % 32:
        raise ValueError("Image size must be a positive multiple of 32")
    if config["geometry"] not in ("square", "native"):
        raise ValueError("Geometry must be square or native")
    if config["mode"] not in ("full", "accuracy", "latency"):
        raise ValueError("Mode must be full, accuracy, or latency")
    if config["evaluation_limit"] < 0 or config["cpu_threads"] < 1:
        raise ValueError("Evaluation limit must be >=0 and CPU threads must be >=1")
    if config["max_detections"] != 100:
        raise ValueError("This protocol fixes max_detections=100 for official COCO comparison")
    if not 0 < config["score_threshold"] < 1 or not 0 < config["nms_threshold"] <= 1:
        raise ValueError("Score threshold must be in (0,1); NMS threshold must be in (0,1]")
    if any(config["benchmark"][field] < 1 for field in ("image_count", "warmup", "iterations")):
        raise ValueError("Benchmark image_count, warmup and iterations must be >=1")
    selected_device = torch.device(config["device"])
    if selected_device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA FP32 are supported")
    if selected_device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; use --device cpu for a pipeline check")
        index = (
            selected_device.index
            if selected_device.index is not None
            else torch.cuda.current_device()
        )
        torch.cuda.get_device_properties(index)
        selected_device = torch.device("cuda", index)
    config["device"] = str(selected_device)
    for name in config["selected_models"]:
        settings = config["models"][name]
        if name in ("lite_rcnn", "yolo26"):
            if not (root / settings["weights"]).is_file():
                raise FileNotFoundError(
                    f"{name} requires local trained weights: {settings['weights']}"
                )
        elif settings["weights"] != "COCO_V1":
            raise ValueError(f"{name} currently supports the explicit COCO_V1 weight enum only")
    return config


def environment(root: Path, device: str) -> dict:
    versions = {}
    for name in ("torch", "torchvision", "ultralytics", "pycocotools", "numpy", "Pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    selected_device = torch.device(device)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "torch_threads": torch.get_num_threads(),
        "versions": versions,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(selected_device)
        if selected_device.type == "cuda"
        else None,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "git_revision": revision,
        "comparison_source_sha256": {
            path.name: file_hash(path) for path in sorted(Path(__file__).parent.glob("*.py"))
        },
    }


def run(args, root: Path) -> Path:
    config = load_config(args, root)
    dataset = CocoImages(root / config["data"]["images"], root / config["data"]["annotations"])
    old_manifest = None
    if args.image_ids:
        selected = json.loads((root / args.image_ids).read_text("utf-8"))
        old_manifest = selected if isinstance(selected, dict) else None
        if old_manifest:
            if old_manifest["annotation_sha256"] != file_hash(dataset.annotations):
                raise ValueError("Previous manifest uses a different annotation file")
            image_ids = old_manifest["evaluation_image_ids"]
        else:
            image_ids = selected
        if args.limit is not None:
            raise ValueError("Use --image-ids or --limit, not both")
    else:
        image_ids = select_ids(list(dataset.records), config["evaluation_limit"], config["seed"])
    dataset.validate_ids(image_ids)
    benchmark_ids = (
        old_manifest["benchmark_image_ids"]
        if old_manifest
        else select_ids(image_ids, config["benchmark"]["image_count"], config["seed"] + 1)
    )
    if not set(benchmark_ids).issubset(image_ids):
        raise ValueError("Benchmark IDs must be drawn from the evaluation IDs")
    dataset.validate_ids(benchmark_ids)
    if config["mode"] != "accuracy" and config["benchmark"]["iterations"] < len(benchmark_ids):
        raise ValueError(
            "Iterations must cover all selected benchmark images; increase --iterations"
        )
    output = root / (
        args.output_dir
        or ("outputs/my_comparison/" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ"))
    )
    output.mkdir(parents=True, exist_ok=False)
    # Set before the lazy Ultralytics import, without modifying the user's home configuration.
    os.environ.setdefault("YOLO_CONFIG_DIR", str(output / "ultralytics_settings"))
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    seed_everything(config["seed"])
    manifest = {
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "annotation_sha256": file_hash(dataset.annotations),
        "evaluation_image_ids": image_ids,
        "benchmark_image_ids": benchmark_ids,
        "evaluation_scope": "full" if len(image_ids) == len(dataset.records) else "subset",
        "environment": environment(root, config["device"]),
    }
    write_json(output / "manifest.json", manifest)
    # Only the bounded benchmark sample is preloaded. Accuracy images are decoded one by one.
    images = (
        [(image_id, dataset.read(image_id)) for image_id in benchmark_ids]
        if config["mode"] != "accuracy"
        else []
    )
    reports = []
    for name in config["selected_models"]:
        report = {"model": name, "status": "running", "metadata": {"name": MODEL_NAMES[name]}}
        reports.append(report)
        adapter = None
        try:
            print(f"\n{name}: loading trained detector", flush=True)
            adapter = build_adapter(name, config, dataset.categories, root)
            report["metadata"] = adapter.metadata
            if config["mode"] != "latency":
                report["accuracy"] = evaluate(
                    adapter, dataset, image_ids, output / f"{name}_predictions.json"
                )
            if config["mode"] != "accuracy":
                report["latency"] = benchmark(
                    adapter,
                    images,
                    device=config["device"],
                    **{key: config["benchmark"][key] for key in ("warmup", "iterations")},
                )
            report["status"] = "complete"
        except Exception as error:
            report.update(status="failed", error=f"{type(error).__name__}: {error}")
            manifest.update(status="failed", failed_model=name)
            raise
        finally:
            write_json(output / f"{name}.json", report)
            write_json(output / "manifest.json", manifest)
            save_comparison(output, reports, config, manifest)
            del adapter
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    manifest["status"] = "complete"
    write_json(output / "manifest.json", manifest)
    save_comparison(output, reports, config, manifest)
    print(f"\nComparison saved: {output / 'REPORT.md'}", flush=True)
    return output


def main(root: Path | None = None) -> None:
    root = root or Path(__file__).resolve().parents[3]
    args = parser().parse_args()
    try:
        run(args, root)
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error
