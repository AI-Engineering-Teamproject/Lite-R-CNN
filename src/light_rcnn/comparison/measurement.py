"""One official COCO evaluator and one latency boundary for all adapters."""

from __future__ import annotations

import statistics
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from light_rcnn.comparison.protocol import CocoImages, write_json

COCO_METRICS = (
    "AP_50_95",
    "AP_50",
    "AP_75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_1",
    "AR_10",
    "AR_100",
    "AR_small",
    "AR_medium",
    "AR_large",
)
TIMED_SCOPE = (
    "Decoded CPU RGB uint8 HWC image -> common geometry -> framework preprocessing -> "
    "host-to-device -> detector and native postprocess -> original-image boxes, scores "
    "and original COCO labels on CPU; file decode, JSON serialization and geometry hooks excluded"
)


def official_metrics(annotations: Path, image_ids: list[int], rows: list[dict]) -> dict:
    ground_truth = COCO(str(annotations))
    # Some valid minimal COCO fixtures omit info; annotations (including crowds) stay intact.
    ground_truth.dataset.setdefault("info", {})
    if rows:
        detections = ground_truth.loadRes(rows)
    else:
        detections = COCO()
        detections.dataset = {
            "images": deepcopy(ground_truth.dataset["images"]),
            "categories": deepcopy(ground_truth.dataset["categories"]),
            "annotations": [],
        }
        detections.createIndex()
    evaluator = COCOeval(ground_truth, detections, "bbox")
    evaluator.params.imgIds = list(image_ids)
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    metrics = {
        key: float(value) if value >= 0 else None
        for key, value in zip(COCO_METRICS, evaluator.stats)
    }
    return {
        **metrics,
        "images_evaluated": len(image_ids),
        "detections": len(rows),
        "evaluator": "pycocotools.COCOeval bbox; original annotations including iscrowd",
        "max_dets": [1, 10, 100],
        "unit": "fraction_0_to_1; null means no eligible ground truth",
    }


def evaluate(adapter, dataset: CocoImages, image_ids: list[int], output: Path) -> dict:
    category_ids = {category["id"] for category in dataset.categories}
    rows = []
    for image_id in tqdm(image_ids, desc="COCO predictions"):
        prediction = adapter.predict(dataset.read(image_id))
        rows.extend(prediction.coco_rows(image_id, category_ids))
    # Save original-coordinate predictions so evaluation can be inspected and rerun.
    write_json(output, rows)
    metrics = official_metrics(dataset.annotations, image_ids, rows)
    metrics["predictions_file"] = output.name
    return metrics


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(
    adapter, images: list[tuple[int, np.ndarray]], *, warmup: int, iterations: int, device: str
) -> dict:
    if warmup < 1 or not images or iterations < len(images):
        raise ValueError("Warmup must be >=1 and iterations must cover every benchmark image")
    selected_device = torch.device(device)
    # Capture every distinct image's actual tensor shape outside measured intervals.
    geometry = [{"image_id": image_id, **adapter.inspect_geometry(rgb)} for image_id, rgb in images]
    for index in range(warmup):
        adapter.predict(images[index % len(images)][1])
    synchronize(selected_device)
    samples = []
    for index in tqdm(range(iterations), desc="CPU-to-CPU latency"):
        image_id, rgb = images[index % len(images)]
        synchronize(selected_device)
        start = time.perf_counter_ns()
        result = adapter.predict(rgb)
        synchronize(selected_device)
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        samples.append(
            {"image_id": image_id, "latency_ms": elapsed_ms, "detections": len(result.scores)}
        )
        # Releasing the previous output cannot be charged to the next measured inference.
        del result
    values = [sample["latency_ms"] for sample in samples]
    return {
        "p50_ms": statistics.median(values),
        "p95_ms": float(np.percentile(values, 95, method="linear")),
        "mean_ms": statistics.mean(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "percentile_method": "numpy linear interpolation",
        "warmup": warmup,
        "iterations": iterations,
        "image_count": len(images),
        "batch_size": 1,
        "torch_threads": torch.get_num_threads(),
        "setup_and_shape_probes_excluded": True,
        "timed_scope": TIMED_SCOPE,
        "device": str(selected_device),
        "precision": "fp32",
        "cuda_synchronize_before_and_after": selected_device.type == "cuda",
        "geometry": geometry,
        "samples": samples,
    }
