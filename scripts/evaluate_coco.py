"""Evaluate a Faster R-CNN model on COCO val2017 with pycocotools AP metrics (schema-v2 report)."""

import argparse
import json
import sys
from pathlib import Path
from urllib.error import URLError

import torch
import yaml

from light_rcnn.bench import (
    DEFAULT_WEIGHTS,
    load_comparison_model,
    resolve_device,
    set_detection_limits,
)
from light_rcnn.coco_eval import STAT_NAMES
from light_rcnn.data import CocoTensorDataset
from light_rcnn.evaluation import evaluate_coco, subset_record
from light_rcnn.reporting import (
    build_report,
    collect_environment,
    dataset_info,
    default_report_path,
    display_path,
    file_sha256,
    write_report,
)

SHARED_CONFIG = Path(__file__).resolve().parents[1] / "configs/coco.yaml"
# Checkpoint configs saved before comparison.score_threshold/max_detections existed lack them.
DEFAULT_SCORE_THRESHOLD = 0.001
DEFAULT_MAX_DETECTIONS = 100
DIAGNOSTICS = (
    "images_with_detections",
    "max_detection_score",
    "mean_best_iou_any_class",
    "mean_best_iou_same_class",
    "images_with_iou50_any_class",
    "images_with_iou50_same_class",
)


def read_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_config(config_path: str | None, weights: str | None) -> tuple[dict, str]:
    """--config when given, else the config saved in the checkpoint, else configs/coco.yaml."""
    if not config_path and weights:
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), dict):
            return checkpoint["config"], f"{display_path(weights)} (checkpoint config)"
    path = config_path or SHARED_CONFIG
    return read_yaml(path), display_path(path)


def subset_seed(config: dict, config_path: str | None) -> int:
    """Seed for --limit: from --config, else configs/coco.yaml as in evaluate_yolo26.py.

    A checkpoint's training seed can differ (continuous_research.py varies it per round), which
    would score a different subset than the other models.
    """
    if not config_path and SHARED_CONFIG.is_file():
        return read_yaml(SHARED_CONFIG).get("seed", 0)
    return config.get("seed", 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Dataset/comparison config; defaults to the config saved "
                                         "in --weights (the architecture always comes from it) "
                                         "with the --limit seed from configs/coco.yaml")
    parser.add_argument("--model", choices=("light_rcnn", "faster_rcnn"), required=True)
    parser.add_argument("--weights", help="Required for trained Light R-CNN; "
                                          "optional for the COCO-pretrained baseline")
    parser.add_argument("--device", help="PyTorch device; defaults to the config device")
    parser.add_argument("--imgsz", "--size", dest="size", type=int,
                        help="Override config image_size")
    parser.add_argument("--images", help="Defaults to the config val_images")
    parser.add_argument("--annotations", help="Defaults to the config val_annotations")
    parser.add_argument("--limit", type=int, default=0,
                        help="Seeded random subset (seed from --config, else configs/coco.yaml); "
                             "0 evaluates the full val set")
    parser.add_argument("--batch-size", type=int,
                        help="Evaluation batch size; defaults to comparison.batch_size (1). "
                             "Batches above 1 pad images and change R-CNN detections")
    parser.add_argument("--score-thresh", type=float, help="Override comparison.score_threshold")
    parser.add_argument("--max-dets", type=int, help="Override comparison.max_detections")
    parser.add_argument("--output", help="Defaults to outputs/reports/<model>_accuracy.json")
    args = parser.parse_args()
    if args.model == "light_rcnn" and not args.weights:
        parser.error("--weights is required to evaluate trained Light R-CNN accuracy")

    config, config_source = load_config(args.config, args.weights)
    comparison = config.get("comparison") or {}
    image_size = int(args.size if args.size is not None else config["image_size"])
    score_threshold = (args.score_thresh if args.score_thresh is not None
                       else comparison.get("score_threshold", DEFAULT_SCORE_THRESHOLD))
    max_detections = (args.max_dets if args.max_dets is not None
                      else comparison.get("max_detections", DEFAULT_MAX_DETECTIONS))
    # Not train.val_batch_size: padding to a shared batch size changes R-CNN detections and AP.
    batch_size = (args.batch_size if args.batch_size is not None
                  else comparison.get("batch_size", 1))
    if args.limit < 0 or max_detections < 1 or batch_size < 1 or image_size < 1:
        parser.error("--limit must be >= 0; --max-dets, --batch-size and --imgsz must be >= 1")
    images_dir = args.images or config["data"]["val_images"]
    annotations = args.annotations or config["data"]["val_annotations"]
    seed = subset_seed(config, args.config)

    try:
        device = resolve_device(args.device or config.get("device", "cuda"), "evaluating")
    except ValueError as exc:
        parser.error(f"--device: {exc}")
    try:
        model = load_comparison_model(args.model, config, args.weights, image_size)
    except URLError as exc:
        raise SystemExit(f"Could not download default {args.model} weights ({exc}). "
                         "Pass --weights <checkpoint>.") from exc
    set_detection_limits(model, score_threshold, max_detections)
    model.to(device).eval()

    dataset = CocoTensorDataset(images_dir, annotations)
    metrics = evaluate_coco(model, dataset, device, args.limit, subset_seed=seed,
                            batch_size=batch_size, detailed=True)
    if not metrics["detections"]:
        print(f"WARNING: {args.model} returned no detections above score {score_threshold} on "
              f"{metrics['images_evaluated']} images; writing an all-zero AP report.",
              file=sys.stderr)

    accuracy = {
        "evaluator": "pycocotools",
        **subset_record(metrics["image_ids"], len(dataset.ids), seed),
        "score_threshold": score_threshold,
        "max_detections": max_detections,
        "eval_seconds": metrics["inference_seconds"],  # inference only, as in evaluate_yolo26.py
        "dataset": dataset_info(annotations, images_dir),
        **{key: metrics[key] for key in STAT_NAMES},
        "per_category_ap": metrics["per_category_ap"],
        "detections": metrics["detections"],
        "coco_max_dets": metrics["coco_max_dets"],
        "diagnostics": {key: metrics[key] for key in DIAGNOSTICS},
    }
    report = build_report(
        "accuracy",
        args.model,
        weights=display_path(args.weights) if args.weights else DEFAULT_WEIGHTS[args.model],
        weights_sha256=file_sha256(args.weights),
        config=config_source,
        image_size=image_size,
        batch_size=batch_size,
        precision="fp32",
        environment=collect_environment(str(device)),
        accuracy=accuracy,
        # Pre-v2 reports kept this at the top level; older readers still look for it there.
        images_evaluated=accuracy["images_evaluated"],
    )
    output = write_report(report, args.output or default_report_path("accuracy", args.model))
    hidden = {"per_category_ap", "image_ids"}
    print(json.dumps({
        "output": str(output),
        "model": report["model"],
        "weights": report["weights"],
        "config": report["config"],
        "device": str(device),
        "image_size": report["image_size"],
        "batch_size": report["batch_size"],
        "precision": report["precision"],
        "accuracy": {key: value for key, value in accuracy.items() if key not in hidden},
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
