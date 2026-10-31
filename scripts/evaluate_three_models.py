"""Evaluate Faster R-CNN, Lite R-CNN v2 and v3 on one shared COCO subset."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml
from pycocotools.cocoeval import COCOeval
from torchvision.ops import box_iou
from tqdm import tqdm

from light_rcnn.data import CocoTensorDataset
from light_rcnn.model import build_detector
from light_rcnn.model.detector import light_rcnn_from_config

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_checkpoint(path: Path, device: torch.device) -> tuple[dict, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint, checkpoint.get("config", {})


def build_model(name: str, weight_path: Path | None, config_path: Path, device: torch.device):
    config = load_yaml(config_path)
    checkpoint = {}
    saved_config = {}
    if weight_path is not None:
        checkpoint, saved_config = load_checkpoint(weight_path, device)
    if name == "Faster R-CNN":
        cfg = config
        model = build_detector(
            "faster_rcnn", pretrained=weight_path is None,
            num_classes=cfg["num_classes"], min_size=cfg["image_size"], max_size=cfg["image_size"],
        )
    else:
        cfg = saved_config or config
        model = light_rcnn_from_config(cfg)
    if weight_path is not None:
        model.load_state_dict(checkpoint.get("model", checkpoint))
    return model.to(device).eval(), int(cfg.get("image_size", 640))


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def coco_stats(dataset, image_ids: list[int], records: list[dict]) -> dict:
    if records:
        detections = dataset.coco.loadRes(records)
    else:
        from copy import deepcopy
        from pycocotools.coco import COCO

        detections = COCO()
        detections.dataset = {
            "images": deepcopy(dataset.coco.dataset["images"]),
            "categories": deepcopy(dataset.coco.dataset["categories"]),
            "annotations": [],
        }
        detections.createIndex()
    evaluator = COCOeval(dataset.coco, detections, "bbox")
    evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    values = evaluator.stats
    return {
        "AP": float(values[0]), "mAP_50_95": float(values[0]),
        "AP50": float(values[1]), "AP75": float(values[2]),
        "AP_S": float(values[3]), "AP_M": float(values[4]), "AP_L": float(values[5]),
        "AR@1": float(values[6]), "AR@10": float(values[7]), "AR@100": float(values[8]),
        "AR_S": float(values[9]), "AR_M": float(values[10]), "AR_L": float(values[11]),
    }


def precision_recall(prediction: dict, target: dict, iou_threshold: float = 0.5,
                     score_threshold: float = 0.5) -> tuple[int, int, int]:
    boxes = prediction["boxes"].detach().cpu()
    labels = prediction["labels"].detach().cpu()
    scores = prediction["scores"].detach().cpu()
    gt_boxes, gt_labels = target["boxes"], target["labels"]
    order = torch.argsort(scores, descending=True)
    matched: set[int] = set()
    tp = fp = 0
    for index in order.tolist():
        if float(scores[index]) < score_threshold:
            continue
        candidates = (gt_labels == labels[index]).nonzero(as_tuple=True)[0]
        candidates = [int(i) for i in candidates if int(i) not in matched]
        if candidates:
            overlaps = box_iou(boxes[index:index + 1], gt_boxes[candidates])[0]
            best_offset = int(overlaps.argmax())
            if float(overlaps[best_offset]) >= iou_threshold:
                matched.add(candidates[best_offset])
                tp += 1
                continue
        fp += 1
    return tp, fp, len(gt_boxes) - len(matched)


def proposal_recall(proposals: torch.Tensor, gt_boxes: torch.Tensor, limits=(100, 300)) -> dict:
    if not len(gt_boxes):
        return {f"R@{count}_IoU50": None for count in limits} | {
            f"R@{count}_IoU75": None for count in limits
        }
    result = {}
    for count in limits:
        candidate = proposals[:count].detach().cpu()
        overlaps = box_iou(gt_boxes, candidate) if len(candidate) else torch.zeros((len(gt_boxes), 0))
        best = overlaps.max(dim=1).values if len(candidate) else torch.zeros(len(gt_boxes))
        result[f"R@{count}_IoU50"] = float((best >= 0.5).float().mean())
        result[f"R@{count}_IoU75"] = float((best >= 0.75).float().mean())
    return result


def count_params(model) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def get_rpn_limit(model) -> dict | None:
    rpn = getattr(model, "rpn", None)
    # torchvision versions differ: some expose a dict publicly, while newer
    # versions expose post_nms_top_n() as a method backed by this private dict.
    for attribute in ("_post_nms_top_n", "post_nms_top_n"):
        value = getattr(rpn, attribute, None)
        if isinstance(value, dict):
            return value
    return None


def set_rpn_limit(model, value: int) -> None:
    limits = get_rpn_limit(model)
    if limits is not None:
        limits["testing"] = value


def stage_profile(model, image: torch.Tensor, device: torch.device, repeats: int = 5) -> dict:
    samples = {key: [] for key in ("backbone", "bifpn", "rpn", "roi_head")}
    backbone = getattr(model, "backbone", None)
    backbone_stage = getattr(backbone, "body", None) if getattr(backbone, "neck", None) is not None else backbone
    modules = {
        "backbone": backbone_stage,
        "bifpn": getattr(backbone, "neck", None),
        "rpn": getattr(model, "rpn", None),
        "roi_head": getattr(model, "roi_heads", None),
    }
    handles = []
    starts = {}
    for key, module in modules.items():
        if module is None:
            continue
        def before(_module, _inputs, stage=key):
            sync(device)
            starts[stage] = time.perf_counter()
        def after(_module, _inputs, _output, stage=key):
            sync(device)
            samples[stage].append((time.perf_counter() - starts[stage]) * 1000)
        handles.append(module.register_forward_pre_hook(before))
        handles.append(module.register_forward_hook(after))
    try:
        with torch.inference_mode():
            for _ in range(repeats):
                model([image])
    finally:
        for handle in handles:
            handle.remove()
    return {key: statistics.median(value) if value else None for key, value in samples.items()}


def benchmark(model, images: list[torch.Tensor], device: torch.device, warmup: int,
              iterations: int, proposal_limit: int) -> dict:
    limits = get_rpn_limit(model)
    original = limits.get("testing") if limits is not None else None
    if limits is not None:
        limits["testing"] = proposal_limit
    try:
        with torch.inference_mode():
            for index in tqdm(range(warmup), desc=f"{proposal_limit} proposals warmup",
                              unit="iter", dynamic_ncols=True):
                model([images[index % len(images)]])
            sync(device)
            times = []
            for index in tqdm(range(iterations), desc=f"{proposal_limit} proposals latency",
                              unit="iter", dynamic_ncols=True):
                image = images[index % len(images)]
                sync(device)
                start = time.perf_counter()
                model([image])
                sync(device)
                times.append((time.perf_counter() - start) * 1000)
    finally:
        if limits is not None and original is not None:
            limits["testing"] = original
    p50 = statistics.median(times)
    return {"latency_p50_ms": p50, "latency_p95_ms": percentile(times, 0.95),
            "fps": 1000.0 / p50, "latency_iterations": iterations}


def estimate_flops(model, image: torch.Tensor, device: torch.device) -> dict:
    """Torch profiler FLOPs cover supported Conv/MatMul operators, not all detection ops."""
    try:
        from torch.profiler import ProfilerActivity, profile

        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        with profile(activities=activities, with_flops=True) as prof:
            with torch.inference_mode():
                model([image])
        total = sum(int(event.flops or 0) for event in prof.key_averages())
        return {"FLOPs_G_supported_ops": total / 1e9,
                "FLOPs_coverage": "torch.profiler-supported Conv2d/MatMul ops; detection, ROIAlign and NMS may be omitted"}
    except Exception as exc:  # profiler availability varies by PyTorch build
        return {"FLOPs_G_supported_ops": None, "FLOPs_error": str(exc)}


def evaluate_one(name: str, weight_path: Path | None, config_path: Path, dataset,
                 indices: list[int], device: torch.device, args) -> dict:
    model, _image_size = build_model(name, weight_path, config_path, device)
    if hasattr(model, "transform"):
        model.transform.min_size = (args.image_size,)
        model.transform.max_size = args.image_size
    total_params, trainable_params = count_params(model)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    records, precisions = [], [0, 0, 0]
    proposal_sums = {"R@100_IoU50": 0.0, "R@100_IoU75": 0.0,
                     "R@300_IoU50": 0.0, "R@300_IoU75": 0.0}
    proposal_images = 0
    rpn_output = {}
    handle = model.rpn.register_forward_hook(
        lambda _module, _inputs, output: rpn_output.update(proposals=output[0])
    )
    started = time.perf_counter()
    try:
        for index in tqdm(indices, desc=f"{name} validation", unit="image", dynamic_ncols=True):
            image, target = dataset[index]
            image = image.to(device)
            with torch.inference_mode():
                prediction = model([image])[0]
            image_id = int(target["image_id"].item())
            for box, score, label in zip(prediction["boxes"].cpu().tolist(),
                                          prediction["scores"].cpu().tolist(),
                                          prediction["labels"].cpu().tolist()):
                x1, y1, x2, y2 = box
                records.append({"image_id": image_id, "category_id": int(label),
                                "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score)})
            tp, fp, fn = precision_recall(prediction, target)
            precisions[0] += tp
            precisions[1] += fp
            precisions[2] += fn
            proposals = rpn_output.get("proposals", [torch.empty((0, 4))])[0]
            recall_values = proposal_recall(proposals, target["boxes"])
            if len(target["boxes"]):
                proposal_images += 1
                for key in proposal_sums:
                    proposal_sums[key] += recall_values[key]
    finally:
        handle.remove()
    sync(device)
    elapsed = time.perf_counter() - started
    image_ids = [dataset.ids[index] for index in indices]
    metrics = coco_stats(dataset, image_ids, records)
    tp, fp, fn = precisions
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    print("Validation inference complete; measuring latency and model stages...", flush=True)
    timing_images = [dataset[i][0].to(device) for i in indices[:min(8, len(indices))]]
    latency = benchmark(model, timing_images, device, args.warmup, args.iterations, 300)
    lat100 = benchmark(model, timing_images, device, args.warmup, args.iterations, 100)
    stages = stage_profile(model, timing_images[0], device, repeats=max(3, args.stage_repeats))
    flops = estimate_flops(model, timing_images[0], device)
    vram = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
    return {
        "model": name, "checkpoint": str(weight_path) if weight_path else "torchvision COCO_V1",
        "images_evaluated": len(indices), "image_size": args.image_size,
        **metrics, "Precision_IoU50_score0.5": precision, "Recall_IoU50_score0.5": recall,
        "precision_recall_definition": "global detection counts; score >= 0.5; class-aware greedy IoU >= 0.5",
        "parameters": total_params, "parameters_M": total_params / 1e6,
        "trainable_parameters": trainable_params,
        "checkpoint_size_MB": weight_path.stat().st_size / 1024**2 if weight_path else None,
        "model_state_size_MB_fp32": total_params * 4 / 1024**2,
        "model_size_MB": total_params * 4 / 1024**2,
        "peak_VRAM_allocated_MB": vram,
        **latency, "latency_p50_100_proposals_ms": lat100["latency_p50_ms"],
        "latency_p50_300_proposals_ms": latency["latency_p50_ms"],
        "speedup_vs_faster_rcnn": None,
        "delta_AP_vs_faster_rcnn_pp": None,
        "stage_latency_ms": stages,
        "backbone_latency_ms": stages["backbone"],
        "bifpn_latency_ms": stages["bifpn"],
        "rpn_latency_ms": stages["rpn"], "roi_head_latency_ms": stages["roi_head"],
        "proposal_recall": {key: value / proposal_images if proposal_images else None
                             for key, value in proposal_sums.items()},
        "proposal_recall_images_with_gt": proposal_images,
        **flops,
        "FLOPs_G": flops.get("FLOPs_G_supported_ops"),
        "evaluation_wall_time_seconds": elapsed,
        "device": str(device), "precision_mode": "fp32",
        "throughput_scope": "batch=1, model forward including transform and NMS; image decode excluded",
    }


def write_reports(results: list[dict], out_dir: Path, metadata: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "results": results}
    (out_dir / "evaluation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    flat = []
    for row in results:
        copy = dict(row)
        copy.pop("stage_latency_ms", None)
        copy.pop("proposal_recall", None)
        for key, value in row["stage_latency_ms"].items():
            copy[f"stage_{key}_ms"] = value
        for key, value in row["proposal_recall"].items():
            copy[f"proposal_{key}"] = value
        flat.append(copy)
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    lines = ["# Three-model detection evaluation", "",
             f"- Images: {metadata['images_evaluated']} shared validation images (seed {metadata['seed']})",
             f"- Device: {metadata['device']}; image size: {metadata['image_size']}; batch size: 1", "",
             "| Model | AP/mAP | AP50 | AP75 | AP_S | AP_M | AP_L | Precision | Recall | p50 ms | FPS | Params M | FLOPs G* |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in results:
        lines.append("| {model} | {AP:.4f} | {AP50:.4f} | {AP75:.4f} | {AP_S:.4f} | {AP_M:.4f} | {AP_L:.4f} | {Precision_IoU50_score0.5:.4f} | {Recall_IoU50_score0.5:.4f} | {latency_p50_ms:.2f} | {fps:.2f} | {parameters_M:.3f} | {flops} |".format(
            **row, flops=(f"{row['FLOPs_G_supported_ops']:.3f}" if row["FLOPs_G_supported_ops"] is not None else "N/A")))
    lines.extend(["", "* FLOPs sums operators counted by PyTorch profiler; ROIAlign, NMS and other unsupported ops can be omitted.",
                  "Precision/Recall use score >= 0.5 and class-aware IoU >= 0.5. AP is COCO AP@[.50:.95] and equals mAP_50_95.",
                  "Faster R-CNN has no BiFPN module; its BiFPN latency is null. VRAM is peak allocated CUDA memory.",
                  "See evaluation.json and summary.csv for stage times, proposal recall, AR@100, speedup and AP deltas."])
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-images", default="data/coco/images/val2017")
    parser.add_argument("--annotations", default="data/coco/annotations/instances_val2017.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--images", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--faster-config", default="configs/coco.yaml")
    parser.add_argument("--v2-config", default="configs/coco.yaml")
    parser.add_argument("--v3-config", default="configs/coco_v3.yaml")
    parser.add_argument("--faster-weights", help="Optional Faster R-CNN state dict; default torchvision COCO_V1")
    parser.add_argument("--v2-weights", default="outputs/coco_4090_v2/best.pth")
    parser.add_argument("--v3-weights", default="outputs/coco_4090_v3/best.pth")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--stage-repeats", type=int, default=5)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable. Use --device cpu or install CUDA PyTorch.")
    image_root = Path(args.val_images)
    annotation_path = Path(args.annotations)
    if not image_root.is_absolute():
        image_root = ROOT / image_root
    if not annotation_path.is_absolute():
        annotation_path = ROOT / annotation_path
    for path in (image_root, annotation_path):
        if not path.exists():
            raise SystemExit(f"Required COCO validation path does not exist: {path}")
    dataset = CocoTensorDataset(image_root, str(annotation_path))
    count = min(args.images, len(dataset))
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
    weights = {
        "Faster R-CNN": Path(args.faster_weights).resolve() if args.faster_weights else None,
        "Lite R-CNN v2": (ROOT / args.v2_weights).resolve(),
        "Lite R-CNN v3": (ROOT / args.v3_weights).resolve(),
    }
    configs = {"Faster R-CNN": ROOT / args.faster_config,
               "Lite R-CNN v2": ROOT / args.v2_config, "Lite R-CNN v3": ROOT / args.v3_config}
    for name, path in weights.items():
        if path is not None and not path.is_file():
            raise SystemExit(f"Checkpoint not found for {name}: {path}")
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "outputs" / "evaluation_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"images_evaluated": count, "seed": args.seed, "device": str(device),
                "image_size": args.image_size, "evaluation_timestamp": datetime.now().isoformat(timespec="seconds"),
                "subset_image_ids": [int(dataset.ids[index]) for index in indices],
                "latency_warmup": args.warmup, "latency_iterations": args.iterations}
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
    results = []
    for name in ("Faster R-CNN", "Lite R-CNN v2", "Lite R-CNN v3"):
        print(f"\n=== {name}: evaluating {count} shared images ===", flush=True)
        results.append(evaluate_one(name, weights[name], configs[name], dataset, indices, device, args))
        partial_metadata = {**metadata, "models_completed": [row["model"] for row in results]}
        (output_dir / "partial_results.json").write_text(
            json.dumps({"metadata": partial_metadata, "results": results}, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(results[-1], indent=2), flush=True)
    baseline = next(row for row in results if row["model"] == "Faster R-CNN")
    for row in results:
        row["speedup_vs_faster_rcnn"] = baseline["latency_p50_ms"] / row["latency_p50_ms"]
        row["delta_AP_vs_faster_rcnn_pp"] = (row["AP"] - baseline["AP"]) * 100
    write_reports(results, output_dir, metadata)
    print(f"\nReports written to: {output_dir}", flush=True)
    print(f"  {output_dir / 'REPORT.md'}\n  {output_dir / 'summary.csv'}\n  {output_dir / 'evaluation.json'}", flush=True)


if __name__ == "__main__":
    main()
