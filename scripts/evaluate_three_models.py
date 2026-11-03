"""Evaluate Faster R-CNN, Lite R-CNN v2 and v3 on one shared COCO subset."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from urllib.error import URLError

import torch
import yaml
from torchvision.models.detection.transform import resize_boxes
from torchvision.ops import box_iou
from torchvision.ops.misc import FrozenBatchNorm2d
from tqdm import tqdm

from light_rcnn.bench import load_comparison_model
from light_rcnn.coco_eval import summarize_detections
from light_rcnn.data import CocoTensorDataset
from light_rcnn.evaluation import evaluation_subset
from light_rcnn.model.deploy import fuse_conv_bn
from light_rcnn.reporting import collect_environment, dataset_info, git_commit, percentile

ROOT = Path(__file__).resolve().parents[1]
FASTER_RCNN = "Faster R-CNN"
MODELS = (FASTER_RCNN, "Lite R-CNN v2", "Lite R-CNN v3")
# Test-time RPN proposals of every model for AP, proposal recall, stage times, FLOPs and the main
# latency (Lite R-CNN's trained budget; Faster R-CNN's own default of 1000 is overridden).
PROPOSALS = 300
LATENCY_PROPOSALS = (PROPOSALS, 100)
# Report column -> light_rcnn.coco_eval stat.
COCO_COLUMNS = {
    "AP": "AP_50_95", "mAP_50_95": "AP_50_95", "AP50": "AP_50", "AP75": "AP_75",
    "AP_S": "AP_small", "AP_M": "AP_medium", "AP_L": "AP_large",
    "AR@1": "AR_1", "AR@10": "AR_10", "AR@100": "AR_100",
    "AR_S": "AR_small", "AR_M": "AR_medium", "AR_L": "AR_large",
}
FLOPS_COVERAGE = ("torch.profiler-supported Conv2d/MatMul ops; detection, ROIAlign and NMS may be "
                  "omitted")
# Last column of the original summary.csv layout; columns added later are written after it.
LAST_ORIGINAL_COLUMN = "throughput_scope"
# What REPORT.md says about the protocol changes since the 2026-11-01 runs.
REPORT_NOTES = (
    ("지연 시간은 같은 이미지들을 차례로 반복해 잽니다. p50/p95/p99는 전체 측정값의 백분위수(선형 보간)이고, "
     "'p95 of image medians'는 이미지별 중앙값의 95번째 백분위수라 느린 이미지와 일시적인 지연을 구분해 "
     "줍니다. 원시 측정값과 이미지 ID는 evaluation.json의 latency_runs에 있습니다."),
    (f"Faster R-CNN의 기본 RPN 제안 수(1000개)를 Lite R-CNN과 같은 {PROPOSALS}개로 맞춰 AP와 지연 시간을 "
     "같은 설정에서 잽니다. 이전 버전 스크립트는 Faster R-CNN의 AP를 1000개로, 지연 시간을 300개로 쟀습니다."),
    ("제안 재현율(proposal_R@...)은 RPN 제안을 원본 이미지 좌표로 되돌린 뒤(torchvision resize_boxes) 정답 "
     "상자와 비교합니다. 이전 버전 스크립트는 리사이즈된 좌표의 제안을 원본 좌표의 정답과 비교해, 크기가 바뀐 "
     "이미지의 재현율을 낮게 쟀습니다."),
    ("Conv+BN 융합을 적용하면 모든 모델(Faster R-CNN의 FrozenBatchNorm2d 포함)의 BatchNorm을 앞 Conv에 합친 "
     "배포용 그래프로 AP와 지연 시간을 잽니다. 결과는 부동소수점 오차 범위에서 같고, 매개변수 수는 융합 전 "
     "모델 기준입니다."),
    ("최대 VRAM은 검증 추론(배치 1) 동안의 최대 할당량으로, 지연 시간 측정용 이미지를 장치에 올리기 전에 "
     "읽습니다."),
)


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_model(name: str, weight_path: Path | None, config_path: Path, device: torch.device,
                image_size: int) -> torch.nn.Module:
    """A Lite R-CNN checkpoint's own config decides its architecture; Faster R-CNN uses
    torchvision's COCO weights unless a state dict is given."""
    kind = "faster_rcnn" if name == FASTER_RCNN else "light_rcnn"
    weights = str(weight_path) if weight_path is not None else None
    try:
        model = load_comparison_model(kind, load_yaml(config_path), weights, image_size)
    except URLError as exc:
        raise SystemExit(
            f"{name} 기본 가중치(torchvision COCO_V1)를 내려받지 못했습니다: {exc}\n"
            "인터넷 연결을 확인하거나 --faster-weights <state_dict.pth>로 로컬 가중치를 지정하세요. "
            "Faster R-CNN 없이 Lite R-CNN v2/v3만 평가하려면 --skip-faster-rcnn을 붙이세요."
        ) from exc
    return model.to(device).eval()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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


def get_rpn_limit(model, stage: str = "post") -> dict | None:
    """The RPN's {"training", "testing"} top-n dict for ``stage`` 'post' or 'pre' NMS."""
    rpn = getattr(model, "rpn", None)
    # torchvision versions differ: some expose a dict publicly, while newer
    # versions expose post_nms_top_n() as a method backed by this private dict.
    for attribute in (f"_{stage}_nms_top_n", f"{stage}_nms_top_n"):
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


def latency_stats(samples_ms: list[float], sample_image_ids: list[int],
                  image_ids: list[int]) -> dict:
    """Percentiles plus the raw samples and the image each one timed.

    Cycling over a few images, p95/p99 of all samples mix slow images with random stalls; the
    p95 of per-image medians shows the slow images alone.
    """
    per_image: dict[int, list[float]] = {image_id: [] for image_id in image_ids}
    for value, image_id in zip(samples_ms, sample_image_ids, strict=True):
        per_image[image_id].append(value)
    medians = [statistics.median(values) if values else None for values in per_image.values()]
    return {
        "iterations": len(samples_ms),
        "p50_ms": statistics.median(samples_ms),
        "p95_ms": percentile(samples_ms, 95),
        "p99_ms": percentile(samples_ms, 99),
        "p95_of_image_medians_ms": percentile([m for m in medians if m is not None], 95),
        "image_ids": list(image_ids),
        "image_medians_ms": medians,
        "samples_ms": [round(value, 4) for value in samples_ms],
        "sample_image_ids": list(sample_image_ids),
    }


def benchmark(model, images: list[torch.Tensor], image_ids: list[int], device: torch.device,
              warmup: int, iterations: int, proposal_limit: int) -> dict:
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
            times, timed_ids = [], []
            for index in tqdm(range(iterations), desc=f"{proposal_limit} proposals latency",
                              unit="iter", dynamic_ncols=True):
                position = index % len(images)
                sync(device)
                start = time.perf_counter()
                model([images[position]])
                sync(device)
                times.append((time.perf_counter() - start) * 1000)
                timed_ids.append(image_ids[position])
    finally:
        if limits is not None and original is not None:
            limits["testing"] = original
    return {"proposals": proposal_limit, "warmup": warmup,
            **latency_stats(times, timed_ids, image_ids)}


def estimate_flops(model, image: torch.Tensor, device: torch.device) -> dict:
    """Torch profiler FLOPs cover supported Conv/MatMul operators, not all detection ops."""
    try:
        from torch.profiler import ProfilerActivity, profile

        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        with profile(activities=activities, with_flops=True) as prof, torch.inference_mode():
            model([image])
        total = sum(int(event.flops or 0) for event in prof.key_averages())
        return {"FLOPs_G_supported_ops": total / 1e9, "FLOPs_coverage": FLOPS_COVERAGE}
    except Exception as exc:  # noqa: BLE001 - profiler availability varies by PyTorch build
        return {"FLOPs_G_supported_ops": None, "FLOPs_coverage": FLOPS_COVERAGE,
                "FLOPs_error": str(exc)}


def evaluate_one(name: str, weight_path: Path | None, config_path: Path, dataset,
                 indices: list[int], timing_indices: list[int], warmup: int,
                 device: torch.device, args) -> dict:
    model = build_model(name, weight_path, config_path, device, args.image_size)
    total_params, trainable_params = count_params(model)  # as trained, before Conv+BN folding
    native_proposals = (get_rpn_limit(model) or {}).get("testing")
    set_rpn_limit(model, PROPOSALS)  # score AP with the proposal budget latency is timed at
    fused_layers = fuse_conv_bn(model) if args.fuse else 0
    norms_left = any(isinstance(module, (torch.nn.BatchNorm2d, FrozenBatchNorm2d))
                     for module in model.modules())
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    records, precisions = [], [0, 0, 0]
    proposal_sums = {"R@100_IoU50": 0.0, "R@100_IoU75": 0.0,
                     "R@300_IoU50": 0.0, "R@300_IoU75": 0.0}
    proposal_images = 0
    rpn_output = {}

    def capture(_module, inputs, output):
        rpn_output.update(proposals=output[0], image_sizes=inputs[0].image_sizes)

    handle = model.rpn.register_forward_hook(capture)
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
            # Proposals are in the resized input frame and the targets in the original image
            # frame: map the proposals back the way the transform maps detections back.
            proposals = resize_boxes(rpn_output["proposals"][0], rpn_output["image_sizes"][0],
                                     image.shape[-2:])
            recall_values = proposal_recall(proposals, target["boxes"])
            if len(target["boxes"]):
                proposal_images += 1
                for key in proposal_sums:
                    proposal_sums[key] += recall_values[key]
    finally:
        handle.remove()
    sync(device)
    elapsed = time.perf_counter() - started
    # Inference footprint at batch 1, read before the timing images are put on the device.
    vram = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
    image_ids = [int(dataset.ids[index]) for index in indices]
    coco = summarize_detections(dataset.coco, records, image_ids)
    metrics = {column: coco[stat] for column, stat in COCO_COLUMNS.items()}
    tp, fp, fn = precisions
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    print("Validation inference complete; measuring latency and model stages...", flush=True)
    timing_images = [dataset[index][0].to(device) for index in timing_indices]
    timing_ids = [int(dataset.ids[index]) for index in timing_indices]
    runs = [benchmark(model, timing_images, timing_ids, device, warmup, args.iterations, limit)
            for limit in LATENCY_PROPOSALS]
    latency, lat100 = runs
    stages = stage_profile(model, timing_images[0], device, repeats=max(3, args.stage_repeats))
    flops = estimate_flops(model, timing_images[0], device)
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
        "latency_p50_ms": latency["p50_ms"], "latency_p95_ms": latency["p95_ms"],
        "fps": 1000.0 / latency["p50_ms"], "latency_iterations": args.iterations,
        "latency_p50_100_proposals_ms": lat100["p50_ms"],
        "latency_p50_300_proposals_ms": latency["p50_ms"],
        "speedup_vs_faster_rcnn": None,
        "delta_AP_vs_faster_rcnn_pp": None,
        "stage_latency_ms": stages,
        "backbone_latency_ms": stages["backbone"],
        "bifpn_latency_ms": stages["bifpn"],
        "rpn_latency_ms": stages["rpn"], "roi_head_latency_ms": stages["roi_head"],
        "proposal_recall": {key: value / proposal_images if proposal_images else None
                             for key, value in proposal_sums.items()},
        "proposal_recall_images_with_gt": proposal_images,
        "FLOPs_G_supported_ops": flops["FLOPs_G_supported_ops"],
        "FLOPs_coverage": flops["FLOPs_coverage"],
        "FLOPs_G": flops["FLOPs_G_supported_ops"],
        "evaluation_wall_time_seconds": elapsed,
        "device": str(device), "precision_mode": "fp32",
        "throughput_scope": "batch=1, model forward including transform and NMS; image decode excluded",
        # Added after the 2026-11-01 runs; the CSV keeps them after the original columns.
        "latency_p99_ms": latency["p99_ms"],
        "latency_p95_of_image_medians_ms": latency["p95_of_image_medians_ms"],
        "latency_p95_100_proposals_ms": lat100["p95_ms"],
        "latency_p99_100_proposals_ms": lat100["p99_ms"],
        "latency_p95_of_image_medians_100_proposals_ms": lat100["p95_of_image_medians_ms"],
        "latency_warmup": warmup,
        "latency_timing_images": len(timing_ids),
        "rpn_pre_nms_top_n_test": (get_rpn_limit(model, "pre") or {}).get("testing"),
        "rpn_post_nms_top_n_test": (get_rpn_limit(model) or {}).get("testing"),
        "rpn_post_nms_top_n_native": native_proposals,
        "box_score_thresh": float(model.roi_heads.score_thresh),
        "detections_per_img": int(model.roi_heads.detections_per_img),
        "conv_bn_fused": fused_layers > 0 and not norms_left,
        "conv_bn_fused_layers": fused_layers,
        "proposal_recall_frame": "original_image",
        "FLOPs_error": flops.get("FLOPs_error"),
        "latency_timing_image_ids": timing_ids,
        "latency_runs": runs,
        "per_category_ap": coco["per_category_ap"],
    }


def csv_row(row: dict) -> dict:
    """Scalar columns in the original order, the flattened stage/recall columns where they used
    to end the file, then newer columns; lists and other nested values stay in the JSON."""
    flat = {}
    for key, value in row.items():
        if isinstance(value, (dict, list)):
            continue
        flat[key] = value
        if key == LAST_ORIGINAL_COLUMN:
            flat.update({f"stage_{stage}_ms": ms for stage, ms in row["stage_latency_ms"].items()})
            flat.update({f"proposal_{metric}": recall
                         for metric, recall in row["proposal_recall"].items()})
    return flat


def fmt(value, spec: str = ".4f") -> str:
    return "N/A" if value is None else format(value, spec)


def report_markdown(results: list[dict], metadata: dict) -> str:
    timing, iterations = metadata["latency_timing_images"], metadata["latency_iterations"]
    lines = ["# Three-model detection evaluation", "",
             f"- Images: {metadata['images_evaluated']} shared validation images (seed {metadata['seed']})",
             f"- Device: {metadata['device']}; image size: {metadata['image_size']}; batch size: 1",
             (f"- 지연 시간: 평가 이미지 중 시드 {metadata['seed']}로 고른 {timing}장, "
              f"예열 {metadata['latency_warmup']}회 뒤 {iterations}회 측정 (이미지당 약 {iterations // timing}회)"),
             (f"- RPN 제안 수: 모든 모델 {metadata['rpn_post_nms_top_n_test']}개 "
              "(AP, 제안 재현율, 단계별 시간, FLOPs, 지연 시간 공통)"),
             "- Conv+BN 융합: " + ("적용 (--fuse)" if metadata["fuse_conv_bn"] else "미적용 (--no-fuse)"),
             "",
             ("| Model | AP/mAP | AP50 | AP75 | AP_S | AP_M | AP_L | Precision | Recall | p50 ms | FPS "
              "| Params M | FLOPs G* | p95 ms | p99 ms | p95 of image medians ms | RPN proposals "
              "| Conv+BN fused |"),
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in results:
        proposals = str(row["rpn_post_nms_top_n_test"])
        if row["rpn_post_nms_top_n_native"] not in (None, row["rpn_post_nms_top_n_test"]):
            proposals += f" (native {row['rpn_post_nms_top_n_native']})"
        cells = [row["model"],
                 *(fmt(row[key]) for key in ("AP", "AP50", "AP75", "AP_S", "AP_M", "AP_L",
                                             "Precision_IoU50_score0.5", "Recall_IoU50_score0.5")),
                 fmt(row["latency_p50_ms"], ".2f"), fmt(row["fps"], ".2f"),
                 fmt(row["parameters_M"], ".3f"), fmt(row["FLOPs_G_supported_ops"], ".3f"),
                 fmt(row["latency_p95_ms"], ".2f"), fmt(row["latency_p99_ms"], ".2f"),
                 fmt(row["latency_p95_of_image_medians_ms"], ".2f"), proposals,
                 "yes" if row["conv_bn_fused"] else "no"]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "* FLOPs sums operators counted by PyTorch profiler; ROIAlign, NMS and other unsupported ops can be omitted.",
                  "Precision/Recall use score >= 0.5 and class-aware IoU >= 0.5. AP is COCO AP@[.50:.95] and equals mAP_50_95.",
                  "Faster R-CNN has no BiFPN module; its BiFPN latency is null. VRAM is peak allocated CUDA memory.",
                  "See evaluation.json and summary.csv for stage times, proposal recall, AR@100, speedup and AP deltas.",
                  "", *(f"- {note}" for note in REPORT_NOTES)])
    if metadata["skipped_models"]:
        lines.append(f"- 제외한 모델: {', '.join(metadata['skipped_models'])} (--skip-faster-rcnn). "
                     "Faster R-CNN 대비 속도 향상과 AP 차이는 비어 있습니다.")
    return "\n".join(lines) + "\n"


def write_reports(results: list[dict], out_dir: Path, metadata: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "results": results}
    (out_dir / "evaluation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    flat = [csv_row(row) for row in results]
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in flat for key in row)))
        writer.writeheader()
        writer.writerows(flat)
    (out_dir / "REPORT.md").write_text(report_markdown(results, metadata), encoding="utf-8")


def build_parser(description: str | None = __doc__, device: str = "cpu",
                 images: int = 1000) -> argparse.ArgumentParser:
    """Command-line options; evaluate_three_models_gpu_full.py changes only these defaults."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--val-images", default="data/coco/images/val2017")
    parser.add_argument("--annotations", default="data/coco/annotations/instances_val2017.json")
    parser.add_argument("--device", default=device, help="PyTorch device (default: %(default)s)")
    parser.add_argument("--images", type=int, default=images,
                        help="Shared validation images (default: %(default)s); a seeded random "
                             "subset unless it covers every image")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--faster-config", default="configs/coco.yaml")
    parser.add_argument("--v2-config", default="configs/coco.yaml")
    parser.add_argument("--v3-config", default="configs/coco_v3.yaml")
    parser.add_argument("--faster-weights", help="Optional Faster R-CNN state dict; default torchvision COCO_V1")
    parser.add_argument("--skip-faster-rcnn", action="store_true",
                        help="Evaluate only Lite R-CNN v2/v3, e.g. offline where the torchvision "
                             "COCO weights cannot be downloaded (speedup and AP delta stay empty)")
    parser.add_argument("--v2-weights", default="outputs/coco_4090_v2/best.pth")
    parser.add_argument("--v3-weights", default="outputs/coco_4090_v3/best.pth")
    parser.add_argument("--fuse", action=argparse.BooleanOptionalAction, default=True,
                        help="Fold Conv+BN (FrozenBatchNorm2d included) into every model before "
                             "scoring and timing, as deployed; --no-fuse keeps the training graph")
    parser.add_argument("--timing-images", type=int, default=64,
                        help="Seeded random sample of the evaluated images used for latency")
    parser.add_argument("--warmup", type=int, default=64,
                        help="Untimed forwards before each latency run (raised to --timing-images "
                             "so every image is warm)")
    parser.add_argument("--iterations", type=int, default=640,
                        help="Timed forwards cycling over the timing images (default 10 per image)")
    parser.add_argument("--stage-repeats", type=int, default=5)
    parser.add_argument("--output-dir")
    return parser


def main(argv: list[str] | None = None, parser: argparse.ArgumentParser | None = None) -> None:
    parser = parser or build_parser()
    args = parser.parse_args(argv)
    if min(args.images, args.image_size, args.timing_images, args.stage_repeats) < 1 or args.warmup < 0:
        parser.error("--images, --image-size, --timing-images and --stage-repeats must be >= 1; "
                     "--warmup must be >= 0")
    if args.iterations < args.timing_images:
        parser.error("--iterations must be at least --timing-images so every timing image is timed")
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
    indices = evaluation_subset(dataset.ids, count, args.seed)
    # Latency uses one seeded sample of the evaluated images, shared by every model.
    timing_order = torch.randperm(len(indices), generator=torch.Generator().manual_seed(args.seed))
    timing_indices = [indices[k] for k in timing_order[:args.timing_images].tolist()]
    warmup = max(args.warmup, len(timing_indices))
    names = [name for name in MODELS if not (args.skip_faster_rcnn and name == FASTER_RCNN)]
    weights = {
        FASTER_RCNN: Path(args.faster_weights).resolve() if args.faster_weights else None,
        "Lite R-CNN v2": (ROOT / args.v2_weights).resolve(),
        "Lite R-CNN v3": (ROOT / args.v3_weights).resolve(),
    }
    configs = {FASTER_RCNN: ROOT / args.faster_config,
               "Lite R-CNN v2": ROOT / args.v2_config, "Lite R-CNN v3": ROOT / args.v3_config}
    for name in names:
        if weights[name] is not None and not weights[name].is_file():
            raise SystemExit(f"Checkpoint not found for {name}: {weights[name]}")
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "outputs" / "evaluation_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    metadata = {"images_evaluated": count, "seed": args.seed, "device": str(device),
                "image_size": args.image_size, "evaluation_timestamp": datetime.now().isoformat(timespec="seconds"),
                "subset_image_ids": [int(dataset.ids[index]) for index in indices],
                "latency_warmup": warmup, "latency_iterations": args.iterations,
                "latency_timing_images": len(timing_indices),
                "latency_timing_image_ids": [int(dataset.ids[index]) for index in timing_indices],
                "latency_proposals": list(LATENCY_PROPOSALS),
                "latency_percentile_method": "linear interpolation (numpy default)",
                "rpn_post_nms_top_n_test": PROPOSALS,
                "proposal_recall_frame": "original_image",
                "fuse_conv_bn": args.fuse,
                "models": names, "skipped_models": [name for name in MODELS if name not in names],
                "dataset": dataset_info(annotation_path, image_root),
                "environment": collect_environment(str(device)), "git_commit": git_commit(ROOT)}
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
    if args.skip_faster_rcnn:
        print("Faster R-CNN을 제외하고 Lite R-CNN v2/v3만 평가합니다 (--skip-faster-rcnn).", flush=True)
    results = []
    for name in names:
        print(f"\n=== {name}: evaluating {count} shared images ===", flush=True)
        results.append(evaluate_one(name, weights[name], configs[name], dataset, indices,
                                    timing_indices, warmup, device, args))
        partial_metadata = {**metadata, "models_completed": [row["model"] for row in results]}
        output_dir.mkdir(parents=True, exist_ok=True)  # not before: a failed load leaves no folder
        (output_dir / "partial_results.json").write_text(
            json.dumps({"metadata": partial_metadata, "results": results}, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(csv_row(results[-1]), indent=2), flush=True)
    baseline = next((row for row in results if row["model"] == FASTER_RCNN), None)
    if baseline is not None:
        for row in results:
            row["speedup_vs_faster_rcnn"] = baseline["latency_p50_ms"] / row["latency_p50_ms"]
            if row["AP"] is not None and baseline["AP"] is not None:
                row["delta_AP_vs_faster_rcnn_pp"] = (row["AP"] - baseline["AP"]) * 100
    write_reports(results, output_dir, metadata)
    print(f"\nReports written to: {output_dir}", flush=True)
    print(f"  {output_dir / 'REPORT.md'}\n  {output_dir / 'summary.csv'}\n  {output_dir / 'evaluation.json'}", flush=True)


if __name__ == "__main__":
    main()
