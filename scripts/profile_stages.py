"""Break R-CNN latency into pipeline stages from decoded host images to detections (schema v2)."""

import argparse
import json
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.error import URLError

import torch
import yaml
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from light_rcnn.bench import (
    DEFAULT_WEIGHTS,
    IMAGE_EXTENSIONS,
    load_comparison_model,
    load_images,
    resolve_device,
    set_detection_limits,
)
from light_rcnn.model.deploy import prepare_for_inference
from light_rcnn.reporting import (
    STAGE_LABELS,
    build_report,
    collect_environment,
    default_report_path,
    display_path,
    file_sha256,
    percentile,
    write_report,
)

# R-CNN kinds light_rcnn.bench.load_comparison_model builds (YOLO26 has no stage breakdown).
MODEL_CHOICES = ("light_rcnn", "faster_rcnn", "mobilenet_faster_rcnn")
SCOPE_ID = "rcnn_stage_breakdown"
TOP_STAGES = list(STAGE_LABELS)  # input, preprocess, backbone, rpn, roi_heads, postprocess
SUBSTAGES = {  # stage -> (parent, label); already included in the parent's time
    "backbone.body": ("backbone", "백본 본체 (특징 추출)"),
    "backbone.neck": ("backbone", "FPN/BiFPN 특징 융합"),
    "rpn.filter_proposals": ("rpn", "제안 필터링 + NMS"),
    "roi_heads.postprocess_detections": ("roi_heads", "클래스별 NMS"),
}
# Backbone sub-modules timed when the backbone has them, by the attribute names tried in order:
# Light R-CNN v2/v3 name their fusion 'neck', torchvision's BackboneWithFPN (ResNet50-FPN,
# MobileNetV3-Large-FPN) 'fpn'. A backbone without one keeps that time in 'backbone' only.
BACKBONE_PARTS = {"backbone.body": ("body",), "backbone.neck": ("neck", "fpn")}
OVERHEAD_NOTE = (
    "단계마다 시간을 재느라 약간의 오버헤드가 더해져 단계 합계가 지연 시간 탭의 값보다 "
    "조금 클 수 있습니다. 병목은 비중으로 판단하고 절대 시간은 지연 시간 탭을 기준으로 "
    "보세요. 상위 단계가 적힌 세부 단계 행은 상위 단계 시간에 이미 포함되어 있으니 "
    "더하지 마세요."
)


def load_source(source: str | Path, count: int) -> tuple[list[Path], list[torch.Tensor]]:
    """One image file or the first ``count`` images of a folder, as scripts/benchmark.py reads."""
    path = Path(source)
    if not path.is_file():
        return load_images(path, count)
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise SystemExit(f"Unsupported image file: {source}")
    with Image.open(path) as image:
        return [path], [pil_to_tensor(image.convert("RGB")).contiguous()]


class StageTimer:
    """Accumulate wall-clock milliseconds per stage name for the current call."""

    def __init__(self, device: torch.device):
        self.device = device
        self.current: dict[str, float] = {}

    def sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def wrap(self, stage: str, fn: Callable) -> Callable:
        def timed(*args, **kwargs):
            self.sync()
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            self.sync()
            elapsed = (time.perf_counter() - start) * 1000
            self.current[stage] = self.current.get(stage, 0.0) + elapsed
            return result
        return timed


def backbone_parts(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    """The backbone sub-modules this model has, by substage ({} for a backbone without them)."""
    parts = {}
    for stage, names in BACKBONE_PARTS.items():
        modules = [getattr(model.backbone, name, None) for name in names]
        module = next((part for part in modules if isinstance(part, torch.nn.Module)), None)
        if module is not None:
            parts[stage] = module
    return parts


def unavailable_reason(stage: str, present: bool) -> str:
    """Why a backbone substage has no time of its own (recorded in the report)."""
    names = "/".join(BACKBONE_PARTS[stage])
    where = (f"백본 forward가 {names} 모듈을 따로 부르지 않아" if present
             else f"백본에 {names} 모듈이 없어")
    return f"{where} '{STAGE_LABELS['backbone']}' 시간에만 들어 있습니다"


@contextmanager
def instrument(model: torch.nn.Module, timer: StageTimer) -> Iterator[None]:
    """Shadow the stage callables on the model's sub-module instances; restore them on exit.

    roi_heads covers RoIAlign, the box head (LightRoIHead for Light R-CNN) and the predictor.
    The backbone's body and FPN/BiFPN are timed as substages when it has them (backbone_parts).
    """
    targets = [
        (model.transform, "forward", "preprocess"),
        (model.backbone, "forward", "backbone"),
        (model.rpn, "forward", "rpn"),
        (model.roi_heads, "forward", "roi_heads"),
        (model.transform, "postprocess", "postprocess"),
        *((module, "forward", stage) for stage, module in backbone_parts(model).items()),
        (model.rpn, "filter_proposals", "rpn.filter_proposals"),
        (model.roi_heads, "postprocess_detections", "roi_heads.postprocess_detections"),
    ]
    patched = []
    try:
        for owner, name, stage in targets:
            patched.append((owner, name, name in vars(owner), vars(owner).get(name)))
            setattr(owner, name, timer.wrap(stage, getattr(owner, name)))
        yield
    finally:
        for owner, name, had_instance_attr, previous in reversed(patched):
            if had_instance_attr:
                setattr(owner, name, previous)
            else:
                delattr(owner, name)


def profile_call(model: torch.nn.Module, image: torch.Tensor, device: torch.device,
                 timer: StageTimer) -> tuple[dict[str, float], float, list[dict]]:
    timer.current = {}
    timer.sync()
    start = time.perf_counter()
    x = image.to(device)
    x = x.float().div_(255)
    timer.sync()
    timer.current["input"] = (time.perf_counter() - start) * 1000
    outputs = model([x])
    timer.sync()
    wall = (time.perf_counter() - start) * 1000
    return dict(timer.current), wall, outputs


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "p50_ms": round(percentile(values, 50), 4),
        "mean_ms": round(sum(values) / len(values), 4),
        "p95_ms": round(percentile(values, 95), 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/coco.yaml",
                        help="Comparison settings; a Light R-CNN checkpoint's own config still "
                             "decides its architecture")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="light_rcnn")
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
                        help="Override comparison.channels_last (unset = Ultralytics auto rule)")
    parser.add_argument("--output", help="Defaults to outputs/reports/<model>_stages.json")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    comparison = config["comparison"]
    image_size = int(args.size if args.size is not None else config["image_size"])
    warmup = args.warmup if args.warmup is not None else comparison["warmup"]
    iterations = args.iterations if args.iterations is not None else comparison["iterations"]
    image_count = args.image_count if args.image_count is not None else comparison["image_count"]
    score_threshold = (args.score_thresh if args.score_thresh is not None
                       else comparison["latency_score_threshold"])
    max_detections = args.max_dets if args.max_dets is not None else comparison["max_detections"]
    channels_last = (args.channels_last if args.channels_last is not None
                     else comparison.get("channels_last"))
    if warmup < 0 or iterations < 1 or image_count < 1 or max_detections < 1 or image_size < 1:
        parser.error("--warmup must be >= 0; --iterations, --image-count, --max-dets and --size "
                     "must be >= 1")

    # Same settings as scripts/benchmark.py so the breakdown matches the end-to-end latency run.
    torch.backends.cudnn.benchmark = False
    try:
        device = resolve_device(args.device or config.get("device", "cuda"), "profiling")
    except ValueError as exc:
        parser.error(f"--device: {exc}")
    paths, images = load_source(args.source or config["data"]["val_images"], image_count)
    try:
        model = load_comparison_model(args.model, config, args.weights, image_size)
    except URLError as exc:
        raise SystemExit(f"Could not download default {args.model} weights ({exc}). "
                         "Pass --weights <checkpoint>.") from exc
    set_detection_limits(model, score_threshold, max_detections)
    model.to(device).eval()
    layout = prepare_for_inference(model, device, channels_last)

    timer = StageTimer(device)
    records, walls = [], []
    parts = backbone_parts(model)
    with torch.inference_mode(), instrument(model, timer):
        first, _, _ = profile_call(model, images[0], device, timer)
        missing = [stage for stage in [*TOP_STAGES, *SUBSTAGES]
                   if stage not in first and stage not in BACKBONE_PARTS]
        if missing:
            raise SystemExit(f"Stage hooks did not fire for: {', '.join(missing)}. "
                             "The installed torchvision GeneralizedRCNN layout may have changed.")
        for index in range(warmup):
            profile_call(model, images[index % len(images)], device, timer)
        for index in range(iterations):
            record, wall, _ = profile_call(model, images[index % len(images)], device, timer)
            records.append(record)
            walls.append(wall)

    # A backbone part this model lacks (or never calls) stays inside 'backbone'; say so.
    timed = {stage: entry for stage, entry in SUBSTAGES.items() if stage in first}
    unavailable = [{"stage": stage, "parent": parent, "label": label,
                    "reason": unavailable_reason(stage, stage in parts)}
                   for stage, (parent, label) in SUBSTAGES.items() if stage not in timed]
    stage_values = {stage: [record[stage] for record in records]
                    for stage in [*TOP_STAGES, *timed]}
    top_means = {stage: sum(stage_values[stage]) / iterations for stage in TOP_STAGES}
    top_total = sum(top_means.values())
    totals = [sum(record[stage] for stage in TOP_STAGES) for record in records]
    items = [{
        "stage": stage,
        "label": STAGE_LABELS[stage],
        **summarize(stage_values[stage]),
        "share": round(top_means[stage] / top_total, 4) if top_total > 0 else None,
    } for stage in TOP_STAGES]
    substages = [{
        "stage": stage,
        "parent": parent,
        "label": label,
        **summarize(stage_values[stage]),
    } for stage, (parent, label) in timed.items()]
    total_summary = summarize(totals)
    wall_summary = summarize(walls)

    report = build_report(
        "stages",
        args.model,
        weights=display_path(args.weights) if args.weights else DEFAULT_WEIGHTS[args.model],
        weights_sha256=file_sha256(args.weights),
        image_size=image_size,
        batch_size=1,
        precision="fp32",
        environment=collect_environment(str(device)),
        stages={
            "scope_id": SCOPE_ID,
            "iterations": iterations,
            "warmup": warmup,
            "image_count": len(paths),
            "score_threshold": score_threshold,
            "max_detections": max_detections,
            **layout,
            "images": [str(path) for path in paths[:20]],
            "items": items,
            "substages": substages,
            "unavailable_substages": unavailable,
            "total_p50_ms": total_summary["p50_ms"],
            "total_mean_ms": total_summary["mean_ms"],
            # wall_* minus total_* is the GeneralizedRCNN.forward glue between timed stages
            "wall_p50_ms": wall_summary["p50_ms"],
            "wall_mean_ms": wall_summary["mean_ms"],
            "overhead_note": OVERHEAD_NOTE,
        },
    )
    output = write_report(report, args.output or default_report_path("stages", args.model))
    section = report["stages"]
    print(json.dumps({
        "output": str(output),
        "model": report["model"],
        "weights": report["weights"],
        "device": str(device),
        "image_size": report["image_size"],
        "iterations": iterations,
        "warmup": warmup,
        "image_count": section["image_count"],
        **layout,
        "items": [{key: value for key, value in item.items() if key != "label"}
                  for item in section["items"]],
        "substages": [{key: value for key, value in item.items() if key != "label"}
                      for item in section["substages"]],
        "unavailable_substages": [item["stage"] for item in section["unavailable_substages"]],
        "total_p50_ms": section["total_p50_ms"],
        "total_mean_ms": section["total_mean_ms"],
        "wall_p50_ms": section["wall_p50_ms"],
        "wall_mean_ms": section["wall_mean_ms"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
