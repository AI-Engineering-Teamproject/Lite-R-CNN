"""Before/after table of inference presets for the torchvision Faster R-CNN baselines (ResNet50-FPN
and MobileNetV3-Large-FPN) and Light R-CNN v2/v3 (schema v2).

Every (model, preset) pair gets a freshly loaded model with the preset applied by
light_rcnn.inference_opts, then the same three measurements: COCO AP on one seeded subset (batch 1,
pycocotools, the preset's own score floor unless --score-thresh), latency from decoded host uint8
images to detections (scripts/benchmark.py timed_call) on a seeded sample of the evaluated images,
and parameters/GFLOPs as scripts/benchmark.py counts them.

Latency is timed per model in interleaved blocks: every preset of the model stays loaded and each
round visits them in an order rotated by the round, timing every latency image once per visit, so
machine drift hits all presets alike. Each preset's p50 change against baseline and deploy gets a
95% CI from a paired bootstrap over rounds. --sequential-latency times each (model, preset) in its
own window instead (the earlier design), so its deltas absorb whatever the machine did in between.

Light R-CNN v2 and v3 share the report model key 'light_rcnn', so the reports go to one folder per
version (reports/v2, reports/v3), each with the reports of both baselines as well. The dashboard
reads one such folder, or reports/ itself through its 'Light R-CNN 체크포인트' selector. Every
report of one run records the same top-level comparison_run.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import io
import math
import statistics
import sys
import time
import traceback
import unicodedata
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "scripts"):  # also runs without an editable install (Colab)
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
import torch
import yaml
from benchmark import count_flops, timed_call
from evaluate_coco import DIAGNOSTICS
from plot_results import export_images

from light_rcnn import viz
from light_rcnn.bench import DEFAULT_WEIGHTS, load_comparison_model, load_images, resolve_device
from light_rcnn.coco_eval import STAT_NAMES
from light_rcnn.data import CocoTensorDataset
from light_rcnn.evaluation import evaluate_coco, evaluation_subset, subset_record
from light_rcnn.inference_opts import (
    PRESET_LABELS,
    PRESETS,
    apply_inference_options,
    resolve_options,
)
from light_rcnn.model.deploy import use_channels_last
from light_rcnn.reporting import (
    BASELINE_VARIANT,
    COLD_START_FIRST_FORWARD,
    MODELS,
    SCOPE_HOST_TO_DETECTIONS,
    SCOPE_LABELS,
    ComparisonWarning,
    build_report,
    collect_environment,
    comparability_warnings,
    dataset_info,
    describe_environment,
    display_path,
    file_sha256,
    improvement_warnings,
    latency_summary,
    load_reports,
    percentile,
    select_reports,
    write_report,
)

# --models key -> report model key; always measured in this order (the baselines first).
MODEL_KINDS = {"faster_rcnn": "faster_rcnn", "mobilenet": "mobilenet_faster_rcnn",
               "v2": "light_rcnn", "v3": "light_rcnn"}
MODEL_LABELS = {"faster_rcnn": MODELS["faster_rcnn"],
                "mobilenet": MODELS["mobilenet_faster_rcnn"],
                "v2": f"{MODELS['light_rcnn']} v2", "v3": f"{MODELS['light_rcnn']} v3"}
# torchvision COCO baselines: shared by every Light R-CNN version, so their reports are copied into
# each version's folder; without a local checkpoint a failed weight download skips them.
BASELINE_TAGS = ("faster_rcnn", "mobilenet")
WEIGHT_FLAGS = {"faster_rcnn": "--faster-weights", "mobilenet": "--mobilenet-weights"}
BASELINES_GROUP = "baselines"  # the report folder of both baselines when no Light R-CNN runs
DEFAULT_PRESETS = "baseline,deploy,fast,accurate"
DEPLOY_PRESET = "deploy"  # what the team's other latency tables measure (Conv+BN fused)
# (latency images, warmup, iterations) unless given: 10 passes over 64 images on a GPU.
GPU_COUNTS = (64, 64, 640)
CPU_COUNTS = (16, 16, 128)
# latency["design"] of the reports and the paired statistics of the interleaved design.
DESIGN_INTERLEAVED = "interleaved_blocks"
DESIGN_SEQUENTIAL = "sequential"
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0
PAIRED_METHOD = "paired bootstrap over rounds"
AP_COLUMNS = (("ap_pct", "AP_50_95"), ("ap50_pct", "AP_50"), ("ap75_pct", "AP_75"),
              ("ap_small_pct", "AP_small"), ("ap_medium_pct", "AP_medium"),
              ("ap_large_pct", "AP_large"), ("ar100_pct", "AR_100"))
CSV_COLUMNS = (
    "model", "model_label", "preset", "preset_label", "status", "note",
    *(name for name, _ in AP_COLUMNS),
    "p50_ms", "p95_ms", "p99_ms", "mean_ms", "fps", "cold_start_ms",
    "delta_ap_pp", "delta_p50_pct", "delta_p50_vs_deploy_pct",
    "score_threshold", "max_detections", "detections_per_image", "latency_mean_detections",
    "images_evaluated", "latency_images", "warmup", "iterations",
    "parameters_m", "deployed_parameters_m", "gflops", "precision", "memory_format",
    "conv_bn_fused", "device", "inference_summary", "accuracy_report", "latency_report",
    "delta_p50_ci_low", "delta_p50_ci_high", "delta_p50_vs_deploy_ci_low",
    "delta_p50_vs_deploy_ci_high",
)
STATUS_LABELS = {"skipped": "건너뜀", "failed": "실패"}


class WeightsUnavailable(Exception):
    """A baseline's default torchvision COCO weights could not be downloaded."""


@dataclass
class Setup:
    config: dict
    config_path: Path
    device: torch.device
    image_size: int
    seed: int
    weights: dict[str, str | None]
    images_dir: Path
    annotations: Path
    dataset: CocoTensorDataset
    indices: list[int]
    limit: int
    latency_paths: list[Path]
    latency_ids: list[int]
    latency_images: list[torch.Tensor]
    warmup: int
    iterations: int
    score_thresh: float | None
    output_dir: Path
    groups: list[str]
    environment: dict
    rounds: int | None  # interleaved rounds per model; None = --sequential-latency
    comparison_run: str


@dataclass
class Prepared:
    """One preset applied to a freshly loaded model, with its cold start and COCO AP measured."""

    tag: str
    preset: str
    label: str
    model: torch.nn.Module
    record: dict
    parameters_m: float
    cold_start_ms: float
    accuracy: dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="모델 × 추론 설정마다 같은 COCO 이미지의 AP와 같은 방식의 지연 시간을 재서 "
                    "개선 전/후 표, 그래프, 대시보드용 보고서를 만듭니다.")
    parser.add_argument("--config", default="configs/coco.yaml",
                        help="데이터 경로, 입력 크기, 부분 평가 seed (Light R-CNN 구조는 "
                             "체크포인트에 저장된 설정을 따름)")
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_KINDS),
                        default=list(MODEL_KINDS),
                        help="비교할 모델 (기본: 전부, 항상 faster_rcnn → mobilenet → v2 → v3 "
                             "순서로 측정)")
    parser.add_argument("--presets", nargs="+", default=[DEFAULT_PRESETS],
                        help=f"쉼표로 구분한 추론 설정 ({', '.join(PRESETS)}). baseline은 항상 "
                             f"맨 앞에 포함 (기본: {DEFAULT_PRESETS})")
    parser.add_argument("--v2-weights", default="outputs/coco_4090_v2/best.pth")
    parser.add_argument("--v3-weights", default="outputs/coco_4090_v3/best.pth")
    parser.add_argument("--faster-weights",
                        help="Faster R-CNN 체크포인트. 생략하면 torchvision COCO 가중치를 내려받고, "
                             "내려받지 못하면 Faster R-CNN만 건너뜀")
    parser.add_argument("--mobilenet-weights",
                        help="MobileNetV3 Faster R-CNN(torchvision fasterrcnn_mobilenet_v3_large_fpn) "
                             "체크포인트. 생략하면 torchvision COCO 가중치를 내려받고, 내려받지 "
                             "못하면 MobileNet만 건너뜀")
    parser.add_argument("--device", help="PyTorch 장치 (기본: config의 device)")
    parser.add_argument("--images", help="평가 이미지 폴더 (기본: config의 val_images)")
    parser.add_argument("--annotations", help="COCO 주석 JSON (기본: config의 val_annotations)")
    parser.add_argument("--limit", type=int, default=0,
                        help="config seed로 고른 부분집합 크기 (evaluate_coco.py와 같은 규칙), "
                             "0 = 전체")
    parser.add_argument("--score-thresh", type=float,
                        help="모든 설정·모델에 같은 검출 점수 하한 (생략하면 각 설정의 값, 기본 "
                             "0.05 = 모델 기본값·학습 검증·팀 세 모델 평가와 같은 하한. "
                             "docs/MEASUREMENT.md 비교 프로토콜은 0.001)")
    parser.add_argument("--latency-images", type=int,
                        help=f"지연 시간을 잴 평가 이미지 수 = 교차 측정의 블록 크기 (기본 GPU "
                             f"{GPU_COUNTS[0]}, CPU {CPU_COUNTS[0]})")
    parser.add_argument("--warmup", type=int,
                        help=f"설정마다 예열 횟수, 지연 시간 이미지 수보다 적으면 그만큼 늘림 "
                             f"(기본 GPU {GPU_COUNTS[1]}, CPU {CPU_COUNTS[1]})")
    parser.add_argument("--iterations", type=int,
                        help=f"설정마다 측정 횟수 (기본 GPU {GPU_COUNTS[2]}, CPU {CPU_COUNTS[2]}). "
                             "교차 측정은 지연 시간 이미지 수의 배수로 올림 (라운드 수 = 이 값 ÷ "
                             "이미지 수, 올림)")
    parser.add_argument("--sequential-latency", action="store_true",
                        help="이전 방식: 모델·설정마다 따로 이어서 지연 시간을 잼 (신뢰구간 없음). "
                             "기본은 교차 측정 — 한 모델의 설정을 모두 올려 두고 라운드마다 순서를 "
                             "돌려 가며 블록 단위로 번갈아 재고, Δp50에 95%% 신뢰구간을 붙임")
    parser.add_argument("--output-dir", help="결과 폴더 (기본: outputs/improvements/<실행 시각>)")
    return parser


def parse_presets(values: list[str], parser: argparse.ArgumentParser) -> list[str]:
    names = [BASELINE_VARIANT]
    for value in values:
        for raw in value.split(","):
            name = raw.strip().lower().replace("-", "_")
            if not name:
                continue
            if name not in PRESETS:
                parser.error(f"알 수 없는 추론 설정: {raw.strip()} (가능: {', '.join(PRESETS)})")
            if name not in names:
                names.append(name)
    return names


def preset_label(preset: str, score_thresh: float | None) -> str:
    label = PRESET_LABELS[preset]
    return f"{label} · 점수 하한 {score_thresh:g}" if score_thresh is not None else label


def cuda_only(preset: str) -> bool:
    return PRESETS[preset].cuda_graph or PRESETS[preset].fp16


def report_groups(tags: list[str]) -> list[str]:
    """One report folder per Light R-CNN version; without one, the baselines' own folder."""
    light = [tag for tag in tags if tag not in BASELINE_TAGS]
    baselines = [tag for tag in tags if tag in BASELINE_TAGS]
    return light or (baselines if len(baselines) == 1 else [BASELINES_GROUP])


def groups_for(tag: str, setup: Setup) -> list[str]:
    """Report folders of one model: a baseline goes into every folder of the run (each Light
    R-CNN version's), a Light R-CNN version into its own."""
    return setup.groups if tag in BASELINE_TAGS else [tag]


def group_label(group: str, tags: list[str]) -> str:
    """What one report folder holds: its Light R-CNN version, then the baselines among ``tags``."""
    own = [group] if group in MODEL_LABELS and group not in BASELINE_TAGS else []
    return " · ".join(MODEL_LABELS[tag]
                      for tag in [*own, *(tag for tag in BASELINE_TAGS if tag in tags)])


def reported(results: list[dict]) -> list[str]:
    """The --models keys with a report written (a skipped baseline has none), in run order."""
    return [tag for tag in MODEL_KINDS
            if any(result["tag"] == tag and result.get("reports") for result in results)]


def latency_sample(dataset: CocoTensorDataset, images_dir: Path, indices: list[int], count: int,
                   seed: int) -> tuple[list[Path], list[int], list[torch.Tensor]]:
    """A seeded random sample of the evaluated images, decoded once into host uint8 tensors."""
    order = torch.randperm(len(indices), generator=torch.Generator().manual_seed(seed))[:count]
    ids = [int(dataset.ids[indices[position]]) for position in order.tolist()]
    paths = [images_dir / dataset.coco.loadImgs(image_id)[0]["file_name"] for image_id in ids]
    images = [load_images(path, 1)[1][0] for path in paths]
    return paths, ids, images


def build_model(tag: str, setup: Setup) -> torch.nn.Module:
    weights = setup.weights[tag]
    try:
        return load_comparison_model(MODEL_KINDS[tag], setup.config, weights, setup.image_size)
    except (OSError, RuntimeError) as exc:  # URLError/HTTPError, proxy refusals, bad hash
        if tag not in BASELINE_TAGS or weights:
            raise
        raise WeightsUnavailable(
            f"torchvision COCO 가중치를 내려받지 못했습니다 ({type(exc).__name__}: {exc}). "
            f"{WEIGHT_FLAGS[tag]} <체크포인트>로 로컬 가중치를 지정하면 함께 측정합니다."
        ) from exc


def graph_note(status: dict) -> str:
    """Warning for a CUDA-graph run whose calls did not all replay the graph ('' if they did)."""
    if not status or (status["status"] == "captured" and not status["eager_calls"]):
        return ""
    counts = f"재생 {status['replays']}회, eager {status['eager_calls']}회"
    if status["status"] == "captured":  # the graph only replays the input shape it captured
        return (f"CUDA 그래프가 일부 호출에만 쓰임 ({counts}, 입력 크기가 캡처 때와 다른 호출은 "
                "eager) — 지연 시간에 eager 호출이 섞였을 수 있음")
    return f"CUDA 그래프 미사용: {status['status']} ({counts}) — 지연 시간은 그래프 없이 잰 값"


def prepare_run(tag: str, preset: str, setup: Setup) -> Prepared:
    """Load, apply the preset, time the first forward and measure COCO AP (no latency yet)."""
    device, images = setup.device, setup.latency_images
    model = build_model(tag, setup)
    parameters_m = sum(param.numel() for param in model.parameters()) / 1e6
    record = apply_inference_options(
        model, resolve_options(preset, box_score_thresh=setup.score_thresh), device)
    applied = record["applied"]
    print(f"  적용: {record['summary']}", flush=True)

    with torch.inference_mode():
        cold_start_ms, _ = timed_call(model, images[0], device)
    with contextlib.redirect_stdout(io.StringIO()):  # pycocotools prints every summary
        metrics = evaluate_coco(model, setup.dataset, device, setup.limit,
                                subset_seed=setup.seed, batch_size=1, detailed=True)
    accuracy = {
        "evaluator": "pycocotools",
        **subset_record(metrics["image_ids"], len(setup.dataset.ids), setup.seed),
        "score_threshold": applied["box_score_thresh"],
        "score_threshold_source": "--score-thresh" if setup.score_thresh is not None else "preset",
        "max_detections": applied["detections_per_img"],
        "eval_seconds": metrics["inference_seconds"],
        "dataset": dataset_info(setup.annotations, setup.images_dir),
        **{key: metrics[key] for key in STAT_NAMES},
        "per_category_ap": metrics["per_category_ap"],
        "detections": metrics["detections"],
        "coco_max_dets": metrics["coco_max_dets"],
        "diagnostics": {key: metrics[key] for key in DIAGNOSTICS},
    }
    per_image = accuracy["detections"] / max(accuracy["images_evaluated"], 1)
    print(f"  COCO: AP {_number(_pct(accuracy['AP_50_95']))} · "
          f"AP50 {_number(_pct(accuracy['AP_50']))} · AP75 {_number(_pct(accuracy['AP_75']))} · "
          f"이미지당 검출 {per_image:.1f}개 ({accuracy['images_evaluated']}장, "
          f"점수 하한 {accuracy['score_threshold']:g}, {accuracy['eval_seconds']:.0f}초)",
          flush=True)
    return Prepared(tag, preset, preset_label(preset, setup.score_thresh), model, record,
                    parameters_m, cold_start_ms, accuracy)


def model_call(model: torch.nn.Module, setup: Setup) -> Callable[[int], tuple[float, int]]:
    """Time latency image ``index`` once (benchmark.timed_call): (milliseconds, detections)."""
    def call(index: int) -> tuple[float, int]:
        elapsed, outputs = timed_call(model, setup.latency_images[index], setup.device)
        return elapsed, len(outputs[0]["boxes"])

    return call


def sequential_latency(call: Callable[[int], tuple[float, int]], count: int, warmup: int,
                       iterations: int) -> dict:
    """--sequential-latency: warm-up, then call i times image i % count, in this preset's window."""
    for index in range(warmup):
        call(index % count)
    samples, detections = [], []
    for index in range(iterations):
        elapsed, found = call(index % count)
        samples.append(elapsed)
        detections.append(found)
    return {"samples": samples, "detections": detections, "rounds": None}


def visit_order(names: list[str], round_index: int) -> list[str]:
    """Round r visits the presets rotated by r: over a multiple of len(names) rounds (main rounds
    up to one) each takes every slot equally often. Within a round a preset keeps the same
    predecessor; the untimed call after each switch softens that carry-over."""
    shift = round_index % len(names) if names else 0
    return names[shift:] + names[:shift]


def interleaved_rounds(iterations: int, images: int, presets: int) -> int:
    """Rounds for at least ``iterations`` timed calls over ``images`` latency images, rounded up
    to whole rotation cycles of ``presets`` (visit_order), so no preset gets a slot more often."""
    return math.ceil(math.ceil(iterations / images) / max(presets, 1)) * max(presets, 1)


def interleaved_latency(calls: dict[str, Callable[[int], tuple[float, int]]], count: int,
                        warmup: int, rounds: int) -> tuple[dict[str, dict], dict[str, str]]:
    """Time the presets of one model in interleaved blocks within one time window.

    Warm-up goes round-robin, ``warmup`` calls per preset. In round r the presets come in an order
    rotated by r; each visit makes one untimed call on image 0 (re-warms that model after the
    switch), then times images 0..count-1 once each. A preset that raises is reported and left
    out of the remaining calls. Returns ({preset: samples, detections and per-round samples},
    {failed preset: error}); sample i still times image i % count.
    """
    names = list(calls)
    timings = {name: {"samples": [], "detections": [], "rounds": []} for name in names}
    failed: dict[str, str] = {}
    plan = [(name, [index % count], False) for index in range(warmup) for name in names]
    plan += [(name, range(count), True) for round_index in range(rounds)
             for name in visit_order(names, round_index)]
    for name, indices, timed in plan:
        if name in failed:
            continue
        try:
            if timed:
                calls[name](0)  # untimed: the first call after switching models
            measured = [calls[name](index) for index in indices]
        except Exception as exc:  # noqa: BLE001 - one failed preset must not cost the others
            traceback.print_exc()
            failed[name] = f"{type(exc).__name__}: {exc}"
            continue
        if timed:
            block = [elapsed for elapsed, _ in measured]
            timings[name]["rounds"].append(block)
            timings[name]["samples"].extend(block)
            timings[name]["detections"].extend(found for _, found in measured)
    return {name: timing for name, timing in timings.items() if name not in failed}, failed


def paired_delta(after_rounds: list[list[float]] | None, before_rounds: list[list[float]] | None,
                 resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict | None:
    """p50 change of ``after`` against ``before`` (%) with a 95% CI from a paired bootstrap.

    Both presets were timed in the same rounds, so each resample draws round indices with
    replacement and recomputes both p50s from those same rounds: a slow round slows both. None
    with fewer than two rounds (one round gives no spread) or mismatched rounds.
    """
    if not after_rounds or not before_rounds:
        return None
    after = np.asarray(after_rounds, dtype=float)
    before = np.asarray(before_rounds, dtype=float)
    if after.ndim != 2 or after.shape != before.shape or len(after) < 2 or before.min() <= 0:
        return None
    rounds = len(after)
    picks = np.random.default_rng(seed).integers(0, rounds, size=(resamples, rounds))
    deltas = np.empty(resamples)
    for start in range(0, resamples, 250):  # bounded memory for long runs
        chunk = picks[start:start + 250]
        new = np.median(after[chunk].reshape(len(chunk), -1), axis=1)
        old = np.median(before[chunk].reshape(len(chunk), -1), axis=1)
        deltas[start:start + len(chunk)] = (new - old) / old * 100
    low, high = np.percentile(deltas, [2.5, 97.5])
    point = _change_pct(percentile(after.ravel(), 50), percentile(before.ravel(), 50))
    return {"delta_p50_pct": round(point, 4), "ci95_pct": [round(float(low), 4),
                                                            round(float(high), 4)],
            "resamples": resamples, "method": PAIRED_METHOD}


def paired_entries(preset: str, timings: dict[str, dict]) -> dict | None:
    """latency['paired'] of one preset against baseline and deploy timed in the same rounds of
    this run; None for baseline itself, with no reference timed or with too few rounds."""
    if preset == BASELINE_VARIANT or preset not in timings:
        return None
    entries = {}
    for key, reference in (("vs_baseline", BASELINE_VARIANT), ("vs_deploy", DEPLOY_PRESET)):
        stats = None
        if reference != preset and reference in timings:
            stats = paired_delta(timings[preset]["rounds"], timings[reference]["rounds"])
        entries[key] = {"variant": reference, **stats} if stats else None
    return entries if any(entries.values()) else None


def freeze_graph_status(run: Prepared) -> str:
    """Snapshot the CUDA-graph status once latency is timed; returns the caution ('' if none).

    The applied summary says 'CUDA 그래프' from apply time; whether it replayed is only known now.
    Frozen before count_flops' deliberate eager call, so the record counts the measured calls.
    """
    graph = dict(run.record.get("cuda_graph_status") or {})
    note = graph_note(graph)
    if graph:
        run.record["cuda_graph_status"] = graph
    if note:
        run.record["summary"] += f" ({note})"
    return note


def report_fields(run: Prepared, setup: Setup) -> dict:
    """Top-level fields both reports of one (model, preset) share."""
    weights = setup.weights[run.tag]
    return {
        "weights": display_path(weights) if weights else DEFAULT_WEIGHTS[MODEL_KINDS[run.tag]],
        "weights_sha256": file_sha256(weights),
        "config": display_path(setup.config_path),
        "image_size": setup.image_size,
        "batch_size": 1,
        "precision": "fp16" if run.record["applied"]["fp16"] else "fp32",
        "environment": setup.environment,
        "variant": run.preset,
        "variant_label": run.label,
        "inference_options": run.record,
        "model_tag": run.tag,
        "model_label": MODEL_LABELS[run.tag],
        "comparison_run": setup.comparison_run,
    }


def accuracy_report(run: Prepared, common: dict) -> dict:
    return build_report("accuracy", MODEL_KINDS[run.tag], **common, accuracy=run.accuracy,
                        images_evaluated=run.accuracy["images_evaluated"])


def write_reports(run: Prepared, setup: Setup, reports: dict[str, dict]) -> dict[str, str]:
    """Write {report kind: report} into every folder of the run's model; the first path per kind."""
    written = {}
    for report_kind, report in reports.items():
        for group in groups_for(run.tag, setup):
            path = (setup.output_dir / "reports" / group
                    / f"{MODEL_KINDS[run.tag]}_{report_kind}_{run.preset}.json")
            write_report(report, path)
            written.setdefault(report_kind, display_path(path))
    return written


def accuracy_only(run: Prepared, note: str, setup: Setup) -> dict:
    """A preset that failed after its COCO pass (latency timing or its report): status 'failed',
    but its accuracy report is still written, so its AP stays (the baseline's anchors every ΔAP
    and is the dashboard's 'before')."""
    print(f"  [실패] {run.label} ({run.preset}): {note} — 정확도(AP) 보고서는 저장합니다",
          flush=True)
    result = {"tag": run.tag, "preset": run.preset, "label": run.label, "status": "failed",
              "note": note}
    try:
        report = accuracy_report(run, report_fields(run, setup))
        written = write_reports(run, setup, {"accuracy": report})
    except Exception:  # noqa: BLE001 - the failure above is what the summary must show
        traceback.print_exc()
        return result
    return {**result, "accuracy": run.accuracy, "reports": written}


def finish_run(run: Prepared, timing: dict, note: str, setup: Setup,
               paired: dict | None = None) -> dict:
    """GFLOPs, the latency record and both reports of one timed preset; returns the result."""
    kind, device, images = MODEL_KINDS[run.tag], setup.device, setup.latency_images
    record, applied = run.record, run.record["applied"]
    # A CUDA graph replays without dispatching ops: count its eager fallback (taken with autograd).
    with torch.enable_grad() if applied["cuda_graph"] else torch.inference_mode():
        gflops, flops_input, flops_method = count_flops(run.model, images[0], device,
                                                        setup.image_size)
    canvas = applied["fixed_canvas"]
    if gflops is not None and canvas and canvas != flops_input:
        flops_method += f"; the backbone ran on the {canvas[0]}x{canvas[1]} fixed canvas"
    samples, detections, rounds = timing["samples"], timing["detections"], timing["rounds"]
    count = len(images)
    design = {"design": DESIGN_SEQUENTIAL} if rounds is None else {
        "design": DESIGN_INTERLEAVED, "rounds": len(rounds), "block_size": count,
        "round_medians_ms": [round(statistics.median(block), 4) for block in rounds],
        "paired": paired,
    }
    latency = {
        "scope_id": SCOPE_HOST_TO_DETECTIONS,
        "timed_scope": SCOPE_LABELS[SCOPE_HOST_TO_DETECTIONS],
        "warmup": setup.warmup,
        "iterations": len(samples),
        "image_count": count,
        "image_selection": "seeded_random_evaluated",
        "image_seed": setup.seed,
        "image_ids": setup.latency_ids,
        "score_threshold": applied["box_score_thresh"],
        "max_detections": applied["detections_per_img"],
        "conv_bn_fused": applied["conv_bn_fused"],
        "memory_format": applied["memory_format"],
        "images": [display_path(path) for path in setup.latency_paths[:20]],
        "mean_detections": sum(detections) / len(detections),
        # Sample i times image i % image_count; a slow image and a random stall differ here.
        "per_image_median_ms": [statistics.median(samples[k::count]) if samples[k::count] else None
                                for k in range(count)],
        **latency_summary(samples, cold_start_ms=run.cold_start_ms),
        "cold_start_scope": COLD_START_FIRST_FORWARD,
        **design,
    }
    model_stats = {
        "parameters_m": run.parameters_m,
        "deployed_parameters_m": sum(param.numel() for param in run.model.parameters()) / 1e6,
        "gflops": gflops,
        "flops_input": flops_input,
        "flops_method": flops_method,
    }
    head = "지연 시간" if rounds is None else f"{run.label} ({run.preset})"
    print(f"  {head}: p50 {latency['p50_ms']:.2f} ms · p95 {latency['p95_ms']:.2f} ms · "
          f"{latency['fps']:.1f} FPS · 첫 추론 {run.cold_start_ms:.1f} ms", flush=True)
    for key, name in (("vs_baseline", "개선 전"), ("vs_deploy", "deploy")):
        entry = (paired or {}).get(key)
        if entry:
            low, high = entry["ci95_pct"]
            print(f"    Δp50 {name} 대비 {entry['delta_p50_pct']:+.1f}% "
                  f"[95% CI {low:+.1f}, {high:+.1f}]", flush=True)
    if note:
        print(f"  [주의] {note}", flush=True)

    common = report_fields(run, setup)
    written = write_reports(run, setup, {
        "accuracy": accuracy_report(run, common),
        "latency": build_report("latency", kind, **common, latency=latency,
                                model_stats=model_stats)})
    return {"tag": run.tag, "preset": run.preset, "label": run.label, "status": "ok",
            "note": note, "summary": record["summary"], "precision": common["precision"],
            "accuracy": run.accuracy, "latency": latency, "model_stats": model_stats,
            "reports": written}


def run_one(tag: str, preset: str, setup: Setup) -> dict:
    """--sequential-latency: load, apply, AP, then this preset's own latency window and reports."""
    run = prepare_run(tag, preset, setup)
    try:
        with torch.inference_mode():
            timing = sequential_latency(model_call(run.model, setup), len(setup.latency_images),
                                        setup.warmup, setup.iterations)
        return finish_run(run, timing, freeze_graph_status(run), setup)
    except Exception as exc:  # noqa: BLE001 - the COCO pass already done must not be lost
        traceback.print_exc()
        return accuracy_only(run, f"{type(exc).__name__}: {exc}", setup)


def run_interleaved(runs: list[Prepared], setup: Setup) -> list[dict]:
    """Latency of one model's prepared presets in interleaved blocks, then GFLOPs and reports."""
    count = len(setup.latency_images)
    print(f"\n[지연 시간] {MODEL_LABELS[runs[0].tag]}: 설정 {len(runs)}개 교차 측정 — 설정마다 "
          f"예열 {setup.warmup}회(번갈아), {setup.rounds}라운드 × 블록 {count}장 "
          "(라운드마다 순서 회전)", flush=True)
    started = time.perf_counter()
    with torch.inference_mode():
        timings, failures = interleaved_latency(
            {run.preset: model_call(run.model, setup) for run in runs}, count, setup.warmup,
            setup.rounds)
    print(f"  ({time.perf_counter() - started:.0f}초)", flush=True)
    # Every status is frozen before any count_flops call, so each counts its timed calls only.
    notes = {run.preset: freeze_graph_status(run) for run in runs if run.preset in timings}
    results = []
    for run in runs:
        note = failures.get(run.preset)
        if note is None:
            try:
                results.append(finish_run(run, timings[run.preset], notes[run.preset], setup,
                                          paired_entries(run.preset, timings)))
                continue
            except Exception as exc:  # noqa: BLE001 - one failed preset must not cost the others
                traceback.print_exc()
                note = f"{type(exc).__name__}: {exc}"
        results.append(accuracy_only(run, note, setup))  # its AP was measured before the timing
    return results


def summary_rows(results: list[dict], device: str) -> list[dict]:
    """One flat row per run; deltas are against the same model's baseline (and deploy) run. A run
    that failed after its COCO pass keeps its AP columns, and a baseline like that its ΔAPs."""
    measured = {(result["tag"], result["preset"]): result for result in results
                if result["status"] == "ok"}
    scored = {(result["tag"], result["preset"]): result for result in results
              if result.get("accuracy")}
    rows = []
    for result in results:
        row = dict.fromkeys(CSV_COLUMNS)
        row.update(model=result["tag"], model_label=MODEL_LABELS[result["tag"]],
                   preset=result["preset"], preset_label=result["label"],
                   status=result["status"], note=result["note"], device=device)
        accuracy = result.get("accuracy")
        if accuracy:
            for name, key in AP_COLUMNS:
                row[name] = _pct(accuracy.get(key))
            row.update(
                score_threshold=accuracy["score_threshold"],
                max_detections=accuracy["max_detections"],
                detections_per_image=accuracy["detections"] / max(accuracy["images_evaluated"], 1),
                images_evaluated=accuracy["images_evaluated"],
                accuracy_report=result["reports"]["accuracy"])
            base = scored.get((result["tag"], BASELINE_VARIANT))
            if base is not None and result["preset"] != BASELINE_VARIANT:
                before_ap = base["accuracy"]["AP_50_95"]
                if before_ap is not None and accuracy["AP_50_95"] is not None:
                    row["delta_ap_pp"] = 100 * (accuracy["AP_50_95"] - before_ap)
        if result["status"] == "ok":
            latency, stats, reports = result["latency"], result["model_stats"], result["reports"]
            paired = latency.get("paired") or {}
            row.update({key: latency[key] for key in ("p50_ms", "p95_ms", "p99_ms", "mean_ms",
                                                      "fps", "cold_start_ms")})
            row.update(
                latency_mean_detections=latency["mean_detections"],
                latency_images=latency["image_count"], warmup=latency["warmup"],
                iterations=latency["iterations"], parameters_m=stats["parameters_m"],
                deployed_parameters_m=stats["deployed_parameters_m"], gflops=stats["gflops"],
                precision=result["precision"], memory_format=latency["memory_format"],
                conv_bn_fused=latency["conv_bn_fused"], inference_summary=result["summary"],
                latency_report=reports["latency"])
            base = measured.get((result["tag"], BASELINE_VARIANT))
            if base is not None and result["preset"] != BASELINE_VARIANT:
                row["delta_p50_pct"] = _change_pct(latency["p50_ms"], base["latency"]["p50_ms"])
                if row["delta_p50_pct"] is not None:
                    row["delta_p50_ci_low"], row["delta_p50_ci_high"] = _ci(
                        paired.get("vs_baseline"), BASELINE_VARIANT)
            # Fusion is exact, so only latency needs this second reference.
            deploy = measured.get((result["tag"], DEPLOY_PRESET))
            if deploy is not None and result["preset"] not in (BASELINE_VARIANT, DEPLOY_PRESET):
                row["delta_p50_vs_deploy_pct"] = _change_pct(latency["p50_ms"],
                                                             deploy["latency"]["p50_ms"])
                if row["delta_p50_vs_deploy_pct"] is not None:
                    row["delta_p50_vs_deploy_ci_low"], row["delta_p50_vs_deploy_ci_high"] = _ci(
                        paired.get("vs_deploy"), DEPLOY_PRESET)
        rows.append(row)
    return rows


def _ci(entry: dict | None, variant: str) -> tuple[float | None, float | None]:
    """The paired 95% CI of an entry against ``variant``, or (None, None)."""
    bounds = entry.get("ci95_pct") if isinstance(entry, dict) and entry.get(
        "variant") == variant else None
    return tuple(bounds) if isinstance(bounds, (list, tuple)) and len(bounds) == 2 else (None, None)


def _change_pct(after: float, before: float) -> float | None:
    return (after - before) / before * 100 if before > 0 else None


def _pct(value) -> float | None:
    return None if value is None else 100 * value  # pycocotools -1 (no ground truth) is None


def _number(value, digits: int = 2, signed: bool = False, missing: str = "–") -> str:
    if value is None:
        return missing
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _delta_cell(row: dict, key: str, missing: str) -> str:
    """'-38.9 [-41.0, -36.5]': the p50 change with its paired 95% CI when there is one."""
    text = _number(row[key], 1, signed=True, missing=missing)
    low, high = (row.get(key.removesuffix("_pct") + suffix) for suffix in ("_ci_low", "_ci_high"))
    if row[key] is None or low is None or high is None:
        return text
    return f"{text} [{low:+.1f}, {high:+.1f}]"


def table_cells(row: dict, missing: str = "–") -> list[str]:
    baseline = row["preset"] == BASELINE_VARIANT
    return [
        row["model_label"],
        f"{row['preset_label']} ({row['preset']})",
        *(_number(row[name], missing=missing) for name in ("ap_pct", "ap50_pct", "ap75_pct",
                                                           "ap_small_pct", "ap_large_pct")),
        _number(row["p50_ms"], missing=missing),
        _number(row["p95_ms"], missing=missing),
        _number(row["fps"], 1, missing=missing),
        "기준" if baseline else _number(row["delta_ap_pp"], 2, signed=True, missing=missing),
        "기준" if baseline else _delta_cell(row, "delta_p50_pct", missing),
        "기준" if row["preset"] == DEPLOY_PRESET
        else _delta_cell(row, "delta_p50_vs_deploy_pct", missing),
    ]


TABLE_HEADER = ["모델", "추론 설정", "AP", "AP50", "AP75", "AP_S", "AP_L", "p50 (ms)", "p95 (ms)",
                "FPS", "ΔAP (%p)", "Δp50 (%) [95% CI]", "Δp50 vs deploy (%) [95% CI]"]
TABLE_NOTE = ("AP 계열은 COCO AP(%), ΔAP는 같은 모델의 개선 전(기본 설정) 대비 변화(%p), "
              "Δp50은 p50 지연 시간 변화율(음수 = 빨라짐), Δp50 vs deploy는 같은 모델의 deploy "
              "(배포 기본) 대비 p50 변화율입니다. [ ] 안은 교차 측정의 95% 신뢰구간(쌍 "
              "부트스트랩)으로, 0을 포함하면 그 차이는 측정 오차 범위입니다 (순차 측정이면 없음).")
VERSIONS_NOTE = ("Light R-CNN v2와 v3는 대시보드에서 같은 'Light R-CNN'으로 표시됩니다. 두 버전이 "
                 "함께 든 reports 폴더를 열면 사이드바 'Light R-CNN 체크포인트'에서 버전을 고르고, "
                 "버전별 폴더를 열면 그 버전만 봅니다.")
DASHBOARD_HOWTO = ("`streamlit run dashboard/app.py`를 실행하고 사이드바 '결과 폴더'에 아래 경로를 "
                   "넣은 뒤 '추론 설정'을 바꿔 가며 보세요 ('개선 전/후' 탭).")


def _width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)


def console_table(rows: list[dict]) -> str:
    """Aligned plain-text table; Hangul takes two terminal columns."""
    body = [table_cells(row, missing="-") for row in rows]
    widths = [max(_width(cell) for cell in column) for column in zip(TABLE_HEADER, *body)]
    lines = []
    for index, cells in enumerate([TABLE_HEADER, *body]):
        padded = [cell + " " * (width - _width(cell)) if column < 2
                  else " " * (width - _width(cell)) + cell
                  for column, (cell, width) in enumerate(zip(cells, widths))]
        lines.append("  ".join(padded).rstrip())
        if index == 0:
            lines.append("  ".join("-" * width for width in widths))
    return "\n".join(lines)


def design_lines(setup: Setup) -> list[str]:
    """How latency was scheduled and how to read its CI, for the protocol section."""
    if setup.rounds is None:
        return [("지연 시간 측정 방식: 순차 측정 (--sequential-latency) — 모델·설정마다 따로, 각자의 "
                 "시간대에 잽니다. 그 사이 기계 상태(발열·클럭·다른 작업)가 바뀌면 Δp50에 그대로 "
                 "섞이므로 (같은 CPU 실행 두 번에서 10%p 넘게 달라짐) 신뢰구간은 없습니다.")]
    lines = [(f"지연 시간 측정 방식: 교차 측정 — 모델마다 추론 설정을 모두 메모리에 올려 두고 번갈아 "
              f"예열(설정마다 {setup.warmup}회)한 뒤 {setup.rounds}라운드를 돕니다. 라운드마다 설정 "
              "순서를 한 칸씩 돌려(순서 회전) 설정마다 블록 하나를 잽니다: 설정을 바꾼 직후 첫 "
              "이미지로 시간을 재지 않는 호출 1번, 이어서 블록 크기 "
              f"{len(setup.latency_images)}장(지연 시간 이미지 전부)을 같은 순서로 한 번씩. 라운드 "
              "수는 설정 수의 배수라 설정마다 순서의 모든 자리를 같은 횟수만큼 돕니다. 측정 중 "
              "기계 상태가 바뀌어도 모든 설정에 고르게 섞입니다.")]
    if setup.rounds < 2:
        lines.append("라운드가 1개뿐이라 Δp50 신뢰구간은 계산하지 않았습니다 (--iterations를 지연 "
                     "시간 이미지 수의 2배 이상으로 주세요).")
        return lines
    few = " 라운드가 5개 미만이라 구간이 실제보다 좁게 나올 수 있습니다." if setup.rounds < 5 else ""
    lines.append(f"Δp50의 [95% CI]: 라운드를 복원 추출하되 두 설정에 같은 라운드를 쓰는 쌍 "
                 f"부트스트랩({BOOTSTRAP_RESAMPLES}회, seed 고정)으로 구한 95% 신뢰구간입니다. "
                 "0을 포함하면 차이가 오차 범위(빨라졌다고도 느려졌다고도 할 수 없음)이고, 포함하지 "
                 "않으면 이 장치에서 실제로 달라진 것입니다. 이 실행 안의 흔들림만 반영하며 다른 "
                 f"날·다른 장치와의 차이는 담지 않습니다.{few}")
    return lines


def protocol_lines(setup: Setup, tags: list[str]) -> list[str]:
    total, evaluated = len(setup.dataset.ids), len(setup.indices)
    subset = (f"{total}장 전체" if evaluated == total
              else f"{total}장 중 {evaluated}장 (config seed {setup.seed} 무작위 부분집합)")
    floor = (f"모든 설정·모델에 {setup.score_thresh:g} (--score-thresh)"
             if setup.score_thresh is not None
             else "각 추론 설정의 값 (기본 0.05 = 모델 기본값, 학습 검증·팀 세 모델 평가와 같음 · "
                  "비교 프로토콜(0.001) 결과와 AP를 섞지 마세요)")
    sync = " · CUDA 동기화 포함" if setup.device.type == "cuda" else ""
    baselines = [MODEL_LABELS[tag] for tag in BASELINE_TAGS if tag in tags]
    same = f" ({', '.join(baselines)} 포함)" if baselines else ""
    count = len(setup.latency_images)
    rounds = "" if setup.rounds is None else f" ({setup.rounds}라운드 × {count}장)"
    latency = (f"지연 시간: 평가 이미지 중 seed {setup.seed}로 고른 {count}장 · "
               f"예열 {setup.warmup}회 · 반복 {setup.iterations}회{rounds} · "
               f"{SCOPE_LABELS[SCOPE_HOST_TO_DETECTIONS]}{sync}")
    # deploy's channels_last=None follows the same automatic rule (on for x86 CPUs, off on CUDA).
    deploy = ("Conv+BN 융합 + channels_last" if use_channels_last(setup.device)
              else "Conv+BN 융합만")
    references = ("개선 전(baseline)은 Conv+BN 융합 없이 contiguous 메모리로 잽니다. 팀의 다른 "
                  "지연 시간 표(benchmark.py·run_comparison.py, profile_stages.py, "
                  "evaluate_three_models.py --fuse)는 융합한 모델을 재므로 그 표와는 deploy 행"
                  f"(이 장치에서는 {deploy})과 'Δp50 vs deploy'로 견주세요.")
    return [
        f"장치: {describe_environment(setup.environment)}",
        f"COCO 평가: {display_path(setup.images_dir)} {subset} · 배치 1 · pycocotools",
        f"검출 점수 하한: {floor} · 이미지당 최대 검출 수는 모델 기본값",
        latency,
        *design_lines(setup),
        f"모델·추론 설정마다 새로 불러온 모델에 같은 옵션을 적용합니다{same}.",
        references,
        f"비교 실행 ID: {setup.comparison_run} (이 실행의 모든 보고서에 comparison_run으로 기록)",
    ]


def group_selections(setup: Setup, presets: list[str]):
    """Per report folder: (folder, its reports' baseline selection, {preset: selection})."""
    for group in setup.groups:
        reports, _ = load_reports(setup.output_dir / "reports" / group)
        if reports:
            yield group, select_reports(reports, variant=BASELINE_VARIANT), {
                preset: select_reports(reports, variant=preset) for preset in presets}


def folder_warnings(baseline: dict, selections: dict) -> list:
    """The dashboard's own comparability checks for every preset of one folder (no info notes)."""
    found = []
    for preset, selected in selections.items():
        found.extend(comparability_warnings(selected))
        if preset != BASELINE_VARIANT and baseline:
            found.extend(improvement_warnings(baseline, selected))
    return unique_warnings(warning for warning in found if warning.level != "info")


def unique_warnings(warnings) -> list:
    seen: dict[str, object] = {}
    for warning in warnings:
        seen.setdefault(warning.message, warning)
    return list(seen.values())


def write_summary(setup: Setup, presets: list[str], results: list[dict], tags: list[str]
                  ) -> list[dict]:
    rows = summary_rows(results, setup.environment.get("device_name") or str(setup.device))
    with open(setup.output_dir / "summary.csv", "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows({key: round(value, 4) if isinstance(value, float) else value
                          for key, value in row.items()} for row in rows)
    measured = [row for row in rows if row["status"] == "ok"]
    lines = ["# 추론 설정 개선 전/후 비교", "",
             f"- 측정 시각: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}",
             *(f"- {line}" for line in protocol_lines(setup, tags)), "", "## 결과", ""]
    lines.append("| " + " | ".join(TABLE_HEADER) + " |")
    lines.append("|" + "|".join(["---", "---"] + ["---:"] * (len(TABLE_HEADER) - 2)) + "|")
    lines.extend("| " + " | ".join(table_cells(row)) + " |" for row in measured)
    lines += ["", TABLE_NOTE, "", "## 추론 설정", ""]
    for preset in presets:
        models: dict[str, list[str]] = {}
        for result in results:
            if result["preset"] == preset and result["status"] == "ok":
                models.setdefault(result["summary"], []).append(MODEL_LABELS[result["tag"]])
        title = f"- {preset_label(preset, setup.score_thresh)} (`{preset}`)"
        if len(models) > 1:  # e.g. a CUDA graph that only one model could replay
            lines.append(f"{title}:")
            lines.extend(f"  - {', '.join(labels)}: {text}" for text, labels in models.items())
        else:
            lines.append(f"{title}: {next(iter(models), '측정한 결과 없음')}")
    others = [row for row in rows if row["status"] != "ok"]
    if others:
        lines += ["", "## 건너뛰거나 실패한 측정", ""]
        lines.extend(f"- {row['model_label']} · {row['preset_label']} (`{row['preset']}`): "
                     f"{STATUS_LABELS[row['status']]} — {row['note']}"
                     + (f" (정확도는 측정해 보고서에 저장: AP {row['ap_pct']:.2f})"
                        if row["ap_pct"] is not None else "") for row in others)
    cautions = [f"{row['model_label']} · {row['preset_label']} (`{row['preset']}`): {row['note']}"
                for row in measured if row["note"]]
    warnings = unique_warnings(warning for _, baseline, selections in group_selections(setup, presets)
                               for warning in folder_warnings(baseline, selections))
    if cautions or warnings:
        lines += ["", "## 비교 전에 확인할 점", ""]
        lines.extend(f"- [{viz.LEVEL_LABELS['warning']}] {caution}" for caution in cautions)
        lines.extend(f"- [{viz.LEVEL_LABELS.get(w.level, w.level)}] {w.message}" for w in warnings)
    folders = report_folders(setup)
    if folders:
        several = len(folders) > 1
        combined = setup.output_dir / "reports"
        example = display_path(combined if several else next(iter(folders.values())))
        lines += ["", "## 대시보드", "", (VERSIONS_NOTE + " " if several else "") + DASHBOARD_HOWTO,
                  ""]
        if several:
            lines.append(f"- `{display_path(combined)}`: 전체 "
                         f"({', '.join(MODEL_LABELS[group] for group in folders)} — 사이드바 "
                         "'Light R-CNN 체크포인트'에서 버전을 고름)")
        lines += [*(f"- `{display_path(folder)}`: {group_label(group, reported(results))}"
                    for group, folder in folders.items()),
                  "", "예:", "",
                  (f"- Windows 명령 프롬프트(cmd): `set LIGHT_RCNN_RESULTS_DIR={example}` 다음 줄에 "
                   "`streamlit run dashboard/app.py`"),
                  f"- bash: `LIGHT_RCNN_RESULTS_DIR={example} streamlit run dashboard/app.py`",
                  (f"- PowerShell: `$env:LIGHT_RCNN_RESULTS_DIR=\"{example}\"; "
                   "streamlit run dashboard/app.py`")]
    (setup.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def report_folders(setup: Setup) -> dict[str, Path]:
    """The report folders written so far, one per Light R-CNN version (or the baselines' own)."""
    folders = {group: setup.output_dir / "reports" / group for group in setup.groups}
    return {group: folder for group, folder in folders.items() if folder.is_dir()}


def write_figures(setup: Setup, presets: list[str], rows: list[dict], reported_tags: list[str]
                  ) -> list[Path]:
    """One HTML page per report folder with a before/after chart per preset (PNG if kaleido works).
    ``reported_tags`` (see reported) names the folders' models in the page titles."""
    figure = getattr(viz, "improvement_figure", None)
    if figure is None or len(presets) < 2:
        return []
    figures_dir = setup.output_dir / "figures"
    written, images_ok = [], True
    for group, baseline, selections in group_selections(setup, presets):
        label = group_label(group, reported_tags)
        figures = {}
        for preset in presets[1:]:
            selected = selections[preset]
            if not baseline or not selected:
                continue
            fig = figure(baseline, selected, "light", preset_label(preset, setup.score_thresh),
                         ap_metric="AP_50_95", percentile="p50")
            if viz.is_empty(fig):
                continue
            fig.update_layout(title_text=f"<b>개선 전/후 비교 · {MODEL_LABELS.get(group, label)} · "
                                         f"{preset_label(preset, setup.score_thresh)}</b>")
            figures[f"improvement_{group}_{preset}"] = fig
        if not figures:
            continue
        members = {*BASELINE_TAGS, group}
        chosen = [row for row in rows if row["status"] == "ok" and row["model"] in members]
        table = [dict(zip(TABLE_HEADER, table_cells(row))) for row in chosen]
        cautions = [ComparisonWarning("warning", f"{row['model_label']} · {row['preset_label']}: "
                                                 f"{row['note']}") for row in chosen if row["note"]]
        figures_dir.mkdir(parents=True, exist_ok=True)
        page = figures_dir / f"improvement_{group}.html"
        # The sources table lists the "before" reports; summary.csv names every report file.
        page.write_text(viz.html_report(figures, table,
                                        cautions + folder_warnings(baseline, selections),
                                        baseline, "light", f"개선 전/후 비교 — {label}"),
                        encoding="utf-8")
        written.append(page)
        if not images_ok:
            continue
        try:
            written.extend(export_images(figures, figures_dir, "light", ["png"], 2.0, 1100))
        except Exception as exc:  # noqa: BLE001 - kaleido/Chrome failures vary; keep the HTML
            images_ok = False
            print(f"[주의] PNG 그래프는 건너뜁니다 ({type(exc).__name__}). kaleido와 Chrome이 "
                  "필요합니다: python -c \"import kaleido; kaleido.get_chrome_sync()\"",
                  file=sys.stderr, flush=True)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    presets = parse_presets(args.presets, parser)
    tags = [tag for tag in MODEL_KINDS if tag in args.models]
    for name in ("latency_images", "warmup", "iterations"):
        value = getattr(args, name)
        if value is not None and value < (0 if name == "warmup" else 1):
            parser.error(f"--{name.replace('_', '-')} 값이 너무 작습니다: {value}")
    if args.limit < 0:
        parser.error("--limit은 0 이상이어야 합니다")
    if args.score_thresh is not None and not 0 <= args.score_thresh <= 1:
        parser.error("--score-thresh는 0과 1 사이여야 합니다")

    config_path = Path(args.config)
    if not config_path.is_file():
        raise SystemExit(f"config 파일이 없습니다: {config_path}")
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    images_dir = Path(args.images or config["data"]["val_images"])
    annotations = Path(args.annotations or config["data"]["val_annotations"])
    missing = [str(path) for path in (images_dir, annotations) if not path.exists()]
    if missing:
        raise SystemExit("COCO 평가 데이터가 없습니다:\n  - " + "\n  - ".join(missing))
    weights = {"faster_rcnn": args.faster_weights, "mobilenet": args.mobilenet_weights,
               "v2": args.v2_weights, "v3": args.v3_weights}
    for tag in tags:
        if weights[tag] and not Path(weights[tag]).is_file():
            raise SystemExit(f"{MODEL_LABELS[tag]} 체크포인트가 없습니다: {weights[tag]}")
    try:
        device = resolve_device(args.device or config.get("device", "cuda"), "measuring")
    except ValueError as exc:
        parser.error(f"--device: {exc}")
    # COCO images vary in shape, so cuDNN autotuning would re-tune per shape inside the timings.
    torch.backends.cudnn.benchmark = False

    seed = int(config.get("seed", 0))
    dataset = CocoTensorDataset(images_dir, str(annotations))
    indices = evaluation_subset(dataset.ids, args.limit, seed)
    if not indices:
        raise SystemExit(f"평가할 이미지가 없습니다: {annotations}")
    defaults = GPU_COUNTS if device.type == "cuda" else CPU_COUNTS
    count, warmup, iterations = (default if value is None else value for value, default in zip(
        (args.latency_images, args.warmup, args.iterations), defaults))
    latency_paths, latency_ids, latency_images = latency_sample(
        dataset, images_dir, indices, min(count, len(indices)), seed)
    skipped_presets = [preset for preset in presets if cuda_only(preset) and device.type != "cuda"]
    # Interleaved: whole rounds, each timing every latency image once per preset.
    timed = len(presets) - len(skipped_presets)
    rounds = None if args.sequential_latency else interleaved_rounds(
        iterations, len(latency_images), timed)
    output_dir = Path(args.output_dir or Path("outputs/improvements")
                      / datetime.now().strftime("%Y%m%d_%H%M%S"))
    if any((output_dir / "reports").rglob("*.json")):
        print(f"[주의] {output_dir}/reports에 이전 보고서가 있습니다. 같은 모델·설정의 보고서는 "
              "새 결과로 덮어쓰지만 다른 보고서는 남으니 새 폴더를 쓰는 편이 안전합니다.",
              file=sys.stderr, flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup = Setup(
        config=config, config_path=config_path, device=device,
        image_size=int(config["image_size"]), seed=seed, weights=weights,
        images_dir=images_dir, annotations=annotations, dataset=dataset, indices=indices,
        limit=args.limit, latency_paths=latency_paths, latency_ids=latency_ids,
        latency_images=latency_images, warmup=max(warmup, len(latency_images)),
        iterations=iterations if rounds is None else rounds * len(latency_images),
        score_thresh=args.score_thresh, output_dir=output_dir,
        groups=report_groups(tags), environment=collect_environment(str(device)),
        rounds=rounds,
        # Unique per run: a rerun into the same folder must not pass for the same comparison.
        comparison_run=f"{output_dir.resolve().name}-{uuid.uuid4().hex[:8]}")

    print("=== 추론 설정 개선 전/후 비교 ===", flush=True)
    print(f"모델: {', '.join(MODEL_LABELS[tag] for tag in tags)}")
    print("추론 설정: " + ", ".join(f"{preset_label(p, args.score_thresh)} ({p})" for p in presets))
    for line in protocol_lines(setup, tags):
        print(line)
    if any(value is None for value in (args.latency_images, args.warmup, args.iterations)):
        where = "GPU" if device.type == "cuda" else "CPU"
        print(f"지정하지 않은 지연 시간 측정 횟수는 {where} 기본값을 씁니다 "
              f"(이미지 {defaults[0]}장, 예열 {defaults[1]}회, 반복 {defaults[2]}회).")
    if setup.warmup > warmup:
        print(f"예열을 지연 시간 이미지 수에 맞춰 {setup.warmup}회로 늘렸습니다 "
              "(모든 입력 크기를 측정 전에 한 번씩).")
    if setup.iterations > iterations:
        print(f"반복을 라운드 단위로 맞춰 {setup.iterations}회로 늘렸습니다 "
              f"({rounds}라운드 × 지연 시간 이미지 {len(latency_images)}장 · 라운드 수는 설정 "
              f"{timed}개의 배수라 설정마다 순서의 모든 자리를 같은 횟수만큼 돕니다).")
    print(f"결과 폴더: {display_path(output_dir)}", flush=True)
    cuda_note = f"CUDA 전용 설정이라 {device.type.upper()}에서는 측정하지 않습니다"
    for preset in skipped_presets:
        print(f"[건너뜀] {PRESET_LABELS[preset]} ({preset}): {cuda_note}.", flush=True)

    results: list[dict] = []
    unavailable: dict[str, str] = {}
    total = len(tags) * (len(presets) - len(skipped_presets))
    number = 0
    for tag in tags:
        finished: list[dict] = []  # this model's results, in preset order once complete
        runs: list[Prepared] = []  # interleaved: built and evaluated, timed together below
        for preset in presets:
            label = preset_label(preset, args.score_thresh)
            entry = {"tag": tag, "preset": preset, "label": label}
            if preset in skipped_presets:
                finished.append({**entry, "status": "skipped", "note": cuda_note})
                continue
            number += 1
            print(f"\n[{number}/{total}] {MODEL_LABELS[tag]} · {label} ({preset})", flush=True)
            if tag in unavailable:
                print(f"  [건너뜀] {unavailable[tag]}", flush=True)
                finished.append({**entry, "status": "skipped", "note": unavailable[tag]})
                continue
            started = time.perf_counter()
            try:
                if rounds is None:
                    finished.append(run_one(tag, preset, setup))
                else:
                    runs.append(prepare_run(tag, preset, setup))
                print(f"  ({time.perf_counter() - started:.0f}초)", flush=True)
            except WeightsUnavailable as exc:
                unavailable[tag] = str(exc)
                print(f"  [건너뜀] {MODEL_LABELS[tag]}: {exc}", flush=True)
                finished.append({**entry, "status": "skipped", "note": str(exc)})
            except Exception as exc:  # noqa: BLE001 - one failed run must not cost the others
                traceback.print_exc()
                finished.append({**entry, "status": "failed",
                                 "note": f"{type(exc).__name__}: {exc}"})
            finally:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            if rounds is None:  # a lost session keeps finished runs
                write_summary(setup, presets, results + finished, tags)
        if runs:
            finished.extend(run_interleaved(runs, setup))
            runs.clear()  # only one model's presets stay loaded at a time
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        finished.sort(key=lambda result: presets.index(result["preset"]))
        results.extend(finished)
        if rounds is not None:
            write_summary(setup, presets, results, tags)

    rows = write_summary(setup, presets, results, tags)
    figures = write_figures(setup, presets, rows, reported(results))
    measured = [row for row in rows if row["status"] == "ok"]
    print("\n=== 개선 전/후 비교 결과 ===")
    if measured:
        print(console_table(measured))
        print(TABLE_NOTE)
    for row in rows:
        if row["status"] != "ok" or row["note"]:  # a measured row's note is a caution
            status = STATUS_LABELS.get(row["status"], viz.LEVEL_LABELS["warning"])
            print(f"- {status}: {row['model_label']} · {row['preset_label']} — {row['note']}")
    print(f"\n표: {display_path(output_dir / 'summary.md')}, {display_path(output_dir / 'summary.csv')}")
    for path in figures:
        print(f"그래프: {display_path(path)}")
    folders = report_folders(setup)
    for group, folder in folders.items():
        print(f"대시보드 결과 폴더 ({group_label(group, reported(results))}): "
              f"{display_path(folder)}")
    if len(folders) > 1:
        print(f"대시보드 결과 폴더 (전체, 'Light R-CNN 체크포인트'에서 버전 선택): "
              f"{display_path(output_dir / 'reports')}")
    if folders:
        print((VERSIONS_NOTE + " " if len(folders) > 1 else "")
              + "streamlit run dashboard/app.py 실행 후 사이드바 '결과 폴더'에 위 경로를 넣으세요.",
              flush=True)
    if not measured:
        print("측정한 결과가 없습니다.", file=sys.stderr)
        return 1
    return 1 if any(row["status"] == "failed" for row in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
