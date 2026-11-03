"""Shared result schema for accuracy, latency and stage-profile reports.

Every measurement script builds one JSON record per run with ``build_report`` and writes it with
``write_report``. The dashboard and the figure exporter read records back with ``load_reports``,
pick one record per (kind, model) with ``select_reports`` and surface comparability problems with
``comparability_warnings``.

A record may also carry the inference preset it was measured with (top-level ``variant``,
``variant_label`` and ``inference_options``, see ``light_rcnn.inference_opts``); records without one
ran the model defaults and count as the ``baseline`` variant. Light R-CNN v2 and v3 share the
``light_rcnn`` key; ``checkpoints`` tells their reports apart by the weights they record.

This module must stay importable without PyTorch: the dashboard only needs pandas, plotly and
streamlit, so torch is imported lazily inside ``collect_environment``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 2

# Fixed model order and display names. Colors in viz.py follow this order, never a rank.
MODELS: dict[str, str] = {
    "faster_rcnn": "Vanilla Faster R-CNN",
    "light_rcnn": "Light R-CNN",
    "yolo26": "YOLO26",
}
MODEL_ORDER = list(MODELS)
KINDS = ("accuracy", "latency", "stages")
KIND_LABELS = {"accuracy": "정확도", "latency": "지연 시간", "stages": "단계별 처리 시간"}

COCO_VAL2017_IMAGES = 5000

# Canonical timing scopes. Latencies are only comparable when their scope_id matches.
SCOPE_HOST_TO_DETECTIONS = "host_uint8_to_detections"
SCOPE_LABELS = {
    SCOPE_HOST_TO_DETECTIONS: "디코드된 uint8 이미지(호스트 메모리) → 최종 검출 결과 "
                              "(장치 복사·정규화·리사이즈·모델·후처리/NMS 포함, 파일 디코드 제외)",
}

# Stage keys written by scripts/profile_stages.py, in pipeline order.
STAGE_LABELS = {
    "input": "입력 복사·정규화",
    "preprocess": "전처리 (리사이즈·정규화)",
    "backbone": "백본 + FPN/BiFPN",
    "rpn": "RPN (제안 + NMS)",
    "roi_heads": "RoI 헤드 (RoIAlign + 분류/회귀 + NMS)",
    "postprocess": "후처리 (원본 크기 복원)",
}

# What the first timed call (cold_start_ms) includes; only equal scopes are comparable.
COLD_START_FIRST_FORWARD = "first_forward_after_load"
COLD_START_FIRST_PREDICT = "first_predict_incl_predictor_setup"
COLD_START_LABELS = {
    COLD_START_FIRST_FORWARD: "장치에 올린 모델의 첫 추론",
    COLD_START_FIRST_PREDICT: "첫 predict 호출 (Ultralytics 예측기 준비·레이어 융합 포함)",
}

LOW_ITERATION_WARNING = 100

# Inference presets (light_rcnn.inference_opts.PRESETS, which needs torch). Reports name their own
# label; the baseline one also covers files written before presets existed.
BASELINE_VARIANT = "baseline"
BASELINE_VARIANT_LABEL = "개선 전 (기본 설정)"
CUSTOM_VARIANT = "custom"  # options that match no preset
VARIANT_ORDER = ("baseline", "deploy", "fast", "accurate", "accurate_fast", "gpu_fp16", "gpu_graph")
CHECKPOINT_MODEL = "light_rcnn"  # the model compared across checkpoints (v2, v3, ...)


# --------------------------------------------------------------------------- writing helpers


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_commit(root: str | Path | None = None) -> str | None:
    """Return the current commit hash, suffixed with ``-dirty`` when tracked files changed."""
    cwd = Path(root) if root else Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
                                text=True, check=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=cwd,
                               capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return f"{commit}-dirty" if dirty else commit


def file_sha256(path: str | Path | None) -> str | None:
    if not path or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown CPU"


def collect_environment(device: str | None = None) -> dict[str, Any]:
    """Describe the software and hardware a measurement ran on."""
    env: dict[str, Any] = {
        "device": str(device) if device is not None else None,
        "device_type": "cpu",
        "device_name": _cpu_name(),
        "cpu_name": _cpu_name(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: _package_version(name) for name in (
            "torch", "torchvision", "ultralytics", "pycocotools", "onnxruntime", "tensorrt")},
    }
    try:
        import torch
    except ImportError:
        return env
    env["torch_num_threads"] = torch.get_num_threads()
    env["cuda_runtime"] = torch.version.cuda
    env["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    env["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    wants_cuda = device is not None and (str(device).startswith("cuda") or str(device).isdigit())
    if wants_cuda and torch.cuda.is_available():
        index = int(str(device).split(":")[1]) if ":" in str(device) else (
            int(device) if str(device).isdigit() else torch.cuda.current_device())
        props = torch.cuda.get_device_properties(index)
        env.update({
            "device_type": "cuda",
            "device_name": torch.cuda.get_device_name(index),
            "gpu_memory_gb": round(props.total_memory / 1024 ** 3, 2),
            "gpu_capability": f"{props.major}.{props.minor}",
        })
    return env


def percentile(values: Iterable[float], q: float) -> float:
    """Linear-interpolation percentile (numpy's default method); q is in [0, 100]."""
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * q / 100.0
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def latency_summary(samples_ms: Iterable[float], *, cold_start_ms: float | None = None) -> dict[str, Any]:
    """Summarize per-call latencies; keeps the raw samples so distributions can be plotted."""
    samples = [float(value) for value in samples_ms]
    if not samples:
        raise ValueError("latency_summary needs at least one sample")
    count = len(samples)
    mean = sum(samples) / count
    std = math.sqrt(sum((value - mean) ** 2 for value in samples) / (count - 1)) if count > 1 else 0.0
    return {
        "n": count,
        "p50_ms": percentile(samples, 50),
        "p90_ms": percentile(samples, 90),
        "p95_ms": percentile(samples, 95),
        "p99_ms": percentile(samples, 99),
        "mean_ms": mean,
        "std_ms": std,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "fps": 1000.0 / mean if mean > 0 else None,
        "cold_start_ms": cold_start_ms,
        "samples_ms": [round(value, 4) for value in samples],
    }


def build_report(kind: str, model: str, **sections: Any) -> dict[str, Any]:
    """Create a schema-v2 record. ``sections`` holds accuracy/latency/stages/model_stats/etc."""
    if kind not in KINDS:
        raise ValueError(f"Unknown report kind: {kind}")
    if model not in MODELS:
        raise ValueError(f"Unknown model key: {model}")
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "model": model,
        "model_name": MODELS[model],
        "created_at": utc_now(),
        "git_commit": git_commit(),
    }
    report.update(sections)
    return report


def write_report(report: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    # ASCII escapes keep the files readable even by tools that assume the locale codec.
    output.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return output


def default_report_path(kind: str, model: str, root: str | Path = "outputs") -> Path:
    return Path(root) / "reports" / f"{model}_{kind}.json"


def display_path(path: str | Path | None) -> str | None:
    """Repository-relative path when possible, so reports do not leak machine-specific prefixes."""
    if path is None:
        return None
    resolved = Path(path).resolve()
    root = Path(__file__).resolve().parents[2]
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def dataset_info(annotations: str | Path, images_dir: str | Path) -> dict[str, Any]:
    """Fingerprint the evaluation set so reports scored on different annotations are caught."""
    return {
        "annotations": display_path(annotations),
        "annotations_sha256": file_sha256(annotations),
        "images_dir": display_path(images_dir),
    }


# --------------------------------------------------------------------------- reading helpers


@dataclass
class Report:
    kind: str
    model: str
    data: dict[str, Any]
    source: str
    created_at: str
    legacy: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def model_name(self) -> str:
        return MODELS[self.model]

    @property
    def section(self) -> dict[str, Any]:
        return self.data.get(self.kind) or {}

    @property
    def environment(self) -> dict[str, Any]:
        return self.data.get("environment") or {}

    @property
    def inference_options(self) -> dict[str, Any]:
        options = self.data.get("inference_options")
        return options if isinstance(options, dict) else {}

    @property
    def variant(self) -> str:
        """Inference preset of the run; reports without one ran the model defaults."""
        value = self.data.get("variant")
        if (value is None or not str(value).strip()) and "preset" in self.inference_options:
            # apply_inference_options' record: a tuned run must never pass for the baseline.
            value = self.inference_options["preset"] or CUSTOM_VARIANT
        return ("" if value is None else str(value).strip()) or BASELINE_VARIANT

    @property
    def variant_label(self) -> str:
        for label in (self.data.get("variant_label"), self.inference_options.get("label")):
            if isinstance(label, str) and label.strip():
                return label.strip()
        return _default_variant_label(self.variant)


def _default_variant_label(variant: str) -> str:
    return BASELINE_VARIANT_LABEL if variant == BASELINE_VARIANT else variant


def identify_model(raw: dict[str, Any]) -> str | None:
    key = str(raw.get("model", "")).lower()
    if key in MODELS:
        return key
    text = " ".join(str(raw.get(name, "")) for name in ("model_name", "model", "model_family", "architecture"))
    compact = re.sub(r"[^a-z0-9]+", "", text.lower())
    if compact.startswith("yolo") or "yolo26" in compact:
        return "yolo26"
    if "light" in compact:
        return "light_rcnn"
    if "fasterrcnn" in compact:
        return "faster_rcnn"
    return None


def _first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if mapping.get(key) is not None:
            return mapping[key]
    return None


_RUN_FOLDER = re.compile(r"^\d{8}_\d{6}$")  # run_comparison.py: outputs/comparison_runs/<실행 시각>


def _created_at(raw: dict[str, Any], mtime: float, source: str = "") -> tuple[str, str | None]:
    """Measurement time plus a note when it had to be inferred (v1 files have no created_at)."""
    stamp = raw.get("created_at")
    if isinstance(stamp, str) and stamp:
        return stamp, None
    folder = Path(source).parent.name
    if _RUN_FOLDER.match(folder):
        try:
            # The folder is the measuring machine's local time; its zone is unknown, so no offset.
            run_time = datetime.strptime(folder, "%Y%m%d_%H%M%S").isoformat(timespec="seconds")
            return run_time, "측정 시각 미기록: 실행 폴더 이름(현지 시각)에서 추정"
        except ValueError:
            pass
    return (datetime.fromtimestamp(mtime, timezone.utc).isoformat(timespec="seconds"),
            "측정 시각 미기록: 파일 수정 시각으로 대체(실제 측정 시각과 다를 수 있음)")


def _as_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None


def _legacy_device_type(device: Any) -> str | None:
    text = "" if device is None else str(device).lower()
    if text == "cpu":
        return "cpu"
    return "cuda" if text.startswith("cuda") or text.isdigit() else None


def _legacy_reports(raw: dict[str, Any], model: str, source: str, created: str) -> list[Report]:
    """Translate the flat v1 JSON files written before schema v2 existed."""
    reports = []
    common = {
        "model": model,
        "model_name": MODELS[model],
        "weights": raw.get("weights"),
        "image_size": _first(raw, "image_size", "imgsz"),
        "precision": raw.get("precision"),
        "batch_size": raw.get("batch_size"),
        # v1 "device" is "cpu"/"cuda:0", not a hardware name, so it must not land in device_name.
        "environment": {"device": raw.get("device"), "device_type": _legacy_device_type(raw.get("device"))},
    }
    accuracy_src = raw.get("accuracy") if isinstance(raw.get("accuracy"), dict) else raw
    ap = _first(accuracy_src, "AP_50_95", "map", "mAP")
    if ap is not None:
        images = _as_int(raw.get("images_evaluated"))
        evaluator = "ultralytics" if model == "yolo26" else "pycocotools"
        accuracy = {
            "evaluator": evaluator,
            "images_evaluated": images,
            "images_total": None,
            "is_partial": None if images is None else images < COCO_VAL2017_IMAGES,
            "AP_50_95": ap,
            "AP_50": _first(accuracy_src, "AP_50", "map50"),
            "AP_75": _first(accuracy_src, "AP_75", "map75"),
            "AP_small": accuracy_src.get("AP_small"),
            "AP_medium": accuracy_src.get("AP_medium"),
            "AP_large": accuracy_src.get("AP_large"),
        }
        reports.append(Report("accuracy", model, {**common, "kind": "accuracy", "accuracy": accuracy},
                              source, created, legacy=True,
                              notes=["schema v1 파일: 평가 이미지 수와 평가기 정보가 추정값입니다."]))
    latency_src = raw.get("latency") if isinstance(raw.get("latency"), dict) else raw
    p50 = _first(latency_src, "p50_ms", "median_ms")
    if p50 is not None:
        latency = {
            "scope_id": None,
            "timed_scope": raw.get("timed_scope"),
            "warmup": raw.get("warmup"),
            "iterations": raw.get("iterations"),
            "image_count": raw.get("image_count"),
            "n": raw.get("iterations"),
            "p50_ms": p50,
            "p95_ms": latency_src.get("p95_ms"),
        }
        reports.append(Report("latency", model, {**common, "kind": "latency", "latency": latency},
                              source, created, legacy=True,
                              notes=["schema v1 파일: 원시 샘플이 없어 분포 그래프를 그릴 수 없습니다."]))
    return reports


def normalize(raw: Any, source: str, mtime: float = 0.0) -> list[Report]:
    """Turn one parsed JSON payload (dict or list of dicts) into zero or more reports."""
    items = raw if isinstance(raw, list) else [raw]
    reports: list[Report] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        model = identify_model(item)
        if model is None:
            continue
        created, stamp_note = _created_at(item, mtime, source)
        if item.get("schema_version") == SCHEMA_VERSION and item.get("kind") in KINDS:
            kind = item["kind"]
            if not isinstance(item.get(kind), dict):
                raise ValueError(f"'{kind}' 항목이 객체가 아닙니다")
            if not isinstance(item.get("environment"), (dict, type(None))):
                raise ValueError("'environment' 항목이 객체가 아닙니다")
            found = [Report(kind, model, item, source, created)]
        else:
            found = _legacy_reports(item, model, source, created)
        if stamp_note:
            for report in found:
                report.notes.append(stamp_note)
        reports.extend(found)
    return reports


def load_reports(root: str | Path = "outputs", extra: Iterable[tuple[str, bytes | str]] = ()
                 ) -> tuple[list[Report], list[str]]:
    """Read every result JSON under ``root`` (skipping hidden folders) plus uploaded payloads."""
    reports: list[Report] = []
    errors: list[str] = []
    base = Path(root)
    paths = sorted(base.rglob("*.json")) if base.exists() else []
    for path in paths:
        if any(part.startswith(".") for part in path.relative_to(base).parts):
            continue  # e.g. outputs/.ultralytics/settings.json
        # A malformed file is listed as unreadable instead of taking the whole dashboard down.
        try:
            raw = json.loads(path.read_text(encoding="utf-8-sig"))  # -sig: Windows tools add a BOM
            reports.extend(normalize(raw, str(path), path.stat().st_mtime))
        except (OSError, TypeError, ValueError, AttributeError, KeyError) as exc:
            errors.append(f"{path}: {exc}")
    for name, payload in extra:
        try:
            text = payload.decode("utf-8-sig") if isinstance(payload, bytes) else payload.lstrip("\ufeff")
            reports.extend(normalize(json.loads(text), f"업로드: {name}"))
        except (TypeError, ValueError, AttributeError, KeyError) as exc:
            errors.append(f"{name}: {exc}")
    return reports, errors


def _rank(report: Report) -> tuple:
    # Schema-v2 reports always beat v1 files: v1 lacks the shared evaluator and timing scope.
    if report.kind == "accuracy":
        section = report.section
        complete = section.get("is_partial") is False
        return (not report.legacy, complete, _as_int(section.get("images_evaluated")) or 0,
                report.created_at)
    return (not report.legacy, report.created_at)


UNKNOWN_DEVICE = "장치 미기록"


def device_label(report: Report) -> str:
    """Hardware a timing ran on; v1 files only know 'cpu'/'cuda', so they say so."""
    env = report.environment
    name = env.get("device_name")
    if name:
        return str(name)
    # v1 YOLO files store the CUDA index ("0"); the normalised type groups them with "cuda".
    device = (env.get("device_type") if report.legacy else None) or env.get("device")
    return f"{device} (기기명 미기록)" if device else UNKNOWN_DEVICE


def timing_devices(reports: Iterable[Report]) -> list[str]:
    """Devices that latency/stage reports were measured on, newest first."""
    newest: dict[str, str] = {}
    for report in reports:
        if report.kind in ("latency", "stages"):
            label = device_label(report)
            newest[label] = max(newest.get(label, ""), report.created_at)
    return sorted(newest, key=newest.get, reverse=True)


def variants(reports: Iterable[Report]) -> list[str]:
    """Inference presets present in ``reports``: the baseline first, then the presets in
    inference_opts order, then any other name alphabetically."""
    found = {report.variant for report in reports}
    known = [name for name in VARIANT_ORDER if name in found]
    return known + sorted(found - set(known))


def variant_labels(reports: Iterable[Report]) -> dict[str, str]:
    """Display label per variant, in ``variants`` order: the newest label a report records."""
    reports = list(reports)
    best: dict[str, tuple[bool, str, str]] = {}
    for report in reports:
        label = report.variant_label
        rank = (label != _default_variant_label(report.variant), report.created_at, label)
        best[report.variant] = max(best.get(report.variant, rank), rank)
    return {name: best[name][2] for name in variants(reports)}


def select_reports(reports: Iterable[Report], device: str | None = None,
                   variant: str | None = None, checkpoint: str | None = None
                   ) -> dict[tuple[str, str], Report]:
    """Pick one report per (kind, model): schema v2 first, then complete and larger accuracy runs,
    then the newest. With ``device``, timings (latency, stages) come only from that hardware;
    accuracy does not depend on the device, so it is always eligible. With ``variant``, only
    reports measured with that inference preset count (no preset recorded = ``baseline``). With
    ``checkpoint`` (a key or label from ``checkpoints``), Light R-CNN reports come only from that
    checkpoint; the other models are left as they are."""
    if checkpoint is not None:
        reports = checkpoint_reports(reports, checkpoint)
    chosen: dict[tuple[str, str], Report] = {}
    for report in reports:
        if variant is not None and report.variant != variant:
            continue
        if device is not None and report.kind != "accuracy" and device_label(report) != device:
            continue
        key = (report.kind, report.model)
        if key not in chosen or _rank(report) > _rank(chosen[key]):
            chosen[key] = report
    return chosen


def select_baseline(reports: Iterable[Report], after: dict[tuple[str, str], Report],
                    device: str | None = None, checkpoint: str | None = None
                    ) -> dict[tuple[str, str], Report]:
    """The 'before' of ``after`` (select_reports of another preset): per (kind, model), the
    baseline report of the after-report's own comparison_run when there is one, else
    select_reports' baseline pick, with the same device and checkpoint rules. A partial
    compare_improvements.py run so keeps its own baseline instead of a larger evaluation's."""
    reports = list(reports)
    before = select_reports(reports, device=device, variant=BASELINE_VARIANT, checkpoint=checkpoint)
    for key, report in after.items():
        run = report.data.get("comparison_run")
        if not run:
            continue
        same_run = [r for r in reports if r.data.get("comparison_run") == run]
        picked = select_reports(same_run, device=device, variant=BASELINE_VARIANT,
                                checkpoint=checkpoint).get(key)
        if picked is not None:
            before[key] = picked
    return before


# --------------------------------------------------------------------------- comparability


@dataclass
class ComparisonWarning:
    level: str  # "error" breaks the comparison, "warning" weakens it, "info" is context
    message: str


def _hashable(value: Any) -> Any:
    # Hand-edited files can hold a list or dict where a scalar belongs.
    if isinstance(value, list):
        return tuple(_hashable(item) for item in value)
    return str(value) if isinstance(value, (dict, set)) else value


def _distinct(values: Iterable[Any]) -> set:
    return {_hashable(value) for value in values if value is not None}


def _join(values: Iterable[Any], sep: str = ", ") -> str:
    """Sort mixed-type values without raising: numbers numerically, everything else as text."""
    numeric = (int, float)
    ordered = sorted(values, key=lambda v: (not isinstance(v, numeric),
                                            v if isinstance(v, numeric) else str(v)))
    return sep.join(str(value) for value in ordered)


def is_cpu_run(report: Report) -> bool:
    env = report.environment
    return env.get("device_type") == "cpu" or env.get("device") == "cpu"


def on_target_device(report: Report) -> bool:
    """The latency target is set for a T4 GPU; other or unrecorded hardware is only a reference."""
    name = str(report.environment.get("device_name") or "")
    return not is_cpu_run(report) and bool(re.search(r"\bT4\b", name, re.IGNORECASE))


def _cuda_graph_status(record: Mapping[str, Any] | None) -> Any:
    graph = (record or {}).get("cuda_graph_status")
    return graph.get("status") if isinstance(graph, Mapping) else graph


def cuda_graph_note(record: Mapping[str, Any] | None) -> str | None:
    """Korean caution for a run that asked for a CUDA graph and did not replay it, else None.

    ``record`` is apply_inference_options' record or a report's ``inference_options``. Its summary
    names the CUDA graph at apply time; only ``cuda_graph_status``, final after the first CUDA
    forward, says whether the self-check kept the graph or the backbone ran eagerly.
    """
    if not (record or {}).get("cuda_graph_status"):
        return None
    status = _cuda_graph_status(record)
    if status == "captured":
        return None
    if str(status).startswith("disabled"):
        return "CUDA 그래프 사용 안 됨 (자체 검사에서 재생 결과가 eager와 달라 백본을 eager로 실행)"
    return f"CUDA 그래프 상태: {status or '기록 없음'}"


def comparability_warnings(selected: dict[tuple[str, str], Report],
                           latency_target_ms: float | None = None) -> list[ComparisonWarning]:
    warnings: list[ComparisonWarning] = []
    accuracy = [r for (kind, _), r in selected.items() if kind == "accuracy"]
    latency = [r for (kind, _), r in selected.items() if kind == "latency"]
    stages = [r for (kind, _), r in selected.items() if kind == "stages"]

    # Only reachable when the caller did not filter by variant (select_reports(variant=None)).
    by_variant: dict[str, list[str]] = {}
    for key in sorted(selected, key=lambda k: (MODEL_ORDER.index(k[1]), KINDS.index(k[0]))):
        by_variant.setdefault(selected[key].variant, []).append(
            f"{MODELS[key[1]]} {KIND_LABELS[key[0]]}")
    if len(by_variant) > 1:
        labels = variant_labels(selected.values())
        detail = " / ".join(f"{labels[name]}: {', '.join(items)}" for name, items in by_variant.items())
        warnings.append(ComparisonWarning("error", f"결과마다 추론 설정이 다릅니다 ({detail}). "
                                "같은 추론 설정으로 측정한 결과끼리만 비교하세요."))

    for kind, group, label in (("accuracy", accuracy, "정확도"), ("latency", latency, "지연 시간")):
        missing = [MODELS[m] for m in MODEL_ORDER if (kind, m) not in selected]
        if missing and group:
            warnings.append(ComparisonWarning("info", f"{label} 결과가 없는 모델: {', '.join(missing)}"))

    evaluators = _distinct(r.section.get("evaluator") for r in accuracy)
    if len(evaluators) > 1:
        warnings.append(ComparisonWarning("error", "AP 계산기가 모델마다 다릅니다 "
                                f"({_join(evaluators)}). 같은 pycocotools로 다시 평가해야 공정합니다."))
    image_counts = _distinct(r.section.get("images_evaluated") for r in accuracy)
    if len(image_counts) > 1:
        warnings.append(ComparisonWarning("error", "모델마다 평가한 이미지 수가 다릅니다 "
                                f"({_join(image_counts)}장)."))
    hashes = _distinct(r.section.get("image_ids_sha256") for r in accuracy)
    if len(hashes) > 1:
        warnings.append(ComparisonWarning("error", "모델마다 평가한 이미지가 다릅니다 (같은 장수라도 "
                                "다른 사진). 같은 config seed와 --limit으로 다시 평가하세요."))
    # Training imports and older reports carry no id hash; fall back to how the subset was drawn.
    subset_reports = [r for r in accuracy if r.section.get("is_partial")]
    if (len(accuracy) > 1 and len(hashes) <= 1
            and any(not r.section.get("image_ids_sha256") for r in subset_reports)):
        subsets = _distinct((r.section.get("subset_selection"), r.section.get("subset_seed"))
                            for r in subset_reports if r.section.get("subset_selection"))
        if len(subsets) > 1:
            warnings.append(ComparisonWarning("error", "부분 평가의 이미지 선택 방식이나 seed가 "
                                    "모델마다 다릅니다. 같은 seed와 --limit으로 다시 평가하세요."))
        unknown = [r.model_name for r in subset_reports if not r.section.get("subset_selection")]
        if unknown:
            warnings.append(ComparisonWarning("warning", "부분 평가 이미지가 다른 모델과 같은지 확인할 수 "
                                    f"없습니다 ({', '.join(unknown)}). 같은 seed로 다시 평가하세요."))
    batch_sizes = _distinct(r.data.get("batch_size") for r in accuracy)
    if len(batch_sizes) > 1:
        warnings.append(ComparisonWarning("warning", f"정확도 평가 배치 크기가 다릅니다 ({_join(batch_sizes)}). "
                                "R-CNN은 여러 장을 묶으면 패딩 때문에 AP가 1%p 안팎 달라질 수 있습니다."))
    partial = [r.model_name for r in accuracy if r.section.get("is_partial")]
    if partial:
        warnings.append(ComparisonWarning("warning", f"부분 평가 결과입니다 ({', '.join(partial)}). "
                                "전체 COCO val2017(5000장) 결과와 직접 비교하지 마세요."))
    totals = _distinct(r.section.get("images_total") for r in accuracy)
    small = [total for total in totals if total != COCO_VAL2017_IMAGES]
    if small:
        warnings.append(ComparisonWarning("warning", "평가 세트가 COCO val2017 전체(5000장)가 아닙니다 "
                                f"({_join(small)}장). 스모크 테스트 결과로만 보세요. "
                                "학습에 쓰인 이미지가 섞여 있으면 AP가 실제보다 높게 나옵니다."))
    datasets = [r.section.get("dataset") for r in accuracy]
    fingerprints = _distinct(d.get("annotations_sha256") for d in datasets if isinstance(d, dict))
    if len(fingerprints) > 1:
        warnings.append(ComparisonWarning("error", "모델마다 다른 주석(annotation) 파일로 평가했습니다."))
    thresholds = _distinct(r.section.get("score_threshold") for r in accuracy)
    if len(thresholds) > 1:
        warnings.append(ComparisonWarning("warning", "검출 점수 임계값이 다릅니다 "
                                f"({_join(thresholds)}). 낮을수록 AP가 유리합니다."))

    for group, kind in ((latency, "latency"), (stages, "stages")):
        label = KIND_LABELS[kind]
        # v1 files only know "cpu"/"cuda:0", so compare device types first and names only within one.
        device_types = _distinct(r.environment.get("device_type") for r in group)
        devices = _distinct(r.environment.get("device_name") for r in group)
        if len(device_types) > 1:
            devices = device_types
        elif len(group) > 1 and any(r.environment.get("device_name") is None for r in group):
            warnings.append(ComparisonWarning("warning", f"일부 {label} 결과에 장치 이름이 없어 "
                                    "같은 장치인지 확인할 수 없습니다."))
        if len(devices) > 1:
            warnings.append(ComparisonWarning("error", f"{label}을 서로 다른 장치에서 측정했습니다 "
                                    f"({_join(devices, ' / ')})."))
        precisions = _distinct(r.data.get("precision") for r in group)
        if len(precisions) > 1:
            warnings.append(ComparisonWarning("error", f"{label} 정밀도가 다릅니다 ({_join(precisions)})."))
        threads = _distinct(r.environment.get("torch_num_threads") for r in group
                            if r.environment.get("device_type") == "cpu")
        if len(threads) > 1:
            warnings.append(ComparisonWarning("error", f"{label} 측정의 CPU 스레드 수가 모델마다 다릅니다 "
                                    f"({_join(threads)}). 같은 스레드 수로 다시 측정하세요."))
    scopes = {_hashable(r.section.get("scope_id")) for r in latency}
    if latency and (len(scopes) > 1 or None in scopes):
        warnings.append(ComparisonWarning("warning", "타이밍 범위(scope)가 같다고 확인되지 않았습니다. "
                                "schema v2 벤치마크로 다시 측정하면 같은 범위로 맞춰집니다."))
    formats = _distinct(r.section.get("memory_format") for r in latency)
    if len(formats) > 1:
        warnings.append(ComparisonWarning("error", f"메모리 형식이 모델마다 다릅니다 ({_join(formats)}). "
                                "CPU에서는 channels_last 여부만으로 1.5배 안팎 차이가 납니다."))
    fused = _distinct(r.section.get("conv_bn_fused") for r in latency)
    if len(fused) > 1:
        warnings.append(ComparisonWarning("warning", "Conv+BN 융합 여부가 모델마다 다릅니다."))
    unrecorded = [r.model_name for r in latency
                  if not r.legacy and r.section.get("memory_format") is None]
    if unrecorded:
        warnings.append(ComparisonWarning("warning", "메모리 형식이 기록되지 않았습니다 "
                                f"({', '.join(unrecorded)}). 다시 측정하세요."))
    for key, text in (("score_threshold", "점수 임계값이 다릅니다 ({}). 낮을수록 후처리/NMS 시간이 늘어납니다."),
                      ("max_detections", "최대 검출 수가 다릅니다 ({})."),
                      ("image_count", "이미지 수가 다릅니다 ({}장).")):
        values = _distinct(r.section.get(key) for r in latency)
        if len(values) > 1:
            warnings.append(ComparisonWarning("warning", "지연 시간 측정의 " + text.format(_join(values))))
    cold_scopes = _distinct(r.section.get("cold_start_scope") for r in latency)
    if len(cold_scopes) > 1:
        warnings.append(ComparisonWarning("info", "첫 추론 시간(cold start)은 모델마다 포함 범위가 다릅니다: "
                                + _join((COLD_START_LABELS.get(s, s) for s in cold_scopes), " / ")))
    for model in MODEL_ORDER:
        picked = [selected[(kind, model)] for kind in KINDS if (kind, model) in selected]
        modes = _distinct(r.section.get("inference_mode") for r in picked if r.kind != "stages")
        if len(modes) > 1:
            warnings.append(ComparisonWarning("error", f"{MODELS[model]}의 정확도와 지연 시간이 서로 다른 "
                                    f"추론 방식으로 측정됐습니다 ({_join(modes)})."))
        # Compare checkpoints by hash; fall back to the path only when no report has one (v1 files).
        # A path is weaker evidence (relative vs absolute, renamed labels), so it only warns, and
        # only for paths that name different files (as the checkpoint selector groups them).
        shas = _distinct(r.data.get("weights_sha256") for r in picked)
        paths = _distinct(r.data.get("weights") for r in picked)
        kinds = "·".join(KIND_LABELS[r.kind] for r in picked)
        if len(shas) > 1:
            warnings.append(ComparisonWarning("error", f"{MODELS[model]}의 {kinds} 결과가 서로 다른 "
                                    "가중치(체크포인트)로 측정됐습니다. 같은 가중치로 다시 측정하세요."))
        elif not shas and len(_checkpoint_keys(picked, model)[0]) > 1:
            warnings.append(ComparisonWarning("warning", f"{MODELS[model]}의 {kinds} 결과에 기록된 가중치 "
                                    f"경로가 다릅니다 ({_join(paths)}). 같은 체크포인트인지 확인하세요."))
    sizes = _distinct(r.data.get("image_size") for r in [*accuracy, *latency, *stages])
    if len(sizes) > 1:
        warnings.append(ComparisonWarning("warning", f"입력 크기가 다릅니다 ({_join(sizes)})."))
    few = [r.model_name if r.kind == "latency" else f"{r.model_name} {KIND_LABELS[r.kind]}"
           for r in [*latency, *stages]
           if (_as_int(r.section.get("n") or r.section.get("iterations")) or 0) < LOW_ITERATION_WARNING]
    if few:
        warnings.append(ComparisonWarning("warning", f"반복 횟수가 {LOW_ITERATION_WARNING}회 미만입니다 ({', '.join(few)}). "
                                "p95/p99는 통계적으로 불안정합니다."))
    empty = [r.model_name for r in latency if r.section.get("mean_detections") == 0]
    if empty:
        warnings.append(ComparisonWarning("warning", f"검출이 0개인 상태로 측정한 지연 시간입니다 ({', '.join(empty)}). "
                                "학습되지 않은 가중치일 수 있고, NMS·후처리 비용이 빠져 실제보다 조금 짧게 나옵니다."))
    off_target = [r for r in latency if not on_target_device(r)]
    if off_target:
        target = f"({latency_target_ms:g} ms)" if latency_target_ms is not None else ""
        where = "CPU" if all(is_cpu_run(r) for r in off_target) else "T4가 아닌 장치"
        warnings.append(ComparisonWarning("warning", f"{where}에서 측정한 지연 시간이 있습니다. 지연 시간 목표{target}는 "
                                "T4 GPU 기준이므로 목표 대비 차이를 그대로 해석하지 마세요."))
    # gpu_graph's summary names the CUDA graph at apply time; only the status says it replayed.
    for model in MODEL_ORDER:
        statuses = [_cuda_graph_status(selected[(kind, model)].inference_options)
                    for kind in ("latency", "stages") if (kind, model) in selected]
        unused = [status for status in statuses if status not in (None, "", "captured")]
        if unused:
            warnings.append(ComparisonWarning("warning", f"CUDA 그래프를 켰지만 쓰지 못한 결과가 있습니다 "
                                    f"({MODELS[model]}: {unused[0]}). 백본을 eager로 실행한 지연 시간이므로 "
                                    "CUDA 그래프의 효과로 보지 마세요."))
    if any(r.legacy for r in selected.values()):
        warnings.append(ComparisonWarning("info", "이전 형식(schema v1) 결과 파일이 섞여 있습니다. 환경 정보가 부족합니다."))
    return warnings


def _dataset(report: Report) -> dict[str, Any]:
    value = report.section.get("dataset")
    return value if isinstance(value, dict) else {}


# Setup an inference preset never changes: (level, label, show values, read). Precision, memory
# format and score/NMS settings are left out on purpose, presets may change them.
_SAME_SETUP_CHECKS = {
    "accuracy": (
        ("error", "평가 이미지 수", True, lambda r: _as_int(r.section.get("images_evaluated"))),
        ("error", "평가 이미지 목록", False, lambda r: r.section.get("image_ids_sha256")),
        ("error", "주석 파일", False, lambda r: _dataset(r).get("annotations_sha256")),
        ("error", "AP 계산기", True, lambda r: r.section.get("evaluator")),
        ("error", "가중치", False, lambda r: r.data.get("weights_sha256")),
        ("warning", "평가 배치 크기", True, lambda r: _as_int(r.data.get("batch_size"))),
    ),
    "latency": (
        ("error", "측정 장치", True, device_label),
        ("error", "CPU 스레드 수", True,
         lambda r: r.environment.get("torch_num_threads") if is_cpu_run(r) else None),
        ("error", "가중치", False, lambda r: r.data.get("weights_sha256")),
        ("warning", "측정 이미지 수", True, lambda r: _as_int(r.section.get("image_count"))),
        ("warning", "타이밍 범위", False, lambda r: r.section.get("scope_id")),
    ),
}


def improvement_warnings(before: dict[tuple[str, str], Report],
                         after: dict[tuple[str, str], Report]) -> list[ComparisonWarning]:
    """Before/after pairs (same model and kind) whose setup differs in something a preset never
    changes, so their difference is not the preset's effect alone."""
    warnings: list[ComparisonWarning] = []
    for model in MODEL_ORDER:
        for kind, checks in _SAME_SETUP_CHECKS.items():
            old_report, new_report = before.get((kind, model)), after.get((kind, model))
            if old_report is None or new_report is None:
                continue
            found: dict[str, list[str]] = {}
            for level, label, show, read in checks:
                old, new = _hashable(read(old_report)), _hashable(read(new_report))
                if old is not None and new is not None and old != new:
                    found.setdefault(level, []).append(f"{label} {old} → {new}" if show else label)
            for level, items in found.items():
                warnings.append(ComparisonWarning(level, f"{MODELS[model]} {KIND_LABELS[kind]}의 개선 전/후 "
                                        f"측정 조건이 다릅니다 ({', '.join(items)}). 변화량에 추론 설정 "
                                        "말고 다른 차이도 섞여 있습니다."))
    return warnings


# --------------------------------------------------------------------------- checkpoints


def model_label(report: Report) -> str | None:
    """Checkpoint name a report records (compare_improvements.py: 'Light R-CNN v3'); the bare
    model name it writes for Faster R-CNN names no checkpoint."""
    label = report.data.get("model_label")
    label = label.strip() if isinstance(label, str) else ""
    return label if label and label != report.model_name else None


def same_file(first: str, second: str) -> bool:
    """Whether two '/'-separated weights paths name one file; v1 files record it absolute,
    relative or with Windows separators."""
    a, b = first.split("/"), second.split("/")
    n = min(len(a), len(b))
    return a[-n:] == b[-n:]


def _checkpoint_keys(found: Iterable[Report], model: str
                     ) -> tuple[dict[str, list[Report]], dict[str, set[str]]]:
    """One model's reports per checkpoint key (weights hash, else weights path) and the paths
    each checkpoint is recorded under."""
    by_key: dict[str, list[Report]] = {}
    paths: dict[str, set[str]] = {}
    # Hashed reports first, so a path-only (v1) file can join the checkpoint it names.
    for report in sorted((r for r in found if r.model == model),
                         key=lambda r: not r.data.get("weights_sha256")):
        sha = report.data.get("weights_sha256")
        path = str(report.data.get("weights") or "").replace("\\", "/")
        if sha:
            key = str(sha)
        elif path:
            key = next((known for known, seen in paths.items()
                        if any(same_file(path, other) for other in seen)), path)
        else:
            continue
        by_key.setdefault(key, []).append(report)
        if path:
            paths.setdefault(key, set()).add(path)
    return by_key, paths


def _newest_label(members: list[Report]) -> str | None:
    newest = sorted(members, key=lambda r: r.created_at, reverse=True)
    return next(filter(None, map(model_label, newest)), None)


def checkpoint_groups(found: list[Report]) -> dict[str, dict[str, list[Report]]]:
    """Each model's reports per checkpoint, keyed by its label (model_label, else weights path).
    A weights hash identifies a checkpoint; a report without one joins the checkpoint whose path
    it records, or forms one per path. Reports recording neither are left out."""
    groups: dict[str, dict[str, list[Report]]] = {}
    for model in MODEL_ORDER:
        by_key, paths = _checkpoint_keys(found, model)
        labelled: dict[str, list[Report]] = {}
        for key, members in by_key.items():
            label = (_newest_label(members)
                     or (min(paths[key], key=len) if key in paths else f"sha256 {key[:12]}"))
            labelled[f"{label} ({key[:12]})" if label in labelled else label] = members
        groups[model] = labelled
    return groups


def _checkpoint_labels(by_key: dict[str, list[Report]], paths: dict[str, set[str]]
                       ) -> dict[str, str]:
    """Label per checkpoint key: the newest model_label, else the weights file stem under as many
    folders as it takes to tell the checkpoints apart (best, coco_4090_v2/best)."""
    labels: dict[str, str] = {}
    files: dict[str, tuple[tuple[str, ...], str]] = {}  # key -> (folders, file stem)
    for key, members in by_key.items():
        name = _newest_label(members)
        if name is None and key in paths:
            path = PurePosixPath(min(paths[key], key=lambda p: (len(p), p)))
            files[key] = (tuple(part for part in path.parent.parts if part != "/"), path.stem)
        labels[key] = name or f"sha256 {key[:12]}"
    depth = dict.fromkeys(files, 0)
    while True:
        labels.update({key: "/".join([*folders[len(folders) - depth[key]:], stem])
                       for key, (folders, stem) in files.items()})
        counts = Counter(labels.values())
        grow = [key for key, (folders, _) in files.items()
                if counts[labels[key]] > 1 and depth[key] < len(folders)]
        if not grow:
            break
        for key in grow:
            depth[key] += 1
    return {key: f"{label} ({key[:12]})" if counts[label] > 1 else label
            for key, label in labels.items()}


def checkpoints(reports: Iterable[Report]) -> list[tuple[str, str]]:
    """Light R-CNN checkpoints in ``reports`` as (key, label), newest first: the key is the
    weights hash (else the weights path), the label the model_label ('Light R-CNN v3'), else the
    weights file stem. Schema-v2 results rank ahead of old-format files, as in select_reports."""
    by_key, paths = _checkpoint_keys(list(reports), CHECKPOINT_MODEL)
    labels = _checkpoint_labels(by_key, paths)
    order = sorted(by_key, key=lambda key: max((not r.legacy, r.created_at) for r in by_key[key]),
                   reverse=True)
    return [(key, labels[key]) for key in order]


def checkpoint_paths(reports: Iterable[Report]) -> dict[str, set[str]]:
    """The weights paths ('/'-separated) each Light R-CNN checkpoint key of ``checkpoints`` is
    recorded under; a path-only checkpoint's key is one of them."""
    return _checkpoint_keys(list(reports), CHECKPOINT_MODEL)[1]


def checkpoint_reports(reports: Iterable[Report], checkpoint: str) -> list[Report]:
    """``reports`` with Light R-CNN narrowed to one checkpoint (a key or label from
    ``checkpoints``); the other models are left as they are."""
    reports = list(reports)
    by_key, paths = _checkpoint_keys(reports, CHECKPOINT_MODEL)
    if checkpoint not in by_key:
        labels = _checkpoint_labels(by_key, paths)
        checkpoint = next((key for key, label in labels.items() if label == checkpoint), checkpoint)
    members = {id(report) for report in by_key.get(checkpoint, [])}
    return [r for r in reports if r.model != CHECKPOINT_MODEL or id(r) in members]


def own_folder(members: list[Report], others: list[Report]) -> Path | None:
    """The folder holding most of ``members``' files and none of ``others``' (uploads have none)."""
    def files(group: list[Report]) -> list[Path]:
        return [Path(r.source) for r in group if not r.source.startswith("업로드: ")]

    taken = files(others)
    for folder, _ in Counter(path.parent for path in files(members)).most_common():
        if not any(path.is_relative_to(folder) for path in taken):
            return folder
    return None


def checkpoint_warnings(found: list[Report], selection: dict) -> list[ComparisonWarning]:
    """One note per model whose results come from several checkpoints (Colab results hold Light
    R-CNN v2 and v3 under one model key): only one report per model and kind is shown."""
    notes = []
    for model, groups in checkpoint_groups(found).items():
        if len(groups) < 2:
            continue
        label_of = {id(r): label for label, members in groups.items() for r in members}
        used: dict[str, list[str]] = {}
        for kind in KINDS:
            report = selection.get((kind, model))
            if report is not None and id(report) in label_of:
                used.setdefault(label_of[id(report)], []).append(KIND_LABELS[kind])
        in_use = next(iter(used)) if len(used) == 1 else ", ".join(
            f"{label}({'·'.join(kinds)})" for label, kinds in used.items())
        text = (f"결과 폴더에 {MODELS[model]} 체크포인트가 {len(groups)}개({', '.join(groups)}) 섞여 "
                + (f"있어 카드와 그래프는 그중 {in_use}만 씁니다." if used
                   else "있습니다. 카드와 그래프는 모델마다 체크포인트 하나의 결과만 씁니다."))
        hints = []
        for label, members in groups.items():
            folder = own_folder(members, [r for other, rest in groups.items() if other != label
                                          for r in rest])
            if folder is not None:
                hints.append(f"{label} → {folder}")
        text += (" 체크포인트별로 보려면 사이드바의 '결과 폴더'를 바꾸세요: " + " · ".join(hints) if hints
                 else " 체크포인트별로 보려면 한 체크포인트의 결과만 든 폴더를 사이드바의 '결과 폴더'에 "
                      "넣으세요.")
        notes.append(ComparisonWarning("warning", text))
    return notes


def load_targets(config_path: str | Path = "configs/coco.yaml") -> dict[str, float]:
    """Read ``comparison.targets`` from the YAML config; fall back to the README targets."""
    defaults = {"latency_ms": 15.0, "latency_percentile": "p50", "ap_metric": "AP_50_95", "ap": 0.60}
    try:
        import yaml

        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    except (ImportError, OSError, ValueError):
        return defaults
    targets = (config.get("comparison") or {}).get("targets") or {}
    return {**defaults, **{key: value for key, value in targets.items() if value is not None}}


def describe_environment(env: dict[str, Any]) -> str:
    packages = env.get("packages") or {}
    parts = [env.get("device_name") or env.get("device") or "장치 미기록"]
    if packages.get("torch"):
        parts.append(f"torch {packages['torch']}")
    if env.get("cuda_runtime") and env.get("device_type") != "cpu":
        parts.append(f"CUDA {env['cuda_runtime']}")
    if env.get("python"):
        parts.append(f"Python {env['python']}")
    return " · ".join(str(part) for part in parts)


if __name__ == "__main__":  # quick inspection: python -m light_rcnn.reporting [outputs]
    found, problems = load_reports(sys.argv[1] if len(sys.argv) > 1 else "outputs")
    names = variant_labels(found)
    for name, label in names.items():
        if len(names) > 1:
            print(f"== {name} ({label})")
        picked = select_reports(found, variant=name)
        for (kind, model), report in sorted(picked.items()):
            print(f"{kind:9s} {model:12s} {report.source}")
        for warning in comparability_warnings(picked):
            print(f"[{warning.level}] {warning.message}")
    for problem in problems:
        print(f"[unreadable] {problem}")
