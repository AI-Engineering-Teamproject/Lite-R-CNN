import csv
import importlib.util
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path
from urllib.error import URLError

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_mobilenet_v3_large_fpn,
)

from light_rcnn.inference_opts import PRESET_LABELS
from light_rcnn.model.detector import light_rcnn_from_config
from light_rcnn.reporting import (
    SCOPE_HOST_TO_DETECTIONS,
    comparability_warnings,
    display_path,
    file_sha256,
    improvement_warnings,
    load_reports,
    select_reports,
    variants,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/compare_improvements.py"
SIZES = [(120, 160), (160, 120), (100, 160), (128, 128)]  # (height, width), mixed orientations
IMAGE_IDS = [11, 3, 7, 5]
PRESETS = ["baseline", "deploy", "fast"]
OLD_CSV_COLUMNS = (
    "model", "model_label", "preset", "preset_label", "status", "note", "ap_pct", "ap50_pct",
    "ap75_pct", "ap_small_pct", "ap_medium_pct", "ap_large_pct", "ar100_pct", "p50_ms", "p95_ms",
    "p99_ms", "mean_ms", "fps", "cold_start_ms", "delta_ap_pp", "delta_p50_pct",
    "delta_p50_vs_deploy_pct", "score_threshold", "max_detections", "detections_per_image",
    "latency_mean_detections", "images_evaluated", "latency_images", "warmup", "iterations",
    "parameters_m", "deployed_parameters_m", "gflops", "precision", "memory_format",
    "conv_bn_fused", "device", "inference_summary", "accuracy_report", "latency_report",
)
CI_COLUMNS = ("delta_p50_ci_low", "delta_p50_ci_high", "delta_p50_vs_deploy_ci_low",
              "delta_p50_vs_deploy_ci_high")
PAIRED_KEYS = {"variant", "delta_p50_pct", "ci95_pct", "resamples", "method"}

# Drops the editable-install entry so the script has to find src/ on its own (Colab).
RUNNER = """
import runpy, sys
from pathlib import Path
src = Path(sys.argv[1]).resolve()
sys.path[:] = [p for p in sys.path if not p or Path(p).resolve() != src]
sys.argv = [sys.argv[2], "--help"]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


@pytest.fixture(scope="module")
def coco(tmp_path_factory):
    """Four noise photos with two 'person' boxes each, 128 px config (seed 42 from coco.yaml)."""
    root = tmp_path_factory.mktemp("coco")
    images = root / "images"
    images.mkdir()
    rng = np.random.default_rng(0)
    records, annotations = [], []
    for image_id, (height, width) in zip(IMAGE_IDS, SIZES):
        name = f"{image_id:012d}.jpg"
        Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8)).save(images / name)
        records.append({"id": image_id, "file_name": name, "width": width, "height": height})
        for index, box in enumerate(([8, 10, 40, 50], [50, 20, 60, 70])):
            annotations.append({"id": image_id * 10 + index, "image_id": image_id,
                                "category_id": 1, "bbox": box, "area": box[2] * box[3],
                                "iscrowd": 0})
    path = root / "instances.json"
    path.write_text(json.dumps({"images": records, "annotations": annotations,
                                "categories": [{"id": 1, "name": "person"}]}))
    config = yaml.safe_load((ROOT / "configs/coco.yaml").read_text(encoding="utf-8"))
    config["image_size"] = 128
    config_path = root / "coco_128.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return {"images": images, "annotations": path, "config": config_path}


def tiny_checkpoint(path: Path, config_file: str, **narrow) -> Path:
    """A narrow Light R-CNN whose RoI head calls every box 'person' (~0.8), so NMS has work."""
    config = yaml.safe_load((ROOT / config_file).read_text(encoding="utf-8"))
    config["model"].update(narrow)
    torch.manual_seed(0)
    model = light_rcnn_from_config(config)
    with torch.no_grad():
        model.roi_heads.box_predictor.cls_score.bias[1] = 6.0
    torch.save({"model": model.state_dict(), "config": config}, path)
    return path


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    root = tmp_path_factory.mktemp("weights")
    return {
        "v2": tiny_checkpoint(root / "v2.pth", "configs/coco.yaml", bifpn_channels=32,
                              roi_channels=16, roi_head_dim=64),
        "v3": tiny_checkpoint(root / "v3.pth", "configs/coco_v3.yaml", fpn_channels=32,
                              roi_channels=16, roi_head_dim=64),
    }


@pytest.fixture(scope="module")
def mobilenet_checkpoint(tmp_path_factory):
    """torchvision's MobileNetV3-Large-FPN Faster R-CNN with random weights (weights=None,
    weights_backbone=None: nothing downloaded) saved as a local checkpoint for the real loader;
    its box predictor calls every box 'person' (~0.99), so NMS has work."""
    torch.manual_seed(0)
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None, num_classes=91)
    with torch.no_grad():
        model.roi_heads.box_predictor.cls_score.bias[1] = 6.0
    path = tmp_path_factory.mktemp("mobilenet") / "mobilenet.pth"
    torch.save({"model": model.state_dict()}, path)
    return path


def small_baseline(image_size: int) -> torch.nn.Module:
    """A torchvision Faster R-CNN small enough to stand in for a baseline in a unit test."""
    torch.manual_seed(0)
    return fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None,
                                                 num_classes=91, min_size=image_size,
                                                 max_size=image_size)


@pytest.fixture
def script(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))  # the script adds src/ and scripts/
    spec = importlib.util.spec_from_file_location("compare_improvements", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)  # dataclasses look the module up
    spec.loader.exec_module(module)
    exported = []
    # kaleido may or may not find a Chrome; record the PNG requests instead.
    monkeypatch.setattr(module, "export_images",
                        lambda figures, *args: exported.extend(figures) or [])
    module.exported = exported
    return module


@pytest.fixture
def run(script, coco, weights, tmp_path):
    def main(*argv):
        output = tmp_path / "out"
        code = script.main([
            "--config", str(coco["config"]), "--device", "cpu", "--images", str(coco["images"]),
            "--annotations", str(coco["annotations"]), "--v2-weights", str(weights["v2"]),
            "--v3-weights", str(weights["v3"]), "--output-dir", str(output),
            "--latency-images", "2", "--warmup", "1", "--iterations", "3", *map(str, argv)])
        return code, output

    return main


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def assert_dashboard_ready(folder: Path, presets: list[str], models: set[str]) -> None:
    reports, errors = load_reports(folder)
    assert not errors
    assert variants(reports) == presets
    baseline = select_reports(reports, variant="baseline")
    for preset in presets:
        selected = select_reports(reports, variant=preset)
        assert set(selected) == {(kind, model) for kind in ("accuracy", "latency")
                                 for model in models}
        assert {r.variant_label for r in selected.values()} == {PRESET_LABELS[preset]}
        assert improvement_warnings(baseline, selected) == []
        assert [w.message for w in comparability_warnings(selected) if w.level == "error"] == []


def test_cpu_run_writes_reports_summary_and_charts_the_dashboard_reads(run, script, weights,
                                                                       capsys):
    code, output = run("--models", "v2", "v3", "--presets", "baseline,fast", "accurate,gpu_graph",
                       "--limit", 3)
    assert code == 0
    out = capsys.readouterr().out
    assert "[건너뜀] GPU CUDA 그래프 (CUDA 전용) (gpu_graph)" in out
    assert "ΔAP (%p)" in out and "검출 점수 하한: 각 추론 설정의 값" in out
    assert "개선 전(baseline)은 Conv+BN 융합 없이 contiguous 메모리로 잽니다" in out
    presets = ["baseline", "fast", "accurate"]
    hashes, latency_ids = set(), set()
    for version in ("v2", "v3"):
        folder = output / "reports" / version
        gflops = {}
        assert sorted(path.name for path in folder.iterdir()) == sorted(
            f"light_rcnn_{kind}_{preset}.json" for kind in ("accuracy", "latency")
            for preset in presets)
        for preset in presets:
            accuracy = read_json(folder / f"light_rcnn_accuracy_{preset}.json")
            latency = read_json(folder / f"light_rcnn_latency_{preset}.json")
            for report in (accuracy, latency):
                assert (report["schema_version"], report["model"]) == (2, "light_rcnn")
                assert (report["variant"], report["variant_label"]) == (preset,
                                                                        PRESET_LABELS[preset])
                assert report["inference_options"]["preset"] == preset
                assert (report["model_tag"], report["batch_size"]) == (version, 1)
                assert report["weights_sha256"] == file_sha256(weights[version])
            section = accuracy["accuracy"]
            assert (section["images_evaluated"], section["images_total"]) == (3, 4)
            assert section["is_partial"] and section["subset_seed"] == 42
            # The team protocol: the preset's own floor (box default 0.05), recorded.
            assert (section["score_threshold"], section["score_threshold_source"]) == (0.05,
                                                                                       "preset")
            assert section["max_detections"] == 100 and section["detections"] > 0
            hashes.add(section["image_ids_sha256"])
            timing = latency["latency"]
            assert timing["scope_id"] == SCOPE_HOST_TO_DETECTIONS
            # Warmup is raised to the number of timing images so every shape is warm; interleaved
            # timing rounds 3 iterations up to whole rounds (2 x 2 images), then to whole rotation
            # cycles of the 3 timed presets (3 rounds).
            assert (timing["image_count"], timing["warmup"], timing["n"]) == (2, 2, 6)
            assert (timing["design"], timing["rounds"], timing["block_size"]) == (
                "interleaved_blocks", 3, 2)
            assert len(timing["samples_ms"]) == 6 and timing["cold_start_ms"] > 0
            assert set(timing["image_ids"]) <= set(section["image_ids"])
            latency_ids.add(tuple(timing["image_ids"]))
            gflops[preset] = latency["model_stats"]["gflops"]
            assert latency["model_stats"]["flops_input"] == [128, 128]
            assert ("640x640 fixed canvas" in latency["model_stats"]["flops_method"]) is (
                preset == "accurate")
        accurate = read_json(folder / "light_rcnn_latency_accurate.json")["inference_options"]
        assert accurate["applied"]["nms"] == "soft"
        assert accurate["applied"]["fixed_canvas"] == [640, 640]
        assert gflops["accurate"] > 5 * gflops["fast"] > 0  # 128 px images padded to 640 x 640
        assert_dashboard_ready(folder, presets, {"light_rcnn"})
    assert len(hashes) == 1 and len(latency_ids) == 1  # one subset and one timing sample for all

    rows = read_csv(output / "summary.csv")
    assert [(row["model"], row["preset"], row["status"]) for row in rows] == [
        (version, preset, "skipped" if preset == "gpu_graph" else "ok")
        for version in ("v2", "v3") for preset in [*presets, "gpu_graph"]]
    for row in rows:
        measured = row["status"] == "ok" and row["preset"] != "baseline"
        assert bool(row["delta_ap_pp"]) == bool(row["delta_p50_pct"]) == measured
        assert not row["delta_p50_vs_deploy_pct"]  # no deploy run to compare with
        assert bool(row["note"]) is (row["status"] == "skipped")  # no CUDA graph to flag on a CPU
    summary = (output / "summary.md").read_text(encoding="utf-8")
    assert "| 모델 | 추론 설정 | AP |" in summary and "| 기준 | 기준 | – |" in summary
    assert "| Δp50 (%) [95% CI] | Δp50 vs deploy (%) [95% CI] |" in summary
    assert "건너뛰거나 실패한 측정" in summary and "`gpu_graph`" in summary
    assert "reports/v2" in summary and "같은 'Light R-CNN'으로 표시" in summary
    # The combined folder opens too (checkpoint selector), and both shells get an example.
    combined = display_path(output / "reports")
    assert (f"- `{combined}`: 전체 (Light R-CNN v2, Light R-CNN v3 — 사이드바 'Light R-CNN "
            "체크포인트'에서 버전을 고름)") in summary
    assert f"- bash: `LIGHT_RCNN_RESULTS_DIR={combined} streamlit run dashboard/app.py`" in summary
    assert (f"- PowerShell: `$env:LIGHT_RCNN_RESULTS_DIR=\"{combined}\"; streamlit run "
            "dashboard/app.py`") in summary
    assert f"대시보드 결과 폴더 (전체, 'Light R-CNN 체크포인트'에서 버전 선택): {combined}" in out
    for version in ("v2", "v3"):
        page = (output / "figures" / f"improvement_{version}.html").read_text(encoding="utf-8")
        assert f"개선 전/후 비교 — Light R-CNN {version}" in page
    assert script.exported == [f"improvement_{version}_{preset}" for version in ("v2", "v3")
                               for preset in ("fast", "accurate")]


def test_baselines_without_downloadable_weights_are_skipped(run, script, monkeypatch, capsys):
    real, attempts = script.load_comparison_model, []

    def offline(kind, config, weights=None, image_size=None):
        if kind in ("faster_rcnn", "mobilenet_faster_rcnn") and weights is None:
            attempts.append(kind)
            raise URLError("Tunnel connection failed: 403 Forbidden")
        return real(kind, config, weights, image_size)

    monkeypatch.setattr(script, "load_comparison_model", offline)
    code, output = run("--models", "mobilenet", "faster_rcnn", "v2", "--presets", "fast",
                       "--limit", 2, "--score-thresh", 0.3)
    # Each baseline tried once, not once per preset, in the fixed order.
    assert code == 0 and attempts == ["faster_rcnn", "mobilenet_faster_rcnn"]
    out = capsys.readouterr().out
    assert "torchvision COCO 가중치를 내려받지 못했습니다" in out and "--faster-weights" in out
    assert ("[건너뜀] MobileNetV3 Faster R-CNN: torchvision COCO 가중치를 내려받지 못했습니다" in out
            and "--mobilenet-weights <체크포인트>로 로컬 가중치를 지정하면 함께 측정합니다" in out)
    folder = output / "reports" / "v2"
    assert sorted(path.name for path in folder.iterdir()) == sorted(
        f"light_rcnn_{kind}_{preset}.json" for kind in ("accuracy", "latency")
        for preset in ("baseline", "fast"))
    report = read_json(folder / "light_rcnn_accuracy_fast.json")
    # An explicit floor applies to every preset and says so in the label and the record.
    assert (report["accuracy"]["score_threshold"],
            report["accuracy"]["score_threshold_source"]) == (0.3, "--score-thresh")
    assert report["inference_options"]["options"]["box_score_thresh"] == 0.3
    assert report["variant"] == "fast" and report["variant_label"] == "빠른 설정 · 점수 하한 0.3"
    rows = read_csv(output / "summary.csv")
    assert [(row["model"], row["status"]) for row in rows] == [
        ("faster_rcnn", "skipped"), ("faster_rcnn", "skipped"), ("mobilenet", "skipped"),
        ("mobilenet", "skipped"), ("v2", "ok"), ("v2", "ok")]
    assert "--mobilenet-weights" in rows[2]["note"] and "--faster-weights" in rows[0]["note"]
    summary = (output / "summary.md").read_text("utf-8")
    assert "- MobileNetV3 Faster R-CNN · 개선 전 (기본 설정) · 점수 하한 0.3 (`baseline`): 건너뜀 — " \
           "torchvision COCO 가중치를" in summary
    # The folder holds no baseline report, so its label names Light R-CNN v2 alone.
    assert f"- `{display_path(folder)}`: Light R-CNN v2\n" in summary
    assert f"대시보드 결과 폴더 (Light R-CNN v2): {display_path(folder)}" in out


def test_nothing_measured_exits_with_an_error(run, script, monkeypatch, capsys):
    def offline(kind, *args, **kwargs):
        raise URLError("Tunnel connection failed: 403 Forbidden")

    monkeypatch.setattr(script, "load_comparison_model", offline)
    code, output = run("--models", "faster_rcnn", "--presets", "fast")
    assert code == 1 and not (output / "reports").exists()
    assert "측정한 결과가 없습니다" in capsys.readouterr().err
    assert "## 대시보드" not in (output / "summary.md").read_text(encoding="utf-8")


def test_a_failing_run_is_reported_and_the_others_still_run(run, script, monkeypatch, capsys):
    real = script.apply_inference_options

    def flaky(model, options, device):
        if options.rpn_pre_nms_top_n == 500:  # the 'fast' preset
            raise RuntimeError("boom")
        return real(model, options, device)

    monkeypatch.setattr(script, "apply_inference_options", flaky)
    code, output = run("--models", "v2", "--presets", "fast,deploy", "--limit", 2)
    assert code == 1
    assert "RuntimeError: boom" in capsys.readouterr().err  # the traceback
    rows = read_csv(output / "summary.csv")
    assert [(row["preset"], row["status"]) for row in rows] == [
        ("baseline", "ok"), ("fast", "failed"), ("deploy", "ok")]
    assert rows[1]["note"] == "RuntimeError: boom" and rows[2]["delta_p50_pct"]
    assert rows[2]["delta_p50_ci_low"] and rows[2]["delta_p50_ci_high"]
    assert not list((output / "reports/v2").glob("*_fast.json"))
    paired = read_json(output / "reports/v2/light_rcnn_latency_deploy.json")["latency"]["paired"]
    assert paired["vs_baseline"]["variant"] == "baseline" and paired["vs_deploy"] is None
    assert "실패 — RuntimeError: boom" in (output / "summary.md").read_text(encoding="utf-8")


# Calls before the snapshot: cold start 1 + COCO 2 + warmup 2, then 3 timed calls (sequential) or
# 3 rounds (a whole rotation of the 3 presets) of 1 untimed + 2 timed calls (interleaved).
@pytest.mark.parametrize("flags, eager", [((), 14), (("--sequential-latency",), 8)])
def test_an_eager_cuda_graph_run_is_flagged_and_fast_is_also_compared_with_deploy(
        run, script, monkeypatch, capsys, flags, eager):
    real, statuses = script.apply_inference_options, []

    def disabled_graph(model, options, device):
        """v2's 'fast' stands in for gpu_graph (skipped on a CPU) after a failed self-check."""
        record = real(model, options, device)
        if options.rpn_pre_nms_top_n == 500 and not statuses:
            status = {"status": "disabled (graph replay differs from eager)", "replays": 0,
                      "eager_calls": 0}
            forward = model.backbone.forward

            def eager(*args, **kwargs):  # what inference_opts' _GraphedForward does once disabled
                status["eager_calls"] += 1
                return forward(*args, **kwargs)

            model.backbone.forward = eager
            record["applied"]["cuda_graph"] = True
            record["summary"] += " · CUDA 그래프"  # the apply-time summary
            record["cuda_graph_status"] = status
            statuses.append(status)
        return record

    monkeypatch.setattr(script, "apply_inference_options", disabled_graph)
    code, output = run("--models", "v2", "v3", "--presets", "deploy,fast", "--limit", 2, *flags)
    assert code == 0  # still a valid fused measurement: kept, with a caution
    assert statuses[0]["eager_calls"] == eager + 1  # count_flops' eager call came after it
    note = (f"CUDA 그래프 미사용: disabled (graph replay differs from eager) (재생 0회, eager "
            f"{eager}회) — 지연 시간은 그래프 없이 잰 값")
    out = capsys.readouterr().out
    assert f"  [주의] {note}" in out and f"- 주의: Light R-CNN v2 · 빠른 설정 — {note}" in out
    rows = {(row["model"], row["preset"]): row for row in read_csv(output / "summary.csv")}
    flagged = rows["v2", "fast"]
    assert flagged["status"] == "ok" and flagged["note"] == note
    assert flagged["inference_summary"].endswith(f" · CUDA 그래프 ({note})")
    assert [key for key, row in rows.items() if row["note"]] == [("v2", "fast")]
    report = read_json(output / "reports/v2/light_rcnn_latency_fast.json")["inference_options"]
    assert report["summary"] == flagged["inference_summary"]
    assert report["cuda_graph_status"] == {"status": "disabled (graph replay differs from eager)",
                                           "replays": 0, "eager_calls": eager}  # measured calls

    for (model, preset), row in rows.items():
        assert bool(row["delta_p50_vs_deploy_pct"]) is (preset == "fast")
        if preset == "fast":
            before, after = (float(rows[model, name]["p50_ms"]) for name in ("deploy", "fast"))
            assert float(row["delta_p50_vs_deploy_pct"]) == pytest.approx(
                (after - before) / before * 100, abs=0.01)
    summary = (output / "summary.md").read_text(encoding="utf-8")
    cells = {tuple(line.strip("| ").split(" | ")[:2]): line.strip("| ").split(" | ")[-3:]
             for line in summary.splitlines() if line.startswith("| Light R-CNN")}
    for version in ("v2", "v3"):
        model = f"Light R-CNN {version}"
        assert cells[model, f"{PRESET_LABELS['baseline']} (baseline)"] == ["기준", "기준", "–"]
        assert cells[model, f"{PRESET_LABELS['deploy']} (deploy)"][-1] == "기준"
        assert cells[model, f"{PRESET_LABELS['fast']} (fast)"][-1][0] in "+-"
    # Only v2's fast summary carries the note, so that preset lists its summary per model.
    assert "(`deploy`): " in summary and "(`fast`):\n  - Light R-CNN v2: " in summary
    assert f"({note})\n  - Light R-CNN v3: " in summary
    assert f"- [주의] Light R-CNN v2 · 빠른 설정 (`fast`): {note}" in summary
    pages = {version: (output / f"figures/improvement_{version}.html").read_text(encoding="utf-8")
             for version in ("v2", "v3")}
    assert f"Light R-CNN v2 · 빠른 설정: {note}" in pages["v2"] and note not in pages["v3"]


def logged_calls(script, monkeypatch) -> list[int]:
    """Number every model benchmark.timed_call times (first seen = 0) and log each call."""
    real, calls, models = script.timed_call, [], []

    def logged(model, image, device):
        if not any(model is known for known in models):
            models.append(model)  # kept alive, so a later model never reuses its id
        calls.append(next(number for number, known in enumerate(models) if known is model))
        return real(model, image, device)

    monkeypatch.setattr(script, "timed_call", logged)
    return calls


def test_interleaved_run_times_presets_in_rotated_blocks_and_writes_paired_cis(run, script,
                                                                             monkeypatch, capsys):
    calls = logged_calls(script, monkeypatch)
    code, output = run("--models", "v2", "--presets", "baseline,deploy,fast", "--limit", 4,
                       "--latency-images", 2, "--warmup", 2, "--iterations", 6)
    assert code == 0
    out = capsys.readouterr().out
    # Models 0-2 = baseline, deploy, fast: cold starts (each before its COCO pass), round-robin
    # warm-up (2 calls each), then 3 rounds rotated by one; a block = 1 untimed + 2 timed calls.
    rounds = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    assert calls == [0, 1, 2, *[0, 1, 2] * 2,
                     *(model for order in rounds for model in order for _ in range(3))]
    folder = output / "reports/v2"
    latency = {preset: read_json(folder / f"light_rcnn_latency_{preset}.json")
               for preset in PRESETS}
    accuracy = [read_json(folder / f"light_rcnn_accuracy_{preset}.json") for preset in PRESETS]
    [run_id] = {report["comparison_run"] for report in [*latency.values(), *accuracy]}
    assert run_id.startswith("out-")
    for report in latency.values():
        timing = report["latency"]
        assert (timing["design"], timing["rounds"], timing["block_size"]) == (
            "interleaved_blocks", 3, 2)
        assert (timing["iterations"], timing["n"], timing["warmup"]) == (6, 6, 2)
        assert timing["round_medians_ms"] == pytest.approx(
            [statistics.median(timing["samples_ms"][2 * r:2 * r + 2]) for r in range(3)], abs=1e-3)
    p50 = {preset: report["latency"]["p50_ms"] for preset, report in latency.items()}
    assert "paired" in latency["baseline"]["latency"]
    assert latency["baseline"]["latency"]["paired"] is None
    paired = {preset: latency[preset]["latency"]["paired"] for preset in ("deploy", "fast")}
    assert paired["deploy"]["vs_deploy"] is None
    for preset, reference in (("deploy", "baseline"), ("fast", "baseline"), ("fast", "deploy")):
        entry = paired[preset][f"vs_{reference}"]
        assert set(entry) == PAIRED_KEYS
        assert (entry["variant"], entry["resamples"], entry["method"]) == (
            reference, 2000, "paired bootstrap over rounds")
        assert entry["delta_p50_pct"] == pytest.approx(
            (p50[preset] - p50[reference]) / p50[reference] * 100, abs=1e-3)
        assert entry["ci95_pct"][0] <= entry["ci95_pct"][1]

    rows = read_csv(output / "summary.csv")
    assert tuple(rows[0]) == (*OLD_CSV_COLUMNS, *CI_COLUMNS)
    rows = {row["preset"]: row for row in rows}
    assert not any(rows["baseline"][key] for key in CI_COLUMNS)
    assert [bool(rows["deploy"][key]) for key in CI_COLUMNS] == [True, True, False, False]
    assert [float(rows["fast"][key]) for key in CI_COLUMNS] == pytest.approx(
        [*paired["fast"]["vs_baseline"]["ci95_pct"], *paired["fast"]["vs_deploy"]["ci95_pct"]],
        abs=1e-4)

    def cell(preset, reference):  # what the table shows: Δp50 [CI low, CI high]
        low, high = paired[preset][f"vs_{reference}"]["ci95_pct"]
        delta = (p50[preset] - p50[reference]) / p50[reference] * 100
        return f"{delta:+.1f} [{low:+.1f}, {high:+.1f}]"

    summary = (output / "summary.md").read_text(encoding="utf-8")
    cells = {line.strip("| ").split(" | ")[1]: line.strip("| ").split(" | ")[-2:]
             for line in summary.splitlines() if line.startswith("| Light R-CNN v2 |")}
    assert cells[f"{PRESET_LABELS['deploy']} (deploy)"] == [cell("deploy", "baseline"), "기준"]
    assert cells[f"{PRESET_LABELS['fast']} (fast)"] == [cell("fast", "baseline"),
                                                         cell("fast", "deploy")]
    assert re.fullmatch(r"[+-]\d+\.\d \[[+-]\d+\.\d, [+-]\d+\.\d\]", cell("fast", "deploy"))
    for text in ("교차 측정", "3라운드", "블록 크기 2장", "순서 회전", "쌍 부트스트랩(2000회",
                 "0을 포함하면 차이가 오차 범위", "라운드가 5개 미만", f"비교 실행 ID: {run_id}",
                 "라운드 수는 설정 수의 배수라"):
        assert text in summary
    assert "[지연 시간] Light R-CNN v2: 설정 3개 교차 측정" in out and "Δp50 개선 전 대비" in out
    assert "반복을 라운드 단위로" not in out  # 6 iterations are exactly 3 rounds of 2 images


def test_sequential_latency_keeps_one_time_window_per_preset(run, script, monkeypatch, capsys):
    calls = logged_calls(script, monkeypatch)
    code, output = run("--models", "v2", "--presets", "baseline,deploy,fast", "--limit", 4,
                       "--latency-images", 2, "--warmup", 2, "--iterations", 3,
                       "--sequential-latency")
    assert code == 0
    out = capsys.readouterr().out
    # Each preset: cold start, warm-up 2, timed 3, all before the next preset is loaded.
    assert calls == [model for model in range(3) for _ in range(1 + 2 + 3)]
    for preset in PRESETS:
        report = read_json(output / f"reports/v2/light_rcnn_latency_{preset}.json")
        timing = report["latency"]
        assert timing["design"] == "sequential" and report["comparison_run"].startswith("out-")
        assert not {"rounds", "block_size", "round_medians_ms", "paired"} & set(timing)
        assert (timing["iterations"], timing["n"], timing["warmup"]) == (3, 3, 2)  # no rounding
    rows = {row["preset"]: row for row in read_csv(output / "summary.csv")}
    assert rows["fast"]["delta_p50_pct"] and rows["fast"]["delta_p50_vs_deploy_pct"]
    assert not any(row[key] for row in rows.values() for key in CI_COLUMNS)
    summary = (output / "summary.md").read_text(encoding="utf-8")
    assert "지연 시간 측정 방식: 순차 측정 (--sequential-latency)" in summary
    assert "교차 측정 —" not in summary and "쌍 부트스트랩(" not in summary
    fast = next(line for line in summary.splitlines() if "(fast) |" in line)
    assert all(re.fullmatch(r"[+-]\d+\.\d", cell) for cell in fast.strip("| ").split(" | ")[-2:])
    assert "[지연 시간]" not in out and "  지연 시간: p50 " in out


def test_a_preset_failing_while_timed_is_reported_and_the_others_keep_their_cis(
        run, script, monkeypatch, capsys):
    real, fast_calls = script.timed_call, []

    def flaky(model, image, device):
        if model.rpn._pre_nms_top_n["testing"] == 500:  # 'fast': the cold start still works
            fast_calls.append(image)
            if len(fast_calls) > 1:
                raise RuntimeError("timing boom")
        return real(model, image, device)

    monkeypatch.setattr(script, "timed_call", flaky)
    code, output = run("--models", "v2", "--presets", "deploy,fast", "--limit", 2)
    assert code == 1
    assert "RuntimeError: timing boom" in capsys.readouterr().err  # the traceback
    rows = read_csv(output / "summary.csv")
    assert [(row["preset"], row["status"], row["note"]) for row in rows] == [
        ("baseline", "ok", ""), ("deploy", "ok", ""), ("fast", "failed", "RuntimeError: timing boom")]
    assert rows[1]["delta_p50_ci_low"]
    # Its COCO pass came before the timing: the accuracy report and the AP columns stay.
    assert [path.name for path in (output / "reports/v2").glob("*_fast.json")] == [
        "light_rcnn_accuracy_fast.json"]
    assert rows[2]["ap_pct"] and rows[2]["delta_ap_pp"] and not rows[2]["p50_ms"]
    timing = read_json(output / "reports/v2/light_rcnn_latency_deploy.json")["latency"]
    assert timing["rounds"] == 3 and len(timing["samples_ms"]) == 6  # no round lost
    assert timing["paired"]["vs_baseline"]["variant"] == "baseline"
    assert ("실패 — RuntimeError: timing boom (정확도는 측정해 보고서에 저장: AP "
            in (output / "summary.md").read_text("utf-8"))


@pytest.mark.parametrize("flags", [(), ("--sequential-latency",)])
def test_a_baseline_failing_while_timed_still_anchors_every_delta_ap(run, script, monkeypatch,
                                                                     capsys, flags):
    real, models, baseline_calls = script.timed_call, [], []

    def flaky(model, image, device):
        if not models:
            models.append(model)  # the baseline is loaded first
        if model is models[0]:
            baseline_calls.append(image)
            if len(baseline_calls) > 1:  # its cold start (before the COCO pass) still works
                raise RuntimeError("baseline timing boom")
        return real(model, image, device)

    monkeypatch.setattr(script, "timed_call", flaky)
    code, output = run("--models", "v2", "--presets", "deploy,fast", "--limit", 2, *flags)
    assert code == 1
    assert "[실패] 개선 전 (기본 설정) (baseline): RuntimeError: baseline timing boom" in (
        capsys.readouterr().out)
    rows = {row["preset"]: row for row in read_csv(output / "summary.csv")}
    assert [row["status"] for row in rows.values()] == ["failed", "ok", "ok"]
    assert rows["baseline"]["ap_pct"] and not rows["baseline"]["p50_ms"]
    for preset in ("deploy", "fast"):  # the ΔAP survives, the Δp50 against baseline cannot
        before, after = (float(rows[name]["ap_pct"]) for name in ("baseline", preset))
        assert float(rows[preset]["delta_ap_pp"]) == pytest.approx(after - before, abs=1e-3)
        assert not rows[preset]["delta_p50_pct"]
    assert rows["fast"]["delta_p50_vs_deploy_pct"]
    folder = output / "reports/v2"
    assert sorted(path.name for path in folder.glob("*_baseline.json")) == [
        "light_rcnn_accuracy_baseline.json"]
    # The dashboard's 'before' AP is there; its latency panel has nothing to pair.
    reports, _ = load_reports(folder)
    baseline = select_reports(reports, variant="baseline")
    assert set(baseline) == {("accuracy", "light_rcnn")}
    assert improvement_warnings(baseline, select_reports(reports, variant="fast")) == []


def test_without_a_baseline_the_presets_are_still_compared_with_deploy(run, script, monkeypatch):
    real = script.apply_inference_options

    def no_baseline(model, options, device):
        if options == script.PRESETS["baseline"]:
            raise RuntimeError("baseline boom")
        return real(model, options, device)

    monkeypatch.setattr(script, "apply_inference_options", no_baseline)
    code, output = run("--models", "v2", "--presets", "deploy,fast", "--limit", 2)
    assert code == 1
    rows = {row["preset"]: row for row in read_csv(output / "summary.csv")}
    assert [row["status"] for row in rows.values()] == ["failed", "ok", "ok"]
    assert not any(row["delta_p50_pct"] or row["delta_p50_ci_low"] for row in rows.values())
    assert rows["fast"]["delta_p50_vs_deploy_pct"] and rows["fast"]["delta_p50_vs_deploy_ci_low"]
    folder = output / "reports/v2"
    assert read_json(folder / "light_rcnn_latency_deploy.json")["latency"]["paired"] is None
    paired = read_json(folder / "light_rcnn_latency_fast.json")["latency"]["paired"]
    assert paired["vs_baseline"] is None and paired["vs_deploy"]["variant"] == "deploy"


def fake_calls(names, log, fail=None):
    """Stand-in timed calls: each logs (preset, image) and 'takes' its own call number in ms."""
    def call_for(name):
        def call(index):
            log.append((name, index))
            if fail is not None and fail(name, len(log)):
                raise RuntimeError(f"{name} boom")
            return float(len(log)), index

        return call

    return {name: call_for(name) for name in names}


def test_blocks_rotate_the_preset_order_and_leave_the_switch_call_untimed(script):
    log = []
    timings, failed = script.interleaved_latency(fake_calls(PRESETS, log), count=2, warmup=3,
                                                 rounds=4)
    assert failed == {}
    assert log[:9] == [(name, index) for index in (0, 1, 0) for name in PRESETS]  # round-robin
    blocks = [log[start:start + 3] for start in range(9, len(log), 3)]
    assert [block[0][0] for block in blocks] == ["baseline", "deploy", "fast", "deploy", "fast",
                                                 "baseline", "fast", "baseline", "deploy",
                                                 "baseline", "deploy", "fast"]
    for block in blocks:  # one untimed call on the block's first image, then every image once
        assert [index for _, index in block] == [0, 0, 1] and len({name for name, _ in block}) == 1
    for name in PRESETS:
        timing = timings[name]
        # A call 'takes' its own number, so the samples name the calls that were timed: the 2nd
        # and 3rd of each of this preset's blocks, never the untimed first one.
        starts = [9 + 3 * number for number, block in enumerate(blocks) if block[0][0] == name]
        assert timing["samples"] == [float(start + offset) for start in starts for offset in (2, 3)]
        assert timing["rounds"] == [timing["samples"][r * 2:r * 2 + 2] for r in range(4)]
        assert timing["detections"] == [0, 1] * 4  # sample i times image i % count


@pytest.mark.parametrize(("counts", "presets", "rounds"), [
    ((64, 640), 6, 12),  # the Colab T4 notebook: GPU_COUNTS with its 6 presets (10 rounds before)
    ((64, 640), 4, 12),  # GPU defaults
    ((16, 128), 4, 8),  # CPU defaults: already whole cycles
    ((16, 320), 5, 20),  # the final CPU run
    ((2, 2), 1, 1),
])
def test_rounds_are_whole_rotation_cycles_so_every_preset_takes_every_slot_equally(
        script, counts, presets, rounds):
    images, iterations = counts
    assert script.interleaved_rounds(iterations, images, presets) == rounds
    assert rounds * images >= iterations
    names = [f"p{index}" for index in range(presets)]
    slots = {name: [script.visit_order(names, r).index(name) for r in range(rounds)].count
             for name in names}
    assert {slots[name](slot) for name in names for slot in range(presets)} == {rounds // presets}


def test_a_preset_failing_mid_schedule_is_dropped_and_the_others_keep_every_round(script,
                                                                                  capsys):
    log = []
    # deploy raises on its first call of round 1 (3 warm-up calls + 9 in round 0 come first).
    timings, failed = script.interleaved_latency(
        fake_calls(PRESETS, log, fail=lambda name, number: name == "deploy" and number > 12),
        count=2, warmup=1, rounds=4)
    assert failed == {"deploy": "RuntimeError: deploy boom"}
    assert "RuntimeError: deploy boom" in capsys.readouterr().err
    assert sorted(timings) == ["baseline", "fast"]
    assert all(len(timing["rounds"]) == 4 for timing in timings.values())
    # The rotation goes on over the remaining presets; deploy is never called again.
    assert [name for name, _ in log[12:]] == ["deploy", *["fast"] * 3, *["baseline"] * 3,
                                              *["fast"] * 3, *["baseline"] * 3,
                                              *["baseline"] * 3, *["fast"] * 3]


def test_paired_bootstrap_resamples_whole_rounds_with_the_same_indices_for_both(script):
    factors = [1.0, 1.6, 0.8, 1.3, 0.9, 1.5, 1.1, 0.7]  # strong round-to-round drift
    before = [[100 * factor * (1 + 0.01 * k) for k in range(4)] for factor in factors]
    after = [[0.7 * value for value in block] for block in before]
    paired = script.paired_delta(after, before)
    # Every resample keeps each round's pair together, so the drift cancels exactly.
    assert paired["delta_p50_pct"] == pytest.approx(-30.0)
    assert paired["ci95_pct"] == pytest.approx([-30.0, -30.0])
    assert (paired["resamples"], paired["method"]) == (2000, "paired bootstrap over rounds")
    # Pairing each round with another round's reference lets the drift in: a wide interval.
    low, high = script.paired_delta(after, before[1:] + before[:1])["ci95_pct"]
    assert high - low > 20 and low < -30 < high
    assert script.paired_delta(after, before) == paired  # fixed seed
    assert script.paired_delta(after[:1], before[:1]) is None  # one round: nothing to resample
    assert script.paired_delta(after, before[:-1]) is None


def test_interleaving_sees_through_a_drift_that_fakes_a_sequential_delta(script):
    """A machine slowing down by 0.2% per call: sequential windows credit the drift to whichever
    preset ran later, while interleaved blocks and their paired CI recover the true -40%."""
    costs = {"baseline": 100.0, "fast": 60.0}

    def machine():
        clock = [0]

        def call_for(name):
            def call(index):
                clock[0] += 1
                jitter = 1 + 0.02 * math.sin(1.7 * clock[0])
                return costs[name] * (1 + 0.002 * clock[0]) * jitter, 1

            return call

        return {name: call_for(name) for name in costs}

    windows = {name: script.sequential_latency(call, 8, 8, 80)["samples"]
               for name, call in machine().items()}
    sequential = (statistics.median(windows["fast"]) / statistics.median(windows["baseline"])
                  - 1) * 100
    timings, failed = script.interleaved_latency(machine(), count=8, warmup=8, rounds=10)
    paired = script.paired_delta(timings["fast"]["rounds"], timings["baseline"]["rounds"])
    low, high = paired["ci95_pct"]
    assert failed == {} and sequential > -35  # 'fast' ran later, on a slower machine
    assert low <= -40 <= high and high - low < 5
    assert not low <= sequential <= high
    assert paired["delta_p50_pct"] == pytest.approx(-40, abs=1)


def test_faster_rcnn_gets_the_same_options_and_joins_every_version_folder(run, script,
                                                                          monkeypatch, tmp_path):
    real = script.load_comparison_model
    checkpoint = tmp_path / "faster.pth"
    checkpoint.write_bytes(b"stand-in")

    def small_faster(kind, config, weights=None, image_size=None):
        if kind != "faster_rcnn":
            return real(kind, config, weights, image_size)
        assert weights == str(checkpoint)
        torch.manual_seed(0)  # a torchvision Faster R-CNN, small enough for a unit test
        return fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None,
                                                     num_classes=91, min_size=image_size,
                                                     max_size=image_size)

    monkeypatch.setattr(script, "load_comparison_model", small_faster)
    code, output = run("--models", "v3", "faster_rcnn", "v2", "--presets", "fast", "--limit", 2,
                       "--faster-weights", checkpoint, "--iterations", 2)
    assert code == 0
    for preset in ("baseline", "fast"):
        names = [f"faster_rcnn_{kind}_{preset}.json" for kind in ("accuracy", "latency")]
        for name in names:
            copies = [read_json(output / "reports" / version / name) for version in ("v2", "v3")]
            assert copies[0] == copies[1]
            assert copies[0]["weights"] == display_path(checkpoint)
            assert copies[0]["weights_sha256"] == file_sha256(checkpoint)
        faster = read_json(output / "reports/v2" / names[0])["inference_options"]
        light = read_json(output / f"reports/v2/light_rcnn_accuracy_{preset}.json")
        assert faster["options"] == light["inference_options"]["options"]
        assert faster["applied"]["conv_bn_fused"] is (preset == "fast")
    for version in ("v2", "v3"):
        assert_dashboard_ready(output / "reports" / version, ["baseline", "fast"],
                               {"faster_rcnn", "light_rcnn"})
    assert len({read_json(path)["comparison_run"] for path in output.rglob("*.json")}) == 1
    # 2 iterations over 2 images make one round, rounded up to a whole rotation of the 2 presets.
    timing = read_json(output / "reports/v2/faster_rcnn_latency_fast.json")["latency"]
    assert (timing["rounds"], timing["iterations"]) == (2, 4)
    assert timing["paired"]["vs_baseline"]["variant"] == "baseline"
    rows = read_csv(output / "summary.csv")
    assert [(row["model"], row["preset"]) for row in rows] == [
        (model, preset) for model in ("faster_rcnn", "v2", "v3") for preset in ("baseline", "fast")]
    page = (output / "figures/improvement_v3.html").read_text(encoding="utf-8")
    assert "Light R-CNN v3 · Vanilla Faster R-CNN" in page


def test_mobilenet_baseline_loads_its_checkpoint_and_joins_every_version_folder(
        run, script, monkeypatch, tmp_path, mobilenet_checkpoint, capsys):
    """Both baselines with one preset set: the MobileNet goes through the real loader."""
    real, kinds = script.load_comparison_model, []
    faster_checkpoint = tmp_path / "faster.pth"
    faster_checkpoint.write_bytes(b"stand-in")

    def loader(kind, config, weights=None, image_size=None):
        kinds.append(kind)
        if kind == "faster_rcnn":  # the ResNet-50 model would be slow; any torchvision R-CNN does
            return small_baseline(image_size)
        return real(kind, config, weights, image_size)

    monkeypatch.setattr(script, "load_comparison_model", loader)
    code, output = run("--models", "v3", "mobilenet", "faster_rcnn", "v2", "--presets", "fast",
                       "--limit", 2, "--iterations", 2, "--faster-weights", faster_checkpoint,
                       "--mobilenet-weights", mobilenet_checkpoint)
    assert code == 0
    # Baselines first, every preset on a freshly loaded model.
    assert kinds == ["faster_rcnn"] * 2 + ["mobilenet_faster_rcnn"] * 2 + ["light_rcnn"] * 4
    for preset in ("baseline", "fast"):
        for kind in ("accuracy", "latency"):
            name = f"mobilenet_faster_rcnn_{kind}_{preset}.json"
            copies = [read_json(output / "reports" / version / name) for version in ("v2", "v3")]
            assert copies[0] == copies[1]
            report = copies[0]
            assert (report["model"], report["model_name"]) == ("mobilenet_faster_rcnn",
                                                               "MobileNetV3 Faster R-CNN")
            assert (report["model_tag"], report["model_label"]) == ("mobilenet",
                                                                    "MobileNetV3 Faster R-CNN")
            assert report["weights"] == display_path(mobilenet_checkpoint)
            assert report["weights_sha256"] == file_sha256(mobilenet_checkpoint)
        records = {model: read_json(output / f"reports/v2/{model}_latency_{preset}.json")[
            "inference_options"] for model in ("faster_rcnn", "mobilenet_faster_rcnn",
                                               "light_rcnn")}
        # One preset, applied alike to every model.
        assert len({json.dumps(record["options"], sort_keys=True)
                    for record in records.values()}) == 1
        applied = records["mobilenet_faster_rcnn"]["applied"]
        assert applied["conv_bn_fused"] is (preset == "fast")  # its BatchNorm2d folded
        assert applied["rpn_pre_nms_top_n_test"] == (500 if preset == "fast" else 1000)
        # torchvision defaults that stay as they are and are recorded.
        assert (applied["rpn_post_nms_top_n_test"], applied["rpn_score_thresh"]) == (1000, 0.05)
        accuracy = read_json(output / f"reports/v2/mobilenet_faster_rcnn_accuracy_{preset}.json")
        assert accuracy["accuracy"]["detections"] > 0  # the checkpoint's 'person' bias
    for version in ("v2", "v3"):
        assert_dashboard_ready(output / "reports" / version, ["baseline", "fast"],
                               {"faster_rcnn", "mobilenet_faster_rcnn", "light_rcnn"})
    assert len({read_json(path)["comparison_run"] for path in output.rglob("*.json")}) == 1
    rows = read_csv(output / "summary.csv")
    assert [(row["model"], row["preset"], row["status"]) for row in rows] == [
        (model, preset, "ok") for model in ("faster_rcnn", "mobilenet", "v2", "v3")
        for preset in ("baseline", "fast")]
    mobilenet = rows[2]
    assert mobilenet["model_label"] == "MobileNetV3 Faster R-CNN"
    # The full model, not the 320 px one (BatchNorm2d weights counted in a random build).
    assert float(mobilenet["parameters_m"]) == pytest.approx(19.41, abs=0.01)
    assert rows[3]["delta_ap_pp"] and rows[3]["delta_p50_pct"]
    summary = (output / "summary.md").read_text(encoding="utf-8")
    assert "같은 옵션을 적용합니다 (Vanilla Faster R-CNN, MobileNetV3 Faster R-CNN 포함)." in summary
    assert "| MobileNetV3 Faster R-CNN | 개선 전 (기본 설정) (baseline) |" in summary
    label = "Light R-CNN v3 · Vanilla Faster R-CNN · MobileNetV3 Faster R-CNN"
    assert f"`{display_path(output / 'reports/v3')}`: {label}" in summary
    assert label in (output / "figures/improvement_v3.html").read_text(encoding="utf-8")
    assert "모델: Vanilla Faster R-CNN, MobileNetV3 Faster R-CNN, Light R-CNN v2, Light R-CNN v3" in (
        capsys.readouterr().out)


def test_baselines_join_every_folder_and_share_one_without_light_rcnn(script):
    groups = script.report_groups
    assert groups(["faster_rcnn", "mobilenet", "v2", "v3"]) == ["v2", "v3"]
    assert groups(["mobilenet", "v3"]) == ["v3"]
    assert groups(["faster_rcnn"]) == ["faster_rcnn"] and groups(["mobilenet"]) == ["mobilenet"]
    assert groups(["faster_rcnn", "mobilenet"]) == ["baselines"]
    label = script.group_label
    assert label("v2", ["faster_rcnn", "mobilenet", "v2"]) == (
        "Light R-CNN v2 · Vanilla Faster R-CNN · MobileNetV3 Faster R-CNN")
    assert label("v2", ["mobilenet", "v2"]) == "Light R-CNN v2 · MobileNetV3 Faster R-CNN"
    assert label("v3", ["v3"]) == "Light R-CNN v3"
    assert label("baselines", ["faster_rcnn", "mobilenet"]) == (
        "Vanilla Faster R-CNN · MobileNetV3 Faster R-CNN")
    assert label("mobilenet", ["mobilenet"]) == "MobileNetV3 Faster R-CNN"
    # Measured order: the baselines first, whatever --models says; the default is all four.
    args = script.build_parser().parse_args([])
    assert args.models == list(script.MODEL_KINDS) == ["faster_rcnn", "mobilenet", "v2", "v3"]
    assert script.MODEL_KINDS["mobilenet"] == "mobilenet_faster_rcnn"
    assert args.mobilenet_weights is None


def test_a_baselines_only_run_writes_one_folder_with_both(run, script, monkeypatch, tmp_path):
    def loader(kind, config, weights=None, image_size=None):
        assert kind in ("faster_rcnn", "mobilenet_faster_rcnn") and weights
        return small_baseline(image_size)

    monkeypatch.setattr(script, "load_comparison_model", loader)
    stand_in = tmp_path / "stand_in.pth"
    stand_in.write_bytes(b"stand-in")
    code, output = run("--models", "mobilenet", "faster_rcnn", "--presets", "deploy", "--limit", 1,
                       "--latency-images", 1, "--iterations", 2, "--faster-weights", stand_in,
                       "--mobilenet-weights", stand_in)
    assert code == 0
    assert sorted(path.name for path in (output / "reports").iterdir()) == ["baselines"]
    assert sorted(path.name for path in (output / "reports/baselines").iterdir()) == sorted(
        f"{model}_{kind}_{preset}.json" for model in ("faster_rcnn", "mobilenet_faster_rcnn")
        for kind in ("accuracy", "latency") for preset in ("baseline", "deploy"))
    assert_dashboard_ready(output / "reports/baselines", ["baseline", "deploy"],
                           {"faster_rcnn", "mobilenet_faster_rcnn"})
    summary = (output / "summary.md").read_text(encoding="utf-8")
    assert (f"- `{display_path(output / 'reports/baselines')}`: Vanilla Faster R-CNN · "
            "MobileNetV3 Faster R-CNN") in summary
    page = (output / "figures/improvement_baselines.html").read_text(encoding="utf-8")
    assert "개선 전/후 비교 — Vanilla Faster R-CNN · MobileNetV3 Faster R-CNN" in page


def test_deltas_compare_each_model_with_its_own_baseline(script):
    def measured(tag, preset, ap, p50, **paired):
        return {"tag": tag, "preset": preset, "label": PRESET_LABELS[preset], "status": "ok",
                "note": "", "summary": "", "precision": "fp32",
                "accuracy": {"AP_50_95": ap, "AP_50": ap + 0.2, "AP_75": None, "AP_small": None,
                             "AP_medium": None, "AP_large": None, "AR_100": None,
                             "score_threshold": 0.05, "max_detections": 100, "detections": 40,
                             "images_evaluated": 2},
                "latency": {"p50_ms": p50, "p95_ms": p50, "p99_ms": p50, "mean_ms": p50,
                            "fps": 1000 / p50, "cold_start_ms": 1.0, "mean_detections": 20.0,
                            "image_count": 2, "warmup": 2, "iterations": 4,
                            "memory_format": "contiguous", "conv_bn_fused": False,
                            "paired": {key: {"variant": variant, "ci95_pct": ci}
                                       for key, (variant, ci) in paired.items()} or None},
                "model_stats": {"parameters_m": 4.0, "deployed_parameters_m": 4.0, "gflops": 6.0},
                "reports": {"accuracy": "a.json", "latency": "l.json"}}

    rows = script.summary_rows([
        measured("v2", "baseline", 0.40, 100.0),
        measured("v2", "deploy", 0.40, 90.0, vs_baseline=("baseline", [-12.04, -7.96])),
        measured("v2", "fast", 0.41, 80.0, vs_baseline=("baseline", [-22.04, -17.96]),
                 vs_deploy=("deploy", [-13.0, -9.2])),
        # No v3 baseline or deploy ran, so these CIs describe no row of this table.
        measured("v3", "fast", 0.30, 50.0, vs_baseline=("baseline", [-1.0, 1.0]),
                 vs_deploy=("deploy", [-1.0, 1.0])),
        {"tag": "faster_rcnn", "preset": "baseline", "label": "개선 전 (기본 설정)",
         "status": "skipped", "note": "가중치 없음"},
    ], "cpu")
    baseline, deploy, fast, orphan, skipped = rows
    assert baseline["delta_ap_pp"] is None and baseline["delta_p50_pct"] is None
    assert baseline["delta_p50_vs_deploy_pct"] is None and deploy["delta_p50_vs_deploy_pct"] is None
    assert deploy["delta_p50_pct"] == pytest.approx(-10.0)
    assert fast["delta_ap_pp"] == pytest.approx(1.0)
    assert fast["delta_p50_pct"] == pytest.approx(-20.0)  # includes deploy's own -10%
    assert fast["delta_p50_vs_deploy_pct"] == pytest.approx(-100 / 9)  # 80 vs 90 ms
    assert orphan["delta_ap_pp"] is None  # no v3 baseline: never compared with v2's
    assert orphan["delta_p50_vs_deploy_pct"] is None  # nor with v2's deploy run
    assert skipped["status"] == "skipped" and skipped["ap_pct"] is None
    assert fast["ap_pct"] == pytest.approx(41.0) and fast["ap75_pct"] is None
    columns = script.CSV_COLUMNS
    assert columns.index("delta_p50_vs_deploy_pct") == columns.index("delta_p50_pct") + 1
    assert columns == (*OLD_CSV_COLUMNS, *CI_COLUMNS)  # new columns only appended
    assert len(script.table_cells(fast)) == len(script.TABLE_HEADER)
    assert (fast["delta_p50_ci_low"], fast["delta_p50_ci_high"]) == (-22.04, -17.96)
    assert (fast["delta_p50_vs_deploy_ci_low"], fast["delta_p50_vs_deploy_ci_high"]) == (-13.0, -9.2)
    assert deploy["delta_p50_vs_deploy_ci_low"] is None
    assert all(orphan[key] is None for key in CI_COLUMNS)
    assert all(baseline[key] is None for key in CI_COLUMNS)
    assert script._ci({"variant": "deploy", "ci95_pct": [-1.0, 1.0]}, "baseline") == (None, None)
    assert script.table_cells(baseline)[-3:] == ["기준", "기준", "–"]
    assert script.table_cells(deploy)[-3:] == ["+0.00", "-10.0 [-12.0, -8.0]", "기준"]
    assert script.table_cells(fast)[-3:] == ["+1.00", "-20.0 [-22.0, -18.0]", "-11.1 [-13.0, -9.2]"]
    assert script.table_cells(orphan)[-3:] == ["–", "–", "–"]
    lines = script.console_table([baseline, deploy, fast, orphan]).splitlines()
    assert len({script._width(line) for line in lines}) == 1  # Hangul counted as two columns


def test_graph_note_flags_runs_whose_calls_skipped_the_cuda_graph(script):
    assert script.graph_note({}) == ""  # no CUDA graph requested
    assert script.graph_note({"status": "captured", "replays": 9, "eager_calls": 0}) == ""
    partial = script.graph_note({"status": "captured", "replays": 7, "eager_calls": 2})
    assert partial.startswith("CUDA 그래프가 일부 호출에만 쓰임 (재생 7회, eager 2회")
    pending = script.graph_note({"status": "pending (captured on the first CUDA call)",
                                 "replays": 0, "eager_calls": 0})
    assert pending.startswith("CUDA 그래프 미사용: pending") and "그래프 없이 잰 값" in pending


@pytest.mark.parametrize("argv, message", [
    (["--presets", "baseline,turbo"], "알 수 없는 추론 설정: turbo"),
    (["--latency-images", "0"], "--latency-images"),
    (["--score-thresh", "1.5"], "--score-thresh"),
])
def test_rejects_bad_arguments(script, argv, message, capsys):
    with pytest.raises(SystemExit) as exit_info:
        script.main(argv)
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


def test_missing_inputs_stop_before_measuring(script, coco, tmp_path):
    common = ["--config", str(coco["config"]), "--device", "cpu"]
    with pytest.raises(SystemExit, match="COCO 평가 데이터가 없습니다"):
        script.main([*common, "--images", str(tmp_path / "nowhere")])
    with pytest.raises(SystemExit, match="Light R-CNN v3 체크포인트가 없습니다"):
        script.main([*common, "--images", str(coco["images"]), "--annotations",
                     str(coco["annotations"]), "--models", "v3", "--v3-weights",
                     str(tmp_path / "missing.pth")])
    with pytest.raises(SystemExit, match="MobileNetV3 Faster R-CNN 체크포인트가 없습니다"):
        script.main([*common, "--images", str(coco["images"]), "--annotations",
                     str(coco["annotations"]), "--models", "mobilenet", "--mobilenet-weights",
                     str(tmp_path / "missing.pth")])


def test_runs_without_editable_install(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", RUNNER, str(ROOT / "src"), str(SCRIPT)],
        check=False, cwd=tmp_path, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "--presets" in result.stdout and "--faster-weights" in result.stdout
    text = " ".join(result.stdout.split())  # help lines wrap at the terminal width
    assert "--mobilenet-weights" in text and "faster_rcnn → mobilenet → v2 → v3" in text
