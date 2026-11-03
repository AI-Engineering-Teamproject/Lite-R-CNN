import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from light_rcnn import reporting
from light_rcnn.reporting import (
    BASELINE_VARIANT,
    BASELINE_VARIANT_LABEL,
    build_report,
    comparability_warnings,
    improvement_warnings,
    latency_summary,
    load_reports,
    normalize,
    select_reports,
    variant_labels,
    variants,
    write_report,
)

CPU = {"device": "cpu", "device_type": "cpu", "device_name": "Xeon", "torch_num_threads": 4}
T4 = {"device": "cuda:0", "device_type": "cuda", "device_name": "Tesla T4"}
FAST = {"variant": "fast", "variant_label": "빠른 설정",
        "inference_options": {"fuse_conv_bn": True, "rpn_pre_nms_top_n": 500, "rpn_score_thresh": 0.05}}


def accuracy(model="light_rcnn", ap=0.3, created_at=None, images=5000, **top):
    report = build_report("accuracy", model, precision="fp32", image_size=640, environment={},
                          batch_size=1,
                          accuracy={"evaluator": "pycocotools", "images_evaluated": images,
                                    "images_total": 5000, "is_partial": images < 5000,
                                    "score_threshold": 0.001, "AP_50_95": ap}, **top)
    if created_at:
        report["created_at"] = created_at
    return report


def latency(model="light_rcnn", p50=40.0, environment=CPU, created_at=None, **top):
    report = build_report("latency", model, precision="fp32", image_size=640,
                          environment=dict(environment),
                          latency={"scope_id": reporting.SCOPE_HOST_TO_DETECTIONS,
                                   "memory_format": "contiguous", "image_count": 16,
                                   **latency_summary([p50 + i % 3 - 1 for i in range(150)])}, **top)
    if created_at:
        report["created_at"] = created_at
    return report


def load(*raw):
    return [report for index, item in enumerate(raw) for report in normalize(item, f"r{index}.json")]


def test_variant_defaults_to_baseline_and_labels_fall_back():
    [plain] = load(accuracy())
    assert (plain.variant, plain.variant_label) == (BASELINE_VARIANT, BASELINE_VARIANT_LABEL)
    for empty in (None, "", "  "):
        assert load(accuracy(variant=empty))[0].variant == BASELINE_VARIANT
    [fast] = load(accuracy(**FAST))
    assert (fast.variant, fast.variant_label) == ("fast", "빠른 설정")
    [unlabeled] = load(accuracy(variant=" fast ", variant_label=" "))
    assert (unlabeled.variant, unlabeled.variant_label) == ("fast", "fast")
    [legacy] = load({"model": "light_rcnn", "device": "cpu", "p50_ms": 80.0})
    assert legacy.variant == BASELINE_VARIANT


def test_variant_falls_back_to_the_applied_options_record():
    record = {"preset": "fast", "label": "빠른 설정", "summary": "Conv+BN 융합", "options": {}}
    [fast] = load(accuracy(inference_options=record))
    assert (fast.variant, fast.variant_label) == ("fast", "빠른 설정")
    # Options matching no preset are never mistaken for the baseline.
    [custom] = load(accuracy(inference_options={**record, "preset": None, "label": "사용자 지정 설정"}))
    assert (custom.variant, custom.variant_label) == (reporting.CUSTOM_VARIANT, "사용자 지정 설정")
    [named] = load(accuracy(variant="fast_pre300", inference_options={**record, "preset": None}))
    assert named.variant == "fast_pre300" and named.inference_options["summary"] == "Conv+BN 융합"
    [odd] = load(accuracy(inference_options=["not", "a", "record"]))
    assert (odd.variant, odd.inference_options) == (BASELINE_VARIANT, {})


def test_variants_put_baseline_first_then_presets_then_other_names():
    names = ["zzz_custom", "accurate", "fast", None, "deploy", "gpu_graph", "accurate_fast", "abc",
             "gpu_fp16"]
    found = load(*(accuracy(variant=name) for name in names))
    assert variants(found) == ["baseline", "deploy", "fast", "accurate", "accurate_fast",
                               "gpu_fp16", "gpu_graph", "abc", "zzz_custom"]
    assert variants(load(accuracy(**FAST))) == ["fast"]
    assert variants([]) == [] and variant_labels([]) == {}


def test_variant_labels_prefer_the_newest_recorded_label():
    found = load(
        accuracy(variant="fast", variant_label="빠름 (예전 이름)", created_at="2026-01-01T00:00:00+00:00"),
        latency(variant="fast", variant_label="빠른 설정", created_at="2026-02-01T00:00:00+00:00"),
        latency(variant="fast", created_at="2026-03-01T00:00:00+00:00"),  # no label recorded
        accuracy(),
    )
    labels = variant_labels(iter(found))  # any iterable, consumed once
    assert labels == {"baseline": BASELINE_VARIANT_LABEL, "fast": "빠른 설정"}
    assert list(labels) == ["baseline", "fast"]


def test_select_reports_filters_by_variant():
    found = load(
        accuracy(ap=0.30, created_at="2026-01-01T00:00:00+00:00"),
        accuracy(ap=0.29, created_at="2026-02-01T00:00:00+00:00", **FAST),
        latency(p50=40.0, created_at="2026-01-01T00:00:00+00:00"),
        latency(p50=33.0, created_at="2026-02-01T00:00:00+00:00", **FAST),
    )
    base = select_reports(found, variant=BASELINE_VARIANT)
    fast = select_reports(found, variant="fast")
    assert base[("accuracy", "light_rcnn")].section["AP_50_95"] == 0.30
    assert base[("latency", "light_rcnn")].section["p50_ms"] == 40.0
    assert fast[("accuracy", "light_rcnn")].section["AP_50_95"] == 0.29
    assert fast[("latency", "light_rcnn")].section["p50_ms"] == 33.0
    assert all(report.variant == "fast" for report in fast.values())
    assert select_reports(found, variant="accurate") == {}
    # Without a variant the old rule applies (newest wins), whatever preset it used.
    assert {r.variant for r in select_reports(found).values()} == {"fast"}


def test_variant_and_device_filters_combine():
    found = load(
        accuracy(**FAST),
        latency(p50=33.0, **FAST),
        latency(p50=5.0, environment=T4, **FAST),
        latency(p50=6.0, environment=T4),
    )
    picked = select_reports(found, device="Tesla T4", variant="fast")
    assert picked[("latency", "light_rcnn")].section["p50_ms"] == 5.0
    assert picked[("accuracy", "light_rcnn")].variant == "fast"  # accuracy ignores the device
    assert select_reports(found, device="Xeon", variant=BASELINE_VARIANT) == {}
    assert reporting.timing_devices([r for r in found if r.variant == BASELINE_VARIANT]) == ["Tesla T4"]


def test_single_variant_selection_warnings_are_unchanged():
    raw = [accuracy("light_rcnn"), accuracy("yolo26", ap=0.4), latency("light_rcnn"),
           latency("yolo26", p50=30.0)]
    before = comparability_warnings(select_reports(load(*raw)))
    tagged = load(*({**report, **FAST} for report in raw))
    after = comparability_warnings(select_reports(tagged, variant="fast"))
    assert [(w.level, w.message) for w in after] == [(w.level, w.message) for w in before]
    assert not any("추론 설정" in w.message for w in before)


def test_mixed_variants_in_one_selection_are_an_error():
    found = load(
        accuracy("light_rcnn"), accuracy("faster_rcnn", ap=0.37),
        latency("light_rcnn", created_at="2026-01-01T00:00:00+00:00"),
        latency("light_rcnn", p50=33.0, created_at="2026-02-01T00:00:00+00:00", **FAST),
        latency("faster_rcnn", p50=900.0),
    )
    [error] = [w for w in comparability_warnings(select_reports(found))
               if "추론 설정이 다릅니다" in w.message]
    assert error.level == "error"
    assert f"{BASELINE_VARIANT_LABEL}: Vanilla Faster R-CNN 정확도" in error.message
    assert "빠른 설정: Light R-CNN 지연 시간" in error.message
    for name in (BASELINE_VARIANT, "fast"):
        picked = select_reports(found, variant=name)
        assert not any("추론 설정" in w.message for w in comparability_warnings(picked))


def test_improvement_warnings_flag_setup_differences_but_not_preset_effects():
    same = improvement_warnings(select_reports(load(accuracy(), latency())),
                                select_reports(load(accuracy(**FAST), latency(**FAST))))
    assert same == []
    # Precision, memory format and score thresholds are what presets change: never flagged.
    tuned = latency(**FAST)
    tuned["precision"] = "fp16"
    tuned["latency"].update(memory_format="channels_last", score_threshold=0.05)
    assert improvement_warnings(select_reports(load(latency())), select_reports(load(tuned))) == []

    smoke = accuracy(images=128)
    smoke["batch_size"] = 4
    smoke["accuracy"]["image_ids_sha256"] = "a" * 64
    full = accuracy(**FAST)
    full["accuracy"]["image_ids_sha256"] = "b" * 64
    threads = latency(**FAST, environment={**CPU, "torch_num_threads": 8})
    found = improvement_warnings(select_reports(load(smoke, latency())),
                                 select_reports(load(full, threads)))
    messages = {(w.level, w.message.split(" (")[0]): w.message for w in found}
    accuracy_error = messages[("error", "Light R-CNN 정확도의 개선 전/후 측정 조건이 다릅니다")]
    assert "평가 이미지 수 128 → 5000" in accuracy_error and "평가 이미지 목록" in accuracy_error
    assert "a" * 64 not in accuracy_error  # hashes are named, not printed
    assert "평가 배치 크기 4 → 1" in messages[("warning", "Light R-CNN 정확도의 개선 전/후 측정 조건이 다릅니다")]
    assert "CPU 스레드 수 4 → 8" in messages[("error", "Light R-CNN 지연 시간의 개선 전/후 측정 조건이 다릅니다")]
    # A model measured on one side only has nothing to compare.
    assert improvement_warnings(select_reports(load(smoke)), {}) == []


def test_variant_fields_survive_a_round_trip(tmp_path):
    write_report(latency(**FAST), tmp_path / "fast" / "light_rcnn_latency.json")
    write_report(latency(), tmp_path / "light_rcnn_latency.json")
    found, errors = load_reports(tmp_path)
    assert not errors
    [fast] = [report for report in found if report.variant == "fast"]
    assert fast.data["inference_options"] == FAST["inference_options"]
    assert variant_labels(found) == {"baseline": BASELINE_VARIANT_LABEL, "fast": "빠른 설정"}


def test_module_inspection_lists_each_variant(tmp_path):
    write_report(latency(**FAST), tmp_path / "fast.json")
    write_report(latency(), tmp_path / "baseline.json")
    src = str(Path(reporting.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONIOENCODING": "utf-8",
           "PYTHONPATH": os.pathsep.join([src, os.environ.get("PYTHONPATH", "")])}
    result = subprocess.run([sys.executable, "-m", "light_rcnn.reporting", str(tmp_path)], env=env,
                            capture_output=True, encoding="utf-8", timeout=120, check=True)
    assert f"== baseline ({BASELINE_VARIANT_LABEL})" in result.stdout
    assert "== fast (빠른 설정)" in result.stdout
    assert "추론 설정이 다릅니다" not in result.stdout


@pytest.mark.skipif(importlib.util.find_spec("light_rcnn.inference_opts") is None,
                    reason="light_rcnn.inference_opts not available")
def test_preset_names_and_baseline_label_match_inference_opts():
    pytest.importorskip("torch")
    from light_rcnn import inference_opts

    # Same names in the same order, so the dashboard lists presets as inference_opts defines them.
    assert [name for name in inference_opts.PRESETS if name in reporting.VARIANT_ORDER] == list(
        reporting.VARIANT_ORDER)
    assert inference_opts.PRESET_LABELS[BASELINE_VARIANT] == BASELINE_VARIANT_LABEL
    assert reporting.CUSTOM_VARIANT not in inference_opts.PRESETS


def test_a_cuda_graph_run_that_ran_eagerly_is_flagged_once_per_model():
    disabled = "disabled (graph replay differs from eager)"
    pending = "pending (captured on the first CUDA call)"

    def graph(status, value=None):
        record = {"preset": "gpu_graph", "label": "GPU CUDA 그래프 (CUDA 전용)",
                  "summary": "Conv+BN 융합 · 640×640 고정 캔버스 · CUDA 그래프"}
        if status is not None:
            record["cuda_graph_status"] = value or {"status": status, "replays": 0, "eager_calls": 9}
        return {"variant": "gpu_graph", "inference_options": record}

    def flagged(*raw):
        selected = select_reports(load(*raw), variant="gpu_graph")
        return [w.message for w in comparability_warnings(selected)
                if w.level == "warning" and "CUDA 그래프" in w.message]

    assert flagged(accuracy(**graph(disabled)), latency(environment=T4, **graph(disabled)),
                   latency("faster_rcnn", environment=T4, **graph("captured"))) == [
        (f"CUDA 그래프를 켰지만 쓰지 못한 결과가 있습니다 (Light R-CNN: {disabled}). 백본을 eager로 "
         "실행한 지연 시간이므로 CUDA 그래프의 효과로 보지 마세요.")]
    both = flagged(latency(environment=T4, **graph(disabled)),
                   latency("faster_rcnn", environment=T4, **graph(pending)))
    assert len(both) == 2  # model order
    assert f"(Vanilla Faster R-CNN: {pending})" in both[0] and f"(Light R-CNN: {disabled})" in both[1]
    stages = build_report("stages", "light_rcnn", environment=T4, stages={"items": []},
                          **graph(disabled))
    assert len(flagged(stages)) == 1
    hand_edited = graph("", "disabled (hand-edited report)")  # read the way cuda_graph_note reads it
    assert flagged(latency(environment=T4, **hand_edited)) == [
        ("CUDA 그래프를 켰지만 쓰지 못한 결과가 있습니다 (Light R-CNN: disabled (hand-edited report)). "
         "백본을 eager로 실행한 지연 시간이므로 CUDA 그래프의 효과로 보지 마세요.")]
    # Only timings count (a graph never changes AP), and only a recorded status that is not captured.
    assert flagged(accuracy(**graph(disabled)), latency(environment=T4, **graph("captured"))) == []
    for status in ("captured", "", None):
        assert flagged(latency(environment=T4, **graph(status))) == []
    assert flagged(latency(environment=T4, **graph("x", {"replays": 3}))) == []
    assert reporting.cuda_graph_note(graph(disabled)["inference_options"]).startswith(
        "CUDA 그래프 사용 안 됨")
