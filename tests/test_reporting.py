import json

import numpy as np
import pytest

from light_rcnn import reporting
from light_rcnn.reporting import (
    build_report,
    comparability_warnings,
    latency_summary,
    load_reports,
    normalize,
    percentile,
    select_reports,
)


def latency_report(model="light_rcnn", device_name="Tesla T4", samples=None, **extra):
    samples = samples or [10.0 + i * 0.1 for i in range(200)]
    return build_report("latency", model, precision="fp32", image_size=640,
                        environment={"device": "cuda:0", "device_type": "cuda", "device_name": device_name},
                        latency={"scope_id": reporting.SCOPE_HOST_TO_DETECTIONS, **latency_summary(samples)},
                        **extra)


def accuracy_report(model="light_rcnn", images=5000, total=5000, evaluator="pycocotools", ap=0.3):
    return build_report("accuracy", model, precision="fp32", image_size=640, environment={},
                        accuracy={"evaluator": evaluator, "images_evaluated": images, "images_total": total,
                                  "is_partial": images < total, "score_threshold": 0.001, "AP_50_95": ap})


@pytest.mark.parametrize("q", [0, 1, 25, 50, 90, 95, 99, 100])
def test_percentile_matches_numpy_linear(q):
    values = [3.0, 1.0, 4.0, 1.5, 9.0, 2.6, 5.3]
    assert percentile(values, q) == pytest.approx(np.percentile(values, q))


def test_latency_summary_fields_and_samples():
    summary = latency_summary([10, 20, 30, 40], cold_start_ms=99)
    assert summary["n"] == 4
    assert summary["p50_ms"] == pytest.approx(25)
    assert summary["mean_ms"] == pytest.approx(25)
    assert summary["std_ms"] == pytest.approx(np.std([10, 20, 30, 40], ddof=1))
    assert summary["fps"] == pytest.approx(40)
    assert summary["cold_start_ms"] == 99
    assert summary["samples_ms"] == [10, 20, 30, 40]
    with pytest.raises(ValueError):
        latency_summary([])


def test_legacy_v1_files_are_translated():
    legacy_latency = {"model": "light_rcnn", "device": "cpu", "p50_ms": 80.0, "p95_ms": 120.0, "iterations": 30}
    legacy_yolo_accuracy = {"model": "yolo26", "images_evaluated": 128,
                            "accuracy": {"AP_50_95": 0.48, "AP_50": 0.65, "AP_75": 0.53}}
    [lat] = normalize(legacy_latency, "a.json")
    [acc] = normalize(legacy_yolo_accuracy, "b.json")
    assert lat.kind == "latency" and lat.legacy and lat.section["p50_ms"] == 80.0
    assert acc.kind == "accuracy" and acc.section["evaluator"] == "ultralytics"
    assert acc.section["is_partial"] is True


def test_unknown_models_and_non_dicts_are_ignored():
    assert normalize({"foo": 1}, "x.json") == []
    assert normalize([1, "a", None], "x.json") == []


def test_select_prefers_complete_accuracy_then_newest():
    partial = normalize(accuracy_report(images=500), "partial.json")[0]
    full_old = normalize({**accuracy_report(), "created_at": "2026-01-01T00:00:00+00:00"}, "full_old.json")[0]
    full_new = normalize({**accuracy_report(), "created_at": "2026-02-01T00:00:00+00:00"}, "full_new.json")[0]
    chosen = select_reports([full_new, partial, full_old])
    assert chosen[("accuracy", "light_rcnn")].source == "full_new.json"
    assert select_reports([partial, full_old])[("accuracy", "light_rcnn")].source == "full_old.json"


def test_comparability_flags_unfair_setups():
    reports = [
        *normalize(accuracy_report("light_rcnn"), "a"),
        *normalize(accuracy_report("yolo26", evaluator="ultralytics"), "b"),
        *normalize(latency_report("light_rcnn", "Tesla T4"), "c"),
        *normalize(latency_report("yolo26", "RTX 4090"), "d"),
    ]
    messages: dict[str, list[str]] = {}
    for warning in comparability_warnings(select_reports(reports)):
        messages.setdefault(warning.level, []).append(warning.message)
    errors = " ".join(messages.get("error", []))
    assert "AP 계산기" in errors
    assert "서로 다른 장치" in errors


def test_fair_setup_has_no_errors():
    reports = [*normalize(accuracy_report("light_rcnn"), "a"), *normalize(accuracy_report("yolo26"), "b"),
               *normalize(latency_report("light_rcnn"), "c"), *normalize(latency_report("yolo26"), "d")]
    levels = {w.level for w in comparability_warnings(select_reports(reports))}
    assert "error" not in levels


def test_load_reports_skips_hidden_dirs_and_reports_bad_json(tmp_path):
    (tmp_path / ".ultralytics").mkdir()
    (tmp_path / ".ultralytics" / "settings.json").write_text(json.dumps({"model": "yolo26", "p50_ms": 1}))
    (tmp_path / "reports").mkdir()
    reporting.write_report(latency_report(), tmp_path / "reports" / "light_rcnn_latency.json")
    (tmp_path / "broken.json").write_text("{not json")
    reports, errors = load_reports(tmp_path, [("upload.json", json.dumps(accuracy_report("yolo26")).encode())])
    assert sorted((r.kind, r.model) for r in reports) == [("accuracy", "yolo26"), ("latency", "light_rcnn")]
    assert len(errors) == 1 and "broken.json" in errors[0]


def test_build_report_validates_keys():
    with pytest.raises(ValueError):
        build_report("speed", "light_rcnn")
    with pytest.raises(ValueError):
        build_report("latency", "resnet")


def test_schema_v2_beats_legacy_even_with_fewer_images():
    legacy = normalize({"model": "yolo26", "images_evaluated": 128, "accuracy": {"AP_50_95": 0.48}}, "old.json")[0]
    partial_v2 = normalize(accuracy_report("yolo26", images=16, total=128), "new.json")[0]
    assert select_reports([legacy, partial_v2])[("accuracy", "yolo26")].source == "new.json"


def test_mini_set_threads_and_mode_mismatches_are_flagged():
    acc = accuracy_report("yolo26", images=128, total=128)
    acc["accuracy"]["inference_mode"] = "NMS"
    lat_yolo = latency_report("yolo26", "Xeon", inference_mode="ignored")
    lat_yolo["environment"].update({"device_type": "cpu", "torch_num_threads": 3})
    lat_yolo["latency"]["inference_mode"] = "end2end (NMS-free)"
    lat_light = latency_report("light_rcnn", "Xeon")
    lat_light["environment"].update({"device_type": "cpu", "torch_num_threads": 4})
    reports = [*normalize(acc, "a"), *normalize(lat_yolo, "b"), *normalize(lat_light, "c")]
    text = " ".join(w.message for w in comparability_warnings(select_reports(reports)))
    assert "5000장" in text
    assert "스레드" in text
    assert "추론 방식" in text


def test_display_path_is_repo_relative():
    from pathlib import Path

    root = Path(reporting.__file__).resolve().parents[2]
    assert reporting.display_path(root / "outputs" / "x.json") == "outputs/x.json"
    assert reporting.display_path(None) is None


def _warnings(*reports):
    loaded = [r for i, rep in enumerate(reports) for r in normalize(rep, str(i))]
    return comparability_warnings(select_reports(loaded))


def test_annotation_fingerprint_mismatch_is_an_error():
    light, yolo = accuracy_report("light_rcnn"), accuracy_report("yolo26")
    light["accuracy"]["dataset"] = {"annotations_sha256": "a" * 64}
    yolo["accuracy"]["dataset"] = {"annotations_sha256": "b" * 64}
    assert any(w.level == "error" and "주석" in w.message for w in _warnings(light, yolo))
    yolo["accuracy"]["dataset"] = {"annotations_sha256": "a" * 64}
    assert not any("주석" in w.message for w in _warnings(light, yolo))


def test_cold_start_scope_difference_is_info():
    light, yolo = latency_report("light_rcnn"), latency_report("yolo26")
    light["latency"]["cold_start_scope"] = reporting.COLD_START_FIRST_FORWARD
    yolo["latency"]["cold_start_scope"] = reporting.COLD_START_FIRST_PREDICT
    [note] = [w for w in _warnings(light, yolo) if "cold start" in w.message]
    assert note.level == "info"


def test_dataset_info_fingerprints_annotations(tmp_path):
    ann = tmp_path / "instances.json"
    ann.write_text("{}")
    info = reporting.dataset_info(ann, tmp_path)
    assert info["annotations_sha256"] == reporting.file_sha256(ann)
    assert set(info) == {"annotations", "annotations_sha256", "images_dir"}


def test_v1_run_folder_gives_time_and_one_cuda_device(tmp_path):
    # Mirrors outputs/comparison_runs/20261019_225947: old YOLO files store the CUDA index "0".
    run = tmp_path / "comparison_runs" / "20261019_225947"
    run.mkdir(parents=True)
    for name, raw in {
        "faster_rcnn_latency.json": {"model": "faster_rcnn", "device": "cuda", "p50_ms": 7.7},
        "light_rcnn_latency.json": {"model": "light_rcnn", "device": "cuda", "p50_ms": 4.3},
        "yolo26_latency.json": {"model_family": "YOLO26", "imgsz": 640, "device": "0",
                                "iterations": 500, "p50_ms": 3.2, "p95_ms": 8.5},
    }.items():
        (run / name).write_text(json.dumps(raw), encoding="utf-8")
    reports, errors = load_reports(tmp_path)
    assert not errors and len(reports) == 3
    for report in reports:
        assert report.created_at == "2026-10-19T22:59:47"
        assert "측정 시각 미기록: 실행 폴더 이름(현지 시각)에서 추정" in report.notes
    assert reporting.timing_devices(reports) == ["cuda (기기명 미기록)"]
    selected = select_reports(reports, device="cuda (기기명 미기록)")
    latency_models = sorted(model for kind, model in selected if kind == "latency")
    assert latency_models == sorted(reporting.MODELS)


def test_missing_created_at_falls_back_to_mtime_with_note():
    legacy = {"model": "light_rcnn", "device": "cpu", "p50_ms": 80.0}
    [lat] = normalize(legacy, "outputs/a.json", 0.0)
    assert lat.created_at == "1970-01-01T00:00:00+00:00"
    assert any("파일 수정 시각" in note for note in lat.notes)
    assert reporting.device_label(lat) == "cpu (기기명 미기록)"
    [mps] = normalize({"model": "light_rcnn", "device": "mps", "p50_ms": 9.0}, "b.json")
    assert reporting.device_label(mps) == "mps (기기명 미기록)"
    [v2] = normalize(latency_report(), "outputs/comparison_runs/20261019_225947/c.json")
    assert v2.created_at != "2026-10-19T22:59:47" and v2.notes == []
