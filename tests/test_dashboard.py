import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest

from light_rcnn import reporting
from light_rcnn.reporting import build_report, latency_summary, write_report

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")
CPU = {"device": "cpu", "device_type": "cpu", "device_name": "Xeon", "torch_num_threads": 4}
GPU = {"device": "cuda:0", "device_type": "cuda", "device_name": "NVIDIA GeForce RTX 4090"}
DEVICE_LABEL = "지연 시간 측정 장치"


@pytest.fixture(autouse=True)
def results_dir(monkeypatch, tmp_path):
    """Default folder for runs that do not type one; keeps tests off the repo's outputs/."""
    folder = tmp_path / "default_results"
    monkeypatch.setenv("LIGHT_RCNN_RESULTS_DIR", str(folder))
    return folder


def write_results(root: Path) -> None:
    scopes = {"light_rcnn": reporting.COLD_START_FIRST_FORWARD,
              "yolo26": reporting.COLD_START_FIRST_PREDICT}
    stats = {"light_rcnn": {"parameters_m": 3.26, "gflops": 5.49},
             # older YOLO26 report: parameters_m/gflops are the checkpoint, deployed_* is timed
             "yolo26": {"parameters_m": 2.57, "gflops": 6.24, "deployed_parameters_m": 2.41,
                        "deployed_gflops": 5.48}}
    for index, model in enumerate(("light_rcnn", "yolo26")):
        latency = latency_summary([30.0 + index * 10 + i % 5 for i in range(150)],
                                  cold_start_ms=900.0)
        latency.update(scope_id=reporting.SCOPE_HOST_TO_DETECTIONS, cold_start_scope=scopes[model],
                       mean_detections=0.0 if model == "light_rcnn" else 2.5)
        if model == "yolo26":
            latency["p50_ms"] = str(latency["p50_ms"])  # hand-edited JSON
        write_report(build_report("latency", model, precision="fp32", image_size=640,
                                  environment=CPU, latency=latency, model_stats=stats[model]),
                     root / f"{model}_latency.json")
        write_report(build_report(
            "accuracy", model, precision="fp32", image_size=640, environment=CPU,
            accuracy={"evaluator": "pycocotools", "images_evaluated": "128", "images_total": 128,
                      "AP_50_95": 0.2 + index / 10, "AP_50": 0.4,
                      "per_category_ap": {"person": 0.5 + index / 10, "dog": 0.2,
                                          "cat": 0.4 - index / 10}}),
            root / f"{model}_accuracy.json")
    write_report(build_report("stages", "light_rcnn", environment=CPU, stages={"items": None}),
                 root / "light_rcnn_stages.json")


def write_gpu_latency(root: Path, model: str = "light_rcnn", environment: dict = GPU,
                      created_at: str = "2999-01-01T00:00:00+00:00") -> None:
    """A newer GPU run in its own per-run folder, as run_comparison.py lays them out."""
    latency = latency_summary([4.0 + i % 3 * 0.1 for i in range(150)], cold_start_ms=80.0)
    latency.update(scope_id=reporting.SCOPE_HOST_TO_DETECTIONS,
                   cold_start_scope=reporting.COLD_START_FIRST_FORWARD, mean_detections=3.0)
    report = build_report("latency", model, precision="fp32", image_size=640,
                          environment=environment, latency=latency)
    report["created_at"] = created_at
    run = "".join(ch for ch in created_at[:10] if ch.isdigit()) + "_000000"
    write_report(report, root / "comparison_runs" / run / f"{model}_latency.json")


def run_app(folder: str | None = None) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    if folder is not None:
        app.text_input[0].set_value(folder).run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def device_box(app: AppTest):
    return next(box for box in app.sidebar.selectbox if box.label == DEVICE_LABEL)


def metric_values(app: AppTest, suffix: str) -> list[str]:
    return [m.value for m in app.metric if m.label.endswith(suffix)]


def test_cpu_kpis_tables_and_robust_values(tmp_path):
    write_results(tmp_path)
    app = run_app(str(tmp_path))
    metrics = [m for m in app.metric if m.label.endswith("지연")]
    assert len(metrics) == 2
    assert all(m.proto.color == m.proto.MetricColor.GRAY for m in metrics)  # not red
    captions = [c.value for c in app.caption]
    assert captions.count("CPU 측정 · 목표(T4 GPU)와 직접 비교 불가") == 2
    assert captions.count("측정 장치: Xeon") == 2
    assert any("파라미터 2.4M" in c and "평가 128장" in c for c in captions)
    assert any(c.startswith("첫 추론 범위:") and "직접 비교할 수 없습니다" in c for c in captions)
    frames = {tuple(df.value.columns): df.value for df in app.dataframe}
    latency = next(frame for columns, frame in frames.items() if "평균 검출 수" in columns)
    assert list(latency["평균 검출 수"]) == [0.0, 2.5]
    per_class = next(frame for columns, frame in frames.items() if columns[0] == "클래스")
    assert set(per_class["클래스"]) == {"person", "dog", "cat"}
    assert "모델 간 차이 (%p)" in per_class.columns
    assert "클래스별 AP 표" in [e.label for e in app.expander]
    assert not any("정확도(AP) 결과가 없어" in i.value for i in app.info)


def test_device_selector_never_mixes_cpu_and_gpu_timings(tmp_path):
    write_results(tmp_path)
    write_gpu_latency(tmp_path)
    app = run_app(str(tmp_path))
    box = device_box(app)
    assert box.options == ["모델 1개 · NVIDIA GeForce RTX 4090", "모델 2개 · Xeon"]  # newest first
    assert box.value == "NVIDIA GeForce RTX 4090"
    assert not any("서로 다른 장치" in e.value for e in app.error)
    captions = [c.value for c in app.caption]
    assert "측정 장치: NVIDIA GeForce RTX 4090" in captions
    # The latency target is set for a T4, so a 4090 timing is a reference, not a pass.
    assert "T4 외 장치 · 목표(T4 GPU)와 직접 비교 불가" in captions
    gpu_metric = next(m for m in app.metric if m.label.endswith("지연") and m.value == "4.1 ms")
    assert gpu_metric.proto.color == gpu_metric.proto.MetricColor.GRAY
    assert "이 장치에서 측정한 결과 없음" in captions  # YOLO26 was only timed on the CPU
    assert metric_values(app, "지연") == ["4.1 ms", "–"]
    # Accuracy does not depend on the device, so both AP cards stay filled.
    assert metric_values(app, "AP@[.50:.95]") == ["20.0%", "30.0%"]

    box.set_value("Xeon").run()
    assert not app.exception
    assert [c.value for c in app.caption].count("측정 장치: Xeon") == 2
    assert not any("서로 다른 장치" in e.value for e in app.error)
    assert metric_values(app, "지연") == ["32.0 ms", "42.0 ms"]


def test_default_folder_comes_from_env(results_dir):
    write_results(results_dir)
    app = run_app()
    assert app.text_input[0].value == str(results_dir)
    assert device_box(app).value == "Xeon"
    assert app.get("plotly_chart")


def test_legacy_v1_latency_files(tmp_path):
    run = tmp_path / "comparison_runs" / "20261019_225947"
    run.mkdir(parents=True)
    for name, raw in {
        "light_rcnn_latency.json": {"model": "light_rcnn", "device": "cuda", "precision": "fp32",
                                    "image_size": 640, "iterations": 500, "p50_ms": 4.26,
                                    "p95_ms": 5.2},
        "faster_rcnn_latency.json": {"model": "faster_rcnn", "device": "cuda", "precision": "fp32",
                                     "image_size": 640, "iterations": 500, "p50_ms": 21.0,
                                     "p95_ms": 23.0},
        # the old run_comparison passed the CUDA index to Ultralytics, as in the committed run
        "yolo26_latency.json": {"model_family": "YOLO26", "imgsz": 640, "device": "0",
                                "precision": "fp32", "iterations": 500, "p50_ms": 3.2,
                                "p95_ms": 8.5},
    }.items():
        (run / name).write_text(json.dumps(raw), encoding="utf-8")
    app = run_app(str(tmp_path))
    box = device_box(app)
    assert box.options == ["모델 3개 · cuda (기기명 미기록)"]
    assert box.value == "cuda (기기명 미기록)"
    assert metric_values(app, "지연") == ["21.0 ms", "4.3 ms", "3.2 ms"]
    captions = [c.value for c in app.caption]
    assert "측정 장치: cuda (기기명 미기록)" in captions
    assert captions.count("T4 여부 미기록 · 목표(T4 GPU)와 직접 비교 불가") == 3
    assert all(m.proto.color == m.proto.MetricColor.GRAY
               for m in app.metric if m.label.endswith("지연"))
    assert any("import_training_results.py" in c.value for c in app.code)


def test_t4_timings_are_graded_against_the_target(tmp_path):
    write_gpu_latency(tmp_path, environment={**GPU, "device_name": "Tesla T4"})
    app = run_app(str(tmp_path))
    metric = next(m for m in app.metric if m.label.endswith("지연") and m.value != "–")
    assert metric.proto.color == metric.proto.MetricColor.GREEN  # 4.1 ms is under 15 ms
    assert not any("직접 비교 불가" in c.value for c in app.caption)


def test_card_points_to_other_devices_and_missing_accuracy(tmp_path):
    xeon = {**CPU, "device_name": "Intel(R) Xeon(R) Processor @ 2.10GHz"}
    write_gpu_latency(tmp_path)
    write_gpu_latency(tmp_path, "yolo26", environment=xeon,
                      created_at="2998-01-01T00:00:00+00:00")
    app = run_app(str(tmp_path))
    box = device_box(app)
    assert box.options == ["모델 1개 · NVIDIA GeForce RTX 4090",
                           "모델 1개 · Intel Xeon Processor @ 2.10GHz"]
    assert "NVIDIA GeForce RTX 4090" in [c.value for c in app.sidebar.caption]
    captions = [c.value for c in app.caption]
    assert ("이 장치에서 측정한 결과 없음 · 사이드바에서 다른 장치 선택: "
            "Intel(R) Xeon(R) Processor @ 2.10GHz") in captions  # YOLO26, timed only on the CPU
    assert "결과 없음" in captions  # Faster R-CNN was never measured
    assert any("정확도(AP) 결과가 없어" in i.value for i in app.info)


def test_no_results_disables_device_selector(tmp_path):
    app = run_app(str(tmp_path))
    box = device_box(app)
    assert box.value is None and box.disabled
    assert any("아직 비교 결과가 없습니다" in i.value for i in app.info)


@pytest.mark.parametrize("targets, warned", [
    ({"latency_ms": 15, "latency_percentile": "p99", "ap_metric": "AP_50", "ap": 60}, False),
    ({"latency_ms": "fast", "latency_percentile": "p75", "ap_metric": "mAP", "ap": 0.6}, True),
])
def test_config_targets_outside_widget_options(monkeypatch, targets, warned):
    monkeypatch.setattr(reporting, "load_targets", lambda path: dict(targets))
    app = run_app()
    sidebar_warnings = [w.value for w in app.sidebar.warning]
    assert bool(sidebar_warnings) == warned
    accuracy_target = next(n for n in app.number_input if n.label == "정확도 목표 (%)")
    assert accuracy_target.value == 60.0
    if warned:
        assert "latency_percentile='p75'" in sidebar_warnings[0]
        assert app.segmented_control[0].value == "p50"
    else:
        assert app.segmented_control[0].value == "p99"


def test_quoted_and_missing_results_folder(tmp_path):
    write_results(tmp_path)
    quoted = run_app(f'"{tmp_path}"')
    assert quoted.get("plotly_chart")
    assert not any("찾을 수 없습니다" in w.value for w in quoted.warning)
    missing = run_app(str(tmp_path / "nope"))
    assert any("결과 폴더를 찾을 수 없습니다" in w.value for w in missing.warning)
    assert not missing.info
