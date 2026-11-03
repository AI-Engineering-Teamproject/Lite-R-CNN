import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from light_rcnn import viz
from light_rcnn.reporting import (
    SCOPE_HOST_TO_DETECTIONS,
    build_report,
    device_label,
    latency_summary,
    write_report,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "plot_results.py"

# Drops the editable-install entry so the script has to find src/ on its own.
RUNNER = """
import runpy, sys
from pathlib import Path
src = Path(sys.argv[1]).resolve()
sys.path[:] = [p for p in sys.path if not p or Path(p).resolve() != src]
sys.argv = [sys.argv[2], "--help"]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def test_plot_results_runs_without_editable_install(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", RUNNER, str(ROOT / "src"), str(SCRIPT)],
        check=False, cwd=tmp_path, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "--reports" in result.stdout


def _write_latency(root, name, model, device_name, device_type, created_at):
    report = build_report("latency", model, precision="fp32", image_size=640, created_at=created_at,
                          environment={"device": device_type, "device_type": device_type,
                                       "device_name": device_name},
                          latency={"scope_id": SCOPE_HOST_TO_DETECTIONS,
                                   **latency_summary([10.0 + i * 0.1 for i in range(200)])})
    write_report(report, root / name)


def _plot(monkeypatch, tmp_path, *extra) -> dict:
    """Run the script once; returns what it handed to the HTML report."""
    spec = importlib.util.spec_from_file_location("plot_results_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls, baselines = [], []
    build_all = module.viz.build_all
    monkeypatch.setattr(module, "export_images", lambda *args, **kwargs: [])
    monkeypatch.setattr(module.viz, "build_all", lambda selected, targets, theme, baseline, label:
                        baselines.append(baseline) or build_all(selected, targets, theme, baseline,
                                                                label))
    monkeypatch.setattr(module.viz, "html_report",
                        lambda charts, rows, warnings, selected, theme, title:
                        calls.append({"selected": selected, "title": title, "charts": charts,
                                      "warnings": warnings}) or "")
    monkeypatch.setattr(sys, "argv", ["plot_results.py", "--reports", str(tmp_path / "runs"),
                                      "--out", str(tmp_path / "figures"), *extra])
    module.main()
    [call] = calls
    return {**call, "baseline": baselines[0]}


def _run_plot(monkeypatch, tmp_path, *extra, figures=False):
    call = _plot(monkeypatch, tmp_path, *extra)
    if figures:
        return call["selected"], call["title"], call["charts"]
    return call["selected"], call["title"]


def _timing_devices(selected):
    return {device_label(r) for (kind, _), r in selected.items() if kind == "latency"}


def test_plot_results_uses_one_device_for_timings(monkeypatch, tmp_path):
    runs = tmp_path / "runs"
    gpu, cpu = "2026-10-19T00:00:00+00:00", "2026-10-25T00:00:00+00:00"
    _write_latency(runs, "gpu_light.json", "light_rcnn", "RTX 4090", "cuda", gpu)
    _write_latency(runs, "gpu_yolo.json", "yolo26", "RTX 4090", "cuda", gpu)
    _write_latency(runs, "cpu_light.json", "light_rcnn", "AMD EPYC", "cpu", cpu)

    # Default: the newest device only, so the CPU run never shares a chart with the GPU runs.
    selected, title = _run_plot(monkeypatch, tmp_path)
    assert _timing_devices(selected) == {"AMD EPYC"}
    assert ("latency", "yolo26") not in selected
    assert title.endswith("(AMD EPYC)")

    selected, title = _run_plot(monkeypatch, tmp_path, "--device", "RTX 4090")
    assert _timing_devices(selected) == {"RTX 4090"}
    assert ("latency", "yolo26") in selected and title.endswith("(RTX 4090)")

    with pytest.raises(SystemExit, match="AMD EPYC"):
        _run_plot(monkeypatch, tmp_path, "--device", "Tesla T4")


def test_plot_results_leaves_out_old_v1_gpu_run_by_default(monkeypatch, tmp_path):
    # outputs/ keeps the v1 RTX 4090 run, whose Faster R-CNN cannot be re-measured offline.
    old_run = tmp_path / "runs" / "comparison_runs" / "20261019_225947"
    old_run.mkdir(parents=True)
    (old_run / "faster_rcnn_latency.json").write_text(
        json.dumps({"model": "faster_rcnn", "device": "cuda", "p50_ms": 7.7, "p95_ms": 9.0}),
        encoding="utf-8")
    _write_latency(tmp_path / "runs", "cpu_light.json", "light_rcnn", "AMD EPYC", "cpu",
                   "2026-10-25T00:00:00+00:00")

    selected, title = _run_plot(monkeypatch, tmp_path)
    assert _timing_devices(selected) == {"AMD EPYC"}
    assert ("latency", "faster_rcnn") not in selected
    assert title.endswith("(AMD EPYC)")

    selected, _ = _run_plot(monkeypatch, tmp_path, "--device", "cuda (기기명 미기록)")
    assert ("latency", "faster_rcnn") in selected and ("latency", "light_rcnn") not in selected


def test_plot_results_draws_one_inference_preset_at_a_time(monkeypatch, tmp_path):
    # compare_improvements.py folders hold every preset; the newest one must not win each chart.
    environment = {"device": "cpu", "device_type": "cpu", "device_name": "AMD EPYC"}
    for preset, label, created_at, ap, p50 in (
            ("baseline", "개선 전 (기본 설정)", "2026-11-01T00:00:00+00:00", 0.30, 20.0),
            ("fast", "빠른 설정", "2026-11-01T00:05:00+00:00", 0.31, 12.0)):
        common = {"precision": "fp32", "image_size": 640, "created_at": created_at,
                  "environment": environment, "variant": preset, "variant_label": label}
        write_report(build_report("accuracy", "light_rcnn", **common,
                                  accuracy={"AP_50_95": ap, "images_evaluated": 5000,
                                            "is_partial": False}),
                     tmp_path / "runs" / f"accuracy_{preset}.json")
        write_report(build_report("latency", "light_rcnn", **common,
                                  latency={"scope_id": SCOPE_HOST_TO_DETECTIONS,
                                           **latency_summary([p50] * 50)}),
                     tmp_path / "runs" / f"latency_{preset}.json")

    selected, title, charts = _run_plot(monkeypatch, tmp_path, figures=True)
    assert {report.variant for report in selected.values()} == {"baseline"}
    assert title.endswith("(AMD EPYC)") and "improvement" not in charts

    selected, title, charts = _run_plot(monkeypatch, tmp_path, "--variant", "fast", figures=True)
    assert {report.variant for report in selected.values()} == {"fast"}
    assert title.endswith("(빠른 설정 · AMD EPYC)")
    assert not viz.is_empty(charts["improvement"])  # the before/after chart against baseline

    with pytest.raises(SystemExit, match="baseline, fast"):
        _run_plot(monkeypatch, tmp_path, "--variant", "accurate")


def _write_compare_run(root, presets):
    """compare_improvements.py's layout: reports/v2 and reports/v3 with the same Faster R-CNN files;
    both Light R-CNN versions share the 'light_rcnn' key. AP and p50 differ per checkpoint."""
    environment = {"device": "cpu", "device_type": "cpu", "device_name": "AMD EPYC"}
    for preset, versions in presets.items():
        for version in versions:
            created_at = f"2026-11-0{version[-1]}T00:00:00+00:00"
            for model in ("faster_rcnn", "light_rcnn"):
                light = model == "light_rcnn"
                ap = {"v2": 0.40, "v3": 0.30}[version] + (0.05 if preset != "baseline" else 0)
                common = {"precision": "fp32", "image_size": 640, "environment": environment,
                          "created_at": created_at if light else "2026-11-01T00:00:00+00:00",
                          "variant": preset, "comparison_run": "colab_20261105",
                          "model_label": f"Light R-CNN {version}" if light else "Vanilla Faster R-CNN",
                          "weights": (f"outputs/coco_4090_{version}/best.pth" if light
                                      else "torchvision_coco_v1"),
                          "weights_sha256": version[-1] * 64 if light else None}
                folder = root / "reports" / version
                write_report(build_report("accuracy", model, **common,
                                          accuracy={"AP_50_95": ap if light else 0.37,
                                                    "images_evaluated": 5000, "is_partial": False}),
                             folder / f"{model}_accuracy_{preset}.json")
                p50 = {"v2": 40.0, "v3": 30.0}[version] if light else 900.0
                write_report(build_report("latency", model, **common,
                                          latency={"scope_id": SCOPE_HOST_TO_DETECTIONS,
                                                   **latency_summary([p50] * 120)}),
                             folder / f"{model}_latency_{preset}.json")


def _light(selected) -> dict[str, str]:
    return {kind: report.data["model_label"] for (kind, model), report in selected.items()
            if model == "light_rcnn"}


def test_plot_results_draws_one_light_rcnn_checkpoint(monkeypatch, tmp_path, capsys):
    _write_compare_run(tmp_path / "runs", {"baseline": ("v2", "v3"), "fast": ("v2", "v3"),
                                           "accurate": ("v2",)})
    # Several checkpoints and none chosen: warn and draw the newest, as the dashboard does.
    call = _plot(monkeypatch, tmp_path)
    assert _light(call["selected"]) == {"accuracy": "Light R-CNN v3", "latency": "Light R-CNN v3"}
    assert call["title"] == "Light R-CNN 성능 비교 보고서 (Light R-CNN v3 · AMD EPYC)"
    note = ("결과 폴더에 Light R-CNN 체크포인트가 2개(Light R-CNN v3, Light R-CNN v2) 섞여 있어 가장 최근 "
            "체크포인트인 Light R-CNN v3만 그렸습니다. 다른 체크포인트는 --checkpoint로 고르세요.")
    assert (call["warnings"][0].level, call["warnings"][0].message) == ("warning", note)
    out = capsys.readouterr().out
    assert f"[warning] {note}" in out
    assert ("[info] Light R-CNN 체크포인트: Light R-CNN v3 (다른 체크포인트는 --checkpoint로 선택: "
            "Light R-CNN v2)") in out
    drawn = [fig for fig in call["charts"].values() if not viz.is_empty(fig)]
    assert drawn and all(fig.layout.title.text.endswith("</b> · Light R-CNN v3") for fig in drawn)

    # Chosen by a label part: before and after both come from that checkpoint.
    call = _plot(monkeypatch, tmp_path, "--checkpoint", "V2", "--variant", "fast")
    assert _light(call["selected"]) == {"accuracy": "Light R-CNN v2", "latency": "Light R-CNN v2"}
    assert call["title"] == "Light R-CNN 성능 비교 보고서 (Light R-CNN v2 · fast · AMD EPYC)"
    assert not any("체크포인트가" in w.message for w in call["warnings"])
    assert _light(call["baseline"]) == {"accuracy": "Light R-CNN v2", "latency": "Light R-CNN v2"}
    assert {report.variant for report in call["baseline"].values()} == {"baseline"}
    assert call["charts"]["improvement"].layout.title.text.endswith(" · Light R-CNN v2")
    assert "[warning]" not in capsys.readouterr().out.split("wrote")[0]
    # Chosen by a key: the weights hash or its start.
    assert _light(_plot(monkeypatch, tmp_path, "--checkpoint", "3" * 64)["selected"]) == {
        "accuracy": "Light R-CNN v3", "latency": "Light R-CNN v3"}
    assert _light(_plot(monkeypatch, tmp_path, "--checkpoint", "2222")["selected"])["latency"] == (
        "Light R-CNN v2")

    # A preset only one checkpoint has: the newest checkpoint measured with it.
    call = _plot(monkeypatch, tmp_path, "--variant", "accurate")
    assert _light(call["selected"]) == {"accuracy": "Light R-CNN v2", "latency": "Light R-CNN v2"}
    assert "이 추론 설정(accurate)으로 측정한 것 중 가장 최근 체크포인트인 Light R-CNN v2만" in (
        call["warnings"][0].message)
    for args, error in ((("--checkpoint", "v3", "--variant", "accurate"), "'Light R-CNN v3' has no"),
                        (("--checkpoint", "light r-cnn"), "matches several checkpoints"),
                        (("--checkpoint", "v9"), "Light R-CNN v3 \\(333333333333\\)")):
        with pytest.raises(SystemExit, match=error):
            _plot(monkeypatch, tmp_path, *args)


def test_plot_results_keeps_one_checkpoint_folders_as_they_were(monkeypatch, tmp_path):
    _write_compare_run(tmp_path / "runs", {"baseline": ("v2",)})
    call = _plot(monkeypatch, tmp_path)
    assert call["title"] == "Light R-CNN 성능 비교 보고서 (AMD EPYC)"
    assert not any("체크포인트" in w.message for w in call["warnings"])
    assert all("Light R-CNN v2" not in fig.layout.title.text for fig in call["charts"].values())
    # Asking for it by name still labels the charts.
    call = _plot(monkeypatch, tmp_path, "--checkpoint", "v2")
    assert call["title"] == "Light R-CNN 성능 비교 보고서 (Light R-CNN v2 · AMD EPYC)"


def test_plot_results_pairs_a_preset_with_its_own_runs_baseline(monkeypatch, tmp_path):
    # A smoke run (128 images) next to v2's full-val2017 import, as under outputs/: the import
    # outranks the run's baseline, yet the before/after chart must use the run's own.
    runs = tmp_path / "runs"
    _write_compare_run(runs, {"baseline": ("v2",), "fast": ("v2",)})
    for path in (runs / "reports" / "v2").glob("light_rcnn_accuracy_*.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        report["accuracy"].update(images_evaluated=128, is_partial=True)
        path.write_text(json.dumps(report), encoding="utf-8")
    write_report(build_report("accuracy", "light_rcnn", precision="fp32", image_size=640,
                              batch_size=4, weights="outputs/coco_4090_v2/best.pth",
                              weights_sha256="2" * 64, created_at="2026-10-19T00:00:00+00:00",
                              accuracy={"AP_50_95": 0.288, "images_evaluated": 5000,
                                        "is_partial": False}),
                 runs / "reports" / "light_rcnn_accuracy_training.json")
    call = _plot(monkeypatch, tmp_path, "--variant", "fast")
    before = call["baseline"]["accuracy", "light_rcnn"]
    assert (before.data["comparison_run"], before.section["images_evaluated"]) == ("colab_20261105", 128)
    assert not any("측정 조건이 다릅니다" in warning.message for warning in call["warnings"])


def test_plot_results_picks_a_checkpoint_by_a_weights_path_it_records(monkeypatch, tmp_path):
    _write_compare_run(tmp_path / "runs", {"baseline": ("v2", "v3")})
    for wanted in ("outputs/coco_4090_v2/best.pth", "outputs\\coco_4090_v2\\best.pth",
                   "./outputs/coco_4090_v2/best.pth",
                   "C:\\Users\\82103\\Desktop\\Lite-R-CNN\\outputs\\coco_4090_v2\\best.pth"):
        assert _light(_plot(monkeypatch, tmp_path, "--checkpoint", wanted)["selected"]) == {
            "accuracy": "Light R-CNN v2", "latency": "Light R-CNN v2"}, wanted
    with pytest.raises(SystemExit, match="matches several checkpoints"):
        _plot(monkeypatch, tmp_path, "--checkpoint", "best.pth")


def test_plot_results_names_path_only_checkpoints_by_a_path_that_matches(monkeypatch, tmp_path):
    # The repo's outputs/: old-format v3 files record the 4090's absolute path and a Windows one.
    home = "/home/user/Desktop/Lite-R-CNN/outputs"
    for name, weights, p50 in (("a_v3.json", f"{home}/coco_4090_v3/best.pth", 50.0),
                               ("b_v3.json", "outputs\\coco_4090_v3\\best.pth", 51.0),
                               ("c_v2.json", f"{home}/coco_4090_v2/best.pth", 40.0)):
        (tmp_path / "runs").mkdir(exist_ok=True)
        (tmp_path / "runs" / name).write_text(json.dumps({
            "model": "light_rcnn", "device": "cpu", "iterations": 200, "p50_ms": p50,
            "weights": weights}), encoding="utf-8")
    for wanted in ("outputs\\coco_4090_v3\\best.pth", "outputs/coco_4090_v3/best.pth"):
        call = _plot(monkeypatch, tmp_path, "--checkpoint", wanted)
        assert call["selected"]["latency", "light_rcnn"].section["p50_ms"] in (50.0, 51.0)  # v3
        assert "coco_4090_v3/best" in call["title"]
    with pytest.raises(SystemExit) as error:
        _plot(monkeypatch, tmp_path, "--checkpoint", "v9")
    assert "coco_4090_v3/best (outputs/coco_4090_v3/best.pth)" in str(error.value)
    assert "(/home/user/D)" not in str(error.value)
