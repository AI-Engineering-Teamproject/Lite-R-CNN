import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

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


def _run_plot(monkeypatch, tmp_path, *extra):
    spec = importlib.util.spec_from_file_location("plot_results_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    monkeypatch.setattr(module, "export_images", lambda *args, **kwargs: [])
    monkeypatch.setattr(module.viz, "html_report",
                        lambda figures, rows, warnings, selected, theme, title:
                        calls.append((selected, title)) or "")
    monkeypatch.setattr(sys, "argv", ["plot_results.py", "--reports", str(tmp_path / "runs"),
                                      "--out", str(tmp_path / "figures"), *extra])
    module.main()
    [(selected, title)] = calls
    return selected, title


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
