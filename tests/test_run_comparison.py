import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_comparison.py"


@pytest.fixture
def run(tmp_path, monkeypatch):
    """Run main() inside a scratch repository and record each step's command."""
    spec = importlib.util.spec_from_file_location("run_comparison", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    (tmp_path / "data/coco/images/val2017").mkdir(parents=True)
    (tmp_path / "data/coco/annotations").mkdir(parents=True)
    (tmp_path / "data/coco/annotations/instances_val2017.json").write_text("{}")
    weights = tmp_path / "light.pth"
    weights.touch()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    commands = {}

    def fake_step(label, command, env=None, required=True, hint=""):
        model = value(command, "--model") if "--model" in command else None
        commands[Path(command[1]).name, model] = command
        return True

    monkeypatch.setattr(module, "run_step", fake_step)

    def main(*argv):
        monkeypatch.setattr(sys, "argv", ["run_comparison.py", *argv, "--light-weights",
                                          str(weights), "--no-dashboard"])
        module.main()
        return commands

    return main


def value(command, flag):
    return command[command.index(flag) + 1]


def timing(command):
    return [int(value(command, flag)) for flag in ("--warmup", "--iterations", "--image-count")]


def test_default_cpu_run_times_with_cpu_preset_counts(run, capsys):
    commands = run()

    for step in (("benchmark.py", "faster_rcnn"), ("profile_stages.py", "light_rcnn"),
                 ("benchmark_yolo26.py", None)):
        assert timing(commands[step]) == [5, 30, 16]
        assert value(commands[step], "--config").endswith("configs/coco.yaml")
    out = capsys.readouterr().out
    assert "예열 5회, 반복 30회, 이미지 최대 16장 (cpu)" in out
    assert "--config configs/coco_cpu.yaml" in out


def test_explicit_count_overrides_cpu_preset(run):
    commands = run("--iterations", "100")

    assert timing(commands["benchmark.py", "light_rcnn"]) == [5, 100, 16]


def test_gpu_device_keeps_config_counts(run, capsys):
    commands = run("--device", "cuda")

    assert timing(commands["benchmark.py", "light_rcnn"]) == [50, 500, 64]
    assert "coco_cpu.yaml" not in capsys.readouterr().out


def test_explicit_gpu_config_on_cpu_warns_instead_of_switching(run, capsys):
    commands = run("--config", "configs/coco.yaml", "--models", "light_rcnn")

    assert timing(commands["profile_stages.py", "light_rcnn"]) == [50, 500, 64]
    assert "GPU 기준" in capsys.readouterr().err


def test_cpu_config_needs_no_fallback(run, capsys):
    commands = run("--config", "configs/coco_cpu.yaml", "--models", "yolo26")

    assert timing(commands["benchmark_yolo26.py", None]) == [5, 30, 16]
    captured = capsys.readouterr()
    assert "GPU 기준" not in captured.out + captured.err


def test_rcnn_accuracy_is_evaluated_at_comparison_batch_size(run):
    commands = run("--device", "cuda")

    for model in ("faster_rcnn", "light_rcnn"):
        assert value(commands["evaluate_coco.py", model], "--batch-size") == "1"
