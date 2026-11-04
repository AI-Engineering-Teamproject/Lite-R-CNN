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
    commands = {}  # in step order: (script, --model) -> command
    hints = {}

    def fake_step(label, command, env=None, required=True, hint=""):
        model = value(command, "--model") if "--model" in command else None
        commands[Path(command[1]).name, model] = command
        hints[Path(command[1]).name, model] = hint
        return True

    monkeypatch.setattr(module, "run_step", fake_step)

    def main(*argv):
        monkeypatch.setattr(sys, "argv", ["run_comparison.py", *argv, "--light-weights",
                                          str(weights), "--no-dashboard"])
        module.main()
        return commands

    main.hints = hints
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


def model_order(commands) -> list[str]:
    """Models in the order their steps ran (YOLO26's scripts take no --model)."""
    return list(dict.fromkeys(model or "yolo26" for script, model in commands
                              if script != "plot_results.py"))


def test_mobilenet_baseline_runs_right_after_faster_rcnn_by_default(run):
    commands = run("--device", "cuda")

    assert model_order(commands) == ["faster_rcnn", "mobilenet_faster_rcnn", "light_rcnn", "yolo26"]
    for script in ("evaluate_coco.py", "benchmark.py", "profile_stages.py"):
        command = commands[script, "mobilenet_faster_rcnn"]
        assert "--weights" not in command  # torchvision COCO_V1, as for Faster R-CNN
        assert value(command, "--config") == value(commands[script, "faster_rcnn"], "--config")
    assert timing(commands["benchmark.py", "mobilenet_faster_rcnn"]) == [50, 500, 64]
    assert value(commands["evaluate_coco.py", "mobilenet_faster_rcnn"], "--batch-size") == "1"
    # Its reports land in the same run folder the dashboard and the chart export open.
    outputs = {kind: Path(value(commands[script, "mobilenet_faster_rcnn"], "--output"))
               for kind, script in (("accuracy", "evaluate_coco.py"), ("latency", "benchmark.py"),
                                    ("stages", "profile_stages.py"))}
    run_folder = Path(value(commands["benchmark.py", "faster_rcnn"], "--output")).parent
    assert {path.parent for path in outputs.values()} == {run_folder}
    assert {kind: path.name for kind, path in outputs.items()} == {
        kind: f"mobilenet_faster_rcnn_{kind}.json" for kind in outputs}
    assert value(commands["plot_results.py", None], "--reports") == str(run_folder)


def test_models_always_run_baselines_first_with_their_own_weights(run, tmp_path):
    mobilenet = tmp_path / "mobilenet.pth"
    mobilenet.touch()
    commands = run("--models", "yolo26", "light_rcnn", "mobilenet_faster_rcnn", "faster_rcnn",
                   "--mobilenet-weights", str(mobilenet))

    assert model_order(commands) == ["faster_rcnn", "mobilenet_faster_rcnn", "light_rcnn", "yolo26"]
    for script in ("evaluate_coco.py", "benchmark.py", "profile_stages.py"):
        assert value(commands[script, "mobilenet_faster_rcnn"], "--weights") == str(mobilenet)
        assert "--weights" not in commands[script, "faster_rcnn"]
    hint = run.hints["benchmark.py", "mobilenet_faster_rcnn"]
    assert "skip MobileNetV3 Faster R-CNN with --models faster_rcnn light_rcnn yolo26" in hint
    assert "--mobilenet-weights" not in hint  # a local checkpoint was given
    assert "pass a local checkpoint with --faster-weights" in run.hints["benchmark.py",
                                                                        "faster_rcnn"]


def test_mobilenet_without_local_weights_hints_at_its_own_flag(run):
    run("--models", "mobilenet_faster_rcnn", "light_rcnn")

    hint = run.hints["evaluate_coco.py", "mobilenet_faster_rcnn"]
    assert "pass a local checkpoint with --mobilenet-weights" in hint
    assert "--faster-weights" not in hint
    assert "skip MobileNetV3 Faster R-CNN with --models light_rcnn" in hint


def test_missing_mobilenet_checkpoint_stops_before_measuring(run, tmp_path):
    with pytest.raises(SystemExit, match="MobileNetV3 Faster R-CNN checkpoint not found"):
        run("--models", "mobilenet_faster_rcnn", "--mobilenet-weights",
            str(tmp_path / "missing.pth"))
