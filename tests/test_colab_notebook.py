import ast
import contextlib
import importlib.util
import json
import os
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import light_rcnn.inference_opts  # noqa: F401 - the step B cell imports it; load it up front

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/Lite_RCNN_T4_Colab.ipynb"
SIMULATED = 1_000_000.0  # file times the tests set; a mark written by the notebook is far newer
DEFAULT_A = ["faster_rcnn", "mobilenet_faster_rcnn", "light_rcnn", "yolo26"]
DEFAULT_B = ["faster_rcnn", "mobilenet", "v2", "v3"]


def cells() -> dict[str, dict]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return {cell["id"]: {"type": cell["cell_type"], "source": "".join(cell["source"])}
            for cell in notebook["cells"]}


def run_code(code, namespace: dict) -> dict:
    exec(code, namespace)  # noqa: S102 - the notebook's own cells
    return namespace


def run_parts(source: str, namespace: dict, functions=(), names=(), mentions=()) -> dict:
    """Runs only the cell's top-level defs in `functions`, assignments to `names` and other
    statements mentioning one of `mentions` (cell 2 clones and installs, so never all of it)."""
    picked = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef):
            keep = node.name in functions
        elif isinstance(node, ast.Assign):
            keep = any(isinstance(target, ast.Name) and target.id in names
                       for target in node.targets)
        else:
            keep = any(word in ast.get_source_segment(source, node) for word in mentions)
        if keep:
            picked.append(node)
    return run_code(compile(ast.Module(body=picked, type_ignores=[]), str(NOTEBOOK), "exec"),
                    namespace)


def write(path: Path, text: str, mtime: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def wait_for(condition, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def text_of(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.is_file() else None


def finish_measurement(folder: Path, text: str, mtime: float) -> None:
    """Replaces summary.csv in one step (one mtime change), as after each finished measurement."""
    os.replace(write(folder.parent / "summary.staged", text, mtime), folder / "summary.csv")


def mirror_function(**modules):
    namespace = {"contextlib": contextlib, "shutil": shutil, **modules}
    return run_parts(cells()["setup"]["source"], namespace,
                     functions={"mirror_to_drive"})["mirror_to_drive"]


def settings() -> dict:
    """The settings form's defaults (the cell only assigns values)."""
    return run_code(compile(cells()["params"]["source"], "params", "exec"), {})


def script_module(name: str, monkeypatch):
    """scripts/<name>.py loaded for its constants and parser."""
    monkeypatch.setattr(sys, "path", list(sys.path))  # compare_improvements adds src/, scripts/
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)  # dataclasses look the module up
    spec.loader.exec_module(module)
    return module


def option(command, flag):
    """The values after ``flag`` up to the next option."""
    start = command.index(flag) + 1
    end = next((index for index in range(start, len(command))
                if str(command[index]).startswith("--")), len(command))
    return [str(part) for part in command[start:end]]


def with_model_choice(namespace: dict, with_mobilenet: bool) -> dict:
    """The setup cell's chosen_models, as the step cells call it."""
    namespace["WITH_MOBILENET"] = with_mobilenet
    return run_parts(cells()["setup"]["source"], namespace, functions={"chosen_models"},
                     names={"MOBILENET_MODELS"})


def test_every_code_cell_compiles():
    for cell_id, cell in cells().items():
        if cell["type"] == "code":
            compile(cell["source"], f"{NOTEBOOK.name}:{cell_id}", "exec")


def test_results_zip_leaves_out_quick_test_runs_and_keeps_earlier_full_steps(tmp_path):
    setup = cells()["setup"]["source"]
    namespace = run_parts(setup, {"Path": Path, "REPO_DIR": tmp_path / "Lite-R-CNN"},
                          functions={"new_result_files"},
                          names={"STARTED_MARK", "RESULT_FOLDERS"})
    mark, clock = namespace["STARTED_MARK"], [SIMULATED]

    def tick() -> float:
        clock[0] += 100
        return clock[0]

    def start(run_id: str, quick: bool) -> bool:
        """Cell 2 of a new 'Run all'; True when results.zip starts counting afresh."""
        namespace.update(RUN_ID=run_id, QUICK_TEST=quick)
        run_parts(setup, namespace, names={"LOG_DIR", "MODE"}, mentions=("STARTED_MARK",))
        restarted = mark.stat().st_mtime > 2 * SIMULATED
        if restarted:
            moment = tick()
            os.utime(mark, (moment, moment))
        return restarted

    def produce(*names: str) -> list[Path]:
        """Files a step writes: logs into this run's log folder, results under outputs/."""
        moment = tick()
        return [write(namespace["LOG_DIR"] / name if name.endswith(".log")
                      else namespace["REPO_DIR"] / name, "x", moment) for name in names]

    write(mark, "", SIMULATED)  # touched by an older notebook: no mode recorded
    assert start("q1", quick=True)
    quick = produce("outputs/comparison_runs/q1/faster_rcnn_accuracy.json", "step_a_v2.log")
    assert namespace["new_result_files"]() == sorted(quick)
    assert not start("q2", quick=True)

    assert start("f1", quick=False)  # quick test -> full run: the quick files stay out
    full = produce("outputs/comparison_runs/f1/faster_rcnn_accuracy.json", "step_a_v2.log")
    assert namespace["new_result_files"]() == sorted(full)

    assert not start("f2", quick=False)  # same mode: steps finished earlier stay in
    full += produce("outputs/evaluation_runs/colab_f2/REPORT.md", "step_c.log")
    assert namespace["new_result_files"]() == sorted(full)
    assert namespace["LOG_DIR"] == tmp_path / "Lite-R-CNN/outputs/colab_logs/f2"


def test_mirror_to_drive_copies_each_finished_run_then_the_final_state(tmp_path):
    copies = []

    def copytree(source, target, dirs_exist_ok):
        shutil.copytree(source, target, dirs_exist_ok=dirs_exist_ok)
        copies.append(text_of(target / "summary.csv"))

    folder, target = tmp_path / "colab_1", tmp_path / "drive/step_b_1"
    write(folder / "reports/v2/light_rcnn_accuracy_baseline.json", "{}")
    stop = threading.Event()
    thread = threading.Thread(target=mirror_function(shutil=SimpleNamespace(copytree=copytree)),
                              args=(folder, target, stop), kwargs={"interval": 0.01}, daemon=True)
    thread.start()
    try:
        time.sleep(0.1)
        assert not target.exists()  # no summary.csv yet: nothing has finished
        finish_measurement(folder, "run 1", 1000)
        wait_for(lambda: copies == ["run 1"])
        assert (target / "reports/v2/light_rcnn_accuracy_baseline.json").is_file()
        finish_measurement(folder, "run 2", 2000)
        wait_for(lambda: copies == ["run 1", "run 2"])
        write(folder / "figures/improvement.html", "graph")  # figures come after the last run
        time.sleep(0.1)
        assert len(copies) == 2  # copies only when a measurement finishes
    finally:
        stop.set()
        thread.join(5)
    assert not thread.is_alive()
    assert len(copies) == 3 and text_of(target / "figures/improvement.html") == "graph"


def test_mirror_to_drive_skips_a_step_that_finished_nothing(tmp_path):
    stop = threading.Event()
    stop.set()
    mirror_function()(tmp_path / "colab_1", tmp_path / "drive/step_b_1", stop, interval=0.01)
    assert not (tmp_path / "drive").exists()


def test_mirror_to_drive_retries_after_a_drive_error(tmp_path):
    calls = []

    def flaky_copytree(source, target, dirs_exist_ok):
        calls.append(target)
        if len(calls) == 1:
            raise OSError("Transport endpoint is not connected")
        return shutil.copytree(source, target, dirs_exist_ok=dirs_exist_ok)

    folder, target = tmp_path / "colab_1", tmp_path / "drive/step_b_1"
    write(folder / "summary.csv", "run 1")
    stop = threading.Event()
    thread = threading.Thread(
        target=mirror_function(shutil=SimpleNamespace(copytree=flaky_copytree)),
        args=(folder, target, stop), kwargs={"interval": 0.01}, daemon=True)
    thread.start()
    try:
        wait_for(lambda: text_of(target / "summary.csv") == "run 1")
    finally:
        stop.set()
        thread.join(5)
    assert len(calls) >= 2


class Interrupted(Exception):
    """Stands in for the KeyboardInterrupt that Colab's stop button raises inside sh()."""


def run_step_b(tmp_path, monkeypatch, drive=True, quick=False, fail=False, models="v2 v3",
               with_mobilenet=True):
    """Runs the real step B cell with the measurement replaced by a stub that writes a summary."""
    monkeypatch.chdir(tmp_path)
    repo, drive_dir = tmp_path / "Lite-R-CNN", tmp_path / "drive"
    repo.mkdir()
    drive_dir.mkdir()
    before = set(threading.enumerate())
    copying, packed, shown, commands = [], [], [], []

    def fake_run_step(key, command, log_name):
        commands.append(command)
        out_dir = repo / command[command.index("--output-dir") + 1]
        write(out_dir / "summary.csv", "model,preset\n")
        write(out_dir / "summary.md", "# 추론 설정 개선 전/후 비교\n")
        copying.extend(thread for thread in threading.enumerate() if thread not in before)
        if fail:
            raise Interrupted
        namespace["STATUS"][key] = {"status": "완료", "minutes": 0.0}
        return True

    namespace = {
        "contextlib": contextlib, "datetime": datetime, "os": os, "shutil": shutil, "sys": sys,
        "threading": threading, "REPO_DIR": repo, "DEVICE": "cpu", "STEP_B_MODELS": models,
        "STEP_B_PRESETS": "deploy,fast", "LIMIT_ARGS": [], "TIMING_B": [], "RUN_STEP_B": True,
        "STATUS": {}, "DRIVE_DIR": drive_dir if drive else None, "QUICK_TEST": quick,
        "run_step": fake_run_step, "safely": lambda label, func, *args: func(*args),
        "show_markdown": shown.append, "pack_results": lambda: packed.append(True),
        "commands": commands,
    }
    run_parts(cells()["setup"]["source"], namespace, functions={"mirror_to_drive"})
    with_model_choice(namespace, with_mobilenet)
    code = compile(cells()["step-b"]["source"], "step-b", "exec")
    with pytest.raises(Interrupted) if fail else contextlib.nullcontext():
        run_code(code, namespace)
    for thread in copying:
        thread.join(5)
    assert not any(thread.is_alive() for thread in copying)  # copying stops with the step
    assert [path.name for path in shown] == ([] if fail else ["summary.md"])
    return namespace, sorted(drive_dir.iterdir()), copying, packed


@pytest.mark.parametrize("quick", [False, True])
def test_step_b_copies_to_drive_while_it_runs(tmp_path, monkeypatch, quick):
    namespace, copies, copying, packed = run_step_b(tmp_path, monkeypatch, quick=quick)
    assert len(copying) == 1 and packed == [True]
    folder = tmp_path / "Lite-R-CNN" / namespace["STATUS"]["B"]["folder"]
    stamp = folder.name.removeprefix("colab_")
    assert [copy.name for copy in copies] == [f"step_b_{'quick_' if quick else ''}{stamp}"]
    assert sorted(path.name for path in copies[0].iterdir()) == ["summary.csv", "summary.md"]


def test_step_b_stops_copying_when_the_stop_button_is_pressed(tmp_path, monkeypatch):
    namespace, copies, copying, packed = run_step_b(tmp_path, monkeypatch, fail=True)
    assert len(copying) == 1 and not packed and "B" not in namespace["STATUS"]
    # The measurements finished before the stop are still copied once more on the way out.
    assert sorted(path.name for path in copies[0].iterdir()) == ["summary.csv", "summary.md"]


def test_step_b_without_drive_starts_no_copy(tmp_path, monkeypatch):
    namespace, copies, copying, packed = run_step_b(tmp_path, monkeypatch, drive=False)
    assert not copying and not copies and packed == [True]
    assert namespace["STATUS"]["B"]["status"] == "완료"


def test_step_b_does_not_call_its_score_floor_the_comparison_protocol():
    # Step A and docs/MEASUREMENT.md define the comparison protocol at 0.001.
    lines = [line for line in cells()["b-md"]["source"].splitlines()
             if "0.05" in line and "하한" in line]
    assert lines and not any("프로토콜" in line for line in lines)


def test_the_settings_form_includes_the_mobilenet_baseline_by_default(monkeypatch):
    values = settings()
    assert values["WITH_MOBILENET"] is True
    assert values["STEP_A_MODELS"].split() == DEFAULT_A
    assert values["STEP_B_MODELS"].split() == DEFAULT_B
    # Names the scripts accept, in the order they measure anyway (the baselines first).
    assert DEFAULT_A == list(script_module("run_comparison", monkeypatch).MODELS)
    assert DEFAULT_B == list(script_module("compare_improvements", monkeypatch).MODEL_KINDS)
    form = cells()["params-md"]["source"]
    assert "| `WITH_MOBILENET` |" in form and "`faster_rcnn mobilenet v2 v3`" in form
    intro = cells()["intro"]["source"]
    assert "feat/personal-comparison-ci" in intro and "fasterrcnn_mobilenet_v3_large_fpn" in intro
    assert "4개 모델" in intro and "| 85~120분 |" in intro


@pytest.mark.parametrize("with_mobilenet", [True, False])
def test_chosen_models_drop_the_mobilenet_baseline_when_it_is_turned_off(with_mobilenet):
    chosen = with_model_choice({}, with_mobilenet)["chosen_models"]
    expected_a = DEFAULT_A if with_mobilenet else ["faster_rcnn", "light_rcnn", "yolo26"]
    expected_b = DEFAULT_B if with_mobilenet else ["faster_rcnn", "v2", "v3"]
    assert chosen(" ".join(DEFAULT_A)) == expected_a
    assert chosen(" ".join(DEFAULT_B)) == expected_b


@pytest.mark.parametrize("with_mobilenet", [True, False])
def test_step_b_compares_the_mobilenet_baseline_unless_it_is_turned_off(tmp_path, monkeypatch,
                                                                       with_mobilenet):
    namespace, _, _, _ = run_step_b(tmp_path, monkeypatch, models=" ".join(DEFAULT_B),
                                    with_mobilenet=with_mobilenet)
    [command] = namespace["commands"]
    expected = DEFAULT_B if with_mobilenet else ["faster_rcnn", "v2", "v3"]
    assert option(command, "--models") == expected
    assert namespace["GROUP_LABELS"]["mobilenet"] == "MobileNetV3 Faster R-CNN"
    # compare_improvements.py accepts exactly these arguments.
    script = script_module("compare_improvements", monkeypatch)
    args = script.build_parser().parse_args([str(part) for part in command[2:]])
    assert args.models == expected


def run_step(cell_id: str, tmp_path, monkeypatch, with_mobilenet: bool, **values) -> list:
    """Runs a step cell with run_step replaced by a stub; returns the (key, command) calls."""
    monkeypatch.chdir(tmp_path)
    repo = tmp_path / "Lite-R-CNN"
    repo.mkdir(exist_ok=True)
    calls = []

    def fake_run_step(key, command, log_name):
        calls.append((key, command))
        namespace["STATUS"][key] = {"status": "완료", "minutes": 0.0}
        return True

    namespace = {"datetime": datetime, "os": os, "sys": sys, "REPO_DIR": repo, "DEVICE": "cuda",
                 "LIMIT": 0, "LIMIT_ARGS": [], "TIMING_A": [], "TIMING_C": [], "STATUS": {},
                 "run_step": fake_run_step, "pack_results": lambda: None,
                 "safely": lambda label, func, *args: func(*args), "show": print,
                 "show_markdown": print, **values}
    with_model_choice(namespace, with_mobilenet)
    run_code(compile(cells()[cell_id]["source"], cell_id, "exec"), namespace)
    return calls


@pytest.mark.parametrize("with_mobilenet", [True, False])
def test_step_a_runs_the_fair_comparison_with_the_mobilenet_baseline_then_v3(
        tmp_path, monkeypatch, with_mobilenet):
    values = settings()
    calls = run_step("step-a", tmp_path, monkeypatch, with_mobilenet,
                     RUN_STEP_A=True, STEP_A_MODELS=values["STEP_A_MODELS"],
                     STEP_A_WITH_V3=values["STEP_A_WITH_V3"])
    (v2_key, v2), (v3_key, v3) = calls
    assert (v2_key, v3_key) == ("A (v2)", "A (v3)")
    expected = DEFAULT_A if with_mobilenet else ["faster_rcnn", "light_rcnn", "yolo26"]
    assert option(v2, "--models") == expected
    assert option(v2, "--light-weights") == ["outputs/coco_4090_v2/best.pth"]
    assert option(v2, "--yolo-weights") == ["yolo26n.pt"]
    assert option(v3, "--models") == ["light_rcnn"]
    assert option(v3, "--light-weights") == ["outputs/coco_4090_v3/best.pth"]
    assert set(expected) <= set(script_module("run_comparison", monkeypatch).MODELS)


@pytest.mark.parametrize("with_mobilenet", [True, False])
def test_step_c_adds_the_mobilenet_baseline_as_a_fourth_model(tmp_path, monkeypatch,
                                                              with_mobilenet):
    [(key, command)] = run_step("step-c", tmp_path, monkeypatch, with_mobilenet, RUN_STEP_C=True)
    assert key == "C" and Path(command[1]).name == "evaluate_three_models_gpu_full.py"
    assert ("--with-mobilenet" in command) is with_mobilenet
    gpu = script_module("evaluate_three_models_gpu_full", monkeypatch)
    args = gpu.build_parser().parse_args([str(part) for part in command[2:]])
    assert args.with_mobilenet is with_mobilenet and args.device == "cuda"


def test_run_info_records_whether_the_mobilenet_baseline_ran(tmp_path):
    namespace = {"json": json, "subprocess": __import__("subprocess"), "datetime": datetime,
                 "Path": Path, "REPO_DIR": tmp_path, "RUN_ID": "r1", "LOG_DIR": tmp_path / "logs",
                 "REPO_URL": "url", "BRANCH": "b", "GPU_NAME": "Tesla T4",
                 "torch": SimpleNamespace(__version__="2", version=SimpleNamespace(cuda="12")),
                 "RUN_LIMIT": 0, "LIMIT": 0, "QUICK_TEST": False, "RUN_STEP_A": True,
                 "RUN_STEP_B": True, "RUN_STEP_C": False, "STATUS": {}, **settings()}
    run_parts(cells()["setup"]["source"], namespace, functions={"write_run_info"})
    namespace["write_run_info"]()
    info = json.loads((tmp_path / "logs/colab_run_r1.json").read_text(encoding="utf-8"))
    assert info["with_mobilenet"] is True and info["step_b_models"] == " ".join(DEFAULT_B)


GUARDED_CELLS = ("data", "step-a", "step-b", "step-c", "finish")


@pytest.mark.parametrize("cell", GUARDED_CELLS)
def test_later_cells_stop_with_a_hint_before_importing_when_setup_did_not_finish(cell):
    source = cells()[cell]["source"]
    guard = source.index('globals().get("SETUP_OK", True)')
    imports = [source.find(word) for word in ("from light_rcnn", "import light_rcnn") if word in source]
    assert all(guard < position for position in imports)  # no ModuleNotFoundError first
    for namespace in ({}, {"REPO_DIR": Path("/nowhere"), "SETUP_OK": False}):
        with pytest.raises(SystemExit, match="준비 완료"):
            run_code(source, dict(namespace))


def test_setup_marks_itself_finished_only_at_the_very_end():
    lines = [line for line in cells()["setup"]["source"].splitlines() if line.strip()]
    assert lines[-1] == "SETUP_OK = True"
    assert "SETUP_OK = False" in cells()["setup"]["source"]


def git(*args, cwd=None):
    import subprocess

    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                   env=dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@e",
                            GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@e"))


@pytest.fixture
def remote_without_branch(tmp_path):
    """A local 'GitHub' with only main, plus a patch against it (what the user uploads)."""
    remote = tmp_path / "remote"
    git("init", "-q", "-b", "main", str(remote))
    write(remote / "README.md", "main\n")
    git("add", "-A", cwd=remote)
    git("commit", "-q", "-m", "main", cwd=remote)
    work = tmp_path / "content"
    work.mkdir()
    patch = work / "lite-rcnn-perf-improve.patch"
    patch.write_text("diff --git a/NEW.txt b/NEW.txt\nnew file mode 100644\n--- /dev/null\n+++ b/NEW.txt\n"
                     "@@ -0,0 +1 @@\n+patched\n", encoding="utf-8")
    return remote, work, patch


def setup_namespace(remote, work, patch_file=""):
    import codecs
    import shlex
    import subprocess

    namespace = {"Path": Path, "WORK_DIR": work, "REPO_DIR": work / "Lite-R-CNN",
                 "REPO_URL": str(remote), "BRANCH": "perf/improve", "PATCH_FILE": patch_file,
                 "codecs": codecs, "contextlib": contextlib, "os": os, "shlex": shlex,
                 "shutil": shutil, "subprocess": subprocess, "sys": sys}
    return run_parts(cells()["setup"]["source"], namespace,
                     functions=("sh", "colab_secret", "find_patch", "clone_with_patch", "clone_repo"))


def test_missing_branch_falls_back_to_main_plus_the_uploaded_patch(remote_without_branch, capsys):
    remote, work, _ = remote_without_branch
    namespace = setup_namespace(remote, work)
    namespace["clone_repo"]()
    repo = work / "Lite-R-CNN"
    assert (repo / "NEW.txt").read_text(encoding="utf-8") == "patched\n"
    assert (repo / ".git" / "colab_patch").read_text(encoding="utf-8") == "lite-rcnn-perf-improve.patch"
    namespace["clone_repo"]()  # re-running the cell keeps the patched checkout
    assert "패치로 만든 저장소가 이미 있어" in capsys.readouterr().out


def test_patch_file_setting_skips_the_branch(remote_without_branch):
    remote, work, patch = remote_without_branch
    renamed = patch.rename(work / "my.patch")
    namespace = setup_namespace(remote, work, patch_file="my.patch")
    namespace["clone_repo"]()
    assert (work / "Lite-R-CNN" / "NEW.txt").is_file()
    assert renamed.is_file()


def test_missing_branch_without_a_patch_explains_both_ways(remote_without_branch):
    remote, work, patch = remote_without_branch
    patch.unlink()
    namespace = setup_namespace(remote, work)
    with pytest.raises(SystemExit, match="패치 파일"):
        namespace["clone_repo"]()


def test_a_patch_that_does_not_apply_leaves_no_half_made_checkout(remote_without_branch):
    remote, work, patch = remote_without_branch
    patch.write_text("diff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n"
                     "-not what main has\n+x\n", encoding="utf-8")
    namespace = setup_namespace(remote, work)
    with pytest.raises(SystemExit, match="새 패치"):
        namespace["clone_repo"]()
    assert not (work / "Lite-R-CNN").exists()
