import ast
import contextlib
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


def run_step_b(tmp_path, monkeypatch, drive=True, quick=False, fail=False):
    """Runs the real step B cell with the measurement replaced by a stub that writes a summary."""
    monkeypatch.chdir(tmp_path)
    repo, drive_dir = tmp_path / "Lite-R-CNN", tmp_path / "drive"
    repo.mkdir()
    drive_dir.mkdir()
    before = set(threading.enumerate())
    copying, packed, shown = [], [], []

    def fake_run_step(key, command, log_name):
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
        "threading": threading, "REPO_DIR": repo, "DEVICE": "cpu", "STEP_B_MODELS": "v2 v3",
        "STEP_B_PRESETS": "deploy,fast", "LIMIT_ARGS": [], "TIMING_B": [], "RUN_STEP_B": True,
        "STATUS": {}, "DRIVE_DIR": drive_dir if drive else None, "QUICK_TEST": quick,
        "run_step": fake_run_step, "safely": lambda label, func, *args: func(*args),
        "show_markdown": shown.append, "pack_results": lambda: packed.append(True),
    }
    run_parts(cells()["setup"]["source"], namespace, functions={"mirror_to_drive"})
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
