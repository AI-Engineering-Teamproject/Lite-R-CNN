import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch
import yaml

from light_rcnn.reporting import (
    build_report,
    default_report_path,
    file_sha256,
    load_reports,
    select_reports,
    write_report,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/import_training_results.py"
STATS = ("AP_50_95", "AP_50", "AP_75", "AP_small", "AP_medium", "AP_large")
RESULTS_MD = """# Lite R-CNN 결과 — synthetic

- 상태: **완료** (3/3 epoch)
- 갱신 시각: 2026-10-19T21:40:59+09:00
- 코드 revision: `abc1234` (worktree dirty)
- 장치: `cuda` (NVIDIA GeForce RTX 4090); PyTorch `2.7.1+cu128`, CUDA runtime `12.8`
- 모델: `light_rcnn`, 입력 최대 변 640px
- 데이터: train 118,287장, val 5,000장; batch 8, val batch 4, AMP True
"""


@pytest.fixture
def importer():
    spec = importlib.util.spec_from_file_location("import_training_results", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validation(ap, images=5000, scope="full"):
    metrics = {key: ap * (index + 1) / 10 for index, key in enumerate(STATS)}
    metrics["AP_50_95"] = ap
    return {"images_evaluated": images, "evaluation_batch_size": 4, "detections": 1234,
            "images_with_detections": images - 1, "evaluation_scope": scope,
            "evaluation_subset_seed": 42 if scope == "subset" else None, **metrics}


def make_run(path, rows, *, results=RESULTS_MD, annotations="data/missing/instances.json",
             model=None):
    path.mkdir()
    config = {"seed": 42, "device": "cuda", "image_size": 640,
              "data": {"val_images": "data/missing/val2017", "val_annotations": annotations},
              "model": model or {"architecture": "light_rcnn", "box_score_thresh": 0.05},
              "train": {"epochs": len(rows), "val_batch_size": 4}}
    (path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (path / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows),
                                        encoding="utf-8")
    if results is not None:
        (path / "RESULTS.md").write_text(results, encoding="utf-8")
    return path


def test_imports_best_epoch_as_v2_accuracy_report(importer, tmp_path, capsys):
    rows = [
        {"epoch": 1, "validation": validation(0.20), "best_ap": 0.20, "best_epoch": 1},
        {"epoch": 2, "validation": validation(0.30), "best_ap": 0.30, "best_epoch": 2},
        {"epoch": 3, "validation": None, "best_ap": 0.30, "best_epoch": 2},
    ]
    run = make_run(tmp_path / "run", rows)
    output = tmp_path / "reports/light_rcnn_accuracy_training.json"
    importer.main([str(run), "--output", str(output)])

    report = json.loads(output.read_text(encoding="utf-8"))
    accuracy = report["accuracy"]
    assert report["schema_version"] == 2 and report["kind"] == "accuracy"
    assert report["model"] == "light_rcnn"
    assert accuracy["evaluator"] == "pycocotools"
    assert accuracy["source"] == "training_validation"
    assert (accuracy["epoch"], accuracy["epoch_rule"], accuracy["epochs_completed"]) == (
        2, "best_epoch", 3)
    assert {key: accuracy[key] for key in STATS} == {
        key: rows[1]["validation"][key] for key in STATS}
    assert (accuracy["images_evaluated"], accuracy["images_total"]) == (5000, 5000)
    assert accuracy["is_partial"] is False and accuracy["subset_selection"] == "full"
    assert accuracy["score_threshold"] == 0.05 and accuracy["max_detections"] == 100
    assert accuracy["diagnostics"] == {"images_with_detections": 4999}
    # Training stores six AP stats only; AR and per-class AP must not be invented.
    assert "AR_100" not in accuracy and "per_category_ap" not in accuracy
    assert accuracy["dataset"]["annotations"] == "data/missing/instances.json"
    assert accuracy["dataset"]["annotations_sha256"] is None
    assert report["weights"] == str(run / "best.pth") and report["weights_sha256"] is None
    assert accuracy["weights_epoch"] is None
    assert report["precision"] == "fp32" and report["image_size"] == 640
    assert report["batch_size"] == 4
    assert report["environment"]["device_name"] == "NVIDIA GeForce RTX 4090"
    assert report["environment"]["device_type"] == "cuda"
    assert report["environment"]["packages"] == {"torch": "2.7.1+cu128"}
    assert report["environment"]["cuda_runtime"] == "12.8"
    assert report["created_at"] == "2026-10-19T12:40:59+00:00"
    assert report["created_at_source"] == "RESULTS.md"
    assert report["git_commit"] == "abc1234-dirty"
    assert json.loads(capsys.readouterr().out)["accuracy"]["epoch"] == 2

    reports, errors = load_reports(tmp_path / "reports")
    assert not errors
    chosen = select_reports(reports)[("accuracy", "light_rcnn")]
    assert chosen.section["AP_50_95"] == pytest.approx(0.30) and not chosen.legacy


def test_falls_back_to_max_ap_and_records_subset_and_weights(importer, tmp_path):
    rows = [  # pre-best_epoch history format
        {"epoch": 1, "validation": validation(0.25, images=16, scope="subset")},
        {"epoch": 2, "validation": validation(0.15, images=16, scope="subset")},
    ]
    annotations = tmp_path / "instances.json"
    annotations.write_text('{"images": []}', encoding="utf-8")
    run = make_run(tmp_path / "run", rows, annotations=str(annotations),
                   model={"box_score_thresh": 0.3, "box_detections_per_img": 50})
    torch.save({"epoch": 1, "config": {}}, run / "best.pth")
    output = tmp_path / "out.json"
    importer.main([str(run), "--output", str(output)])

    report = json.loads(output.read_text(encoding="utf-8"))
    accuracy = report["accuracy"]
    assert (accuracy["epoch"], accuracy["epoch_rule"], accuracy["weights_epoch"]) == (
        1, "max_AP_50_95", 1)
    assert accuracy["AP_50_95"] == pytest.approx(0.25)
    assert (accuracy["images_evaluated"], accuracy["images_total"]) == (16, 5000)
    assert accuracy["is_partial"] is True and accuracy["subset_selection"] == "seeded_random"
    assert accuracy["subset_seed"] == 42
    assert (accuracy["score_threshold"], accuracy["max_detections"]) == (0.3, 50)
    assert accuracy["dataset"]["annotations_sha256"] == file_sha256(annotations)
    assert report["weights_sha256"] == file_sha256(run / "best.pth")


def resumed_run(tmp_path, checkpoint):
    """train.py --resume into a new folder: best.pth (epoch 2) is copied, history starts at 3."""
    rows = [
        {"epoch": 3, "validation": validation(0.20), "best_ap": 0.30, "best_epoch": 2},
        {"epoch": 4, "validation": validation(0.25), "best_ap": 0.30, "best_epoch": 2},
    ]
    run = make_run(tmp_path / "run", rows)
    torch.save(checkpoint, run / "best.pth")
    return run


def test_epoch_mismatch_imports_checkpoint_metrics(importer, tmp_path):
    run = resumed_run(tmp_path, {"epoch": 2, "config": {}, "metrics": validation(0.30)})
    output = tmp_path / "out.json"
    importer.main([str(run), "--output", str(output)])

    report = json.loads(output.read_text(encoding="utf-8"))
    accuracy = report["accuracy"]
    assert (accuracy["epoch"], accuracy["epoch_rule"], accuracy["epochs_completed"]) == (
        2, "checkpoint_metrics", 4)
    assert accuracy["weights_epoch"] == 2 and accuracy["AP_50_95"] == pytest.approx(0.30)
    assert report["weights_sha256"] == file_sha256(run / "best.pth")


def test_epoch_mismatch_without_checkpoint_metrics_exits(importer, tmp_path):
    run = resumed_run(tmp_path, {"epoch": 2, "config": {}})
    output = tmp_path / "out.json"
    with pytest.raises(SystemExit, match="without its validation metrics"):
        importer.main([str(run), "--output", str(output)])
    assert not output.exists()


def test_without_results_md_keeps_environment_minimal(importer, tmp_path):
    rows = [{"epoch": 1, "validation": validation(0.1), "best_ap": 0.1, "best_epoch": 1}]
    run = make_run(tmp_path / "run", rows, results=None)
    output = tmp_path / "out.json"
    importer.main([str(run), "--output", str(output)])

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["environment"] == {"device": "cuda", "device_type": "cuda",
                                     "device_name": None, "recorded_from": "config"}
    assert report["git_commit"] is None
    assert report["accuracy"]["images_total"] == 5000  # full scope: every val image
    assert report["accuracy"]["is_partial"] is False
    assert report["created_at"] == importer.UNKNOWN_CREATED_AT
    assert report["created_at_source"] == "unknown"


def test_run_without_results_md_loses_to_timestamped_reports(importer, tmp_path):
    untimed = make_run(tmp_path / "untimed", [
        {"epoch": 1, "validation": validation(0.40), "best_ap": 0.40, "best_epoch": 1}],
        results=None)
    # A fresh checkout: history.jsonl is newer than every rival's measurement.
    checkout = datetime(2030, 1, 1, tzinfo=timezone.utc).timestamp()
    os.utime(untimed / "history.jsonl", (checkout, checkout))
    timed = make_run(tmp_path / "timed", [
        {"epoch": 1, "validation": validation(0.20), "best_ap": 0.20, "best_epoch": 1}])
    reports_dir = tmp_path / "reports"
    importer.main([str(untimed), "--output", str(reports_dir / "untimed.json")])
    importer.main([str(timed), "--output", str(reports_dir / "timed.json")])
    measured = build_report("accuracy", "light_rcnn", created_at="2026-10-20T00:00:00+00:00",
                            accuracy={"evaluator": "pycocotools", "is_partial": False,
                                      "images_evaluated": 5000, "AP_50_95": 0.30})
    write_report(measured, reports_dir / "evaluate_coco.json")

    reports, errors = load_reports(reports_dir)
    assert not errors
    by_name = {Path(report.source).stem: report for report in reports}
    for rival in ("timed", "evaluate_coco"):
        for pair in ([by_name["untimed"], by_name[rival]], [by_name[rival], by_name["untimed"]]):
            assert select_reports(pair)[("accuracy", "light_rcnn")] is by_name[rival]


def test_rejects_history_without_validation(importer, tmp_path):
    run = make_run(tmp_path / "run", [{"epoch": 1, "validation": None}])
    with pytest.raises(SystemExit, match="no validated epoch"):
        importer.main([str(run), "--output", str(tmp_path / "out.json")])


def test_default_output_does_not_overwrite_evaluate_coco_report(importer):
    assert importer.DEFAULT_OUTPUT != default_report_path("accuracy", "light_rcnn")
    assert importer.DEFAULT_OUTPUT.parent == default_report_path("accuracy", "light_rcnn").parent
