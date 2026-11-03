"""Import a training run's best validation epoch as a schema-v2 accuracy report.

Training validates with light_rcnn.evaluation (pycocotools COCOeval) at the model's own
box_score_thresh, not comparison.score_threshold, and history.jsonl keeps only the six AP stats.
The report records exactly that, so the dashboard can flag it next to evaluate_coco.py results.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

from light_rcnn.reporting import (
    build_report,
    dataset_info,
    display_path,
    file_sha256,
    utc_now,
    write_report,
)

ROOT = Path(__file__).resolve().parents[1]
# Kept apart from evaluate_coco.py's outputs/reports/light_rcnn_accuracy.json.
DEFAULT_OUTPUT = Path("outputs/reports/light_rcnn_accuracy_training.json")
# light_rcnn_from_config defaults, used when the run config leaves them out.
DEFAULT_SCORE_THRESH = 0.05
DEFAULT_DETECTIONS_PER_IMG = 100
# created_at when RESULTS.md has no 갱신 시각: history.jsonl's mtime is only the checkout time,
# so the earliest stamp keeps every timestamped report ahead of this one in select_reports.
UNKNOWN_CREATED_AT = "1970-01-01T00:00:00+00:00"
STATS = ("AP_50_95", "AP_50", "AP_75", "AP_small", "AP_medium", "AP_large")
DIAGNOSTICS = (
    "images_with_detections",
    "max_detection_score",
    "mean_best_iou_any_class",
    "mean_best_iou_same_class",
    "images_with_iou50_any_class",
    "images_with_iou50_same_class",
)
# Lines scripts/train.py writes into RESULTS.md.
RESULTS_PATTERNS = {
    key: re.compile(pattern, re.MULTILINE) for key, pattern in {
        "updated": r"^- 갱신 시각: (\S+)",
        "revision": r"^- 코드 revision: `([^`]+)` \(worktree (\w+)\)",
        "device": r"^- 장치: `([^`]+)` \((.+?)\); PyTorch `([^`]+)`(?:, CUDA runtime `([^`]+)`)?",
        "val_images": r"^- 데이터: .*?\bval ([\d,]+)장",
    }.items()
}


def read_history(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"history.jsonl not found: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def best_row(rows: list[dict]) -> tuple[dict, str]:
    """The epoch train.py saved as best.pth (its best_epoch field), else the highest AP_50_95."""
    validated = {row["epoch"]: row for row in rows if row.get("validation")}
    if not validated:
        raise SystemExit("history.jsonl has no validated epoch; nothing to import")
    recorded = rows[-1].get("best_epoch")
    if recorded in validated:
        return validated[recorded], "best_epoch"
    return max(validated.values(), key=lambda row: row["validation"]["AP_50_95"]), "max_AP_50_95"


def to_utc(stamp: str) -> str | None:
    try:
        moment = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    if moment.tzinfo is None:
        return None
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


def parse_results(path: Path) -> dict:
    """Fields from RESULTS.md; missing or reworded lines are skipped."""
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    parsed = {}
    if match := RESULTS_PATTERNS["updated"].search(text):
        parsed["updated"] = to_utc(match.group(1))
    if match := RESULTS_PATTERNS["revision"].search(text):
        revision, state = match.groups()
        if revision != "unavailable":
            parsed["revision"] = f"{revision}-dirty" if state == "dirty" else revision
    if match := RESULTS_PATTERNS["device"].search(text):
        parsed.update(zip(("device", "device_name", "torch", "cuda_runtime"), match.groups()))
    if match := RESULTS_PATTERNS["val_images"].search(text):
        parsed["val_images"] = int(match.group(1).replace(",", ""))
    return parsed


def read_checkpoint(path: Path) -> dict | None:
    """Checkpoint metadata (epoch, config) for cross-checking; None when unreadable."""
    if not path.is_file():
        return None
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (ImportError, OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError):
        return None
    return checkpoint if isinstance(checkpoint, dict) else None


def environment(results: dict, config: dict) -> dict:
    """Training hardware as RESULTS.md recorded it; only the config device when it is absent."""
    device = results.get("device") or config.get("device")
    text = str(device or "").lower()
    env = {
        "device": device,
        "device_type": "cuda" if text.startswith("cuda") else ("cpu" if text == "cpu" else None),
        "device_name": results.get("device_name"),
        "recorded_from": "RESULTS.md" if results.get("device_name") else "config",
    }
    if results.get("torch"):
        env["packages"] = {"torch": results["torch"]}
    if results.get("cuda_runtime") not in (None, "not used", "unknown"):
        env["cuda_runtime"] = results["cuda_runtime"]
    return env


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Training output folder with history.jsonl and config.yaml")
    parser.add_argument("--weights", help="Defaults to <run_dir>/best.pth")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help=f"Defaults to {DEFAULT_OUTPUT.as_posix()}")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    rows = read_history(run_dir / "history.jsonl")
    row, rule = best_row(rows)
    epoch, validation = row["epoch"], row["validation"]
    weights = Path(args.weights) if args.weights else run_dir / "best.pth"
    checkpoint = read_checkpoint(weights) or {}
    weights_epoch = checkpoint.get("epoch")
    if weights_epoch is not None and weights_epoch != epoch:
        # e.g. train.py --resume into a new folder copies best.pth but starts a new history.jsonl.
        if not isinstance(checkpoint.get("metrics"), dict):
            raise SystemExit(f"{weights} holds epoch {weights_epoch} without its validation metrics, "
                             f"and history.jsonl has epoch {epoch}; pass --weights for that epoch")
        epoch, validation, rule = weights_epoch, checkpoint["metrics"], "checkpoint_metrics"
    config_path = run_dir / "config.yaml"
    if config_path.is_file():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        config_source = display_path(config_path)
    elif isinstance(checkpoint.get("config"), dict):
        config = checkpoint["config"]
        config_source = f"{display_path(weights)} (checkpoint config)"
    else:
        raise SystemExit(f"config.yaml not found in {run_dir} and {weights} has no config")
    results = parse_results(run_dir / "RESULTS.md")

    evaluated = int(validation["images_evaluated"])
    scope = validation.get("evaluation_scope")
    total = evaluated if scope == "full" else results.get("val_images")
    partial = {"full": False, "subset": True}.get(scope)
    if partial is None and total is not None:
        partial = evaluated < total
    model_cfg = config.get("model") or {}
    data_cfg = config.get("data") or {}
    dataset = None
    if data_cfg.get("val_annotations") and data_cfg.get("val_images"):
        dataset = dataset_info(ROOT / data_cfg["val_annotations"], ROOT / data_cfg["val_images"])
    accuracy = {
        "evaluator": "pycocotools",
        "source": "training_validation",
        "epoch": epoch,
        "epoch_rule": rule,
        "epochs_completed": rows[-1]["epoch"],
        "weights_epoch": weights_epoch,
        "images_evaluated": evaluated,
        "images_total": total,
        "is_partial": partial,
        "subset_selection": None if partial is None else ("seeded_random" if partial else "full"),
        "subset_seed": validation.get("evaluation_subset_seed"),
        "score_threshold": model_cfg.get("box_score_thresh", DEFAULT_SCORE_THRESH),
        "max_detections": model_cfg.get("box_detections_per_img", DEFAULT_DETECTIONS_PER_IMG),
        "dataset": dataset,
        **{key: validation.get(key) for key in STATS},
        "detections": validation.get("detections"),
        "diagnostics": {key: validation[key] for key in DIAGNOSTICS if key in validation},
    }
    stamp = results.get("updated")
    report = build_report(
        "accuracy",
        "light_rcnn",
        # When training measured it and with which code, not when this script ran.
        created_at=stamp or UNKNOWN_CREATED_AT,
        created_at_source="RESULTS.md" if stamp else "unknown",
        git_commit=results.get("revision"),
        imported_at=utc_now(),
        run_dir=display_path(run_dir),
        weights=display_path(weights),
        weights_sha256=file_sha256(weights),
        config=config_source,
        image_size=config.get("image_size"),
        batch_size=validation.get("evaluation_batch_size")
        or (config.get("train") or {}).get("val_batch_size"),
        precision="fp32",  # train.py validates outside autocast, even with AMP training
        environment=environment(results, config),
        accuracy=accuracy,
        images_evaluated=evaluated,
    )
    output = write_report(report, args.output)
    print(json.dumps({
        "output": str(output),
        "model": report["model"],
        "weights": report["weights"],
        "weights_sha256": report["weights_sha256"],
        "created_at": report["created_at"],
        "git_commit": report["git_commit"],
        "environment": report["environment"],
        "accuracy": accuracy,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
