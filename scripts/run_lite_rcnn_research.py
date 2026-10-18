"""One-command COCO preparation, resumable Lite R-CNN training and reporting."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], cwd: Path = ROOT) -> None:
    print("\n$ " + subprocess.list2cmdline([str(part) for part in command]), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def dataset_ready(config: dict) -> bool:
    data = config["data"]
    paths = tuple(ROOT / data[key] for key in (
        "train_images", "train_annotations", "val_images", "val_annotations"
    ))
    return (
        paths[0].is_dir()
        and any(paths[0].glob("*.jpg"))
        and paths[1].is_file()
        and paths[2].is_dir()
        and any(paths[2].glob("*.jpg"))
        and paths[3].is_file()
    )


def history_summary(history_path: Path) -> tuple[list[dict], dict | None]:
    if not history_path.exists():
        return [], None
    rows = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    best = max(
        (row for row in rows if row.get("validation") and
         row["validation"].get("AP_50_95") is not None),
        key=lambda row: row["validation"]["AP_50_95"],
        default=None,
    )
    return rows, best


def write_report(output_dir: Path, config: dict, checkpoint: dict | None) -> Path:
    history_path = output_dir / "history.jsonl"
    rows, best = history_summary(history_path)
    last = rows[-1] if rows else None
    epoch = checkpoint.get("epoch") if checkpoint else (last or {}).get("epoch", 0)
    total_epochs = config["train"]["epochs"]
    data = config["data"]
    report = [
        "# Lite R-CNN 학습 연구 보고서",
        "",
        f"- 갱신 시각: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- 상태: {'완료' if epoch >= total_epochs else '진행 중 또는 중단됨'} ({epoch}/{total_epochs} epochs)",
        f"- 장치: {config.get('device', '설정값 없음')}",
        f"- 이미지 크기: {config.get('image_size', '설정값 없음')}",
        f"- 학습 annotation: `{data['train_annotations']}`",
        f"- 검증 annotation: `{data['val_annotations']}`",
        "- 비교 대상 측정: 이 자동화 범위에 포함하지 않음 (팀원 담당)",
        f"- 학습 이미지 제한: {config['train'].get('train_limit', 0) or '전체 split'}",
        f"- 검증 이미지 제한: {config['train'].get('val_limit', 0) or '전체 split'}",
        "",
        "## 모델 설정",
        "",
        f"- Backbone: {config['model'].get('backbone')}",
        f"- BiFPN channels: {config['model'].get('bifpn_channels')}",
        f"- RoI channels: {config['model'].get('roi_channels')}",
        f"- Batch size: {config['train'].get('batch_size')}",
        f"- Learning rate: {config['train'].get('learning_rate')}",
        f"- Distillation: {config.get('distillation', {}).get('enabled', False)}",
        "",
        "## 현재까지의 결과",
        "",
    ]
    if best:
        metrics = best["validation"]
        report.extend([
            f"- 최고 검증 AP@[.50:.95]: {metrics.get('AP_50_95', 0.0) * 100:.3f}% (epoch {best['epoch']})",
            f"- 평가 범위: {metrics.get('evaluation_scope', 'unknown')}, {metrics.get('images_evaluated', 'unknown')} validation images",
            f"- 해당 epoch AP50: {metrics.get('AP_50', 0.0) * 100:.3f}%",
            f"- 해당 epoch AP75: {metrics.get('AP_75', 0.0) * 100:.3f}%",
        ])
    else:
        report.append("- COCO 검증 결과가 아직 없습니다.")
    if last:
        report.extend([
            f"- 마지막 epoch: {last['epoch']}",
            f"- 마지막 학습 loss: {last.get('losses', {}).get('loss_total', '기록 없음')}",
        ])
    report.extend([
        "",
        "## 산출물",
        "",
        f"- 최고 체크포인트: `{(output_dir / 'best.pth').as_posix()}`",
        f"- 재개 체크포인트: `{(output_dir / 'last.pth').as_posix()}`",
        f"- epoch별 손실·검증 기록: `{history_path.as_posix()}`",
        "",
        "검증 AP는 설정된 validation split 기준입니다. 비교 모델 결과와 속도 측정은 별도 공통 조건 벤치마크에서 합쳐야 합니다.",
        "",
    ])
    report_path = output_dir / "RESEARCH_REPORT.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/coco_cpu.yaml")
    parser.add_argument("--device", default="cpu", help="Defaults to CPU for this workstation")
    parser.add_argument("--epochs", type=int, help="Override total training epochs")
    parser.add_argument("--output-dir", help="Override the configured output directory")
    parser.add_argument("--train-limit", type=int, help="Optional training subset for a quick smoke run")
    parser.add_argument("--val-limit", type=int, help="Optional validation subset for a quick smoke run")
    parser.add_argument("--skip-data-prepare", action="store_true")
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["device"] = args.device
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.train_limit is not None:
        config["train"]["train_limit"] = args.train_limit
    if args.val_limit is not None:
        config["train"]["val_limit"] = args.val_limit
    if args.output_dir:
        config["train"]["output_dir"] = args.output_dir
    output_dir = (ROOT / config["train"]["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "last.pth"

    try:
        if not dataset_ready(config):
            if args.skip_data_prepare:
                raise FileNotFoundError(
                    "COCO train/val images or annotations are missing; run scripts/prepare_hf_coco.py"
                )
            print("COCO files are incomplete; preparing them from Hugging Face Hub.", flush=True)
            run([sys.executable, "scripts/prepare_hf_coco.py"])
            if not dataset_ready(config):
                raise FileNotFoundError("Hugging Face preparation finished but the configured files are incomplete")

        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            completed = int(checkpoint.get("epoch", 0))
            requested = int(config["train"]["epochs"])
            if completed >= requested:
                print(f"Training already reached {completed}/{requested} epochs; keeping its results.")
            else:
                command = [
                    sys.executable, "scripts/train.py", "--resume", str(checkpoint_path),
                    "--epochs", str(requested), "--device", args.device,
                ]
                if args.train_limit is not None:
                    command += ["--train-limit", str(args.train_limit)]
                if args.val_limit is not None:
                    command += ["--val-limit", str(args.val_limit)]
                run(command)
        else:
            command = [sys.executable, "scripts/train.py", "--config", str(config_path),
                       "--device", args.device]
            if args.epochs is not None:
                command += ["--epochs", str(args.epochs)]
            if args.output_dir:
                command += ["--output-dir", args.output_dir]
            if args.train_limit is not None:
                command += ["--train-limit", str(args.train_limit)]
            if args.val_limit is not None:
                command += ["--val-limit", str(args.val_limit)]
            run(command)
    finally:
        checkpoint = None
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        report_path = write_report(output_dir, config, checkpoint)
        print(f"\nResearch report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
