"""Run a controlled small-data ablation of the GAP and spatial single-FC RoI heads."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "gap": {"roi_head_dim": 0, "roi_head_pool_size": 1},
    "spatial": {"roi_head_dim": 128, "roi_head_pool_size": 3},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/coco_cpu.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-limit", type=int, default=16)
    parser.add_argument("--val-limit", type=int, default=16)
    parser.add_argument("--output-root", default="outputs/roi_head_ablation")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()
    if min(args.epochs, args.train_limit, args.val_limit) < 0 or args.epochs == 0:
        parser.error("epochs must be positive and image limits nonnegative")

    base = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    output_root = (ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for name in args.variants:
        run_dir = output_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        config = deepcopy(base)
        config["device"] = args.device
        config["model"].update(VARIANTS[name])
        config["train"].update({
            "epochs": args.epochs,
            "train_limit": args.train_limit,
            "val_limit": args.val_limit,
            "output_dir": str(run_dir),
        })
        if config["train"]["train_limit"]:
            subset_steps = math.ceil(
                config["train"]["train_limit"] / config["train"]["batch_size"]
            )
            config["train"]["warmup_steps"] = min(
                config["train"]["warmup_steps"], max(1, subset_steps // 5)
            )
        config_path = run_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        last_path = run_dir / "last.pth"
        if last_path.exists():
            previous = torch.load(last_path, map_location="cpu", weights_only=True)
            saved_config = previous.get("config")
            if not saved_config:
                raise ValueError(f"Checkpoint has no config; use a fresh ablation output root: {last_path}")
            for key in ("model", "data", "num_classes", "image_size", "distillation", "seed"):
                if saved_config.get(key) != config.get(key):
                    raise ValueError(
                        f"Existing {name} checkpoint differs in {key}; use a fresh --output-root"
                    )
            for key in (
                "batch_size",
                "learning_rate",
                "weight_decay",
                "warmup_steps",
                "train_limit",
                "val_limit",
                "subset_selection",
            ):
                if saved_config["train"].get(key) != config["train"].get(key):
                    raise ValueError(
                        f"Existing {name} checkpoint differs in train.{key}; use a fresh --output-root"
                    )
            if int(previous.get("epoch", 0)) < args.epochs:
                command = [sys.executable, "scripts/train.py", "--resume", str(last_path),
                           "--epochs", str(args.epochs), "--device", args.device]
            else:
                command = None
        else:
            command = [sys.executable, "scripts/train.py", "--config", str(config_path)]
        if command:
            print(f"\n=== RoI head ablation: {name} ===", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)

        checkpoint = torch.load(last_path, map_location="cpu", weights_only=True)
        history_path = run_dir / "history.jsonl"
        rows = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        best = checkpoint.get("metrics") or {}
        results.append({
            "variant": name,
            "config": VARIANTS[name],
            "train_limit": args.train_limit,
            "val_limit": args.val_limit,
            "epochs": checkpoint.get("epoch"),
            "best_ap_50_95": checkpoint.get("best_ap"),
            "best_epoch": checkpoint.get("best_epoch"),
            "last_loss": rows[-1]["losses"].get("loss_total") if rows else None,
            "learning_rate": rows[-1].get("learning_rate") if rows else None,
            "evaluation_scope": best.get("evaluation_scope"),
            "checkpoint": str((run_dir / "best.pth").resolve()),
        })

    (output_root / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# Lite R-CNN RoI head ablation",
        "",
        f"- Run time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- Device: {args.device}; train images: {args.train_limit or 'all'}; validation images: {args.val_limit or 'all'}; epochs: {args.epochs}",
        "- This is an internal Lite R-CNN ablation, not a Faster R-CNN or YOLO26 comparison.",
        "- Small-subset AP is diagnostic only and must not be reported as full COCO accuracy.",
        "",
        "| RoI head | Pool grid | FC width | Last train loss | Best AP@[.50:.95] | Eval scope |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in results:
        ap = row["best_ap_50_95"]
        lines.append(
            f"| {row['variant']} | {row['config']['roi_head_pool_size']}×{row['config']['roi_head_pool_size']} | "
            f"{row['config']['roi_head_dim'] or 'none'} | {row['last_loss']:.4f} | "
            f"{'—' if ap is None else f'{ap * 100:.3f}%'} | {row['evaluation_scope']} |"
        )
    lines.extend(["", "Detailed results and checkpoint paths are in `results.json`.", ""])
    (output_root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nAblation report: {output_root / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
