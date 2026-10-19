"""Train, validate, resume and optionally distill a Light R-CNN detector."""

import argparse
import json
import math
import random
import shutil
import subprocess
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from light_rcnn.data import CocoTensorDataset, detection_collate
from light_rcnn.distillation import DistillationTrainer, build_teacher
from light_rcnn.evaluation import evaluate_coco
from light_rcnn.model.detector import light_rcnn_from_config
from light_rcnn.utils import seed_everything


def rng_state():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    name, keys, position, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached))
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_save(checkpoint, path):
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def write_results_report(output_dir, config, history_path, device, train_images, val_images):
    """Write a portable Markdown summary after every completed training epoch."""
    rows = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        return
    evaluated = [row for row in rows if row.get("validation") is not None]
    best_row = (
        max(evaluated, key=lambda row: row["validation"]["AP_50_95"])
        if evaluated
        else None
    )
    last_row = rows[-1]
    last_metrics = last_row.get("validation")
    completed = last_row["epoch"] >= config["train"]["epochs"]
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
        cuda_runtime = torch.version.cuda or "unknown"
    else:
        device_name, cuda_runtime = "CPU", "not used"
    try:
        git_revision = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        worktree_state = "dirty" if dirty else "clean"
    except (OSError, subprocess.CalledProcessError):
        git_revision, worktree_state = "unavailable", "unavailable"

    def percent(metrics, key):
        return "—" if metrics is None else f"{metrics[key] * 100:.2f}%"

    def number(losses, key):
        value = losses.get(key)
        return "—" if value is None else f"{value:.4f}"

    validation_scope = (
        "전체 val split"
        if config["train"].get("val_limit", 0) == 0
        else f"고정 subset {config['train']['val_limit']}장"
    )
    lines = [
        f"# Lite R-CNN 결과 — {output_dir.name}",
        "",
        f"- 상태: **{'완료' if completed else '진행 중'}** "
        f"({last_row['epoch']}/{config['train']['epochs']} epoch)",
        f"- 갱신 시각: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- 코드 revision: `{git_revision}` (worktree {worktree_state})",
        f"- 장치: `{device}` ({device_name}); PyTorch `{torch.__version__}`, CUDA runtime `{cuda_runtime}`",
        f"- 모델: `{config['model'].get('architecture', 'light_rcnn')}`, 입력 최대 변 {config['image_size']}px",
        f"- 데이터: train {train_images:,}장, val {val_images:,}장; "
        f"batch {config['train']['batch_size']}, "
        f"val batch {config['train'].get('val_batch_size', 1)}, "
        f"AMP {config['train'].get('amp', False)}",
        f"- Seed: {config['seed']}; validation 범위: {validation_scope}",
        "",
        "## 요약",
        "",
    ]
    if best_row:
        best_metrics = best_row["validation"]
        lines.extend(
            [
                f"- 최고 AP@[.50:.95]: **{percent(best_metrics, 'AP_50_95')}** (epoch {best_row['epoch']})",
                f"- 최고 epoch 지표: AP50 {percent(best_metrics, 'AP_50')}, "
                f"AP75 {percent(best_metrics, 'AP_75')}, "
                f"AP-small/medium/large {percent(best_metrics, 'AP_small')} / "
                f"{percent(best_metrics, 'AP_medium')} / {percent(best_metrics, 'AP_large')}",
            ]
        )
    if last_metrics:
        lines.append(
            f"- 마지막 validation (epoch {last_row['epoch']}): "
            f"AP@[.50:.95] {percent(last_metrics, 'AP_50_95')}, "
            f"AP50 {percent(last_metrics, 'AP_50')}, AP75 {percent(last_metrics, 'AP_75')}"
        )
    lines.extend(
        [
            "",
            "## Epoch 기록",
            "",
            "| Epoch | LR | Train loss | Classifier | Box | RPN objectness | RPN box | AP@[.50:.95] | AP50 | AP75 | AP-small | AP-medium | AP-large |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        losses = row["losses"]
        metrics = row.get("validation")
        values = [
            row["epoch"],
            f"{row['learning_rate']:.2e}",
            number(losses, "loss_total"),
            number(losses, "loss_classifier"),
            number(losses, "loss_box_reg"),
            number(losses, "loss_objectness"),
            number(losses, "loss_rpn_box_reg"),
            percent(metrics, "AP_50_95"),
            percent(metrics, "AP_50"),
            percent(metrics, "AP_75"),
            percent(metrics, "AP_small"),
            percent(metrics, "AP_medium"),
            percent(metrics, "AP_large"),
        ]
        lines.append("| " + " | ".join(map(str, values)) + " |")
    lines.extend(
        [
            "",
            "## 다른 컴퓨터에서 이어가기",
            "",
            "이 run 폴더의 `config.yaml`, `history.jsonl`, `last.pth`, `best.pth`, `RESULTS.md`를 함께 보관하십시오. "
            "전체 COCO 이미지와 annotation은 별도 준비가 필요합니다. 총 epoch 수를 늘려 재개할 때는 저장된 설정을 그대로 사용하고 "
            "`python scripts/train.py --resume outputs/<run>/last.pth --epochs <총목표epoch>`를 실행합니다.",
            "",
            "`best.pth`는 최고 validation AP 모델이며 `last.pth`는 마지막 완료 epoch의 optimizer/scheduler 상태를 포함한 재개 파일입니다. "
            "상세 재현 설정은 [config.yaml](config.yaml), 원시 epoch 기록은 [history.jsonl](history.jsonl)입니다.",
            "",
        ]
    )
    report_path = output_dir / "RESULTS.md"
    temporary = report_path.with_suffix(".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(report_path)


def deterministic_subset(dataset, limit, seed):
    """Choose a reproducible random subset without consuming the training RNG."""
    if limit <= 0 or limit >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:limit].tolist()
    return Subset(dataset, indices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Defaults to configs/coco.yaml, or saved config on resume")
    startup = parser.add_mutually_exclusive_group()
    startup.add_argument(
        "--resume", help="Restore the complete training state at an epoch boundary"
    )
    startup.add_argument("--init-weights", help="Initialize student weights for a new run, e.g. KD")
    distillation = parser.add_mutually_exclusive_group()
    distillation.add_argument(
        "--distill", action="store_true", help="Enable feature and RoI distillation"
    )
    distillation.add_argument(
        "--distill-roi-only",
        action="store_true",
        help="Distill RoI class logits only; useful for a classification-focused experiment",
    )
    parser.add_argument(
        "--device",
        help="Training device: cpu, cuda, or cuda:0. CPU training is supported but much slower.",
    )
    parser.add_argument("--epochs", type=int, help="Total epoch count, including completed epochs")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--train-limit", type=int, default=None,
        help="Deterministic random subset of N training images; 0=all",
    )
    parser.add_argument(
        "--val-limit", type=int, default=None,
        help="Deterministic random subset of N validation images; 0=all",
    )
    parser.add_argument("--validate-every", type=int)
    args = parser.parse_args()
    saved = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if saved and saved.get("format_version") != 1:
        raise ValueError("Legacy checkpoint: use --init-weights for a new run instead of --resume")
    if args.config or not saved:
        with open(args.config or "configs/coco.yaml", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    else:
        config = deepcopy(saved["config"])
    if args.distill or args.distill_roi_only:
        kd = config.setdefault("distillation", {})
        kd["enabled"] = True
        if args.distill_roi_only:
            kd["feature_weight"] = 0.0
    train = config["train"]
    if args.device is not None:
        config["device"] = args.device
    for argument, key in (
        (args.epochs, "epochs"),
        (args.output_dir, "output_dir"),
        (args.validate_every, "validate_every"),
    ):
        if argument is not None:
            train[key] = argument
    for argument, key in ((args.train_limit, "train_limit"), (args.val_limit, "val_limit")):
        if argument is not None:
            train[key] = argument
        train.setdefault(key, 0)
    train.setdefault("validate_every", 1)
    train.setdefault("val_batch_size", 1)
    train.setdefault("warmup_steps", 500)
    train.setdefault("lr_milestones", [24, 32])
    train.setdefault("lr_gamma", 0.1)
    train.setdefault("max_grad_norm", 10.0)
    train.setdefault("horizontal_flip_prob", 0.0)
    train.setdefault("box_batch_size_per_image", 512)
    train.setdefault("subset_selection", "seeded_random")
    if train["train_limit"]:
        subset_steps = math.ceil(train["train_limit"] / train["batch_size"])
        train["warmup_steps"] = min(train["warmup_steps"], max(1, subset_steps // 5))
    if (
        train["epochs"] < 1
        or train["batch_size"] < 1
        or train["workers"] < 0
        or train["validate_every"] < 1
        or train["val_batch_size"] < 1
        or train["train_limit"] < 0
        or train["val_limit"] < 0
        or train["warmup_steps"] < 0
        or train["max_grad_norm"] <= 0
        or not 0.0 <= train["horizontal_flip_prob"] <= 1.0
    ):
        raise ValueError("Invalid epoch, batch, worker, validation or subset settings")
    if saved:
        for key in ("model", "num_classes", "image_size", "distillation", "data", "seed"):
            if config.get(key) != saved["config"].get(key):
                raise ValueError(
                    f"Cannot change {key} on resume; start a new run with --init-weights"
                )
        for key in (
            "batch_size",
            "learning_rate",
            "weight_decay",
            "amp",
            "warmup_steps",
            "lr_milestones",
            "lr_gamma",
            "train_limit",
            "val_limit",
            "validate_every",
            "max_grad_norm",
            "horizontal_flip_prob",
        ):
            if train.get(key) != saved["config"]["train"].get(key):
                raise ValueError(f"Cannot change train.{key} on resume")
        if (train["train_limit"] or train["val_limit"]) and (
            saved["config"]["train"].get("subset_selection") != train["subset_selection"]
        ):
            raise ValueError(
                "Subset selection changed; start a new output directory so resume uses the same images"
            )
    output_dir = Path(train["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if not saved and (output_dir / "last.pth").exists():
        raise ValueError("Output already contains last.pth; use --resume or a new --output-dir")
    seed_everything(config["seed"])
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu (slower; see configs/coco_cpu.yaml)")
    data = config["data"]
    required_data = (
        ("train_images", data["train_images"]),
        ("train_annotations", data["train_annotations"]),
        ("val_images", data["val_images"]),
        ("val_annotations", data["val_annotations"]),
    )
    missing = [f"{name}: {path}" for name, path in required_data if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "COCO training data is incomplete. Missing paths:\n  - "
            + "\n  - ".join(missing)
            + "\nRun scripts/prepare_hf_coco.py first or correct the paths in the config."
        )
    dataset = CocoTensorDataset(
        data["train_images"],
        data["train_annotations"],
        horizontal_flip_prob=train["horizontal_flip_prob"],
    )
    if train["train_limit"]:
        dataset = deterministic_subset(dataset, train["train_limit"], config["seed"])
    if not len(dataset):
        raise ValueError("Training dataset must contain images")
    loader = DataLoader(
        dataset,
        batch_size=train["batch_size"],
        shuffle=True,
        num_workers=train["workers"],
        pin_memory=device.type == "cuda",
        collate_fn=detection_collate,
    )
    validation = CocoTensorDataset(data["val_images"], data["val_annotations"])
    if not len(validation):
        raise ValueError("Validation dataset must contain images")
    student = light_rcnn_from_config(
        config, pretrained=not (saved or args.init_weights) and config["model"]["pretrained"]
    )
    if saved or args.init_weights:
        initial = saved or torch.load(args.init_weights, map_location="cpu", weights_only=True)
        if "config" in initial:
            for key in ("model", "num_classes", "image_size"):
                # Pretraining only changes initialization, not the saved architecture.
                source, target = deepcopy(initial["config"][key]), deepcopy(config[key])
                if key == "model":
                    source.pop("pretrained", None)
                    target.pop("pretrained", None)
                if source != target:
                    raise ValueError(f"Initial checkpoint has incompatible {key}")
        student.load_state_dict(initial.get("model", initial))
    kd = config.get("distillation", {})
    trainer = student
    if kd.get("enabled", False):
        trainer = DistillationTrainer(
            student,
            build_teacher(config),
            kd.get("feature_weight", 1.0),
            kd.get("roi_kl_weight", 1.0),
            kd.get("temperature", 2.0),
        )
    trainer.to(device)
    parameters = [parameter for parameter in trainer.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=train["learning_rate"], weight_decay=train["weight_decay"]
    )

    def lr_factor(step):
        warmup = min(1.0, (step + 1) / max(1, train["warmup_steps"]))
        decay = sum(step >= milestone * len(loader) for milestone in train["lr_milestones"])
        return warmup * train["lr_gamma"] ** decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    amp_enabled = device.type == "cuda" and train["amp"]
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    else:  # torch 2.2 compatibility
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    start_epoch, global_step, best_ap, best_epoch = 0, 0, -math.inf, None
    if saved:
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        if kd.get("enabled", False):
            trainer.adapters.load_state_dict(saved["distillation_adapters"])
        start_epoch, global_step = saved["epoch"], saved["global_step"]
        best_ap, best_epoch = saved["best_ap"], saved["best_epoch"]
        restore_rng(saved["rng_state"])
        # Keep the historical winner when continuing in a different directory.
        old_best = Path(args.resume).resolve().parent / "best.pth"
        new_best = (output_dir / "best.pth").resolve()
        if old_best != new_best and best_epoch is not None:
            if not old_best.exists():
                raise ValueError(
                    "Copy best.pth alongside the resume checkpoint to preserve the winner"
                )
            shutil.copy2(old_best, new_best)
    if start_epoch >= train["epochs"]:
        raise ValueError("--epochs must exceed the number of completed epochs")
    history = output_dir / "history.jsonl"
    if saved and history.exists():
        rows = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
        history.write_text(
            "".join(json.dumps(row) + "\n" for row in rows if row["epoch"] <= start_epoch),
            encoding="utf-8",
        )
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    print(
        f"Training {len(dataset)} images on {device}; validation limit={train['val_limit']}; "
        f"distillation={kd.get('enabled', False)}; epoch {start_epoch + 1}/{train['epochs']}"
    )
    for epoch in range(start_epoch, train["epochs"]):
        trainer.train()
        totals = defaultdict(float)
        steps = 0
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{train['epochs']}")
        for images, targets in progress:
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [
                {key: value.to(device, non_blocking=True) for key, value in target.items()}
                for target in targets
            ]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                losses = trainer(images, targets)
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch + 1}: {losses}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(parameters, train.get("max_grad_norm", 10.0))
            if not torch.isfinite(norm):
                if not scaler.is_enabled():
                    raise FloatingPointError("Nonfinite gradient norm")
                scaler.update()  # GradScaler recorded overflow during unscale; skip this update.
                continue
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            steps += 1
            for key, value in losses.items():
                totals[key] += float(value.detach())
            totals["loss_total"] += float(loss.detach())
            progress.set_postfix(loss=f"{float(loss.detach()):.3f}")
        if not steps:
            raise FloatingPointError("No optimizer updates completed this epoch")
        metrics = None
        improved = False
        if (epoch + 1) % train["validate_every"] == 0 or epoch + 1 == train["epochs"]:
            metrics = evaluate_coco(
                student,
                validation,
                device,
                train["val_limit"],
                subset_seed=config["seed"],
                batch_size=train["val_batch_size"],
            )
            ap = metrics["AP_50_95"]
            if not math.isfinite(ap) or ap < 0:
                raise ValueError("Validation has no evaluable non-crowd ground truth")
            improved = ap > best_ap
            if improved:
                best_ap, best_epoch = ap, epoch + 1
        checkpoint = {
            "format_version": 1,
            "model": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch + 1,
            "global_step": global_step,
            "config": config,
            "best_ap": best_ap,
            "best_epoch": best_epoch,
            "metrics": metrics,
            "rng_state": rng_state(),
        }
        if kd.get("enabled", False):
            checkpoint["distillation_adapters"] = trainer.adapters.state_dict()
        atomic_save(checkpoint, output_dir / "last.pth")
        if improved:
            atomic_save(checkpoint, output_dir / "best.pth")
        record = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "losses": {key: value / steps for key, value in totals.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation": metrics,
            "best_ap": best_ap if math.isfinite(best_ap) else None,
            "best_epoch": best_epoch,
        }
        with history.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, allow_nan=False) + "\n")
        write_results_report(output_dir, config, history, device, len(dataset), len(validation))
        print(json.dumps(record, allow_nan=False))
    print(
        f"Finished. Best validation AP={best_ap:.6f} at epoch {best_epoch}; {output_dir / 'best.pth'}"
    )


if __name__ == "__main__":
    main()
