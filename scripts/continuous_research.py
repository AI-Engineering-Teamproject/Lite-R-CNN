"""Run resumable Lite R-CNN COCO experiments until stopped, keeping the best checkpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def config_paths_exist(config: dict) -> bool:
    data = config["data"]
    paths = [
        ROOT / data[name]
        for name in ("train_images", "train_annotations", "val_images", "val_annotations")
    ]
    return (
        all(
            path.is_dir() if index in (0, 2) else path.is_file()
            for index, path in enumerate(paths)
        )
        and any(paths[0].glob("*.jpg"))
        and any(paths[2].glob("*.jpg"))
    )


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def run_training(config_path: Path, output_dir: Path, device: str, epochs: int,
                 resume: bool, init_weights: Path | None,
                 train_limit: int | None, val_limit: int | None) -> None:
    if resume:
        command = [
            sys.executable,
            "scripts/train.py",
            "--resume",
            str(output_dir / "last.pth"),
            "--epochs",
            str(epochs),
            "--device",
            device,
        ]
    else:
        command = [
            sys.executable,
            "scripts/train.py",
            "--config",
            str(config_path),
            "--device",
            device,
            "--epochs",
            str(epochs),
        ]
        if init_weights:
            command += ["--init-weights", str(init_weights)]
    if train_limit is not None:
        command += ["--train-limit", str(train_limit)]
    if val_limit is not None:
        command += ["--val-limit", str(val_limit)]
    print("\n$ " + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_report(report_path: Path, records: list[dict], state: dict) -> None:
    best = state.get("incumbent_ap")
    lines = [
        "# Lite R-CNN 지속 학습 연구 기록",
        "",
        f"- 갱신: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- 진행 상태: round {state.get('round', 1)} 학습 중/대기",
        f"- 최고 검증 AP@[.50:.95]: {'미측정' if best is None else f'{best * 100:.3f}%'}",
        f"- 현재 최고 체크포인트: `{state.get('incumbent', '아직 없음')}`",
        f"- 데이터: COCO 2017 (train_limit={state.get('train_limit') or 0}, val_limit={state.get('val_limit') or 0}; 0은 전체 split)",
        "- 범위: Lite R-CNN 모델 설계·학습만 포함. Faster R-CNN/YOLO26 비교는 팀원 담당.",
        "",
        "## 실험 이력",
        "",
        "| Round | 시작점 | Seed | Learning rate | Epochs | Best AP | Epoch | Incumbent 갱신 | 결과 체크포인트 |",
        "|---:|---|---:|---:|---:|---:|---:|:---:|---|",
    ]
    if records:
        for row in records:
            ap = row.get("best_ap")
            lines.append(
                f"| {row['round']} | {row['start_from']} | {row['seed']} | {row['learning_rate']:.3g} | "
                f"{row['epochs']} | {'—' if ap is None else f'{ap * 100:.3f}%'} | "
                f"{row.get('best_epoch', '—')} | {'예' if row.get('promoted') else '아니오'} | "
                f"`{row['checkpoint']}` |"
            )
    else:
        lines.append("| — | — | — | — | — | 학습 결과 없음 | — | — | — |")
    lines.extend([
        "",
        "각 round는 최고 모델의 가중치에서 새 optimizer로 시작합니다. 새 결과가 기존 최고 AP를 넘을 때만 incumbent를 교체합니다. 중단하려면 Ctrl+C를 누르고, 같은 실행 명령으로 다시 시작하면 현재 round의 last.pth부터 이어갑니다.",
        "",
        "## 유의 사항",
        "",
        "이 자동 루프는 정해진 학습률 조정 규칙으로 재학습을 반복합니다. 구조 변경이나 연구적 해석은 기록을 보고 다음 코드 변경으로 반영해야 하며, 모델 코드가 스스로 바뀌지는 않습니다. subset 실험은 전체 COCO 결과로 간주하지 않습니다.",
        "",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/coco_cpu.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cycles", type=int, default=0,
                        help="Completed experiment cycles; 0 means keep running until Ctrl+C")
    parser.add_argument("--epochs", type=int, help="Total epochs per experiment cycle")
    parser.add_argument("--train-limit", type=int, help="Optional fixed training subset")
    parser.add_argument("--val-limit", type=int, help="Optional fixed validation subset")
    parser.add_argument("--output-root", default="outputs/research")
    parser.add_argument("--skip-data-prepare", action="store_true")
    args = parser.parse_args()
    if args.cycles < 0:
        parser.error("--cycles must be nonnegative")
    if args.train_limit is not None and args.train_limit < 0:
        parser.error("--train-limit must be nonnegative")
    if args.val_limit is not None and args.val_limit < 0:
        parser.error("--val-limit must be nonnegative")

    base_config_path = (ROOT / args.config).resolve()
    base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    epochs = args.epochs or int(base_config["train"]["epochs"])
    output_root = (ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "state.json"
    records_path = output_root / "experiments.jsonl"
    report_path = output_root / "RESEARCH_REPORT.md"
    records = read_records(records_path)
    signature = {
        "model": base_config.get("model"),
        "num_classes": base_config.get("num_classes"),
        "image_size": base_config.get("image_size"),
        "data": base_config.get("data"),
        "distillation": base_config.get("distillation"),
        "train": {
            key: base_config["train"].get(key)
            for key in (
                "batch_size", "workers", "learning_rate", "weight_decay", "amp",
                "warmup_steps", "lr_milestones", "lr_gamma", "validate_every",
                "train_limit", "val_limit",
            )
        },
    }

    if not config_paths_exist(base_config):
        if args.skip_data_prepare:
            raise FileNotFoundError("COCO dataset is incomplete; run scripts/prepare_hf_coco.py")
        print("COCO files are incomplete; preparing the configured Hugging Face dataset.", flush=True)
        subprocess.run([sys.executable, "scripts/prepare_hf_coco.py"], cwd=ROOT, check=True)
        if not config_paths_exist(base_config):
            raise FileNotFoundError("Dataset preparation finished, but configured COCO paths are incomplete")

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("signature") not in (None, signature):
            raise ValueError(
                "The saved research state uses a different model/data/training setup. "
                "Start a new campaign with another --output-root."
            )
        for key in ("train_limit", "val_limit"):
            requested = getattr(args, key)
            if requested is not None and requested != state.get(key, 0):
                raise ValueError(
                    f"This campaign uses {key}={state.get(key)}; use a new --output-root to change it."
                )
        state["signature"] = signature
    else:
        base_output = (ROOT / base_config["train"]["output_dir"]).resolve()
        round_one = output_root / "round_001"
        if (base_output / "last.pth").exists():
            initial_output, initial_config = base_output, base_config_path
        elif (round_one / "last.pth").exists() and (round_one / "config.yaml").exists():
            initial_output, initial_config = round_one, round_one / "config.yaml"
        else:
            initial_output, initial_config = round_one, round_one / "config.yaml"
        state = {
            "round": 1,
            "active_output": str(initial_output),
            "active_config": str(initial_config),
            "learning_rate": float(base_config["train"]["learning_rate"]),
            "incumbent": None,
            "incumbent_ap": None,
            "completed_cycles": 0,
            "resume_active": (initial_output / "last.pth").exists(),
            "train_limit": args.train_limit if args.train_limit is not None else base_config["train"].get("train_limit", 0),
            "val_limit": args.val_limit if args.val_limit is not None else base_config["train"].get("val_limit", 0),
            "signature": signature,
        }
        if state["resume_active"]:
            initial = load_checkpoint(initial_output / "last.pth")
            saved_train = initial.get("config", {}).get("train", {})
            state["train_limit"] = saved_train.get("train_limit", state["train_limit"])
            state["val_limit"] = saved_train.get("val_limit", state["val_limit"])
            state["learning_rate"] = float(
                saved_train.get("learning_rate", state["learning_rate"])
            )
            state["incumbent_ap"] = initial.get("best_ap")
            if (initial_output / "best.pth").exists():
                state["incumbent"] = str(initial_output / "best.pth")
        atomic_json(state_path, state)

    lr_floor = float(base_config["train"]["learning_rate"]) / 32.0
    cycles_this_run = 0
    try:
        while args.cycles == 0 or cycles_this_run < args.cycles:
            round_number = int(state["round"])
            active_output = Path(state["active_output"])
            active_config = Path(state["active_config"])
            active_checkpoint = active_output / "last.pth"
            already_complete = False
            if active_checkpoint.exists():
                current = load_checkpoint(active_checkpoint)
                already_complete = int(current.get("epoch", 0)) >= epochs
            if already_complete:
                checkpoint = current
            else:
                if not active_checkpoint.exists():
                    if round_number > 1 and not state.get("incumbent"):
                        raise RuntimeError("Cannot start a new cycle before a best checkpoint exists")
                    config = deepcopy(base_config)
                    config["device"] = args.device
                    config["seed"] = int(base_config.get("seed", 42)) + round_number - 1
                    config["train"]["epochs"] = epochs
                    config["train"]["learning_rate"] = float(state["learning_rate"])
                    if state.get("train_limit") is not None:
                        config["train"]["train_limit"] = state["train_limit"]
                    if state.get("val_limit") is not None:
                        config["train"]["val_limit"] = state["val_limit"]
                    config["train"]["output_dir"] = str(active_output)
                    active_config.parent.mkdir(parents=True, exist_ok=True)
                    active_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
                    state["active_config"] = str(active_config)
                    atomic_json(state_path, state)
                run_training(active_config, active_output, args.device, epochs,
                             resume=active_checkpoint.exists(),
                             init_weights=Path(state["incumbent"]) if round_number > 1 else None,
                             train_limit=state.get("train_limit"),
                             val_limit=state.get("val_limit"))
                checkpoint = load_checkpoint(active_checkpoint)

            candidate_ap = checkpoint.get("best_ap")
            promoted = state.get("incumbent") is None or (
                candidate_ap is not None and
                (state.get("incumbent_ap") is None or candidate_ap > state["incumbent_ap"])
            )
            if promoted:
                state["incumbent"] = str(active_output / "best.pth")
                state["incumbent_ap"] = candidate_ap
            history = read_records(active_output / "history.jsonl")
            best_epoch = checkpoint.get("best_epoch")
            row = {
                "round": round_number,
                "start_from": "COCO pretrained backbone" if round_number == 1 else f"incumbent before round {round_number}",
                "seed": checkpoint.get("config", {}).get("seed", base_config.get("seed", 42)),
                "learning_rate": float(state["learning_rate"]),
                "epochs": int(checkpoint.get("epoch", epochs)),
                "best_ap": candidate_ap,
                "best_epoch": best_epoch,
                "promoted": promoted,
                "checkpoint": str(active_output / "best.pth"),
                "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
            if not any(int(existing["round"]) == round_number for existing in records):
                records.append(row)
                with records_path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")
            state["completed_cycles"] = max(int(state.get("completed_cycles", 0)), round_number)
            state["resume_active"] = False
            cycles_this_run += 1
            print(
                f"Round {round_number} complete: AP={None if candidate_ap is None else f'{candidate_ap * 100:.3f}%'}; "
                f"incumbent={'updated' if promoted else 'kept'} at {state.get('incumbent_ap')}.",
                flush=True,
            )
            next_rate = float(state["learning_rate"])
            if promoted:
                next_rate = max(lr_floor, next_rate * 0.5)
            else:
                next_rate = min(float(base_config["train"]["learning_rate"]), next_rate * 2.0)
            state.update({
                "round": round_number + 1,
                "active_output": str(output_root / f"round_{round_number + 1:03d}"),
                "active_config": str(output_root / f"round_{round_number + 1:03d}" / "config.yaml"),
                "learning_rate": next_rate,
                "resume_active": False,
            })
            atomic_json(state_path, state)
            write_report(report_path, records, state)

        print(f"Completed requested {cycles_this_run} research cycle(s). Report: {report_path}")
    except KeyboardInterrupt:
        print("\nResearch loop stopped by user; the last completed epoch checkpoint is preserved.")
    finally:
        write_report(report_path, records, state)
        print(f"Research report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
