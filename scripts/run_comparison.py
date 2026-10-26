"""Evaluate, benchmark and profile the compared detectors in sequence, then open the dashboard."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

from light_rcnn.reporting import display_path

ROOT = Path(__file__).resolve().parents[1]
# Run order is fixed (baseline -> student -> control) regardless of the --models order.
MODELS = {"faster_rcnn": "Faster R-CNN", "light_rcnn": "Light R-CNN", "yolo26": "YOLO26"}
DEFAULT_CONFIG = "configs/coco.yaml"
CPU_CONFIG = "configs/coco_cpu.yaml"
TIMING_KEYS = ("warmup", "iterations", "image_count")


def run_step(label: str, command: list[str], env: dict[str, str] | None = None,
             required: bool = True, hint: str = "Fix the error above and rerun.") -> bool:
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}", flush=True)
    returncode = subprocess.run(command, cwd=ROOT, env=env, check=False).returncode
    if returncode and required:
        raise SystemExit(f"\n'{label}' failed (exit code {returncode}). {hint}")
    return returncode == 0


def project_path(path: str | Path) -> Path:
    """Relative paths (CLI and config) are relative to the repository, where every step runs."""
    return (ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate, benchmark and profile Faster R-CNN, Light R-CNN and YOLO26, "
        "export the charts, then open Streamlit."
    )
    parser.add_argument("--config",
                        help=f"Shared data/comparison settings passed to every step "
                             f"(default: {DEFAULT_CONFIG})")
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS),
                        help="Models to run (default: all); always run Faster -> Light -> YOLO26")
    parser.add_argument("--device", default="cpu",
                        help="PyTorch device; defaults to cpu. On cpu with the default --config, "
                             f"unset timing counts come from {CPU_CONFIG}")
    parser.add_argument("--source",
                        help="Shared image file or folder used for all model benchmarks; "
                             "defaults to the config val_images")
    parser.add_argument("--single-image", metavar="IMAGE",
                        help="Run the latency benchmarks on exactly this image "
                             "(skips COCO accuracy evaluation)")
    parser.add_argument("--inference-only", action="store_true",
                        help="Skip COCO AP evaluation; only benchmark a shared image file/folder")
    parser.add_argument("--yolo-weights", help="Defaults to comparison.control.weights")
    parser.add_argument("--yolo-nms-free", action="store_true",
                        help="Run YOLO26 with the one-to-one (NMS-free) head")
    parser.add_argument("--light-weights", help="Defaults to <train.output_dir>/best.pth")
    parser.add_argument("--faster-weights",
                        help="Local Faster R-CNN checkpoint; omit to download torchvision weights")
    parser.add_argument("--train-light", action="store_true",
                        help="Train Light R-CNN before comparing")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit COCO evaluation images for every model (0 = all)")
    parser.add_argument("--warmup", type=int, help="Defaults to comparison.warmup")
    parser.add_argument("--iterations", type=int, help="Defaults to comparison.iterations")
    parser.add_argument("--image-count", type=int, help="Defaults to comparison.image_count")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Write reports without launching Streamlit")
    args = parser.parse_args()
    models = [model for model in MODELS if model in args.models]
    config_given = args.config is not None
    args.config = args.config or DEFAULT_CONFIG

    if args.single_image and args.source:
        parser.error("use either --single-image or --source, not both")
    if args.single_image:
        args.source = args.single_image
        args.inference_only = True

    config_path = project_path(args.config)
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {display_path(config_path)}")
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    val_images = project_path(config["data"]["val_images"])
    val_annotations = project_path(config["data"]["val_annotations"])
    source = project_path(args.source) if args.source else val_images
    comparison = config.get("comparison") or {}
    timing = {key: getattr(args, key) for key in TIMING_KEYS}
    gpu_counts = (args.device.split(":")[0] == "cpu" and comparison.get("device") != "cpu"
                  and None in timing.values())
    counts = comparison
    if gpu_counts and not config_given:
        # The default config's counts are sized for a GPU; keep its paths but time with the CPU preset.
        with open(project_path(CPU_CONFIG), encoding="utf-8") as file:
            counts = yaml.safe_load(file)["comparison"]
    timing = {key: counts[key] if value is None else value for key, value in timing.items()}

    required_paths = [source] if args.inference_only else [val_images, val_annotations, source]
    missing = [display_path(path) for path in dict.fromkeys(required_paths) if not path.exists()]
    if missing:
        if args.inference_only:
            raise SystemExit("Benchmark source not found:\n  - " + "\n  - ".join(missing))
        raise SystemExit(
            "COCO comparison prerequisites are missing:\n  - " + "\n  - ".join(missing)
            + "\nDownload/extract COCO val2017 images and annotations, then run this script again."
        )

    profiled = " (R-CNN 계열은 단계별 분석으로 한 번 더)" if models != ["yolo26"] else ""
    print(f"지연 시간 측정: 예열 {timing['warmup']}회, 반복 {timing['iterations']}회, "
          f"이미지 최대 {timing['image_count']}장 ({args.device}){profiled}", flush=True)
    if gpu_counts and config_given:
        print(f"[warn] {display_path(config_path)}의 측정 횟수는 GPU 기준입니다 "
              f"(comparison.device: {comparison.get('device')}). CPU에서는 --config {CPU_CONFIG} "
              "또는 --warmup/--iterations/--image-count를 지정하세요.", file=sys.stderr, flush=True)
    elif gpu_counts:
        print(f"기본 config({DEFAULT_CONFIG})의 측정 횟수는 GPU 기준이라 {CPU_CONFIG}의 횟수를 "
              f"씁니다. CPU 설정 전체를 쓰려면 --config {CPU_CONFIG}, 다른 횟수는 "
              "--warmup/--iterations/--image-count로 지정하세요.", flush=True)

    if args.train_light:
        train_images = project_path(config["data"]["train_images"])
        train_annotations = project_path(config["data"]["train_annotations"])
        missing_train = [display_path(path) for path in (train_images, train_annotations)
                         if not path.exists()]
        if missing_train:
            raise SystemExit(
                "--train-light also requires the COCO train split:\n  - "
                + "\n  - ".join(missing_train)
            )
        run_step("Light R-CNN 학습", [
            sys.executable, "scripts/train.py", "--config", str(config_path),
            "--device", args.device,
        ])
    light_weights = project_path(args.light_weights
                                 or Path(config["train"]["output_dir"]) / "best.pth")
    if "light_rcnn" in models and not light_weights.is_file():
        if args.inference_only and not args.light_weights:
            print(
                f"Warning: Light R-CNN checkpoint {display_path(light_weights)} is missing. "
                "Benchmarking its architecture with an ImageNet-pretrained backbone and an "
                "untrained detector head; accuracy is not meaningful.",
                flush=True,
            )
            light_weights = None
        else:
            raise SystemExit(
                f"Light R-CNN checkpoint not found: {light_weights}\n"
                "Pass a trained checkpoint with --light-weights, train it first with "
                f"`python scripts/train.py --config {args.config}`, rerun this command with "
                "`--train-light`, or skip it with --models."
            )
    faster_weights = project_path(args.faster_weights) if args.faster_weights else None
    if "faster_rcnn" in models and faster_weights and not faster_weights.is_file():
        raise SystemExit(f"Faster R-CNN checkpoint not found: {faster_weights}")

    env = os.environ.copy()
    env["YOLO_CONFIG_DIR"] = str(ROOT / "outputs/.ultralytics")
    (ROOT / "outputs/.ultralytics").mkdir(parents=True, exist_ok=True)
    report_dir = ROOT / "outputs/comparison_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir.mkdir(parents=True, exist_ok=True)
    env["LIGHT_RCNN_RESULTS_DIR"] = str(report_dir)

    common = ["--config", str(config_path), "--device", args.device]
    limit = ["--limit", str(args.limit)] if args.limit else []
    benchmark = ["--source", str(source)]
    for key, value in timing.items():
        benchmark += [f"--{key.replace('_', '-')}", str(value)]
    # YOLO26 always predicts one image at a time; batching pads R-CNN inputs and changes AP.
    batch_size = ["--batch-size", str(comparison.get("batch_size", 1))]

    def output(model: str, kind: str) -> list[str]:
        return ["--output", str(report_dir / f"{model}_{kind}.json")]

    for index, model in enumerate(models, start=1):
        prefix = f"{index}/{len(models)} {MODELS[model]}"
        others = [other for other in models if other != model]
        hint = "Fix the error above"
        hint += f", or skip {MODELS[model]} with --models {' '.join(others)}." if others else "."
        if model == "faster_rcnn" and faster_weights is None:
            hint += (" If the torchvision COCO weights could not be downloaded, pass a local "
                     "checkpoint with --faster-weights.")
        hint += f"\nReports finished so far are in {display_path(report_dir)}."

        if model == "yolo26":
            yolo = [*common]
            if args.yolo_weights:
                yolo += ["--weights", args.yolo_weights]
            if args.yolo_nms_free:
                yolo.append("--nms-free")
            if not args.inference_only:
                run_step(f"{prefix}: COCO 평가", [
                    sys.executable, "scripts/evaluate_yolo26.py", *yolo, *limit,
                    *output(model, "accuracy"),
                ], env, hint=hint)
            run_step(f"{prefix}: 지연 시간 측정", [
                sys.executable, "scripts/benchmark_yolo26.py", *yolo, *benchmark,
                *output(model, "latency"),
            ], env, hint=hint)
            continue

        rcnn = ["--model", model, *common]
        weights = light_weights if model == "light_rcnn" else faster_weights
        if weights is not None:
            rcnn += ["--weights", str(weights)]
        if not args.inference_only:
            run_step(f"{prefix}: COCO 평가", [
                sys.executable, "scripts/evaluate_coco.py", *rcnn, *limit, *batch_size,
                *output(model, "accuracy"),
            ], env, hint=hint)
        run_step(f"{prefix}: 지연 시간 측정", [
            sys.executable, "scripts/benchmark.py", *rcnn, *benchmark, *output(model, "latency"),
        ], env, hint=hint)
        run_step(f"{prefix}: 단계별 지연 시간 분석", [
            sys.executable, "scripts/profile_stages.py", *rcnn, *benchmark,
            *output(model, "stages"),
        ], env, hint=hint)

    # A failed chart export must not block the dashboard; the JSON reports are already written.
    figures_dir = report_dir / "figures"
    exported = run_step("결과 그래프 내보내기", [
        sys.executable, "scripts/plot_results.py", "--config", str(config_path),
        "--reports", str(report_dir), "--out", str(figures_dir),
    ], env, required=False)
    if not exported:
        print(f"[warn] Chart export failed; the reports in {display_path(report_dir)} are still "
              "usable.", file=sys.stderr)

    if args.inference_only:
        print("\nInference-only mode reports latency only; COCO accuracy evaluation was skipped.")
        if args.single_image:
            print("Single-image latency is a smoke check, not a representative distribution.")
    print(f"\nReports saved to {display_path(report_dir)}"
          + (f"; charts in {display_path(figures_dir)}." if exported else "."), flush=True)
    if args.no_dashboard:
        print("Run Streamlit with this folder selected to view them.")
        return

    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "dashboard/app.py"],
        cwd=ROOT,
        env=env,
    )
    print(f"\nDashboard started (PID {server.pid}): http://localhost:8501", flush=True)


if __name__ == "__main__":
    main()
