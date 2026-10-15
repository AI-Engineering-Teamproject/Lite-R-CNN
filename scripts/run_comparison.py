"""Run the three detector evaluations/benchmarks in sequence, then open the dashboard."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_step(label: str, command: list[str]) -> None:
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate and benchmark Faster R-CNN, Light R-CNN and YOLO26, then open Streamlit."
    )
    parser.add_argument("--config", default="configs/coco.yaml")
    parser.add_argument("--device", default="cpu", help="PyTorch device; defaults to cpu")
    parser.add_argument("--source", default="data/coco/images/val2017",
                        help="Shared image file or folder used for all model benchmarks")
    parser.add_argument("--inference-only", action="store_true",
                        help="Skip COCO AP evaluation; only benchmark a shared image/file folder")
    parser.add_argument("--yolo-weights", default="yolo26n.pt")
    parser.add_argument("--light-weights", default="outputs/coco/best.pth")
    parser.add_argument("--train-light", action="store_true", help="Train Light R-CNN before comparing")
    parser.add_argument("--limit", type=int, default=0, help="Limit Faster/Light COCO evaluation images")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--image-count", type=int, default=16)
    parser.add_argument("--no-dashboard", action="store_true", help="Write reports without launching Streamlit")
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve()
    source = Path(args.source)
    if not source.is_absolute():
        source = ROOT / source
    val_images = ROOT / "data/coco/images/val2017"
    val_annotations = ROOT / "data/coco/annotations/instances_val2017.json"
    light_weights = (ROOT / args.light_weights).resolve()

    required_paths = [source] if args.inference_only else [config_path, val_images, val_annotations]
    missing = [str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
               for path in required_paths if not path.exists()]
    if missing:
        if args.inference_only:
            raise SystemExit("Benchmark source not found:\n  - " + "\n  - ".join(missing))
        raise SystemExit("COCO comparison prerequisites are missing:\n  - " + "\n  - ".join(missing)
                         + "\nDownload/extract COCO val2017 images and annotations, then run this script again.")

    if args.train_light:
        train_images = ROOT / "data/coco/images/train2017"
        train_annotations = ROOT / "data/coco/annotations/instances_train2017.json"
        missing_train = [str(path.relative_to(ROOT)) for path in (train_images, train_annotations)
                         if not path.exists()]
        if missing_train:
            raise SystemExit(
                "--train-light also requires the COCO train split:\n  - "
                + "\n  - ".join(missing_train)
            )
        run_step("Train Light R-CNN", [sys.executable, "scripts/train.py", "--config", args.config, "--device", args.device])
    if not light_weights.is_file():
        if args.inference_only:
            print(
                "Warning: Light R-CNN checkpoint is missing. Benchmarking its architecture with an "
                "ImageNet-pretrained backbone and an untrained detector head; accuracy is not meaningful.",
                flush=True,
            )
            light_weights = None
        else:
            raise SystemExit(
                f"Light R-CNN checkpoint not found: {light_weights}\n"
                "Train it first with `python scripts/train.py --config configs/coco.yaml`, "
                "or rerun this command with `--train-light`."
            )

    yolo_env = os.environ.copy()
    yolo_env["YOLO_CONFIG_DIR"] = str(ROOT / "outputs/.ultralytics")
    (ROOT / "outputs/.ultralytics").mkdir(parents=True, exist_ok=True)
    report_dir = ROOT / "outputs/comparison_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir.mkdir(parents=True, exist_ok=True)
    yolo_env["LIGHT_RCNN_RESULTS_DIR"] = str(report_dir)

    common_benchmark = [
        "--source", str(source),
        "--device", args.device,
        "--warmup", str(args.warmup),
        "--iterations", str(args.iterations),
        "--image-count", str(args.image_count),
    ]

    if not args.inference_only:
        run_step("1/3 Faster R-CNN: COCO 평가", [
            sys.executable, "scripts/evaluate_coco.py", "--config", args.config,
            "--model", "faster_rcnn", "--device", args.device,
            "--output", str(report_dir / "faster_rcnn_coco_metrics.json"),
            *( ["--limit", str(args.limit)] if args.limit else [] ),
        ])
    run_step("1/3 Faster R-CNN: 지연 시간 측정", [
        sys.executable, "scripts/benchmark.py", "--model", "faster_rcnn",
        *common_benchmark, "--output", str(report_dir / "faster_rcnn_latency.json"),
    ])

    if not args.inference_only:
        run_step("2/3 Light R-CNN: COCO 평가", [
            sys.executable, "scripts/evaluate_coco.py", "--config", args.config,
            "--model", "light_rcnn", "--weights", str(light_weights), "--device", args.device,
            "--output", str(report_dir / "light_rcnn_coco_metrics.json"),
            *( ["--limit", str(args.limit)] if args.limit else [] ),
        ])
    light_benchmark = [sys.executable, "scripts/benchmark.py", "--model", "light_rcnn"]
    if light_weights is not None:
        light_benchmark.extend(["--weights", str(light_weights)])
    light_benchmark.extend(common_benchmark)
    light_benchmark.extend(["--output", str(report_dir / "light_rcnn_latency.json")])
    run_step("2/3 Light R-CNN: 지연 시간 측정", light_benchmark)

    yolo_device = "0" if args.device == "cuda" else args.device
    if not args.inference_only:
        run_step("3/3 YOLO26: COCO 평가", [
            sys.executable, "scripts/evaluate_yolo26.py", "--weights", args.yolo_weights,
            "--device", yolo_device,
            "--output", str(report_dir / "yolo26_coco_metrics.json"),
            *( ["--limit", str(args.limit)] if args.limit else [] ),
        ])
    run_step("3/3 YOLO26: 지연 시간 측정", [
        sys.executable, "scripts/benchmark_yolo26.py", "--weights", args.yolo_weights,
        "--device", yolo_device, "--source", str(source),
        "--warmup", str(args.warmup), "--iterations", str(args.iterations),
        "--image-count", str(args.image_count),
        "--output", str(report_dir / "yolo26_latency.json"),
    ],)

    if args.no_dashboard:
        print(f"\nReports saved to {report_dir}. Run Streamlit with this folder selected to view them.")
        return

    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "dashboard/app.py"],
        cwd=ROOT,
        env=yolo_env,
    )
    print(f"\nDashboard started (PID {server.pid}): http://localhost:8501", flush=True)


if __name__ == "__main__":
    main()
