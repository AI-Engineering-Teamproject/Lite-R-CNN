"""Plot per-epoch COCO metrics and losses from Lite R-CNN training runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_run(run_dir: Path) -> dict:
    path = run_dir / "history.jsonl"
    if not path.is_file():
        raise SystemExit(f"History not found: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"History is empty: {path}")
    if any(row.get("validation") is None for row in rows):
        raise SystemExit(f"Validation metrics are missing in {path}")
    return {"name": run_dir.name, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["outputs/coco_4090_v2", "outputs/coco_4090_v3"],
        help="Run directories containing history.jsonl",
    )
    parser.add_argument(
        "--output",
        default="outputs/training_curves/lite_rcnn_v2_v3",
        help="Output path without extension; PNG and SVG are written",
    )
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Install plotting dependency with: python -m pip install matplotlib") from exc

    runs = [load_run(Path(value)) for value in args.runs]
    colors = ["#2864a5", "#df7626", "#31936a", "#9b59b6", "#c44569"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    fig.suptitle("Lite R-CNN v2 vs v3 — COCO training history", fontsize=17, fontweight="bold")

    panels = [
        ("AP@[.50:.95]", "AP"),
        ("AP50 and AP75", "AP"),
        ("AP by object size", "AP"),
        ("Training loss", "Loss"),
        ("RPN and distillation loss", "Loss"),
        ("Learning rate", "Learning rate"),
    ]

    for ax, (title, ylabel) in zip(axes.flat, panels):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    for run_index, run in enumerate(runs):
        color = colors[run_index % len(colors)]
        rows = run["rows"]
        epochs = [row["epoch"] for row in rows]
        name = run["name"]
        label = f"{name} (n={len(rows)})"

        axes[0, 0].plot(epochs, [r["validation"]["AP_50_95"] for r in rows], marker="o", ms=3, color=color, label=label)

        for metric, style in (("AP_50", "-"), ("AP_75", "--")):
            axes[0, 1].plot(epochs, [r["validation"][metric] for r in rows], linestyle=style, color=color, label=f"{name} {metric}")

        for key, style in (("loss_total", "-"), ("loss_classifier", "--"), ("loss_box_reg", ":")):
            if key in rows[0]["losses"]:
                axes[1, 0].plot(epochs, [r["losses"][key] for r in rows], linestyle=style, color=color, label=f"{name} {key.removeprefix('loss_')}")

        for key, style in (("loss_objectness", "-."), ("loss_rpn_box_reg", (0, (1, 1)))):
            if key in rows[0]["losses"]:
                axes[1, 1].plot(epochs, [r["losses"][key] for r in rows], linestyle=style, alpha=0.8, color=color, label=f"{name} {key.removeprefix('loss_')}")

        kd_keys = [key for key in rows[0]["losses"] if key.startswith("loss_kd_")]
        if kd_keys:
            kd_total = [sum(row["losses"].get(key, 0.0) for key in kd_keys) for row in rows]
            axes[1, 1].plot(epochs, kd_total, color=color, label=f"{name} KD total")
            for key, style in zip(kd_keys, ("--", ":")):
                axes[1, 1].plot(epochs, [row["losses"].get(key, 0.0) for row in rows], linestyle=style, alpha=0.75, color=color, label=f"{name} {key.removeprefix('loss_kd_')}")

        for metric, marker in (("AP_small", "o"), ("AP_medium", "s"), ("AP_large", "^") ):
            axes[0, 2].plot(epochs, [r["validation"][metric] for r in rows], marker=marker, markevery=max(1, len(rows) // 12), ms=3, color=color, label=f"{name} {metric.removeprefix('AP_')}")

        axes[1, 2].plot(epochs, [row["learning_rate"] for row in rows], color=color, label=label)

    for ax in axes.flat:
        ax.legend(fontsize=8, frameon=True)
        ax.margins(x=0.02)
    axes[1, 2].set_yscale("log")
    fig.text(
        0.01,
        0.005,
        "AP curves use the recorded validation split (check each history.jsonl for evaluation_scope). "
        "Losses are epoch means; v3 KD curves are omitted for runs without distillation.",
        fontsize=9,
        color="#555555",
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        destination = output.with_suffix(f".{extension}")
        fig.savefig(destination, dpi=180, bbox_inches="tight")
        print(f"Wrote {destination}")
    plt.close(fig)


if __name__ == "__main__":
    main()
