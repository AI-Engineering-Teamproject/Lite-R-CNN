"""Plot the recorded CPU comparison of Faster R-CNN and Lite R-CNN v2/v3."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs/evaluation_runs/20261101_054559/evaluation.json"
DEFAULT_OUTPUT = ROOT / "outputs/evaluation_runs/20261101_054559/cpu_three_model_comparison.png"
EXPECTED = ("Faster R-CNN", "Lite R-CNN v2", "Lite R-CNN v3")
COLORS = ("#3569a8", "#ed7d31", "#55a868")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    meta, rows = payload["metadata"], payload["results"]
    if tuple(row["model"] for row in rows) != EXPECTED:
        raise ValueError("Unexpected model names/order in evaluation JSON")
    if meta.get("device") != "cpu" or meta.get("images_evaluated") != 1000:
        raise ValueError("Input must be the CPU evaluation of the shared 1,000-image subset")
    if any(row.get("device") != "cpu" or row.get("images_evaluated") != 1000 for row in rows):
        raise ValueError("All three entries must be CPU results for 1,000 images")

    names = [row["model"] for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    fig.suptitle(
        f"CPU comparison — Faster R-CNN and Lite R-CNN v2/v3\n"
        f"COCO val2017 subset: 1,000 images, {meta['image_size']} px, batch 1, FP32 (seed {meta['seed']})",
        fontsize=17, fontweight="bold")

    ax = axes[0, 0]
    width = 0.23
    for offset, key, label, color in zip((-width, 0, width),
            ("AP", "AP50", "AP75"), ("AP@[.50:.95]", "AP50", "AP75"), COLORS):
        bars = ax.bar(x + offset, [row[key] * 100 for row in rows], width, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    ax.set_title("COCO accuracy on the shared 1,000-image subset")
    ax.set_ylabel("AP (%)")
    ax.set_xticks(x, names)
    ax.set_ylim(0, max(row["AP50"] * 100 for row in rows) * 1.25)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=3, loc="upper right")

    ax = axes[0, 1]
    totals = [row["evaluation_wall_time_seconds"] for row in rows]
    bars = ax.bar(x, totals, color=COLORS, width=0.62)
    labels = [f"{int(round(t)) // 60}:{int(round(t)) % 60:02d}\n({t:.1f} s)" for t in totals]
    ax.bar_label(bars, labels=labels, padding=4, fontsize=9)
    ax.set_title("Measured elapsed time for the full 1,000-image loop")
    ax.set_ylabel("Time (seconds)")
    ax.set_xticks(x, names)
    ax.set_ylim(0, max(totals) * 1.24)
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.02, 0.97, "Includes image loading/decoding, inference, and per-image PR/proposal-recall work",
            transform=ax.transAxes, va="top", fontsize=8, color="#444444")

    ax = axes[1, 0]
    width = 0.32
    p50 = [row["latency_p50_ms"] for row in rows]
    p95 = [row["latency_p95_ms"] for row in rows]
    b50 = ax.bar(x - width / 2, p50, width, label="p50", color="#4c78a8")
    b95 = ax.bar(x + width / 2, p95, width, label="p95", color="#f58518")
    ax.bar_label(b50, fmt="%.0f", padding=2, fontsize=8)
    ax.bar_label(b95, fmt="%.0f", padding=2, fontsize=8)
    ax.set_title("Single-image CPU latency (100 timed iterations)")
    ax.set_ylabel("Latency (ms; logarithmic scale)")
    ax.set_yscale("log")
    ax.set_xticks(x, names)
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.legend()

    ax = axes[1, 1]
    params = [row["parameters_M"] for row in rows]
    bars = ax.bar(x, params, color=COLORS, width=0.62)
    ax.bar_label(bars, fmt="%.2f M", padding=4, fontsize=9)
    ax.set_title("Model parameters")
    ax.set_ylabel("Parameters (millions)")
    ax.set_xticks(x, names)
    ax.set_ylim(0, max(params) * 1.2)
    ax.grid(axis="y", alpha=0.25)

    fig.text(0.5, -0.012,
        "Total-loop time is measured before COCOeval aggregation; it is not model-forward-only time. "
        "The separate latency benchmark excludes image decoding and cycles the first 8 images after 10 warmups. "
        "AP is for this 1,000-image subset, not the full 5,000-image validation set.",
        ha="center", va="top", fontsize=8, color="#444444")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved comparison graph: {args.output}")
    for row in rows:
        print(f"{row['model']}: {row['evaluation_wall_time_seconds']:.2f} seconds for 1,000 images")

if __name__ == "__main__":
    main()
