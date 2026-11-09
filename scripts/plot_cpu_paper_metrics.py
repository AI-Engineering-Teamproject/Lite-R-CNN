"""Create a paper-ready CPU comparison figure from the recorded COCO evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/evaluation_runs/20261101_054559/evaluation.json"
OUTPUT = ROOT / "outputs/evaluation_runs/20261101_054559/cpu_paper_metrics.png"
MODELS = ("Faster R-CNN", "Lite R-CNN v2", "Lite R-CNN v3")
COLORS = ("#2878B5", "#E87522", "#3A923A")


def main() -> None:
    result_file = json.loads(SOURCE.read_text(encoding="utf-8"))
    metadata, records = result_file["metadata"], result_file["results"]
    if metadata.get("device") != "cpu" or metadata.get("images_evaluated") != 1000:
        raise ValueError("Expected the CPU evaluation run with 1,000 COCO validation images")
    if tuple(record["model"] for record in records) != MODELS:
        raise ValueError("Unexpected model names or order in evaluation.json")

    labels = [record["model"] for record in records]
    y = np.arange(len(records))
    fig, axes = plt.subplots(3, 1, figsize=(11.0, 7.8), sharey=True)
    fig.suptitle("CPU Results on COCO val2017", fontsize=21, fontweight="bold", y=0.99)
    fig.text(
        0.5, 0.935,
        "1,000-image subset (seed 42)  |  640 px  |  batch size 1  |  FP32",
        ha="center", va="center", fontsize=14, color="#444444",
    )

    panels = [
        ("AP@[.50:.95]", [record["AP"] * 100 for record in records], "AP@[.50:.95] (%)", 38.0,
         [f"{record['AP'] * 100:.2f}%" for record in records]),
        ("AP50", [record["AP50"] * 100 for record in records], "AP50 (%)", 59.0,
         [f"{record['AP50'] * 100:.2f}%" for record in records]),
        ("Total time for the 1,000-image evaluation loop",
         [record["evaluation_wall_time_seconds"] / 60 for record in records],
         "Elapsed time (minutes)", 50.0,
         [f"{int(round(record['evaluation_wall_time_seconds'])) // 60}:"
          f"{int(round(record['evaluation_wall_time_seconds'])) % 60:02d}"
          for record in records]),
    ]

    for ax, (title, values, xlabel, xmax, annotations) in zip(axes, panels):
        bars = ax.barh(y, values, height=0.58, color=COLORS, edgecolor="white", linewidth=0.7)
        ax.bar_label(bars, labels=annotations, padding=7, fontsize=14, color="#222222")
        ax.set_title(title, loc="left", fontsize=16, fontweight="bold", pad=6)
        ax.set_xlim(0, xmax)
        ax.set_xlabel(xlabel, fontsize=15, labelpad=5)
        ax.set_yticks(y, labels, fontsize=14)
        ax.tick_params(axis="y", labelleft=True)
        ax.invert_yaxis()
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#888888")
        ax.spines["bottom"].set_color("#888888")
        ax.tick_params(axis="both", length=0, labelsize=13)

    fig.text(
        0.5, 0.015,
        "Total loop includes image loading/decoding, model inference, and per-image PR/proposal-recall calculations; "
        "it stops before COCOeval aggregation and is not model-forward-only time.",
        ha="center", va="bottom", fontsize=10, color="#555555", wrap=True,
    )
    fig.subplots_adjust(left=0.27, right=0.96, top=0.85, bottom=0.15, hspace=0.72)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT}")
    for record in records:
        seconds = record["evaluation_wall_time_seconds"]
        print(
            f"{record['model']}: AP={record['AP'] * 100:.2f}%, "
            f"AP50={record['AP50'] * 100:.2f}%, total_loop={seconds:.2f}s"
        )


if __name__ == "__main__":
    main()
