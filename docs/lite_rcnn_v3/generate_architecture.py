"""Generate the source-derived, English-only Lite R-CNN v3 architecture figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MplPath


HERE = Path(__file__).resolve().parent
INK = "#263445"
MUTED = "#536475"
EDGE = "#46596B"
RULE = "#93A1AE"
PAPER = "#FFFFFF"
BLUE = "#DCEAF5"
BLUE_SIDE = "#B8CAD9"
GREEN = "#DDEBDD"
GREEN_SIDE = "#B8CEB8"
ORANGE = "#F3E4CC"
PURPLE = "#E9E1F1"
PALE = "#F1F4F6"


def cuboid(ax, x, y, w, h, *, fill=BLUE, side=BLUE_SIDE, depth=0.16,
           grid=False, label=None, label_size=8):
    """Draw one shallow 3D tensor volume; labels are placed outside when dense."""
    dx, dy = depth, depth * 0.72
    ax.add_patch(Polygon([(x, y + h), (x + dx, y + h + dy),
                          (x + w + dx, y + h + dy), (x + w, y + h)],
                         closed=True, facecolor=side, edgecolor=EDGE, linewidth=0.8, zorder=4))
    ax.add_patch(Polygon([(x + w, y), (x + w + dx, y + dy),
                          (x + w + dx, y + h + dy), (x + w, y + h)],
                         closed=True, facecolor=side, edgecolor=EDGE, linewidth=0.8, zorder=4))
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=EDGE,
                           linewidth=0.9, zorder=5))
    if grid:
        for frac in (0.25, 0.5, 0.75):
            ax.plot([x + w * frac, x + w * frac], [y + 0.06, y + h - 0.06],
                    color=RULE, linewidth=0.45, zorder=6)
        for frac in (0.33, 0.66):
            ax.plot([x + 0.06, x + w - 0.06], [y + h * frac, y + h * frac],
                    color=RULE, linewidth=0.45, zorder=6)
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=label_size, color=INK, zorder=7, linespacing=1.12)


def block(ax, x, y, w, h, title, details="", *, fill=BLUE, dashed=False,
          title_size=8.8, detail_size=7.4, rounding=0.07):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.04,rounding_size={rounding}",
        facecolor=fill, edgecolor=EDGE, linewidth=0.9,
        linestyle="--" if dashed else "-", zorder=4,
    ))
    if details:
        ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=INK, zorder=6)
        ax.text(x + w / 2, y + h * 0.34, details, ha="center", va="center",
                fontsize=detail_size, color=INK, zorder=6, linespacing=1.2)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=INK, zorder=6,
                linespacing=1.17)


def arrow(ax, start, end, *, color=INK, dashed=False, lw=1.15, label=None,
          label_xy=None, label_size=7.4, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=10,
        connectionstyle=f"arc3,rad={rad}", color=color, linewidth=lw,
        linestyle="--" if dashed else "-", shrinkA=2, shrinkB=2, zorder=3,
    ))
    if label:
        lx, ly = label_xy or ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(lx, ly, label, ha="center", va="center", fontsize=label_size,
                color=color, bbox=dict(facecolor=PAPER, edgecolor="none", pad=1.2), zorder=9)


def route_arrow(ax, points, *, color=INK, dashed=False, label=None,
                label_xy=None, label_size=7.4, lw=1.1):
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1)
    path = MplPath(points, codes)
    ax.add_patch(PathPatch(path, fill=False, edgecolor=color, linewidth=lw,
                           linestyle="--" if dashed else "-", zorder=2))
    p0, p1 = points[-2], points[-1]
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=10,
                                 color=color, linewidth=lw,
                                 linestyle="--" if dashed else "-", shrinkA=0,
                                 shrinkB=1, zorder=3))
    if label:
        lx, ly = label_xy or points[len(points) // 2]
        ax.text(lx, ly, label, ha="center", va="center", fontsize=label_size,
                color=color, bbox=dict(facecolor=PAPER, edgecolor="none", pad=1.3), zorder=9)


def main() -> None:
    fig, ax = plt.subplots(figsize=(22, 12.5), dpi=170)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 12.5)
    ax.axis("off")

    ax.text(0.38, 12.05, "LITE R-CNN v3 | IMPLEMENTED ARCHITECTURE",
            fontsize=17, color=INK, fontweight="bold", ha="left", va="center")
    ax.text(0.4, 11.62, "Two-stage anchor-based detector built on TorchVision Faster R-CNN",
            fontsize=10, color=MUTED, ha="left", va="center")
    ax.plot([0.35, 21.65], [11.32, 11.32], color=RULE, linewidth=0.8)
    ax.text(0.42, 11.03, "INFERENCE GRAPH | SOLID ARROWS",
            fontsize=10.2, color=INK, fontweight="bold", ha="left", va="center")

    # Input transform and backbone.
    cuboid(ax, 0.48, 9.11, 1.28, 1.45, fill=GREEN, side=GREEN_SIDE,
           grid=True, label="RGB\nimage", label_size=9)
    block(ax, 2.08, 9.02, 1.66, 1.58, "INPUT TRANSFORM",
          "aspect ratio kept\nlong edge ~ 640\nnormalize + pad", fill=PALE,
          title_size=8.5, detail_size=7.2)
    cuboid(ax, 4.05, 9.00, 2.16, 1.64, fill=BLUE, side=BLUE_SIDE,
           grid=True, label="MobileNetV3-Large\nfeatures[3, 6, 12, 16]", label_size=8.2)
    arrow(ax, (1.83, 9.83), (2.02, 9.83))
    arrow(ax, (3.78, 9.83), (4.00, 9.83))
    arrow(ax, (6.42, 9.83), (6.55, 9.83))

    # Feature taps use 3D tensor icons with readable external labels.
    c_levels = [
        (10.28, "C2 | 24 ch\n160 x 160 | s4"),
        (9.72, "C3 | 40 ch\n80 x 80 | s8"),
        (9.16, "C4 | 112 ch\n40 x 40 | s16"),
        (8.60, "C5 | 960 ch\n20 x 20 | s32"),
    ]
    for y, label in c_levels:
        cuboid(ax, 6.62, y, 0.34, 0.36, fill="#E8EFF5", side=BLUE_SIDE, depth=0.09)
        ax.text(7.13, y + 0.18, label, ha="left", va="center", fontsize=6.8,
                color=INK, linespacing=1.0)

    block(ax, 8.55, 8.63, 2.28, 1.96, "LITE FPN | 64 CH",
          "lateral: Conv1x1 + BN + ReLU\ntop-down: nearest upsample + add\nDWConv3x3 + PWConv1x1 + BN + SiLU",
          fill=GREEN, title_size=9.0, detail_size=7.1)
    arrow(ax, (7.98, 9.78), (8.50, 9.78))

    # Pyramid labels are outside the 3D blocks to stay readable.
    p_levels = [
        (10.35, "P2 | 64 ch\n160 x 160"),
        (9.87, "P3 | 64 ch\n80 x 80"),
        (9.39, "P4 | 64 ch\n40 x 40"),
        (8.91, "P5 | 64 ch\n20 x 20"),
        (8.43, "P6 | 64 ch\n10 x 10"),
        (7.95, "P7 | 64 ch\n5 x 5"),
    ]
    for y, label in p_levels:
        cuboid(ax, 11.07, y, 0.34, 0.31, fill="#E8EFF5", side=BLUE_SIDE, depth=0.09)
        ax.text(11.58, y + 0.155, label, ha="left", va="center", fontsize=7.0,
                color=INK, linespacing=1.0)
    ax.text(12.65, 8.37, "P6 = MaxPool(P5), k=1 s=2\nP7 = MaxPool(P6), k=1 s=2",
            ha="left", va="center", fontsize=6.6, color=MUTED, linespacing=1.2)
    arrow(ax, (10.88, 9.76), (11.00, 9.76))

    # Shared RPN head with actual operations and outputs.
    block(ax, 14.33, 8.77, 2.25, 1.82, "ANCHOR RPN | P2-P7",
          "shared RPNHead\nConv3x3 + ReLU\n1x1 objectness + box deltas\nanchors: 8/16/32/64/128/256\nratios: 0.5/1/2",
          fill=ORANGE, title_size=8.7, detail_size=6.7)
    block(ax, 17.04, 8.77, 2.02, 1.82, "PROPOSALS",
          "pre-NMS top-k <= 1000 / level\npost-NMS <= 300 / image",
          fill=ORANGE, title_size=9.2, detail_size=7.6)
    arrow(ax, (12.80, 9.73), (14.27, 9.73), label="P2-P7", label_xy=(13.52, 10.02))
    arrow(ax, (16.62, 9.68), (16.98, 9.68))

    # The bracket marks the exact P2-P5 subset that supplies RoIAlign.
    ax.plot([12.53, 12.53], [8.84, 10.66], color="#39739C", linewidth=1.4, zorder=3)
    ax.plot([12.39, 12.53], [10.66, 10.66], color="#39739C", linewidth=1.4, zorder=3)
    ax.plot([12.39, 12.53], [8.84, 8.84], color="#39739C", linewidth=1.4, zorder=3)

    # Separate orthogonal routes into RoIAlign: feature maps and proposals.
    ax.text(5.04, 7.48, "RoI CLASSIFICATION AND BOX REGRESSION",
            fontsize=9.6, color=INK, fontweight="bold", ha="left", va="center")
    block(ax, 5.05, 5.10, 1.86, 1.38, "MULTI-SCALE RoIAlign",
          "P2-P5 | 7 x 7\nsampling ratio 2",
          fill=BLUE, title_size=8.2, detail_size=7.3)
    block(ax, 7.31, 5.10, 2.00, 1.38, "COMPACT BOX HEAD",
          "Conv1x1: 64 -> 32\nAdaptiveAvgPool 7 x 7\nflatten: 1568",
          fill=GREEN, title_size=8.5, detail_size=7.1)
    block(ax, 9.72, 5.10, 1.56, 1.38, "FC + ReLU", "1568 -> 256",
          fill=GREEN, title_size=9, detail_size=7.6)
    block(ax, 11.68, 5.10, 1.98, 1.38, "FAST R-CNN PREDICTOR",
          "91 class logits\nclass-specific box deltas",
          fill=PURPLE, title_size=8.5, detail_size=7.1)
    block(ax, 14.06, 5.10, 2.12, 1.38, "POSTPROCESS",
          "score > 0.05\nclass-wise NMS IoU 0.5\nmax 100 detections",
          fill=ORANGE, title_size=9, detail_size=7.1)
    cuboid(ax, 16.65, 5.22, 1.55, 1.12, fill=GREEN, side=GREEN_SIDE,
           grid=True, label="boxes\nlabels\nscores", label_size=8.0)
    for start, end in [((6.95, 5.79), (7.26, 5.79)),
                       ((9.35, 5.79), (9.67, 5.79)),
                       ((11.32, 5.79), (11.63, 5.79)),
                       ((13.70, 5.79), (14.01, 5.79)),
                       ((16.23, 5.79), (16.59, 5.79))]:
        arrow(ax, start, end)

    # Feature route stays on the upper lane; proposal route stays on the lower lane.
    route_arrow(ax, [(12.53, 9.73), (12.53, 7.78), (11.35, 7.78),
                     (11.35, 7.20), (5.78, 7.20), (5.78, 6.54)],
                color="#39739C", label="P2-P5 feature maps", label_xy=(8.5, 7.20), label_size=7.2)
    route_arrow(ax, [(18.02, 8.73), (18.02, 6.84), (6.38, 6.84), (6.38, 6.54)],
                color="#9A6E2D", label="student proposals", label_xy=(12.2, 6.84), label_size=7.2)

    # The lower lane shows the training-only teacher/student data flow.
    ax.plot([0.35, 21.65], [4.56, 4.56], color=RULE, linewidth=0.8)
    ax.text(0.42, 4.28, "TRAINING ONLY | KNOWLEDGE DISTILLATION (TEACHER IS NOT IN THE INFERENCE GRAPH)",
            fontsize=9.1, color=INK, fontweight="bold", ha="left", va="center")

    block(ax, 0.50, 2.18, 3.48, 1.34, "STUDENT TRAINING PASS",
          "same transformed image; backbone, Lite FPN, RPN, RoI head\nP2-P5 maps, student proposals/logits, detector losses",
          fill=GREEN, dashed=True, title_size=8.9, detail_size=7.0)
    block(ax, 0.50, 0.75, 3.48, 1.13, "FROZEN TEACHER PASS",
          "same transformed image; COCO V1 ResNet50-FPN\nteacher RPN skipped; teacher maps and RoI logits",
          fill=PURPLE, dashed=True, title_size=8.9, detail_size=7.0)
    block(ax, 4.74, 1.45, 4.02, 1.75, "DISTILLATION LOSSES",
          "Feature: student post-FPN P2-P5 vs teacher FPN P2-P5\n64 -> 256 adapters; normalized Smooth L1; weight 0.5\nRoI logits on student proposals: KL, temperature 2; weight 1.0",
          fill=BLUE, dashed=True, title_size=9.0, detail_size=7.1)
    block(ax, 9.58, 1.45, 3.18, 1.75, "TOTAL STUDENT LOSS",
          "RPN objectness + RPN box\nRoI classification + RoI box\n+ weighted feature and RoI-logit KD",
          fill=GREEN, dashed=True, title_size=9.0, detail_size=7.4)
    block(ax, 13.58, 1.45, 3.20, 1.75, "OPTIMIZER UPDATE",
          "updates student + feature adapters\nteacher remains frozen",
          fill=PALE, dashed=True, title_size=9.0, detail_size=7.5)

    # Shared, already-transformed inputs branch to the student and teacher passes.
    ax.plot([2.91, 2.91, 0.30, 0.30], [8.96, 7.65, 7.65, 1.30],
            color="#745A90", linewidth=1.0, linestyle="--", zorder=2)
    ax.text(0.48, 3.78, "same transformed / normalized / padded tensor",
            ha="left", va="center", fontsize=6.7, color="#745A90",
            bbox=dict(facecolor=PAPER, edgecolor="none", pad=1.0), zorder=9)
    arrow(ax, (0.30, 2.83), (0.46, 2.83), color="#745A90", dashed=True)
    arrow(ax, (0.30, 1.30), (0.46, 1.30), color="#745A90", dashed=True)

    # Both model passes supply their corresponding targets to the KD losses.
    arrow(ax, (4.02, 2.94), (4.69, 2.67), color="#745A90", dashed=True)
    arrow(ax, (4.02, 1.30), (4.69, 1.82), color="#745A90", dashed=True)
    route_arrow(ax, [(4.00, 3.18), (4.22, 3.78), (10.02, 3.78), (10.02, 3.24)],
                color="#39739C", dashed=True,
                label="student detector losses", label_xy=(7.08, 3.78), label_size=6.8)
    arrow(ax, (8.80, 2.33), (9.53, 2.33), color="#745A90", dashed=True)
    arrow(ax, (12.80, 2.33), (13.53, 2.33), color="#745A90", dashed=True)

    ax.text(0.42, 0.43,
            "Tensor sizes assume a square 640x640 image; non-square inputs preserve aspect ratio.  "
            "Solid arrows: inference. Dashed arrows: training-only paths and parameter updates.",
            ha="left", va="center", fontsize=7.1, color=MUTED)

    fig.subplots_adjust(left=0.02, right=0.99, top=0.99, bottom=0.025)
    svg_path = HERE / "lite_rcnn_v3_architecture.svg"
    png_path = HERE / "lite_rcnn_v3_architecture.png"
    fig.savefig(svg_path, format="svg", facecolor=PAPER, bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=170, facecolor=PAPER, bbox_inches="tight")
    plt.close(fig)
    # Matplotlib may leave trailing spaces in wrapped SVG path data. They are
    # valid XML, but trimming them keeps the committed vector diff clean.
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Wrote {svg_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
