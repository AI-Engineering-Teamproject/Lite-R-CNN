"""Plot full-validation AP and training loss curves from Lite R-CNN run folders."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


COLORS = ("#2864a5", "#df7626", "#31936a", "#9b59b6")
WIDTH, HEIGHT = 1200, 700


def read_runs(run_dirs: list[Path]) -> list[dict]:
    runs = []
    for run_dir in run_dirs:
        history_path = run_dir / "history.jsonl"
        if not history_path.is_file():
            raise SystemExit(f"History not found: {history_path}")
        rows = [
            json.loads(line)
            for line in history_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows or any(row.get("validation") is None for row in rows):
            raise SystemExit(f"No validation AP records found in {history_path}")
        runs.append({"name": run_dir.name, "rows": rows})
    return runs


def panel(
    *,
    title: str,
    key: str,
    x: float,
    y: float,
    width: float,
    height: float,
    ymax: float,
    yticks: list[float],
    runs: list[dict],
    validation: bool,
) -> list[str]:
    out = [f'<text x="{x}" y="{y - 28}" class="panel-title">{html.escape(title)}</text>']
    x0, x1 = x + 56, x + width - 18
    y0, y1 = y + 10, y + height - 42

    def px(epoch: float) -> float:
        return x0 + (epoch - 1) / 35 * (x1 - x0)

    def py(value: float) -> float:
        return y1 - value / ymax * (y1 - y0)

    for tick in yticks:
        yy = py(tick)
        out.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" class="grid"/>')
        label = f"{tick:.2f}" if validation else f"{tick:.1f}"
        out.append(f'<text x="{x0 - 10}" y="{yy + 5:.1f}" text-anchor="end" class="tick">{label}</text>')
    for tick in (1, 5, 10, 15, 20, 25, 30, 36):
        xx = px(tick)
        out.append(f'<line x1="{xx:.1f}" y1="{y1}" x2="{xx:.1f}" y2="{y1 + 5}" class="axis"/>')
        out.append(f'<text x="{xx:.1f}" y="{y1 + 23}" text-anchor="middle" class="tick">{tick}</text>')
    out.extend(
        [
            f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" class="axis"/>',
            f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" class="axis"/>',
            f'<text x="{(x0 + x1) / 2:.1f}" y="{y1 + 40}" text-anchor="middle" class="axis-label">Epoch</text>',
        ]
    )

    for index, run in enumerate(runs):
        color = COLORS[index % len(COLORS)]
        rows = run["rows"]
        values = [
            row["validation"][key] if validation else row["losses"][key]
            for row in rows
        ]
        points = " ".join(
            f"{px(row['epoch']):.1f},{py(value):.1f}"
            for row, value in zip(rows, values)
        )
        out.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        best_index = max(range(len(values)), key=values.__getitem__) if validation else len(values) - 1
        best_row, best_value = rows[best_index], values[best_index]
        bx, by = px(best_row["epoch"]), py(best_value)
        out.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="5" fill="{color}"/>')
        if validation:
            dy = 23 if index == 0 else -12
            label = f"{run['name']} best {best_value:.3f} (e{best_row['epoch']})"
            out.append(
                f'<text x="{bx + 9:.1f}" y="{by + dy:.1f}" class="annotation" '
                f'fill="{color}">{html.escape(label)}</text>'
            )

    legend_y = y + height + 11
    legend_x = x + 64
    for index, run in enumerate(runs):
        color = COLORS[index % len(COLORS)]
        out.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 28}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        out.append(f'<text x="{legend_x + 36}" y="{legend_y + 5}" class="legend">{html.escape(run["name"])}</text>')
        legend_x += 190
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["outputs/coco_4090", "outputs/coco_4090_v2"],
        help="Run folders containing history.jsonl",
    )
    parser.add_argument(
        "--output", default="docs/figures/coco_training_curves.svg", help="Output SVG path"
    )
    args = parser.parse_args()
    runs = read_runs([Path(value) for value in args.runs])
    if len(runs) > len(COLORS):
        raise SystemExit(f"At most {len(COLORS)} runs can be plotted")

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#263238}.title{font-size:25px;font-weight:700}.subtitle{font-size:14px;fill:#54636b}.panel-title{font-size:19px;font-weight:700}.tick{font-size:12px;fill:#54636b}.axis-label{font-size:14px}.legend{font-size:13px}.annotation{font-size:12px;font-weight:700}.grid{stroke:#dce2e6;stroke-width:1}.axis{stroke:#64737b;stroke-width:1.4}</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="50" y="42" class="title">Lite R-CNN training progress on COCO</text>',
        '<text x="50" y="67" class="subtitle">Full val2017 AP@[.50:.95] and per-epoch mean training loss</text>',
    ]
    parts.extend(
        panel(
            title="Validation AP@[.50:.95]",
            key="AP_50_95",
            x=42,
            y=130,
            width=530,
            height=470,
            ymax=0.32,
            yticks=[0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
            runs=runs,
            validation=True,
        )
    )
    parts.extend(
        panel(
            title="Training loss",
            key="loss_total",
            x=625,
            y=130,
            width=530,
            height=470,
            ymax=1.6,
            yticks=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
            runs=runs,
            validation=False,
        )
    )
    parts.append('<text x="50" y="674" class="subtitle">AP lines use full val2017 evaluation; dots mark each run\'s best AP checkpoint.</text>')
    parts.append("</svg>")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
