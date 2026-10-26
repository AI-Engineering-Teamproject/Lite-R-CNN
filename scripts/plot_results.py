"""Export the comparison charts as PNG/SVG images and one self-contained HTML report.

Reads every result JSON under --reports (the same files the dashboard reads), picks one report per
model and kind, and writes outputs/figures/<chart>_<theme>.{png,svg} plus report.html. PNG/SVG need
the optional 'kaleido' package and a local Chrome/Chromium; without them only the HTML is written.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from light_rcnn import viz
from light_rcnn.reporting import (
    comparability_warnings,
    load_reports,
    load_targets,
    select_reports,
    timing_devices,
)


def export_images(figures, out_dir: Path, theme: str, formats, scale: float, width: int) -> list[Path]:
    written = []
    for name, fig in figures.items():
        if viz.is_empty(fig):
            continue
        for fmt in formats:
            path = out_dir / f"{name}_{theme}.{fmt}"
            fig.write_image(str(path), format=fmt, width=width, height=fig.layout.height, scale=scale)
            written.append(path)
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", default="outputs", help="Folder scanned recursively for result JSON")
    parser.add_argument("--out", default="outputs/figures")
    parser.add_argument("--config", default="configs/coco.yaml", help="Reads comparison.targets")
    parser.add_argument("--theme", choices=("light", "dark", "both"), default="light")
    parser.add_argument("--formats", nargs="+", choices=("png", "svg", "pdf"), default=["png", "svg"])
    parser.add_argument("--scale", type=float, default=2.0, help="PNG pixel density multiplier")
    parser.add_argument("--width", type=int, default=1100)
    parser.add_argument("--title", default="Light R-CNN 성능 비교 보고서")
    parser.add_argument("--device", help="지연 시간·단계별 그래프에 쓸 측정 장치 "
                                         "(기본: 가장 최근 장치, 대시보드와 같음)")
    args = parser.parse_args()

    reports, errors = load_reports(args.reports)
    for error in errors:
        print(f"[skip] unreadable result file: {error}", file=sys.stderr)
    if not reports:
        raise SystemExit(f"No result JSON found under {args.reports}. Run the evaluation/benchmark scripts first.")
    devices = timing_devices(reports)
    if args.device and args.device not in devices:
        raise SystemExit(f"No latency/stage results measured on {args.device!r}. "
                         f"Available devices: {', '.join(devices) or 'none'}")
    device = args.device or (devices[0] if devices else None)
    if device:
        print(f"[info] 지연 시간 측정 장치: {device}")
        others = [label for label in devices if label != device]
        if others:
            print(f"[info] 다른 장치 결과는 제외했습니다 (--device로 선택): {', '.join(others)}")
    selected = select_reports(reports, device=device)
    targets = load_targets(args.config)
    warnings = comparability_warnings(selected, targets.get("latency_ms"))
    title = args.title + (f" ({device})" if device else "")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = viz.summary_rows(selected)

    themes = ("light", "dark") if args.theme == "both" else (args.theme,)
    images_ok = True
    for theme in themes:
        figures = viz.build_all(selected, targets, theme)
        report_path = out_dir / ("report.html" if theme == "light" else f"report_{theme}.html")
        report_path.write_text(viz.html_report(figures, rows, warnings, selected, theme, title), encoding="utf-8")
        print(f"wrote {report_path}")
        if not images_ok:
            continue
        try:
            for path in export_images(figures, out_dir, theme, args.formats, args.scale, args.width):
                print(f"wrote {path}")
        except Exception as exc:  # noqa: BLE001 - kaleido/Chrome failures vary; keep the HTML report
            images_ok = False
            print(f"[warn] PNG/SVG export skipped ({type(exc).__name__}: {exc}).\n"
                  "       It needs the kaleido package (python -m pip install kaleido) and a Chrome or "
                  "Edge browser;\n       without one, run: python -c \"import kaleido; "
                  "kaleido.get_chrome_sync()\"", file=sys.stderr)
    for warning in warnings:
        print(f"[{warning.level}] {warning.message}")


if __name__ == "__main__":
    main()
