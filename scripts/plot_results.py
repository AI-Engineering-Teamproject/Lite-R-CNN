"""Export the comparison charts as PNG/SVG images and one self-contained HTML report.

Reads every result JSON under --reports (the same files the dashboard reads), picks one report per
model and kind (one inference preset, timing device and Light R-CNN checkpoint at a time), and writes
outputs/figures/<chart>_<theme>.{png,svg} plus report.html. PNG/SVG need the optional 'kaleido'
package and a local Chrome/Chromium; without them only the HTML is written.
"""

import argparse
import html
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from light_rcnn import viz
from light_rcnn.reporting import (
    BASELINE_VARIANT,
    CHECKPOINT_MODEL,
    MODELS,
    ComparisonWarning,
    checkpoint_paths,
    checkpoint_reports,
    checkpoints,
    comparability_warnings,
    improvement_warnings,
    load_reports,
    load_targets,
    same_file,
    select_baseline,
    select_reports,
    timing_devices,
    variant_labels,
    variants,
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


def match_checkpoint(choices: list[tuple[str, str]], wanted: str,
                     paths: dict[str, set[str]] | None = None) -> tuple[str, str]:
    """--checkpoint: a checkpoint key (or its start), part of its label or a weights path its
    reports record (relative, absolute or with Windows separators: same_file), ignoring case."""
    text = wanted.strip().replace("\\", "/").lower()
    paths = paths or {}
    exact = [choice for choice in choices if text in (choice[0].lower(), choice[1].lower())]
    path = str(PurePosixPath(text))  # './outputs//x.pth' -> 'outputs/x.pth'
    files = [choice for choice in choices
             if any(same_file(path, recorded.lower()) for recorded in paths.get(choice[0], ()))]
    partial = [choice for choice in choices
               if text in choice[1].lower() or choice[0].lower().startswith(text)]
    for matches in (exact, files, partial):
        if len(matches) == 1:
            return matches[0]
        if matches:
            raise SystemExit(f"--checkpoint {wanted!r} matches several checkpoints: "
                             + ", ".join(label for _, label in matches))

    def hint(key: str) -> str:  # a path key's start says nothing ('/home/user/D'): its shortest path
        recorded = paths.get(key, set())
        return min(recorded, key=len) if key in recorded else key[:12]

    raise SystemExit(f"No {MODELS[CHECKPOINT_MODEL]} checkpoint matches --checkpoint {wanted!r}. "
                     "Available checkpoints: "
                     + (", ".join(f"{label} ({hint(key)})" for key, label in choices) or "none"))


def name_checkpoint(figures, label: str) -> None:
    """Put the checkpoint into every chart title; the charts call the model plain 'Light R-CNN'."""
    for fig in figures.values():
        if not viz.is_empty(fig):
            title = fig.layout.title.text
            fig.update_layout(title_text=" · ".join(filter(None, (title, html.escape(label)))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", default="outputs", help="Folder scanned recursively for result JSON")
    parser.add_argument("--out", help="Output folder (default: outputs/figures, "
                                      "outputs/figures/<variant> for another --variant)")
    parser.add_argument("--config", default="configs/coco.yaml", help="Reads comparison.targets")
    parser.add_argument("--theme", choices=("light", "dark", "both"), default="light")
    parser.add_argument("--formats", nargs="+", choices=("png", "svg", "pdf"), default=["png", "svg"])
    parser.add_argument("--scale", type=float, default=2.0, help="PNG pixel density multiplier")
    parser.add_argument("--width", type=int, default=1100)
    parser.add_argument("--title", default="Light R-CNN 성능 비교 보고서")
    parser.add_argument("--device", help="지연 시간·단계별 그래프에 쓸 측정 장치 "
                                         "(기본: 가장 최근 장치, 대시보드와 같음)")
    parser.add_argument("--variant", help="그릴 추론 설정 (기본: baseline, 대시보드와 같음). "
                                          "compare_improvements.py 결과처럼 설정이 여러 개인 폴더에서 "
                                          "다른 설정을 고르면 기본 설정과의 개선 전/후 그래프도 그림")
    parser.add_argument("--checkpoint", help="그릴 Light R-CNN 체크포인트: 이름 일부(예: v2), 가중치 "
                                             "sha256(앞부분만 써도 됨) 또는 가중치 경로(예: "
                                             "outputs/coco_4090_v2/best.pth). 기본: 체크포인트가 여러 "
                                             "개면 가장 최근 것 (대시보드와 같음)")
    args = parser.parse_args()

    reports, errors = load_reports(args.reports)
    for error in errors:
        print(f"[skip] unreadable result file: {error}", file=sys.stderr)
    if not reports:
        raise SystemExit(f"No result JSON found under {args.reports}. Run the evaluation/benchmark scripts first.")
    # One inference preset at a time, so a folder of several presets never mixes them in a chart.
    found = variants(reports)
    if args.variant and args.variant not in found:
        raise SystemExit(f"No results measured with inference preset {args.variant!r}. "
                         f"Available presets: {', '.join(found)}")
    variant = args.variant or found[0]
    if len(found) > 1:
        print(f"[info] 추론 설정: {variant} (다른 설정은 --variant로 선택: "
              f"{', '.join(name for name in found if name != variant)})")
    # Light R-CNN v2 and v3 share one model key (compare_improvements.py measures both): draw one.
    choices = checkpoints(reports)
    checkpoint, checkpoint_label, checkpoint_note = None, None, None
    if args.checkpoint:
        checkpoint, checkpoint_label = match_checkpoint(choices, args.checkpoint,
                                                        checkpoint_paths(reports))
    elif len(choices) > 1:
        measured = [choice for choice in choices if any(
            report.model == CHECKPOINT_MODEL and report.variant == variant
            for report in checkpoint_reports(reports, choice[0]))]
        if measured:  # the newest checkpoint that has results with this preset
            checkpoint, checkpoint_label = measured[0]
            newest = ("가장 최근" if checkpoint == choices[0][0]
                      else f"이 추론 설정({variant})으로 측정한 것 중 가장 최근")
            checkpoint_note = ComparisonWarning(
                "warning", f"결과 폴더에 {MODELS[CHECKPOINT_MODEL]} 체크포인트가 {len(choices)}개"
                           f"({', '.join(label for _, label in choices)}) 섞여 있어 {newest} "
                           f"체크포인트인 {checkpoint_label}만 그렸습니다. 다른 체크포인트는 "
                           "--checkpoint로 고르세요.")
    if checkpoint is not None:
        others = [label for key, label in choices if key != checkpoint]
        print(f"[info] {MODELS[CHECKPOINT_MODEL]} 체크포인트: {checkpoint_label}"
              + (f" (다른 체크포인트는 --checkpoint로 선택: {', '.join(others)})" if others else ""))
        reports = checkpoint_reports(reports, checkpoint)
        if args.checkpoint and not any(report.model == CHECKPOINT_MODEL and report.variant == variant
                                       for report in reports):
            raise SystemExit(f"Checkpoint {checkpoint_label!r} has no results measured with "
                             f"inference preset {variant!r}.")
    devices = timing_devices([report for report in reports if report.variant == variant])
    if args.device and args.device not in devices:
        raise SystemExit(f"No latency/stage results measured on {args.device!r}. "
                         f"Available devices: {', '.join(devices) or 'none'}")
    device = args.device or (devices[0] if devices else None)
    if device:
        print(f"[info] 지연 시간 측정 장치: {device}")
        others = [label for label in devices if label != device]
        if others:
            print(f"[info] 다른 장치 결과는 제외했습니다 (--device로 선택): {', '.join(others)}")
    selected = select_reports(reports, device=device, variant=variant)
    baseline, label = None, None
    if variant != BASELINE_VARIANT:  # the same comparison run's baseline when the folder has it
        baseline = select_baseline(reports, selected, device=device)
        label = variant_labels(reports).get(variant, variant)
    targets = load_targets(args.config)
    warnings = comparability_warnings(selected, targets.get("latency_ms"))
    if baseline:
        warnings += improvement_warnings(baseline, selected)
    if checkpoint_note:
        warnings.insert(0, checkpoint_note)  # says which run every chart comes from
    scope = " · ".join(part for part in (checkpoint_label, label, device) if part)
    title = args.title + (f" ({scope})" if scope else "")
    default_out = "outputs/figures" if variant == BASELINE_VARIANT else f"outputs/figures/{variant}"
    out_dir = Path(args.out or default_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = viz.summary_rows(selected)

    themes = ("light", "dark") if args.theme == "both" else (args.theme,)
    images_ok = True
    for theme in themes:
        figures = viz.build_all(selected, targets, theme, baseline, label)
        if checkpoint_label:
            name_checkpoint(figures, checkpoint_label)
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
