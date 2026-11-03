"""Plotly figure builders shared by the Streamlit dashboard and scripts/plot_results.py.

Design rules (kept deliberately fixed so every chart reads as one system):
- Each model keeps one color and one marker shape everywhere; colors follow the model, never a rank.
  The three categorical colors pass the CVD / normal-vision checks for all pairs in both themes.
- One y-axis per chart (no dual axes); thin marks, hairline solid grid, values labeled selectively.
- Text uses ink colors, never series colors. Every chart has a table twin in the dashboard.
"""

from __future__ import annotations

import html
import math
from datetime import datetime, timezone
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from light_rcnn.reporting import (
    BASELINE_VARIANT,
    BASELINE_VARIANT_LABEL,
    COLD_START_LABELS,
    MODEL_ORDER,
    MODELS,
    STAGE_LABELS,
    Report,
    describe_environment,
    is_cpu_run,
    on_target_device,
)

BAR_MAX_PX = 24  # marks spec: no bar thicker than this at the export size
FONT_FAMILY = ('system-ui, -apple-system, "Segoe UI", "Malgun Gothic", "Apple SD Gothic Neo", '
               '"Noto Sans KR", "Noto Sans CJK KR", NanumGothic, sans-serif')

THEMES: dict[str, dict[str, Any]] = {
    "light": {
        "surface": "#fcfcfb",
        "page": "#f9f9f7",
        "ink": "#0b0b0b",
        "ink_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "wash": "rgba(11,11,11,0.045)",
        "border": "rgba(11,11,11,0.10)",
        "models": {"light_rcnn": "#2a78d6", "faster_rcnn": "#eb6834", "yolo26": "#1baf7a"},
        # Ordinal one-hue ramp for pipeline stages (validated: monotone L, adjacent dL >= 0.06).
        "stages": ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"],
    },
    "dark": {
        "surface": "#1a1a19",
        "page": "#0d0d0d",
        "ink": "#f0efec",
        "ink_secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "wash": "rgba(255,255,255,0.06)",
        "border": "rgba(255,255,255,0.10)",
        "models": {"light_rcnn": "#3987e5", "faster_rcnn": "#d95926", "yolo26": "#199e70"},
        "stages": ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95"],
    },
}

# Secondary encoding so identity never rides on color alone.
MODEL_SYMBOLS = {"light_rcnn": "circle", "faster_rcnn": "diamond", "yolo26": "square"}
# Pipeline stages drawn as stacked segments; 'input' (device copy + scaling) folds into preprocess.
CHART_STAGES = ["preprocess", "backbone", "rpn", "roi_heads", "postprocess"]
CHART_STAGE_LABELS = {**STAGE_LABELS, "preprocess": "입력·전처리"}

AP_METRICS = {
    "AP_50_95": "AP@[.50:.95]",
    "AP_50": "AP50",
    "AP_75": "AP75",
    "AP_small": "AP 소형",
    "AP_medium": "AP 중형",
    "AP_large": "AP 대형",
}
PERCENTILES = [("p50_ms", "p50"), ("p90_ms", "p90"), ("p95_ms", "p95"), ("p99_ms", "p99")]

Selected = dict[tuple[str, str], Report]
# Optional display name per model key; the dashboard names the chosen checkpoint ('Light R-CNN v3').
ModelLabels = dict[str, str] | None


def _theme(name: str) -> dict[str, Any]:
    return THEMES["dark" if name == "dark" else "light"]


def model_color(model: str, theme: str = "light") -> str:
    return _theme(theme)["models"][model]


def _label(model: str, model_labels: ModelLabels = None) -> str:
    return (model_labels or {}).get(model) or MODELS[model]


def _models_with(selected: Selected, kind: str) -> list[str]:
    return [model for model in MODEL_ORDER if (kind, model) in selected]


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _pct(value: Any) -> float | None:
    number = _num(value)
    return None if number is None or number < 0 else number * 100  # pycocotools -1 = no GT in range


def _samples(section: dict[str, Any]) -> list[float]:
    return [v for v in map(_num, section.get("samples_ms") or []) if v is not None]


def _class_ap(section: dict[str, Any]) -> dict[str, float]:
    raw = section.get("per_category_ap")
    if not isinstance(raw, dict):
        return {}
    return {name: ap for name, value in raw.items() if (ap := _num(value)) is not None}


def _off_target_latency(selected: Selected) -> list[Report]:
    return [r for (kind, _), r in selected.items() if kind == "latency" and not on_target_device(r)]


def _target_note(selected: Selected, *, short: bool = False) -> str:
    """The latency target is set for a T4 GPU; flag it when other hardware is graded against it."""
    off_target = _off_target_latency(selected)
    if not off_target:
        return ""
    if short:
        return " (T4 GPU 기준)"
    where = "CPU" if all(is_cpu_run(r) for r in off_target) else "T4 외 장치"
    return f" (T4 GPU 기준 · {where} 측정이라 참고용)"


def _size_stats(stats: dict) -> tuple[float | None, float | None]:
    """(parameters_m, gflops) of one graph: the fused graph when recorded, else the checkpoint."""
    if stats.get("deployed_gflops") is not None:  # older YOLO26 reports: gflops was the checkpoint
        return _num(stats.get("deployed_parameters_m")), _num(stats["deployed_gflops"])
    # R-CNN reports keep the checkpoint count in parameters_m and the fused one in deployed_*.
    params = stats.get("deployed_parameters_m") if "deployed_gflops" not in stats else None
    if params is None:
        params = stats.get("parameters_m")
    return _num(params), _num(stats.get("gflops"))


AXIS_TITLE_PX = 28  # automargin adds this below the plot for the x-axis title


def _cap_bar_gap(fig: go.Figure, rows: int) -> float:
    plot_px = fig.layout.height - fig.layout.margin.t - fig.layout.margin.b - AXIS_TITLE_PX
    return min(0.9, max(0.35, 1 - BAR_MAX_PX * rows / max(plot_px, 1)))


def _base_layout(fig: go.Figure, theme: str, *, title: str, subtitle: str | None = None,
                 height: int = 420, panel_titles: bool = False, panel_title_lines: int = 1
                 ) -> go.Figure:
    """Fixed header band in pixels: title, subtitle, legend row, then the plot (no overlaps)."""
    t = _theme(theme)
    header = 64 if subtitle else 40  # title (+ subtitle) block
    legend_top = header + 6
    plot_top = legend_top + 34 + (10 + 16 * panel_title_lines if panel_titles else 0)
    fig.update_layout(
        title={"text": f"<b>{title}</b>", "x": 0, "xanchor": "left", "xref": "container",
               "y": 1, "yanchor": "top", "yref": "container", "pad": {"t": 14, "l": 16},
               "font": {"size": 17, "color": t["ink"]},
               "subtitle": {"text": subtitle or "", "font": {"size": 13, "color": t["ink_secondary"]}}},
        height=height,
        margin={"l": 16, "r": 24, "t": plot_top, "b": 56},
        paper_bgcolor=t["surface"],
        plot_bgcolor=t["surface"],
        font={"family": FONT_FAMILY, "size": 13, "color": t["ink_secondary"]},
        hoverlabel={"bgcolor": t["surface"], "bordercolor": t["axis"],
                    "font": {"family": FONT_FAMILY, "size": 13, "color": t["ink"]}},
        legend={"orientation": "h", "yref": "container", "y": 1 - legend_top / height,
                "yanchor": "top", "xref": "container", "xanchor": "left", "x": 0.008,
                "font": {"color": t["ink_secondary"]}, "bgcolor": "rgba(0,0,0,0)",
                "itemclick": False, "itemdoubleclick": False},
        bargap=0.35,
        bargroupgap=0.12,
        uniformtext={"minsize": 11, "mode": "hide"},
    )
    axis = {"gridcolor": t["grid"], "gridwidth": 1, "griddash": "solid", "zeroline": False,
            "linecolor": t["axis"], "linewidth": 1, "showline": True, "ticks": "",
            "tickfont": {"color": t["muted"]}, "title": {"font": {"color": t["ink_secondary"]}},
            "automargin": True}
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def empty_figure(message: str, theme: str = "light", *, title: str = "", height: int = 260) -> go.Figure:
    t = _theme(theme)
    fig = go.Figure()
    _base_layout(fig, theme, title=title, height=height)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font={"size": 14, "color": t["muted"]})
    fig.update_layout(meta={"empty": True})
    return fig


def is_empty(fig: go.Figure) -> bool:
    return bool(isinstance(fig.layout.meta, dict) and fig.layout.meta.get("empty"))


# --------------------------------------------------------------------------- trade-off


def tradeoff_figure(selected: Selected, targets: dict[str, Any], theme: str = "light", *,
                    ap_metric: str | None = None, percentile: str | None = None,
                    model_labels: ModelLabels = None) -> go.Figure:
    """AP vs latency: a marker per model at the chosen percentile (p95 whisker when it is p50)."""
    t = _theme(theme)
    ap_metric = ap_metric or targets.get("ap_metric", "AP_50_95")
    percentile = percentile or targets.get("latency_percentile", "p50")
    lat_key = f"{percentile}_ms"
    title = "정확도 대 속도"
    whisker_note = ", 오른쪽 수염 끝 = p95" if percentile == "p50" else ""
    subtitle = (f"점 = {percentile} 지연 시간{whisker_note} · 왼쪽 위로 갈수록 좋음 · "
                f"음영 = 목표 영역")
    models = [m for m in MODEL_ORDER if ("accuracy", m) in selected and ("latency", m) in selected]
    if not models:
        return empty_figure("같은 모델의 정확도 결과와 지연 시간 결과가 둘 다 있어야 그려집니다.",
                            theme, title=title)
    fig = go.Figure()
    _base_layout(fig, theme, title=title, subtitle=subtitle, height=460)
    target_ms = _num(targets.get("latency_ms"))
    target_ap = _pct(targets.get("ap"))
    note = _target_note(selected)
    xs, ys = [], []
    for model in models:
        acc = selected[("accuracy", model)].section
        lat = selected[("latency", model)].section
        x, y = _num(lat.get(lat_key)), _pct(acc.get(ap_metric))
        if x is None or y is None:
            continue
        p95 = _num(lat.get("p95_ms"))
        whisker = max(p95 - x, 0.0) if p95 is not None and percentile == "p50" else None
        xs.append(x + (whisker or 0))
        ys.append(y)
        name = _label(model, model_labels)
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text", name=name,
            text=[name], textposition="top center",
            textfont={"color": t["ink"], "size": 13},
            marker={"size": 14, "color": t["models"][model], "symbol": MODEL_SYMBOLS[model],
                    "line": {"color": t["surface"], "width": 2}},
            error_x={"type": "data", "symmetric": False, "array": [whisker or 0], "arrayminus": [0],
                     "visible": whisker is not None, "color": t["models"][model], "thickness": 2,
                     "width": 6},
            customdata=[[p95 if p95 is not None else float("nan")]], cliponaxis=False,
            hovertemplate=(f"<b>%{{y:.1f}}%</b> {AP_METRICS.get(ap_metric, ap_metric)}<br>"
                           f"<b>%{{x:.2f}} ms</b> {percentile}<br>p95 %{{customdata[0]:.2f}} ms"
                           f"<extra>{name}</extra>"),
        ))
    if not xs:
        return empty_figure("선택한 지표 값이 결과 파일에 없습니다.", theme, title=title)
    x_max = max([*xs, target_ms or 0]) * 1.18 or 1
    y_max = min(max([*ys, target_ap or 0]) * 1.15 + 2, 100)
    if target_ms is not None and target_ap is not None:
        fig.add_shape(type="rect", x0=0, x1=target_ms, y0=target_ap, y1=y_max, layer="below",
                      fillcolor=t["wash"], line={"width": 0})
        fig.add_annotation(x=0, y=y_max, xanchor="left", yanchor="top", showarrow=False,
                           text=f"목표: {percentile} ≤ {target_ms:g} ms, "
                                f"{AP_METRICS.get(ap_metric, ap_metric)} ≥ {target_ap:g}%{note}",
                           font={"size": 12, "color": t["ink_secondary"]}, xshift=6, yshift=-4)
    if target_ms is not None:
        fig.add_vline(x=target_ms, line={"color": t["axis"], "width": 1})
    if target_ap is not None:
        fig.add_hline(y=target_ap, line={"color": t["axis"], "width": 1})
    fig.update_xaxes(title={"text": f"{percentile} 지연 시간 (ms) · 낮을수록 좋음"}, range=[0, x_max],
                     rangemode="tozero")
    fig.update_yaxes(title={"text": f"COCO {AP_METRICS.get(ap_metric, ap_metric)} (%) · 높을수록 좋음"},
                     range=[0, y_max], ticksuffix="%")
    fig.update_layout(showlegend=True)
    return fig


# --------------------------------------------------------------------------- latency


def latency_percentile_figure(selected: Selected, targets: dict[str, Any], theme: str = "light", *,
                              model_labels: ModelLabels = None) -> go.Figure:
    """Dumbbell per model from p50 to p99 so the tail latency is visible at a glance."""
    t = _theme(theme)
    title = "지연 시간 백분위수"
    models = _models_with(selected, "latency")
    if not models:
        return empty_figure("지연 시간 결과가 아직 없습니다.", theme, title=title)
    fig = go.Figure()
    _base_layout(fig, theme, title=title,
                 subtitle="선 = p50에서 p99까지의 범위 · 선이 길수록 가끔 크게 느려짐 (꼬리 지연)",
                 height=150 + 70 * len(models))
    labels = [_label(m, model_labels) for m in models]
    x_max = 0.0
    symbols = {"p50": "circle", "p90": "line-ns", "p95": "diamond", "p99": "square"}
    for model, label in zip(models, labels):
        lat = selected[("latency", model)].section
        values = [(name, _num(lat.get(key))) for key, name in PERCENTILES]
        values = [(name, v) for name, v in values if v is not None]
        if not values:
            continue
        x_max = max(x_max, *(v for _, v in values))
        color = t["models"][model]
        fig.add_trace(go.Scatter(x=[values[0][1], values[-1][1]], y=[label, label], mode="lines",
                                 line={"color": color, "width": 2}, hoverinfo="skip",
                                 showlegend=False))
        for name, value in values:
            labeled = name in ("p50", values[-1][0])
            fig.add_trace(go.Scatter(
                x=[value], y=[label], mode="markers+text" if labeled else "markers",
                text=[f"{name} {value:.1f}"] if labeled else None,
                textposition="bottom center" if name == "p50" else "top center",
                textfont={"color": t["ink_secondary"], "size": 12},
                marker={"size": 11 if name != "p90" else 12, "symbol": symbols[name], "color": color,
                        "line": {"color": t["surface"] if name != "p90" else color, "width": 2}},
                showlegend=False, cliponaxis=False,
                hovertemplate=f"<b>%{{x:.2f}} ms</b> {name}<extra>{label}</extra>",
            ))
    target_ms = _num(targets.get("latency_ms"))
    if target_ms is not None:
        target_label = f"목표 {target_ms:g} ms{_target_note(selected, short=True)}"
        fig.add_vline(x=target_ms, line={"color": t["axis"], "width": 1},
                      annotation={"text": target_label,
                                  "font": {"color": t["ink_secondary"], "size": 12}},
                      annotation_position="top right")
        x_max = max(x_max, target_ms)
    fig.update_xaxes(title={"text": "지연 시간 (ms)"}, range=[0, x_max * 1.15 or 1])
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(labels)), showgrid=False,
                     range=[-0.6, len(labels) - 0.4], tickfont={"color": t["ink"], "size": 13})
    # Legend for the percentile marker shapes (neutral ink so it never reads as a model color).
    for name, symbol in symbols.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name,
                                 marker={"symbol": symbol, "size": 10, "color": t["muted"],
                                         "line": {"color": t["muted"], "width": 2 if name == "p90" else 0}}))
    fig.update_layout(showlegend=True)
    return fig


def latency_distribution_figure(selected: Selected, targets: dict[str, Any], theme: str = "light", *,
                                model_labels: ModelLabels = None) -> go.Figure:
    """Box plot of the raw per-call samples (needs schema v2 latency reports)."""
    t = _theme(theme)
    title = "지연 시간 분포"
    models = [m for m in _models_with(selected, "latency")
              if _samples(selected[("latency", m)].section)]
    if not models:
        return empty_figure("원시 측정값(samples_ms)이 있는 결과가 없습니다. 새 벤치마크로 다시 측정하세요.",
                            theme, title=title)
    fig = go.Figure()
    _base_layout(fig, theme, title=title,
                 subtitle="상자 = 가운데 50% 측정값 · 가운데 선 = 중앙값(p50) · 점 = 튀는 값",
                 height=150 + 80 * len(models))
    labels = []
    for model in models:
        label = _label(model, model_labels)
        labels.append(label)
        samples = _samples(selected[("latency", model)].section)
        color = t["models"][model]
        fig.add_trace(go.Box(
            x=samples, y=[label] * len(samples), name=label, orientation="h",
            marker={"color": color, "size": 5, "opacity": 0.7}, line={"color": color, "width": 2},
            fillcolor=_with_alpha(color, 0.12), boxpoints="outliers", showlegend=False,
            hovertemplate="%{x:.2f} ms<extra>" + label + "</extra>",
        ))
    target_ms = _num(targets.get("latency_ms"))
    if target_ms is not None:
        target_label = f"목표 {target_ms:g} ms{_target_note(selected, short=True)}"
        fig.add_vline(x=target_ms, line={"color": t["axis"], "width": 1},
                      annotation={"text": target_label,
                                  "font": {"color": t["ink_secondary"], "size": 12}},
                      annotation_position="top")
    fig.update_xaxes(title={"text": "1회 추론 지연 시간 (ms)"}, rangemode="tozero")
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(labels)), showgrid=False,
                     tickfont={"color": t["ink"], "size": 13})
    return fig


def latency_stability_figure(selected: Selected, theme: str = "light", *, window: int = 15,
                             model_labels: ModelLabels = None) -> go.Figure:
    """Latency per iteration with a rolling median: reveals warm-up drift or thermal throttling."""
    t = _theme(theme)
    title = "측정 안정성"
    models = [m for m in _models_with(selected, "latency")
              if _samples(selected[("latency", m)].section)]
    if not models:
        return empty_figure("원시 측정값(samples_ms)이 있는 결과가 없습니다.", theme, title=title)
    window = min(window, max(len(_samples(selected[("latency", m)].section)) for m in models))
    fig = go.Figure()
    _base_layout(fig, theme, title=title,
                 subtitle=f"옅은 점 = 매 반복 측정값 · 선 = 최근 {window}회 중앙값 · "
                          "선이 계속 오르내리면 측정 환경이 불안정함 (발열, 다른 작업)",
                 height=420)
    for model in models:
        samples = _samples(selected[("latency", model)].section)
        color = t["models"][model]
        name = _label(model, model_labels)
        index = list(range(1, len(samples) + 1))
        fig.add_trace(go.Scatter(x=index, y=samples, mode="markers", name=name,
                                 marker={"size": 4, "color": color, "opacity": 0.28}, showlegend=False,
                                 hovertemplate="반복 %{x}: %{y:.2f} ms<extra>" + name + "</extra>"))
        fig.add_trace(go.Scatter(x=index, y=_rolling_median(samples, window), mode="lines",
                                 name=name, line={"color": color, "width": 2},
                                 hovertemplate="반복 %{x} 중앙값 %{y:.2f} ms<extra>" + name + "</extra>"))
    fig.update_xaxes(title={"text": "측정 반복 순서"}, rangemode="tozero")
    fig.update_yaxes(title={"text": "지연 시간 (ms)"}, rangemode="tozero")
    fig.update_layout(showlegend=True, hovermode="closest")
    return fig


def _left_align_panel_titles(fig: go.Figure, t: dict[str, Any]) -> None:
    """make_subplots centers panel titles; left-align each over its own panel instead."""
    for index, annotation in enumerate(fig.layout.annotations):
        axis = fig.layout["xaxis" if index == 0 else f"xaxis{index + 1}"]
        annotation.update(x=axis.domain[0], xanchor="left", align="left", yshift=6,
                          font={"color": t["ink_secondary"], "size": 13})


def _rolling_median(values: list[float], window: int) -> list[float | None]:
    """Full windows only (None until the first fills), so warm-up never shows a partial median."""
    window = max(1, min(window, len(values)))
    medians: list[float | None] = []
    for index in range(len(values)):
        if index < window - 1:
            medians.append(None)
            continue
        chunk = sorted(values[index - window + 1: index + 1])
        middle = len(chunk) // 2
        medians.append(chunk[middle] if len(chunk) % 2 else (chunk[middle - 1] + chunk[middle]) / 2)
    return medians


def _with_alpha(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip("#")
    red, green, blue = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


# --------------------------------------------------------------------------- accuracy


def accuracy_figure(selected: Selected, theme: str = "light", *,
                    model_labels: ModelLabels = None) -> go.Figure:
    """Two small multiples: AP by IoU threshold and AP by object size, one bar per model."""
    t = _theme(theme)
    title = "COCO 정확도 세부 지표"
    models = _models_with(selected, "accuracy")
    if not models:
        return empty_figure("정확도 결과가 아직 없습니다.", theme, title=title)
    groups = [("IoU 기준별", ["AP_50_95", "AP_50", "AP_75"]), ("물체 크기별", ["AP_small", "AP_medium", "AP_large"])]
    fig = make_subplots(rows=1, cols=2, subplot_titles=[name for name, _ in groups],
                        shared_yaxes=True, horizontal_spacing=0.06)
    _base_layout(fig, theme, title=title,
                 subtitle="막대가 높을수록 좋음 · AP50은 느슨한 기준, AP75는 엄격한 기준 · 크기별은 작은 물체에 약한지 보여 줌",
                 height=460, panel_titles=True)
    y_max = 0.0
    for col, (_, metrics) in enumerate(groups, start=1):
        for model in models:
            acc = selected[("accuracy", model)].section
            values = [_pct(acc.get(metric)) for metric in metrics]
            y_max = max(y_max, *(v for v in values if v is not None), 0)
            # width is in category units (~24 px at the 1100 px export); a stretched dashboard
            # still widens columns, a hard pixel cap would need horizontal bars or a fixed width.
            fig.add_trace(go.Bar(
                x=[AP_METRICS[m] for m in metrics], y=values, name=_label(model, model_labels),
                width=0.15,
                legendgroup=model, showlegend=col == 1,
                marker={"color": t["models"][model], "cornerradius": 4, "line": {"width": 0}},
                text=[f"{v:.1f}" if v is not None else "" for v in values], textposition="outside",
                # Labels are wider than the thin bars; constraining them would shrink some below
                # uniformtext.minsize, which then hides them.
                textfont={"color": t["ink_secondary"], "size": 12}, cliponaxis=False,
                constraintext="none",
                hovertemplate="<b>%{y:.2f}%</b> %{x}<extra>" + _label(model, model_labels) + "</extra>",
            ), row=1, col=col)
    _left_align_panel_titles(fig, t)
    fig.update_yaxes(range=[0, max(y_max * 1.18, 5)], ticksuffix="%")
    fig.update_yaxes(title={"text": "AP (%)"}, row=1, col=1)
    fig.update_xaxes(showgrid=False, tickfont={"color": t["ink_secondary"]})
    fig.update_layout(barmode="group", showlegend=True)
    return fig


def per_class_figure(selected: Selected, theme: str = "light", *, top: int = 20,
                     model_labels: ModelLabels = None) -> go.Figure:
    """Cleveland dot plot of per-class AP for the classes where the models differ the most."""
    t = _theme(theme)
    title = "클래스별 AP"
    tables = {m: table for m in _models_with(selected, "accuracy")
              if (table := _class_ap(selected[("accuracy", m)].section))}
    models = list(tables)
    if not models:
        return empty_figure("클래스별 AP(per_category_ap)가 있는 결과가 없습니다.", theme, title=title)
    classes = sorted(set().union(*tables.values()))
    def spread(name: str) -> float:
        values = [tables[m][name] for m in models if name in tables[m]]
        return (max(values) - min(values)) if len(values) > 1 else -values[0] if values else 0
    if len(models) > 1:
        chosen = sorted(classes, key=spread, reverse=True)[:top]
        subtitle = f"모델 간 차이가 가장 큰 {len(chosen)}개 클래스 · 점 사이가 멀수록 그 클래스에서 성능 차이가 큼"
    else:
        only = tables[models[0]]
        chosen = sorted(classes, key=lambda name: only.get(name, 0))[:top]
        subtitle = f"AP가 가장 낮은 {len(chosen)}개 클래스 · 개선이 필요한 클래스 후보"
    anchor = models[0]
    chosen.sort(key=lambda name: tables[anchor].get(name, 0))
    fig = go.Figure()
    _base_layout(fig, theme, title=title, subtitle=subtitle, height=160 + 24 * len(chosen))
    for name in chosen:
        values = [tables[m][name] * 100 for m in models if name in tables[m]]
        if len(values) > 1:
            fig.add_trace(go.Scatter(x=[min(values), max(values)], y=[name, name], mode="lines",
                                     line={"color": t["grid"], "width": 2}, hoverinfo="skip", showlegend=False))
    for model in models:
        xs = [tables[model].get(name) for name in chosen]
        label = _label(model, model_labels)
        fig.add_trace(go.Scatter(
            x=[x * 100 if x is not None else None for x in xs], y=chosen, mode="markers", name=label,
            marker={"size": 10, "color": t["models"][model], "symbol": MODEL_SYMBOLS[model],
                    "line": {"color": t["surface"], "width": 2}}, cliponaxis=False,
            hovertemplate="<b>%{x:.1f}%</b> %{y}<extra>" + label + "</extra>",
        ))
    fig.update_xaxes(title={"text": "AP@[.50:.95] (%)"}, rangemode="tozero", ticksuffix="%")
    fig.update_yaxes(showgrid=False, tickfont={"color": t["ink"], "size": 12}, ticklabelstandoff=8)
    fig.update_layout(showlegend=True, hovermode="closest")
    return fig


# --------------------------------------------------------------------------- stages & efficiency


def stage_figure(selected: Selected, theme: str = "light", *, metric: str = "mean_ms",
                 model_labels: ModelLabels = None) -> go.Figure:
    """Stacked horizontal bar of where each R-CNN model spends its time, in pipeline order."""
    t = _theme(theme)
    title = "단계별 처리 시간"
    models = _models_with(selected, "stages")
    if not models:
        return empty_figure("단계별 측정 결과가 없습니다. scripts/profile_stages.py를 실행하세요.",
                            theme, title=title)
    fig = go.Figure()
    _base_layout(fig, theme, title=title,
                 subtitle="막대 전체 = 한 장 처리 시간(평균) · 가장 긴 조각이 먼저 줄여야 할 병목",
                 height=190 + 70 * len(models))
    labels = [_label(m, model_labels) for m in models]
    totals = {}
    per_model = {}
    for model in models:
        section = selected[("stages", model)].section
        items = {item["stage"]: item for item in section.get("items") or []}
        values = {stage: _num(items.get(stage, {}).get(metric)) or 0.0 for stage in CHART_STAGES}
        values["preprocess"] += _num(items.get("input", {}).get(metric)) or 0.0
        per_model[model] = values
        totals[model] = sum(values.values())
    for index, stage in enumerate(CHART_STAGES):
        xs = [per_model[m][stage] for m in models]
        shares = [x / totals[m] * 100 if totals[m] else 0 for x, m in zip(xs, models)]
        color = t["stages"][index]
        light_fill = index < 2  # the two lightest steps need dark ink in both themes
        fig.add_trace(go.Bar(
            x=xs, y=labels, orientation="h", name=CHART_STAGE_LABELS[stage],
            marker={"color": color, "line": {"color": t["surface"], "width": 2}},
            text=[f"{s:.0f}%" for s in shares], textposition="inside", insidetextanchor="middle",
            textangle=0,
            textfont={"color": "#0b0b0b" if light_fill else "#ffffff", "size": 11},
            customdata=list(zip(shares)),
            hovertemplate=("<b>%{x:.2f} ms</b> (%{customdata[0]:.1f}%)<br>" + CHART_STAGE_LABELS[stage]
                           + "<extra>%{y}</extra>"),
        ))
    for model, label in zip(models, labels):
        fig.add_annotation(x=totals[model], y=label, text=f"{totals[model]:.1f} ms", showarrow=False,
                           xanchor="left", xshift=8, font={"color": t["ink"], "size": 13})
    fig.update_layout(barmode="stack", bargap=_cap_bar_gap(fig, len(models)), showlegend=True,
                      legend_traceorder="normal")
    fig.update_xaxes(title={"text": "평균 처리 시간 (ms)"}, range=[0, max(totals.values()) * 1.18 or 1])
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(labels)), showgrid=False,
                     tickfont={"color": t["ink"], "size": 13})
    return fig


def efficiency_figure(selected: Selected, theme: str = "light", *,
                      model_labels: ModelLabels = None) -> go.Figure:
    """Parameters and GFLOPs side by side (two panels, never a shared or dual axis)."""
    t = _theme(theme)
    title = "모델 크기와 연산량"
    rows = []
    for model in _models_with(selected, "latency"):
        stats = selected[("latency", model)].data.get("model_stats") or {}
        rows.append((model, *_size_stats(stats)))
    rows = [row for row in rows if row[1] is not None or row[2] is not None]
    if not rows:
        return empty_figure("파라미터 수/연산량(model_stats)이 있는 결과가 없습니다.", theme, title=title)
    fig = make_subplots(rows=1, cols=2, subplot_titles=["파라미터 수 (백만 개)", "연산량 (GFLOPs, 이미지 1장)"],
                        shared_yaxes=True, horizontal_spacing=0.08)
    _base_layout(fig, theme, title=title, subtitle="짧을수록 가볍고 빠를 가능성이 큼 · 실제 속도는 지연 시간 그래프가 기준",
                 height=200 + 60 * len(rows), panel_titles=True)
    labels = [_label(m, model_labels) for m, _, _ in rows]
    for col, position in ((1, 1), (2, 2)):
        values = [row[position] for row in rows]
        fig.add_trace(go.Bar(
            x=[v if v is not None else 0 for v in values], y=labels, orientation="h",
            showlegend=False,
            marker={"color": [t["models"][m] for m, _, _ in rows], "cornerradius": 4,
                    "line": {"width": 0}},
            text=[f"{v:.1f}" if v is not None else "측정 안 됨" for v in values], textposition="outside",
            textfont={"color": t["ink_secondary"], "size": 12}, cliponaxis=False,
            customdata=[f"{v:.2f}" if v is not None else "측정 안 됨" for v in values],
            hovertemplate="<b>%{customdata}</b><extra>%{y}</extra>",
        ), row=1, col=col)
        top = max((v for v in values if v is not None), default=1)
        fig.update_xaxes(range=[0, top * 1.3], row=1, col=col)
    _left_align_panel_titles(fig, t)
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(labels)), showgrid=False,
                     tickfont={"color": t["ink"], "size": 13})
    fig.update_layout(bargap=_cap_bar_gap(fig, len(rows)))
    return fig


# --------------------------------------------------------------------------- before / after

# (kind, key, table label, higher is better) for the before/after table.
IMPROVEMENT_METRICS = [
    ("accuracy", "AP_50_95", "AP@[.50:.95] (%)", True),
    ("accuracy", "AP_75", "AP75 (%)", True),
    ("latency", "p50_ms", "p50 (ms)", False),
    ("latency", "p95_ms", "p95 (ms)", False),
    ("latency", "fps", "FPS", True),
]
# Verdict noise band, PERFORMANCE_PLAN 5절: paired full val2017 comparison. Changes below it read
# '변화 없음 (오차 범위)': AP in %p, latency and FPS in % of the before value. A p50 with a paired
# 95% CI is judged by the CI instead.
NOISE_AP_PP = 0.1
NOISE_RELATIVE_PCT = 3.0
LOG_SPREAD = 3.0  # latency panel turns logarithmic once the slowest model is this many times the fastest
CI_COLUMN = "95% CI"
# Pixels per character of a 12 px chart label (a wide UI font's digit, ~0.63 em, plus slack) and
# its gap to the marker: the latency labels get this much right margin, at any chart width.
LABEL_CHAR_PX = 7
LABEL_GAP_PX = 16


def _metric(report: Report | None, kind: str, key: str) -> float | None:
    if report is None:
        return None
    value = report.section.get(key)
    return _pct(value) if kind == "accuracy" else _num(value)


def paired_ci(before: Report | None, after: Report | None) -> tuple[float, float] | None:
    """95% CI (%) of ``after``'s p50 change against ``before`` from its latency ``paired`` entry
    (the interleaved design of scripts/compare_improvements.py), or None. Only two latency
    reports of one comparison run and one checkpoint pair up, so a CI never describes another pair."""
    if before is None or after is None or not before.kind == after.kind == "latency":
        return None
    run = after.data.get("comparison_run")
    if not run or before.data.get("comparison_run") != run:
        return None
    weights = [report.data.get("weights_sha256") or report.data.get("weights")
               for report in (before, after)]
    paired = after.section.get("paired")
    if weights[0] != weights[1] or not isinstance(paired, dict):
        return None
    for entry in paired.values():
        if isinstance(entry, dict) and entry.get("variant") == before.variant:
            bounds = entry.get("ci95_pct")
            if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                low, high = (_num(value) for value in bounds)
                if low is not None and high is not None:
                    return min(low, high), max(low, high)
    return None


def _ci_text(ci: tuple[float, float], digits: int = 1, sep: str = "~") -> str:
    # The verdict reads the raw bounds, so a bound that is not 0 never shows as ±0.
    while digits < 4 and any(value and round(value, digits) == 0 for value in ci):
        digits += 1
    return f"{_signed(ci[0], digits)}{sep}{_signed(ci[1], digits)}%"


def _before_after(before: Selected, after: Selected, kind: str, key: str) -> dict[str, tuple[float, float]]:
    pairs = {}
    for model in MODEL_ORDER:
        old, new = _metric(before.get((kind, model)), kind, key), _metric(after.get((kind, model)), kind, key)
        if old is not None and new is not None:
            pairs[model] = (old, new)
    return pairs


def _signed(value: float, digits: int = 1) -> str:
    text = f"{value:+.{digits}f}"
    return f"±{text[1:]}" if float(text) == 0 else text


def _delta_text(kind: str, old: float, new: float, ci: tuple[float, float] | None = None, *,
                wrap: bool = False) -> str:
    """'+1.0%p', '-8.0 ms (-20%)', '-38 ms (-39%, 95% CI -41~-36%)'; ``wrap`` puts the CI on a
    second line (chart labels)."""
    if kind == "accuracy":
        return f"{_signed(new - old)}%p"
    ratio = (new - old) / old * 100 if old > 0 else None
    change = f"{_signed(new - old, 0 if max(old, new) >= 100 else 1)} ms"
    if ratio is None:
        return change
    digits = 0 if abs(ratio) >= 10 else 1
    relative = f"{_signed(ratio, digits)}%"
    if ci is None:
        return f"{change} ({relative})"
    interval = f"95% CI {_ci_text(ci, digits)}"
    return f"{change} ({relative})<br>{interval}" if wrap else f"{change} ({relative}, {interval})"


def _selection_label(selected: Selected) -> str | None:
    return next((report.variant_label for report in selected.values()), None)


def _label_px(texts: list[str]) -> int:
    """Pixel width a 'middle right' label needs right of its marker (its longest line)."""
    longest = max((len(line) for text in texts for line in text.split("<br>")), default=0)
    return LABEL_CHAR_PX * longest + LABEL_GAP_PX


def _dumbbell_range(pairs: dict[str, tuple[float, float]], *, log: bool, min_span: float,
                    floor: float | None = None, label_room: float = 0.6) -> list[float]:
    """A little room on the left, room for the delta labels on the right (log10 units if log)."""
    values = [math.log10(v) if log else v for pair in pairs.values() for v in pair]
    low, high = min(values), max(values)
    span = max(high - low, min_span)
    start = low - span * 0.12
    return [start if floor is None else max(start, floor), high + span * label_room]


def _log_ticks(axis_range: list[float]) -> list[float]:
    low, high = 10 ** axis_range[0], 10 ** axis_range[1]
    return [step * 10.0 ** exponent for exponent in range(-2, 6) for step in (1, 2, 5)
            if low <= step * 10.0 ** exponent <= high]


def improvement_figure(baseline_selected: Selected, variant_selected: Selected, theme: str = "light",
                       variant_label: str | None = None, *, ap_metric: str = "AP_50_95",
                       percentile: str = "p50", model_labels: ModelLabels = None) -> go.Figure:
    """Before -> after dumbbell per model: AP and latency as two small multiples (no dual axis).
    A p50 change measured in one comparison run also names its paired 95% CI."""
    t = _theme(theme)
    title = "개선 전/후 비교"
    before_label = _selection_label(baseline_selected) or BASELINE_VARIANT_LABEL
    after_label = variant_label or _selection_label(variant_selected) or "개선 후"
    panels = [("accuracy", ap_metric), ("latency", f"{percentile}_ms")]
    data = [_before_after(baseline_selected, variant_selected, kind, key) for kind, key in panels]
    models = [m for m in MODEL_ORDER if any(m in pairs for pairs in data)]
    if not models:
        return empty_figure("개선 전과 개선 후 결과가 모두 있는 모델이 없습니다. "
                            "같은 모델을 두 설정으로 측정해야 그려집니다.", theme, title=title)
    cis = {model: ci for model in data[1] if percentile == "p50" and (ci := paired_ci(
        baseline_selected.get(("latency", model)), variant_selected.get(("latency", model))))}
    # Latencies of different models can be 20x apart; on a log axis equal lengths = equal ratios.
    times = [v for pair in data[1].values() for v in pair]
    log = bool(times) and min(times) > 0 and max(times) / min(times) >= LOG_SPREAD
    # The latency labels sit in the right margin, sized in pixels: room in data units shrinks
    # with the browser width and cut the text off (the CI line is the longest).
    latency_labels = [_delta_text("latency", old, new, cis.get(model), wrap=True)
                      for model, (old, new) in data[1].items()]
    # Two lines, so the AP title fits over its panel even once the label margin narrows it.
    panel_titles = [f"{AP_METRICS.get(ap_metric, ap_metric)} (%)<br>높을수록 좋음",
                    f"{percentile} 지연 시간 (ms{', 로그 눈금' if log else ''})<br>낮을수록 좋음"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=panel_titles, shared_yaxes=True,
                        horizontal_spacing=0.1)
    _base_layout(fig, theme, title=title,
                 subtitle=f"빈 점 = {before_label} · 채운 점 = {after_label} · 오른쪽 숫자 = 변화량",
                 height=196 + 70 * len(models), panel_titles=True, panel_title_lines=2)
    if latency_labels:
        fig.update_layout(margin={"r": max(fig.layout.margin.r, _label_px(latency_labels))})
    _left_align_panel_titles(fig, t)
    for col, ((kind, _), pairs) in enumerate(zip(panels, data), start=1):
        unit = "%" if kind == "accuracy" else " ms"
        for model, (old, new) in pairs.items():
            label, color, symbol = _label(model, model_labels), t["models"][model], MODEL_SYMBOLS[model]
            ci = cis.get(model) if kind == "latency" else None
            text, hover = _delta_text(kind, old, new, ci, wrap=True), _delta_text(kind, old, new, ci)
            fig.add_trace(go.Scatter(x=[old, new], y=[label, label], mode="lines", hoverinfo="skip",
                                     line={"color": color, "width": 2}, showlegend=False),
                          row=1, col=col)
            for value, after in ((old, False), (new, True)):
                rightmost = after == (new >= old)  # the delta sits right of the dumbbell
                fig.add_trace(go.Scatter(
                    x=[value], y=[label], mode="markers+text" if rightmost else "markers",
                    text=[text] if rightmost else None, textposition="middle right",
                    textfont={"color": t["ink"], "size": 12},
                    marker={"size": 12, "symbol": symbol if after else f"{symbol}-open", "color": color,
                            "line": {"color": t["surface"] if after else color, "width": 2}},
                    showlegend=False, cliponaxis=False,
                    hovertemplate=(f"<b>%{{x:.2f}}{unit}</b>" + (f" ({hover})" if after else "")
                                   + f"<br>{after_label if after else before_label}<extra>{label}</extra>"),
                ), row=1, col=col)
        if not pairs:
            fig.update_xaxes(range=[0, 1], showticklabels=False, row=1, col=col)
            left = 0.0
        elif kind == "accuracy":
            axis_range = _dumbbell_range(pairs, log=False, min_span=4.0, floor=0.0)
            fig.update_xaxes(range=axis_range, ticksuffix="%", row=1, col=col)
            left = axis_range[0]
        elif log:
            axis_range = _dumbbell_range(pairs, log=True, min_span=math.log10(1.5),
                                         label_room=0.12)
            ticks = _log_ticks(axis_range)
            fig.update_xaxes(type="log", range=axis_range, tickvals=ticks,
                             ticktext=[f"{tick:g}" for tick in ticks], row=1, col=col)
            left = 10 ** axis_range[0]
        else:
            axis_range = _dumbbell_range(pairs, log=False, min_span=max(max(times) * 0.25, 0.01),
                                         floor=0.0, label_room=0.12)
            fig.update_xaxes(range=axis_range, row=1, col=col)
            left = axis_range[0]
        absent = [_label(m, model_labels) for m in models if m not in pairs]
        if absent:
            fig.add_trace(go.Scatter(x=[left] * len(absent), y=absent, mode="text",
                                     text=["비교할 결과 없음"] * len(absent), textposition="middle right",
                                     textfont={"color": t["muted"], "size": 12}, hoverinfo="skip",
                                     showlegend=False, cliponaxis=False), row=1, col=col)
    labels = [_label(m, model_labels) for m in models]
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(labels)), showgrid=False,
                     range=[-0.6, len(labels) - 0.4], tickfont={"color": t["ink"], "size": 13})
    # Legend for the two marker states (neutral ink so it never reads as a model color).
    for name, symbol in ((before_label, "circle-open"), (after_label, "circle")):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name,
                                 marker={"symbol": symbol, "size": 10, "color": t["muted"],
                                         "line": {"color": t["muted"], "width": 2}}))
    fig.update_layout(showlegend=True, hovermode="closest")
    return fig


def improvement_rows(baseline_selected: Selected, variant_selected: Selected, *,
                     model_labels: ModelLabels = None) -> list[dict[str, Any]]:
    """Table twin of improvement_figure: one row per model and metric measured before and after.
    A p50 measured in one comparison run gets its paired 95% CI ('95% CI' column, only when some
    row has one) and is judged by it: a CI that holds 0 is noise."""
    rows = []
    for model in MODEL_ORDER:
        for kind, key, label, higher_is_better in IMPROVEMENT_METRICS:
            old_report, new_report = baseline_selected.get((kind, model)), variant_selected.get((kind, model))
            old, new = _metric(old_report, kind, key), _metric(new_report, kind, key)
            if old is None or new is None:
                continue
            change = new - old
            relative = change / old * 100 if old > 0 else None
            ci = paired_ci(old_report, new_report) if key == "p50_ms" else None
            if ci is not None:  # the raw bounds: a CI that misses 0 by 0.04% still misses it
                verdict = ("변화 없음 (오차 범위)" if ci[0] <= 0 <= ci[1]
                           else "좋아짐" if (ci[0] > 0) == higher_is_better else "나빠짐")
            else:
                # AP: the raw %p change (rounded only past float noise, so an exact 0.1%p stays
                # out); latency and FPS: the relative change as the table shows it (1 decimal).
                if kind == "accuracy":
                    within = abs(round(change, 9)) < NOISE_AP_PP
                else:
                    within = (abs(round(relative, 1)) < NOISE_RELATIVE_PCT if relative is not None
                              else change == 0)
                # The plain '변화 없음' (identical as shown) is for AP only; timings always jitter.
                verdict = ("변화 없음" if kind == "accuracy" and round(change, 2) == 0
                           else "변화 없음 (오차 범위)" if within
                           else "좋아짐" if (change > 0) == higher_is_better else "나빠짐")
            rows.append({"모델": _label(model, model_labels), "지표": label, "개선 전": old,
                         "개선 후": new, "변화": change, "상대 변화 (%)": relative,
                         CI_COLUMN: _ci_text(ci, 1, " ~ ") if ci else None, "판정": verdict})
    if not any(row[CI_COLUMN] for row in rows):
        for row in rows:
            del row[CI_COLUMN]
    return rows


ALL_FIGURES = {
    "tradeoff": "정확도 대 속도",
    "latency_percentiles": "지연 시간 백분위수",
    "latency_distribution": "지연 시간 분포",
    "latency_stability": "측정 안정성",
    "accuracy": "COCO 정확도 세부 지표",
    "per_class": "클래스별 AP",
    "stages": "단계별 처리 시간",
    "efficiency": "모델 크기와 연산량",
}
# Only built for a non-baseline selection with its baseline (see build_all).
VARIANT_FIGURES = {"improvement": "개선 전/후 비교"}


def build_all(selected: Selected, targets: dict[str, Any], theme: str = "light",
              baseline_selected: Selected | None = None,
              variant_label: str | None = None, *, model_labels: ModelLabels = None
              ) -> dict[str, go.Figure]:
    """Every chart for ``selected``; given ``baseline_selected`` and a ``selected`` measured with
    another inference preset, also the before/after chart under 'improvement'."""
    names = {"model_labels": model_labels}
    figures = {"tradeoff": tradeoff_figure(selected, targets, theme, **names)}
    if baseline_selected is not None and any(r.variant != BASELINE_VARIANT for r in selected.values()):
        figures["improvement"] = improvement_figure(
            baseline_selected, selected, theme, variant_label,
            ap_metric=targets.get("ap_metric") or "AP_50_95",
            percentile=targets.get("latency_percentile") or "p50", **names)
    figures.update({
        "latency_percentiles": latency_percentile_figure(selected, targets, theme, **names),
        "latency_distribution": latency_distribution_figure(selected, targets, theme, **names),
        "latency_stability": latency_stability_figure(selected, theme, **names),
        "accuracy": accuracy_figure(selected, theme, **names),
        "per_class": per_class_figure(selected, theme, **names),
        "stages": stage_figure(selected, theme, **names),
        "efficiency": efficiency_figure(selected, theme, **names),
    })
    return figures


def summary_rows(selected: Selected, *, model_labels: ModelLabels = None) -> list[dict[str, Any]]:
    """One row per model with the headline numbers; the table twin of every chart."""
    rows = []
    for model in MODEL_ORDER:
        acc_report = selected.get(("accuracy", model))
        lat_report = selected.get(("latency", model))
        if acc_report is None and lat_report is None:
            continue
        acc = acc_report.section if acc_report else {}
        lat = lat_report.section if lat_report else {}
        stats = (lat_report.data.get("model_stats") or {}) if lat_report else {}
        params, gflops = _size_stats(stats)
        cold_scope = lat.get("cold_start_scope")
        env = (lat_report or acc_report).environment
        rows.append({
            "모델": _label(model, model_labels),
            "AP@[.50:.95] (%)": _pct(acc.get("AP_50_95")),
            "AP50 (%)": _pct(acc.get("AP_50")),
            "AP75 (%)": _pct(acc.get("AP_75")),
            "p50 (ms)": _num(lat.get("p50_ms")),
            "p95 (ms)": _num(lat.get("p95_ms")),
            "p99 (ms)": _num(lat.get("p99_ms")),
            "FPS": _num(lat.get("fps")),
            "첫 추론 (ms)": _num(lat.get("cold_start_ms")),
            "첫 추론 범위": COLD_START_LABELS.get(cold_scope, cold_scope),
            "파라미터 (M)": params,
            "GFLOPs": gflops,
            "평가 이미지": acc.get("images_evaluated"),
            "반복 횟수": lat.get("n") or lat.get("iterations"),
            "장치": env.get("device_name") or env.get("device"),
            "정밀도": (lat_report or acc_report).data.get("precision"),
        })
    return rows


# --------------------------------------------------------------------------- HTML report

LEVEL_LABELS = {"error": "비교 불가", "warning": "주의", "info": "참고"}


def _format(value) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return html.escape(str(value))


def html_report(figures, rows, warnings, selected, theme: str, title: str, *,
                model_labels: ModelLabels = None) -> str:
    """Self-contained HTML page: warnings, summary table, every non-empty chart, sources."""
    t = THEMES[theme]
    parts = [f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  body {{ margin: 0; background: {t['page']}; color: {t['ink']};
         font-family: {FONT_FAMILY}; line-height: 1.5; }}
  main {{ max-width: 1120px; margin: 0 auto; padding: 32px 16px 64px; }}
  h1 {{ font-size: 26px; margin: 0 0 4px; }}
  h2 {{ font-size: 18px; margin: 32px 0 8px; }}
  .meta {{ color: {t['ink_secondary']}; font-size: 14px; margin: 0 0 24px; }}
  .card {{ background: {t['surface']}; border: 1px solid {t['border']}; border-radius: 12px;
          padding: 12px; margin: 0 0 16px; overflow-x: auto; }}
  ul.warnings {{ list-style: none; padding: 0; margin: 0; }}
  ul.warnings li {{ padding: 6px 0; border-bottom: 1px solid {t['grid']}; font-size: 14px; }}
  .tag {{ display: inline-block; min-width: 64px; font-weight: 600; color: {t['ink']}; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ text-align: right; padding: 6px 8px; border-bottom: 1px solid {t['grid']};
           font-variant-numeric: tabular-nums; white-space: nowrap; }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ color: {t['ink_secondary']}; font-weight: 600; }}
</style></head><body><main>
<h1>{html.escape(title)}</h1>
<p class="meta">생성 시각 {datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M %Z')}
 · 같은 결과 파일로 대시보드(streamlit run dashboard/app.py)에서도 확인할 수 있습니다.</p>"""]
    if warnings:
        parts.append("<h2>비교 전에 확인할 점</h2><div class='card'><ul class='warnings'>")
        for warning in warnings:
            parts.append(f"<li><span class='tag'>{LEVEL_LABELS.get(warning.level, warning.level)}</span>"
                         f"{html.escape(warning.message)}</li>")
        parts.append("</ul></div>")
    if rows:
        columns = list(rows[0])
        parts.append("<h2>요약 표</h2><div class='card'><table><thead><tr>")
        parts.extend(f"<th>{html.escape(column)}</th>" for column in columns)
        parts.append("</tr></thead><tbody>")
        for row in rows:
            parts.append("<tr>" + "".join(f"<td>{_format(row[column])}</td>" for column in columns) + "</tr>")
        parts.append("</tbody></table></div>")
    parts.append("<h2>그래프</h2>")
    first = True
    for fig in figures.values():
        if is_empty(fig):
            continue
        parts.append("<div class='card'>")
        parts.append(fig.to_html(full_html=False, include_plotlyjs=first, config={"displaylogo": False}))
        parts.append("</div>")
        first = False
    parts.append("<h2>측정 출처</h2><div class='card'><table><thead><tr><th>종류</th><th>모델</th>"
                 "<th>환경</th><th>측정 시각</th><th>파일</th></tr></thead><tbody>")
    for (kind, model), report in sorted(selected.items()):
        parts.append(f"<tr><td>{html.escape(kind)}</td><td>{html.escape(_label(model, model_labels))}</td>"
                     f"<td>{html.escape(describe_environment(report.environment))}</td>"
                     f"<td>{html.escape(report.created_at)}</td><td>{html.escape(report.source)}</td></tr>")
    parts.append("</tbody></table></div></main></body></html>")
    return "\n".join(parts)
