"""Interactive comparison dashboard for Faster R-CNN, Light R-CNN and YOLO26.

Run with ``streamlit run dashboard/app.py``. Only streamlit, plotly and pandas are required: the
chart and report modules are loaded straight from ``src/`` when the package is not installed.
``scripts/run_comparison.py`` points ``LIGHT_RCNN_RESULTS_DIR`` at the run it just finished.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from light_rcnn import viz
from light_rcnn.reporting import (
    COLD_START_LABELS,
    MODEL_ORDER,
    MODELS,
    SCOPE_LABELS,
    STAGE_LABELS,
    comparability_warnings,
    describe_environment,
    device_label,
    load_reports,
    load_targets,
    on_target_device,
    select_reports,
    timing_devices,
)

PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"],
               "toImageButtonOptions": {"format": "png", "scale": 2}}
LEVEL_ICONS = {"error": ":material/block:", "warning": ":material/warning:",
               "info": ":material/info:"}
SUMMARY_COLUMN_CONFIG = {
    name: st.column_config.NumberColumn(format="%.2f")
    for name in ("AP@[.50:.95] (%)", "AP50 (%)", "AP75 (%)", "p50 (ms)", "p95 (ms)", "p99 (ms)",
                 "FPS", "첫 추론 (ms)", "파라미터 (M)", "GFLOPs")}
PERCENTILE_OPTIONS = [name for _, name in viz.PERCENTILES]
AP_OPTIONS = ["AP_50_95", "AP_50"]

st.set_page_config(page_title="Light R-CNN 성능 비교", page_icon=":material/monitoring:",
                   layout="wide")


def current_theme() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except AttributeError:  # older Streamlit without st.context.theme
        return "light"


def show(fig, key: str) -> None:
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG, key=key)


def table(rows, **kwargs) -> None:
    frame = pd.DataFrame(rows)
    st.dataframe(frame, hide_index=True, width="stretch", **kwargs)


def usable_targets(raw: dict) -> tuple[dict, list[str]]:
    """Targets the sidebar widgets accept; unusable config values fall back to the defaults."""
    fixed, ignored = dict(raw), []
    for key, options in (("latency_percentile", PERCENTILE_OPTIONS), ("ap_metric", AP_OPTIONS)):
        if fixed[key] not in options:
            ignored.append(f"{key}={fixed[key]!r}")
            fixed[key] = options[0]
    for key, fallback in (("ap", 0.60), ("latency_ms", 15.0)):
        value = viz._num(fixed[key])
        if value is None or value < 0:
            ignored.append(f"{key}={fixed[key]!r}")
            value = fallback
        fixed[key] = value
    fixed["ap"] = min(fixed["ap"] / 100 if fixed["ap"] > 1 else fixed["ap"], 1.0)  # "60" = 60%
    return fixed, ignored


def swatch(model: str, theme: str) -> str:
    color = viz.model_color(model, theme)
    return (f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
            f"background:{color};margin-right:6px'></span>")


theme = current_theme()
defaults, ignored = usable_targets(load_targets(ROOT / "configs" / "coco.yaml"))

with st.sidebar:
    st.header("결과 불러오기")
    default_folder = os.environ.get("LIGHT_RCNN_RESULTS_DIR") or str(ROOT / "outputs")
    folder = st.text_input("결과 폴더", value=default_folder).strip().strip("\"'")
    output_dir = Path(folder).expanduser()
    uploaded = st.file_uploader("JSON 결과 파일 추가", type="json", accept_multiple_files=True)
    st.caption("평가·벤치마크 스크립트가 저장한 JSON을 하위 폴더까지 자동으로 읽습니다 "
               "(run_comparison.py는 outputs/comparison_runs/<실행 시각>에 저장). "
               "같은 모델의 결과가 여러 개면 새 형식(v2) > 전체 평가 > 이미지 수가 많은 평가 > "
               "최신 결과 순으로 하나를 고릅니다.")
    reports, errors = load_reports(output_dir,
                                   [(file.name, file.getvalue()) for file in uploaded or []])
    devices = timing_devices(reports)
    timed_models: dict[str, set[str]] = {}
    for report in reports:
        if report.kind != "accuracy":
            timed_models.setdefault(device_label(report), set()).add(report.model)
    device = st.selectbox("지연 시간 측정 장치", devices, disabled=not devices,
                          placeholder="지연 시간 결과 없음",
                          format_func=lambda label: f"모델 {len(timed_models[label])}개 · "
                                                    + re.sub(r"\s*\((R|TM)\)", "", label),
                          help="CPU와 GPU 결과가 섞이지 않도록 지연 시간·병목 분석은 고른 장치에서 "
                               "측정한 결과만 씁니다. 정확도(AP)는 장치와 무관해서 항상 "
                               "표시합니다. 최근에 측정한 장치가 맨 위에 있습니다.")
    if device:
        st.caption(device)
    st.header("목표 기준")
    if ignored:
        st.warning("configs/coco.yaml 목표 값 중 쓸 수 없는 값은 기본값으로 바꿨습니다: "
                   + ", ".join(ignored))
    latency_target = st.number_input("지연 시간 목표 (ms, T4 GPU 기준)", min_value=0.0,
                                     value=float(defaults["latency_ms"]), step=1.0)
    percentile = st.segmented_control("지연 시간 기준", PERCENTILE_OPTIONS,
                                      default=defaults["latency_percentile"],
                                      help="p50 = 보통 속도(중앙값), p95 = 100번 중 95번은 "
                                           "이보다 빠름 (p90·p99도 같은 방식)")
    ap_metric = st.segmented_control("정확도 기준", AP_OPTIONS, default=defaults["ap_metric"],
                                     format_func=lambda key: viz.AP_METRICS[key],
                                     help="AP@[.50:.95]는 COCO 공식 지표, AP50은 더 느슨한 기준")
    ap_target = st.number_input("정확도 목표 (%)", min_value=0.0, max_value=100.0,
                                value=float(defaults["ap"]) * 100, step=1.0)
    per_class_top = st.slider("클래스별 그래프에 보일 클래스 수", 5, 40, 20)

targets = {"latency_ms": latency_target, "latency_percentile": percentile or "p50",
           "ap_metric": ap_metric or "AP_50_95", "ap": ap_target / 100}

# Timings come from one device only; accuracy is device-independent and always eligible.
selected = select_reports(reports, device=device)
warnings = comparability_warnings(selected, latency_target)
missing_dir = not output_dir.is_dir()

st.title("Light R-CNN 성능 비교")
st.caption("바닐라 Faster R-CNN · Light R-CNN · YOLO26 — COCO 정확도, 추론 지연 시간, 병목 단계")

if errors:
    with st.expander(f"읽지 못한 결과 파일 {len(errors)}개"):
        for error in errors:
            st.write(error)

if missing_dir:
    st.warning(f"결과 폴더를 찾을 수 없습니다: `{output_dir}` — 사이드바의 경로를 확인하세요.",
               icon=LEVEL_ICONS["warning"])

if not selected:
    if not missing_dir:
        st.info("아직 비교 결과가 없습니다. 아래 명령으로 평가와 측정을 실행하면 그래프가 "
                "채워집니다.")
        weights = "--light-weights outputs/coco_4090_v2/best.pth --yolo-weights yolo26n.pt"
        light = "--model light_rcnn --weights outputs/coco_4090_v2/best.pth"
        st.code("python scripts/run_comparison.py --config configs/coco.yaml --device cuda "
                f"{weights}\n"
                "# CPU에서 지연 시간만 빠르게 확인:\n"
                "python scripts/run_comparison.py --config configs/coco_cpu.yaml --inference-only "
                f"{weights}\n"
                "# 개별 실행 예:\n"
                f"python scripts/evaluate_coco.py {light}\n"
                f"python scripts/benchmark.py {light} --source data/coco/images/val2017\n"
                f"python scripts/profile_stages.py {light} --source data/coco/images/val2017",
                language="bash")
    st.stop()

# Comparability first: a chart is only as fair as the measurements behind it.
blocking = [w for w in warnings if w.level == "error"]
cautions = [w for w in warnings if w.level == "warning"]
notes = [w for w in warnings if w.level == "info"]
if blocking:
    st.error("**이 상태로는 공정한 비교가 아닙니다.**\n\n"
             + "\n".join(f"- {w.message}" for w in blocking), icon=LEVEL_ICONS["error"])
if cautions:
    st.warning("**해석할 때 주의하세요.**\n\n" + "\n".join(f"- {w.message}" for w in cautions),
               icon=LEVEL_ICONS["warning"])
if notes:
    with st.expander("참고 사항", icon=LEVEL_ICONS["info"]):
        for warning in notes:
            st.write(f"- {warning.message}")

with st.expander("그래프 읽는 법", icon=":material/help:"):
    st.markdown(
        "- **AP (정확도)**: 사진 속 물체를 얼마나 정확히 찾는지. 100%에 가까울수록 좋습니다.\n"
        "- **지연 시간 (ms)**: 사진 한 장을 처리하는 데 걸린 시간. 짧을수록 좋습니다.\n"
        "- **p50 / p95 / p99**: 100번 재서 빠른 순으로 줄 세웠을 때 50번째 / 95번째 / 99번째 값 "
        "(p95 = 100번 중 95번은 이보다 빠름). "
        "p50은 보통 속도, p95·p99는 가끔 느려질 때의 속도입니다.\n"
        "- **목표 영역(음영)**: 지연 시간 목표보다 빠르고 정확도 목표보다 높은 곳. "
        "점이 여기 들어가야 합격입니다. 지연 시간 목표는 T4 GPU 기준이라 T4에서 측정한 결과만 "
        "합격 여부를 가릴 수 있고, 다른 장치의 결과는 참고용입니다.\n"
        "- **측정 장치**: 지연 시간은 사이드바에서 고른 장치의 결과만 비교합니다. "
        "CPU와 GPU 결과는 섞지 않습니다.\n"
        "- 모델 색과 모양은 모든 그래프에서 같습니다: "
        + " · ".join(f"{swatch(m, theme)}{MODELS[m]}" for m in MODEL_ORDER),
        unsafe_allow_html=True,
    )

# KPI row: one card per model with the headline numbers and their gap to the targets.
lat_key = f"{targets['latency_percentile']}_ms"
lat_label = f"{targets['latency_percentile']} 지연"
ap_label = viz.AP_METRICS[targets["ap_metric"]]
cards = st.columns(len(MODEL_ORDER))
for column, model in zip(cards, MODEL_ORDER):
    acc = selected.get(("accuracy", model))
    lat = selected.get(("latency", model))
    with column.container(border=True):
        st.markdown(f"{swatch(model, theme)}**{MODELS[model]}**", unsafe_allow_html=True)
        if acc is None and lat is None:
            elsewhere = [label for label in devices
                         if label != device and model in timed_models[label]]
            st.caption(("이 장치에서 측정한 결과 없음 · 사이드바에서 다른 장치 선택: "
                        + ", ".join(elsewhere)) if elsewhere else "결과 없음")
            continue
        left, right = st.columns(2)
        ap_value = viz._num(acc.section.get(targets["ap_metric"])) if acc else None
        if ap_value is not None:
            gap = ap_value * 100 - ap_target
            left.metric(ap_label, f"{ap_value * 100:.1f}%", f"{gap:+.1f}%p",
                        help=f"아래 숫자 = 목표 {ap_target:g}%와의 차이 (+면 목표 달성)")
        else:
            left.metric(ap_label, "–")
        latency_value = viz._num(lat.section.get(lat_key)) if lat else None
        env = lat.environment if lat else {}
        on_cpu = env.get("device_type") == "cpu" or env.get("device") == "cpu"
        off_target = lat is not None and not on_target_device(lat)
        if latency_value is not None:
            shown = f"{latency_value:.0f} ms" if latency_value >= 100 else f"{latency_value:.1f} ms"
            right.metric(lat_label, shown, f"{latency_value - latency_target:+.1f} ms",
                         delta_color="off" if off_target else "inverse",
                         help=(f"아래 숫자 = 목표 {latency_target:g} ms(T4 GPU 기준)와의 차이. "
                               f"{device_label(lat)}에서 측정했으므로 참고용입니다." if off_target
                               else f"아래 숫자 = 목표 {latency_target:g} ms와의 차이 "
                                    "(-면 목표 달성)"))
        else:
            right.metric(lat_label, "–")
        if lat is not None:
            right.caption(f"측정 장치: {device_label(lat)}")
            if off_target:
                where = ("CPU 측정" if on_cpu
                         else "T4 외 장치" if env.get("device_name") else "T4 여부 미기록")
                right.caption(f"{where} · 목표(T4 GPU)와 직접 비교 불가")
        elif device:
            right.caption("이 장치에서 측정한 결과 없음")
        details = []
        fps = viz._num(lat.section.get("fps")) if lat else None
        if fps:
            details.append(f"{fps:.1f} FPS")
        stats = (lat.data.get("model_stats") or {}) if lat else {}
        params, _ = viz._size_stats(stats)  # the graph that was timed, as in the size chart
        if params:
            details.append(f"파라미터 {params:.1f}M")
        images = viz._num(acc.section.get("images_evaluated")) if acc else None
        if images:
            details.append(f"평가 {images:,.0f}장")
        st.caption(" · ".join(details) or " ")

if not any(("accuracy", m) in selected for m in MODEL_ORDER):
    st.info("정확도(AP) 결과가 없어 AP 카드와 정확도 대 속도 그래프가 비어 있습니다. "
            "학습 중 검증 결과를 가져오거나 직접 평가하세요.", icon=LEVEL_ICONS["info"])
    st.code("python scripts/import_training_results.py outputs/coco_4090_v2\n"
            "python scripts/evaluate_coco.py --model light_rcnn "
            "--weights outputs/coco_4090_v2/best.pth", language="bash")

figures = {
    "tradeoff": viz.tradeoff_figure(selected, targets, theme),
    "latency_percentiles": viz.latency_percentile_figure(selected, targets, theme),
    "latency_distribution": viz.latency_distribution_figure(selected, targets, theme),
    "latency_stability": viz.latency_stability_figure(selected, theme),
    "accuracy": viz.accuracy_figure(selected, theme),
    "per_class": viz.per_class_figure(selected, theme, top=per_class_top),
    "stages": viz.stage_figure(selected, theme),
    "efficiency": viz.efficiency_figure(selected, theme),
}
rows = viz.summary_rows(selected)
device_note = f"측정 장치: {device} (사이드바에서 변경)" if device else None
ms_column = st.column_config.NumberColumn(format="%.2f ms")

overview, latency_tab, accuracy_tab, stages_tab, size_tab, data_tab = st.tabs(
    ["한눈에 보기", "지연 시간", "정확도", "병목 분석", "모델 크기", "데이터·출처"])

with overview:
    show(figures["tradeoff"], "tradeoff")
    show(figures["latency_percentiles"], "latency_percentiles_overview")
    with st.expander("표로 보기"):
        table(rows, column_config=SUMMARY_COLUMN_CONFIG)

with latency_tab:
    if device_note:
        st.caption(device_note)
    show(figures["latency_distribution"], "latency_distribution")
    show(figures["latency_stability"], "latency_stability")
    latency_rows = []
    for model in MODEL_ORDER:
        report = selected.get(("latency", model))
        if report is None:
            continue
        section = report.section
        latency_rows.append({
            "모델": MODELS[model],
            **{name: viz._num(section.get(key)) for key, name in viz.PERCENTILES},
            "평균": viz._num(section.get("mean_ms")), "표준편차": viz._num(section.get("std_ms")),
            "첫 추론": viz._num(section.get("cold_start_ms")),
            "반복": section.get("n") or section.get("iterations"),
            "평균 검출 수": viz._num(section.get("mean_detections")),
            "점수 임계값": section.get("score_threshold"),
            "추론 방식": section.get("inference_mode"),
        })
    table(latency_rows, column_config={
        **{name: ms_column for name in ("p50", "p90", "p95", "p99", "평균", "표준편차", "첫 추론")},
        "평균 검출 수": st.column_config.NumberColumn(format="%.2f")})
    timed = [m for m in MODEL_ORDER if ("latency", m) in selected]
    scopes = {selected[("latency", m)].section.get("scope_id") for m in timed}
    for scope in scopes:
        st.caption(f"측정 범위: {SCOPE_LABELS.get(scope, '기록 없음 (이전 형식 결과)')}")
    cold_scopes = {MODELS[m]: selected[("latency", m)].section.get("cold_start_scope")
                   for m in timed}
    if cold_scopes:
        st.caption("첫 추론 범위: "
                   + " · ".join(f"{name} = {COLD_START_LABELS.get(scope, scope or '기록 없음')}"
                                for name, scope in cold_scopes.items())
                   + (" — 범위가 달라 직접 비교할 수 없습니다."
                      if len(set(cold_scopes.values())) > 1 else ""))

with accuracy_tab:
    show(figures["accuracy"], "accuracy")
    show(figures["per_class"], "per_class")
    # Table twin built from the chart's own markers, so both list the same classes and values.
    markers = [trace for trace in figures["per_class"].data if trace.mode == "markers"]
    if markers:
        spread_label = "모델 간 차이 (%p)"
        per_class_rows = []
        for index in reversed(range(len(markers[0].y))):  # best class first, as the chart reads
            values = {trace.name: trace.x[index] for trace in markers}
            row = {"클래스": markers[0].y[index], **values}
            known = [value for value in values.values() if value is not None]
            if len(markers) > 1:
                row[spread_label] = max(known) - min(known) if len(known) > 1 else None
            per_class_rows.append(row)
        with st.expander("클래스별 AP 표"):
            table(per_class_rows, column_config={
                name: st.column_config.NumberColumn(
                    format="%.1f%%p" if name == spread_label else "%.1f%%")
                for name in per_class_rows[0] if name != "클래스"})
    accuracy_rows = []
    for model in MODEL_ORDER:
        report = selected.get(("accuracy", model))
        if report is None:
            continue
        section = report.section
        accuracy_rows.append({
            "모델": MODELS[model],
            **{label: viz._pct(section.get(key)) for key, label in viz.AP_METRICS.items()},
            "AR@100": viz._pct(section.get("AR_100")),
            "평가 이미지": section.get("images_evaluated"),
            "전체 이미지": section.get("images_total"),
            "평가기": section.get("evaluator"), "점수 임계값": section.get("score_threshold"),
        })
    table(accuracy_rows, column_config={label: st.column_config.NumberColumn(format="%.2f%%")
                                        for label in [*viz.AP_METRICS.values(), "AR@100"]})

with stages_tab:
    if device_note:
        st.caption(device_note)
    show(figures["stages"], "stages")
    stage_rows = []
    for model in MODEL_ORDER:
        report = selected.get(("stages", model))
        if report is None:
            continue
        section = report.section
        for item in [*(section.get("items") or []), *(section.get("substages") or [])]:
            stage_rows.append({"모델": MODELS[model],
                               "단계": item.get("label") or item.get("stage"),
                               "상위 단계": STAGE_LABELS.get(item.get("parent"), item.get("parent") or ""),
                               "p50": viz._num(item.get("p50_ms")),
                               "평균": viz._num(item.get("mean_ms")),
                               "p95": viz._num(item.get("p95_ms")),
                               "비중": viz._pct(item.get("share"))})
        note = section.get("overhead_note")
        if note:
            st.caption(f"{MODELS[model]}: {note}")
    if stage_rows:
        table(stage_rows, column_config={**{name: ms_column for name in ("p50", "평균", "p95")},
                                         "비중": st.column_config.NumberColumn(format="%.1f%%")})

with size_tab:
    show(figures["efficiency"], "efficiency")
    for model in MODEL_ORDER:
        report = selected.get(("latency", model))
        note = (report.data.get("model_stats") or {}).get("flops_method") if report else None
        if note:
            st.caption(f"{MODELS[model]} 연산량 측정 방법: {note}")

with data_tab:
    st.subheader("요약 표")
    table(rows, column_config=SUMMARY_COLUMN_CONFIG)
    st.subheader("사용한 결과 파일")
    source_rows = []
    chosen = {id(report) for report in selected.values()}
    for report in sorted(reports,
                         key=lambda r: (r.kind, MODEL_ORDER.index(r.model), r.created_at)):
        source_rows.append({"사용": id(report) in chosen, "종류": report.kind,
                            "모델": report.model_name, "측정 시각": report.created_at,
                            "환경": describe_environment(report.environment),
                            "커밋": (report.data.get("git_commit") or "")[:12],
                            "파일": report.source, "메모": " ".join(report.notes)})
    table(source_rows, column_config={"사용": st.column_config.CheckboxColumn(disabled=True)})
    report_title = "Light R-CNN 성능 비교 보고서" + (f" ({device})" if device else "")
    st.download_button(
        "HTML 보고서 내려받기",
        data=viz.html_report(viz.build_all(selected, targets, "light"), rows, warnings, selected,
                             "light", report_title),
        file_name="light_rcnn_report.html", mime="text/html", icon=":material/download:")
    with st.expander("선택된 결과 원본 JSON"):
        for (kind, model), report in sorted(selected.items()):
            raw = dict(report.data)
            latency = raw.get("latency")
            if isinstance(latency, dict) and isinstance(latency.get("samples_ms"), list):
                raw["latency"] = {**latency, "samples_ms": f"{len(latency['samples_ms'])}개 생략"}
            st.markdown(f"**{MODELS[model]} · {kind}** — `{report.source}`")
            st.code(json.dumps(raw, indent=2, ensure_ascii=False)[:6000], language="json")
