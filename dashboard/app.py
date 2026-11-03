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
from pathlib import Path, PurePosixPath

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from light_rcnn import viz
from light_rcnn.reporting import (
    BASELINE_VARIANT,
    BASELINE_VARIANT_LABEL,
    CHECKPOINT_MODEL,
    COLD_START_LABELS,
    KIND_LABELS,
    KINDS,
    MODEL_ORDER,
    MODELS,
    SCOPE_LABELS,
    STAGE_LABELS,
    Report,
    checkpoint_groups,
    checkpoint_reports,
    checkpoint_warnings,
    checkpoints,
    comparability_warnings,
    cuda_graph_note,
    describe_environment,
    device_label,
    improvement_warnings,
    load_reports,
    load_targets,
    model_label,
    on_target_device,
    select_baseline,
    select_reports,
    timing_devices,
    variant_labels,
    variants,
)

PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"],
               "toImageButtonOptions": {"format": "png", "scale": 2}}
LEVEL_ICONS = {"error": ":material/block:", "warning": ":material/warning:",
               "info": ":material/info:"}
SUMMARY_COLUMN_CONFIG = {
    name: st.column_config.NumberColumn(format="%.2f")
    for name in ("AP@[.50:.95] (%)", "AP50 (%)", "AP75 (%)", "p50 (ms)", "p95 (ms)", "p99 (ms)",
                 "FPS", "첫 추론 (ms)", "파라미터 (M)", "GFLOPs")}
CI_HELP = ("p50 변화율의 95% 신뢰구간: 같은 실행에서 설정을 번갈아 잰 라운드를 쌍으로 다시 뽑아 "
           "(쌍 부트스트랩) 구합니다. 0을 포함하면 측정 오차 범위입니다.")
DELTA_COLUMN_CONFIG = {
    "개선 전": st.column_config.NumberColumn(format="%.2f"),
    "개선 후": st.column_config.NumberColumn(format="%.2f"),
    "변화": st.column_config.NumberColumn(format="%+.2f"),
    "상대 변화 (%)": st.column_config.NumberColumn(format="%+.1f%%"),
    viz.CI_COLUMN: st.column_config.TextColumn(help=CI_HELP),
}
PERCENTILE_OPTIONS = [name for _, name in viz.PERCENTILES]
AP_OPTIONS = ["AP_50_95", "AP_50"]
VARIANT_LABEL = "추론 설정"
CHECKPOINT_LABEL = "Light R-CNN 체크포인트"
SETTINGS_CI_COLUMN = f"p50 {viz.CI_COLUMN}"
VERSION = re.compile(r"(?<![A-Za-z0-9])[vV](\d+)(?![A-Za-z0-9])")  # 'coco_4090_v3/best' -> 3
# Plain-language verdict rules under the before/after table (viz.improvement_rows applies them).
VERDICT_NOTE = (
    "변화 = 개선 후 − 개선 전 (AP는 %p). 판정: AP는 변화가 {ap:g}%p보다 작으면 '변화 없음 (오차 범위)'입니다. "
    "p50에 95% CI가 있으면 CI로 판정합니다: CI가 0을 포함하면 '변화 없음 (오차 범위)', CI 전체가 0보다 "
    "작으면(빨라짐) '좋아짐', 크면(느려짐) '나빠짐'. CI가 없는 p50과 p95·FPS는 상대 변화가 {rel:g}%보다 "
    "작으면 '변화 없음 (오차 범위)'입니다. 95% CI는 한 번의 실행 안에서 번갈아 잰 측정의 흔들림만 담으므로, "
    "다른 날이나 다른 기계에서 다시 재면 차이가 이보다 클 수 있습니다. AP 기준 {ap:g}%p는 COCO val2017 "
    "전체(5000장)를 같은 이미지로 비교할 때의 값이고, 128장 스모크 세트의 AP 오차는 ±0.5~1%p입니다.")

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
    for column in (viz.CI_COLUMN, SETTINGS_CI_COLUMN):  # rows without a CI: blank, not 'None'
        if column in frame:
            frame[column] = frame[column].fillna("")
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


def checkpoint_names(found: list[Report], choices: list[tuple[str, str]]
                     ) -> tuple[dict[str, str], dict[str, str]]:
    """(selector, chart) name per Light R-CNN checkpoint key of reporting.checkpoints: its
    model_label ('Light R-CNN v3'); else, when the weights path names a version, 'Light R-CNN v3
    (coco_4090_v3/best.pth)' and 'Light R-CNN v3'; else the path label."""
    model = MODELS[CHECKPOINT_MODEL]
    options: dict[str, str] = {}
    charts: dict[str, str] = {}
    for key, label in choices:
        members = [r for r in checkpoint_reports(found, key) if r.model == CHECKPOINT_MODEL]
        named = any(map(model_label, members))
        versions = VERSION.findall(label)
        if named or not versions:
            options[key] = label
            charts[key] = label if named else f"{model} ({label})"
            continue
        paths = [str(r.data["weights"]).replace("\\", "/") for r in members if r.data.get("weights")]
        suffix = PurePosixPath(min(paths, key=len)).suffix if paths else ""
        file = label if label.endswith((")", suffix)) else label + suffix
        options[key], charts[key] = f"{model} v{versions[-1]} ({file})", f"{model} v{versions[-1]}"
    names = list(charts.values())  # two checkpoints of one version: charts name the file too
    return options, {key: options[key] if names.count(name) > 1 else name
                     for key, name in charts.items()}


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
    # Light R-CNN v2 and v3 share one model key; with both in the folder, one is shown at a time.
    checkpoint_choices = checkpoints(reports)
    checkpoint = None
    model_labels: dict[str, str] = {}
    if len(checkpoint_choices) > 1:
        option_names, chart_names = checkpoint_names(reports, checkpoint_choices)
        checkpoint = st.selectbox(CHECKPOINT_LABEL, [key for key, _ in checkpoint_choices],
                                  format_func=option_names.get,
                                  help="결과 폴더에 Light R-CNN 체크포인트(v2·v3 등) 여러 개의 결과가 "
                                       "있으면 하나를 고릅니다. 카드·그래프·표의 Light R-CNN은 고른 "
                                       "체크포인트의 결과이고, 다른 모델은 그대로입니다. 최근에 측정한 "
                                       "체크포인트가 맨 위에 있습니다.")
        st.caption(f"결과 폴더에 Light R-CNN 체크포인트 {len(checkpoint_choices)}개의 결과가 함께 있어 "
                   "고른 체크포인트만 보여 줍니다.")
        model_labels[CHECKPOINT_MODEL] = chart_names[checkpoint]
    elif checkpoint_choices:  # one checkpoint (a per-version folder): named as the selector would
        [(only, _)] = checkpoint_choices
        members = set(map(id, checkpoint_reports(reports, only)))
        light = [report for report in reports if report.model == CHECKPOINT_MODEL]
        if all(id(report) in members for report in light) and any(map(model_label, light)):
            model_labels[CHECKPOINT_MODEL] = checkpoint_names(reports, checkpoint_choices)[1][only]
    pool = checkpoint_reports(reports, checkpoint) if checkpoint else reports
    variant_options = variants(pool)
    variant_names = variant_labels(reports)  # the sources table names every report's setting
    variant = variant_options[0] if variant_options else BASELINE_VARIANT
    if len(variant_options) > 1:
        variant = st.selectbox(VARIANT_LABEL, variant_options, format_func=variant_names.get,
                               help="같은 모델을 여러 추론 설정(기본 설정, Conv+BN 융합, RPN 후보 축소 "
                                    "등)으로 측정한 결과가 있으면 고릅니다. 카드와 그래프는 고른 설정의 "
                                    "결과만 쓰고, '개선 전/후' 탭에서 기본 설정과 비교합니다.")
    variant_label = variant_names.get(variant, BASELINE_VARIANT_LABEL)
    show_variant = len(variant_options) > 1 or variant != BASELINE_VARIANT
    variant_reports = [report for report in pool if report.variant == variant]
    devices = timing_devices(variant_reports)
    timed_models: dict[str, set[str]] = {}
    for report in variant_reports:
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

# Timings come from one device only; accuracy is device-independent and always eligible. Every
# report comes from the chosen inference setting (and Light R-CNN checkpoint); the baseline is the
# "before" of the improvement tab, from the same comparison run when the folder has it.
selected = select_reports(reports, device=device, variant=variant, checkpoint=checkpoint)
baseline_selected = select_baseline(reports, selected, device=device, checkpoint=checkpoint)
# The mixed-checkpoint note goes first: it says which run every card and chart comes from. The
# Light R-CNN checkpoint selector leaves only other models' mixes to warn about.
warnings = [*checkpoint_warnings(pool, selected),
            *comparability_warnings(selected, latency_target)]
improvement_issues = (improvement_warnings(baseline_selected, selected)
                      if variant != BASELINE_VARIANT else [])
missing_dir = not output_dir.is_dir()

checkpoint_of: dict[int, str] = {}  # id(report) -> label of its checkpoint
named: set[int] = set()  # reports whose checkpoint has a model_label ('Light R-CNN v3')
for model_groups in checkpoint_groups(reports).values():
    for label, members in model_groups.items():
        checkpoint_of.update(dict.fromkeys(map(id, members), label))
        if any(map(model_label, members)):
            named.update(map(id, members))


def model_name(model: str) -> str:
    """Display name of a model; Light R-CNN names the checkpoint picked in the sidebar."""
    return model_labels.get(model) or MODELS[model]


def checkpoint_title(model: str, selection: dict) -> str | None:
    """Named checkpoint(s) behind the model's accuracy and latency in ``selection``."""
    picked = [selection.get((kind, model)) for kind in ("accuracy", "latency")]
    if model in model_labels and any(picked):
        return model_labels[model]
    titles = [checkpoint_of[id(r)] for r in picked if r is not None and id(r) in named]
    return " / ".join(dict.fromkeys(titles)) or None


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
        + ("- **추론 설정**: 같은 모델을 다른 추론 옵션(Conv+BN 융합, RPN 후보 축소 등)으로 측정한 "
           "결과는 섞지 않고, 사이드바에서 고른 설정만 보여 줍니다. '개선 전/후' 탭에서 기본 설정과 "
           "비교합니다.\n" if show_variant else "")
        + ("- **Light R-CNN 체크포인트**: v2와 v3처럼 Light R-CNN의 다른 체크포인트 결과도 섞지 않고, "
           "사이드바에서 고른 체크포인트만 보여 줍니다. 그래프와 표에는 그 이름(예: Light R-CNN v3)이 "
           "나옵니다.\n" if checkpoint else "")
        + "- 모델 색과 모양은 모든 그래프에서 같습니다: "
        + " · ".join(f"{swatch(m, theme)}{model_name(m)}" for m in MODEL_ORDER),
        unsafe_allow_html=True,
    )

# KPI row: one card per model with the headline numbers and their gap to the targets.
lat_key = f"{targets['latency_percentile']}_ms"
lat_label = f"{targets['latency_percentile']} 지연"
ap_label = viz.AP_METRICS[targets["ap_metric"]]
if show_variant:
    st.caption(f"추론 설정: **{variant_label}** — 아래 카드와 그래프는 이 설정으로 측정한 결과입니다 "
               "(사이드바에서 변경, 기본 설정과의 비교는 '개선 전/후' 탭).")
measured_with = {name: {r.model for r in pool if r.variant == name} for name in variant_options}
cards = st.columns(len(MODEL_ORDER))
for column, model in zip(cards, MODEL_ORDER):
    acc = selected.get(("accuracy", model))
    lat = selected.get(("latency", model))
    with column.container(border=True):
        title = checkpoint_title(model, selected) or model_name(model)
        st.markdown(f"{swatch(model, theme)}**{title}**", unsafe_allow_html=True)
        if acc is None and lat is None:
            elsewhere = [label for label in devices
                         if label != device and model in timed_models[label]]
            other_settings = [variant_names[name] for name in variant_options
                              if name != variant and model in measured_with[name]]
            if elsewhere:
                st.caption("이 장치에서 측정한 결과 없음 · 사이드바에서 다른 장치 선택: "
                           + ", ".join(elsewhere))
            elif other_settings and model not in measured_with.get(variant, ()):
                st.caption("이 추론 설정으로 측정한 결과 없음 · 사이드바에서 다른 설정 선택: "
                           + ", ".join(other_settings))
            else:
                st.caption("결과 없음")
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

if not any(("accuracy", m) in selected for m in MODEL_ORDER) and variant != BASELINE_VARIANT:
    st.info(f"'{variant_label}' 설정으로 평가한 정확도(AP) 결과가 없어 AP 카드와 정확도 대 속도 "
            "그래프가 비어 있습니다. 같은 추론 설정으로 COCO 평가를 실행하세요.",
            icon=LEVEL_ICONS["info"])
elif not any(("accuracy", m) in selected for m in MODEL_ORDER):
    st.info("정확도(AP) 결과가 없어 AP 카드와 정확도 대 속도 그래프가 비어 있습니다. "
            "학습 중 검증 결과를 가져오거나 직접 평가하세요.", icon=LEVEL_ICONS["info"])
    st.code("python scripts/import_training_results.py outputs/coco_4090_v2\n"
            "python scripts/evaluate_coco.py --model light_rcnn "
            "--weights outputs/coco_4090_v2/best.pth", language="bash")

names = {"model_labels": model_labels}
figures = {
    "tradeoff": viz.tradeoff_figure(selected, targets, theme, **names),
    "latency_percentiles": viz.latency_percentile_figure(selected, targets, theme, **names),
    "latency_distribution": viz.latency_distribution_figure(selected, targets, theme, **names),
    "latency_stability": viz.latency_stability_figure(selected, theme, **names),
    "accuracy": viz.accuracy_figure(selected, theme, **names),
    "per_class": viz.per_class_figure(selected, theme, top=per_class_top, **names),
    "stages": viz.stage_figure(selected, theme, **names),
    "efficiency": viz.efficiency_figure(selected, theme, **names),
}
rows = viz.summary_rows(selected, **names)
checkpoint_note = f"체크포인트: {model_labels[CHECKPOINT_MODEL]}" if checkpoint else None
variant_note = " · ".join(part for part in (
    checkpoint_note, show_variant and f"추론 설정: {variant_label}") if part) or None
device_note = " · ".join(part for part in (variant_note, device and f"측정 장치: {device}") if part)
device_note = f"{device_note} (사이드바에서 변경)" if device_note else None
ms_column = st.column_config.NumberColumn(format="%.2f ms")


def applied_options(chosen: dict) -> list[tuple[str, list[str], dict]]:
    """(model, kinds, record) for each distinct inference_options record in the selection."""
    groups: dict[tuple[str, str], tuple[list[str], dict]] = {}
    for model in MODEL_ORDER:
        for kind in KINDS:
            report = chosen.get((kind, model))
            record = report.inference_options if report else {}
            if record:
                key = (model, json.dumps(record, sort_keys=True, default=str))
                groups.setdefault(key, ([], record))[0].append(KIND_LABELS[kind])
    return [(model, kinds, record) for (model, _), (kinds, record) in groups.items()]


def all_settings_rows() -> list[dict]:
    """Headline before/after numbers of every non-baseline setting on the chosen device, with the
    per-setting view's before/after check ('비교') and, when measured, the paired p50 CI."""
    headline = {"AP@[.50:.95] (%)": "AP 변화 (%p)", "p50 (ms)": "p50 변화 (ms)"}
    overview_rows = []
    for name in variant_options:
        if name == BASELINE_VARIANT:
            continue
        after = select_reports(reports, device=device, variant=name, checkpoint=checkpoint)
        before = select_baseline(reports, after, device=device, checkpoint=checkpoint)
        by_model: dict[str, dict] = {}
        for row in viz.improvement_rows(before, after, **names):
            if row["지표"] in headline:
                entry = by_model.setdefault(row["모델"], {
                    "추론 설정": variant_names[name], "모델": row["모델"], "비교": None,
                    **dict.fromkeys([*headline.values(), "p50 상대 변화 (%)", SETTINGS_CI_COLUMN])})
                entry[headline[row["지표"]]] = row["변화"]
                if row["지표"] == "p50 (ms)":
                    entry["p50 상대 변화 (%)"] = row["상대 변화 (%)"]
                    entry[SETTINGS_CI_COLUMN] = row.get(viz.CI_COLUMN)
        for model in MODEL_ORDER:
            if model_name(model) in by_model:
                levels = {w.level for w in improvement_warnings(
                    {key: r for key, r in before.items() if key[1] == model},
                    {key: r for key, r in after.items() if key[1] == model})}
                by_model[model_name(model)]["비교"] = ("비교 불가" if "error" in levels
                                                     else "주의" if "warning" in levels else "가능")
        overview_rows.extend(by_model.values())
    if not any(row[SETTINGS_CI_COLUMN] for row in overview_rows):  # no CI: the table as before
        for row in overview_rows:
            del row[SETTINGS_CI_COLUMN]
    return overview_rows


overview, improvement_tab, latency_tab, accuracy_tab, stages_tab, size_tab, data_tab = st.tabs(
    ["한눈에 보기", "개선 전/후", "지연 시간", "정확도", "병목 분석", "모델 크기", "데이터·출처"])

with overview:
    show(figures["tradeoff"], "tradeoff")
    show(figures["latency_percentiles"], "latency_percentiles_overview")
    with st.expander("표로 보기"):
        table(rows, column_config=SUMMARY_COLUMN_CONFIG)

with improvement_tab:
    baseline_name = variant_names.get(BASELINE_VARIANT, BASELINE_VARIANT_LABEL)
    others = [variant_names[name] for name in variant_options if name != BASELINE_VARIANT]
    if not others:
        st.info("기본 설정 말고 다른 추론 설정으로 측정한 결과가 아직 없습니다. 결과 JSON에 추론 "
                "설정(variant)이 기록된 측정을 결과 폴더에 추가하면 여기서 개선 전/후를 비교합니다.",
                icon=LEVEL_ICONS["info"])
    elif variant == BASELINE_VARIANT:
        st.info("사이드바의 '추론 설정'에서 개선 설정을 고르면 기본 설정과 그래프로 비교합니다: "
                + ", ".join(others), icon=LEVEL_ICONS["info"])
    else:
        compared = []  # 'Light R-CNN v3', or 'Light R-CNN v2 → Light R-CNN v3' when they differ
        for model in MODEL_ORDER:
            before = checkpoint_title(model, baseline_selected)
            after = checkpoint_title(model, selected)
            if not (before or after) or not any(key[1] == model for key in selected):
                continue
            if not any(key[1] == model for key in baseline_selected):
                compared.append(f"{after or MODELS[model]} (개선 전 결과 없음)")
            else:
                compared.append(after if before == after
                                else f"{before or MODELS[model]} → {after or MODELS[model]}")
        st.caption(f"비교: {baseline_name} → {variant_label}"
                   + (f" · 체크포인트: {', '.join(compared)}" if compared else "")
                   + (f" · 측정 장치: {device}" if device else ""))
        for warning in improvement_issues:
            alert = st.error if warning.level == "error" else st.warning
            alert(warning.message, icon=LEVEL_ICONS[warning.level])
        show(viz.improvement_figure(baseline_selected, selected, theme, variant_label,
                                    ap_metric=targets["ap_metric"],
                                    percentile=targets["latency_percentile"], **names),
             "improvement")
        delta_rows = viz.improvement_rows(baseline_selected, selected, **names)
        if delta_rows:  # every row visible: 35 px per row plus the header
            table(delta_rows, column_config=DELTA_COLUMN_CONFIG, height=35 * (len(delta_rows) + 1) + 3)
            st.caption(VERDICT_NOTE.format(ap=viz.NOISE_AP_PP, rel=viz.NOISE_RELATIVE_PCT))
        applied = applied_options(selected)
        if applied:
            with st.expander("적용된 추론 옵션"):
                for model, kinds, record in applied:
                    st.markdown(f"**{model_name(model)}** · {'·'.join(kinds)}")
                    if isinstance(record.get("summary"), str):
                        st.caption(record["summary"])  # inference_opts' one-line summary
                    graph_note = cuda_graph_note(record)  # the summary names the graph regardless
                    if graph_note:
                        st.warning(graph_note, icon=LEVEL_ICONS["warning"])
                    st.code(json.dumps(record, indent=2, ensure_ascii=False, default=str),
                            language="json")
    if others and (variant == BASELINE_VARIANT or len(others) > 1):
        settings_rows = all_settings_rows()
        if settings_rows:
            with st.expander("모든 추론 설정 한눈에 (기본 설정 대비)",
                             expanded=variant == BASELINE_VARIANT):
                table(settings_rows, column_config={
                    "AP 변화 (%p)": st.column_config.NumberColumn(format="%+.2f"),
                    "p50 변화 (ms)": st.column_config.NumberColumn(format="%+.2f"),
                    "p50 상대 변화 (%)": st.column_config.NumberColumn(format="%+.1f%%"),
                    SETTINGS_CI_COLUMN: st.column_config.TextColumn(help=CI_HELP)})
                if any(row["비교"] != "가능" for row in settings_rows):
                    st.caption("'비교 불가'는 개선 전/후의 가중치·평가 이미지 등이 달라 변화량이 추론 "
                               "설정만의 효과가 아닌 행이고, '주의'는 평가 배치 크기·측정 이미지 수 등이 "
                               "달라 다른 차이가 조금 섞였을 수 있는 행입니다. 사이드바 '추론 설정'에서 "
                               "그 설정을 고르면 이유가 표시됩니다.")

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
            "모델": model_name(model),
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
    cold_scopes = {model_name(m): selected[("latency", m)].section.get("cold_start_scope")
                   for m in timed}
    if cold_scopes:
        st.caption("첫 추론 범위: "
                   + " · ".join(f"{name} = {COLD_START_LABELS.get(scope, scope or '기록 없음')}"
                                for name, scope in cold_scopes.items())
                   + (" — 범위가 달라 직접 비교할 수 없습니다."
                      if len(set(cold_scopes.values())) > 1 else ""))

with accuracy_tab:
    if variant_note:
        st.caption(f"{variant_note} (사이드바에서 변경)")
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
            "모델": model_name(model),
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
            stage_rows.append({"모델": model_name(model),
                               "단계": item.get("label") or item.get("stage"),
                               "상위 단계": STAGE_LABELS.get(item.get("parent"), item.get("parent") or ""),
                               "p50": viz._num(item.get("p50_ms")),
                               "평균": viz._num(item.get("mean_ms")),
                               "p95": viz._num(item.get("p95_ms")),
                               "비중": viz._pct(item.get("share"))})
        note = section.get("overhead_note")
        if note:
            st.caption(f"{model_name(model)}: {note}")
    if stage_rows:
        table(stage_rows, column_config={**{name: ms_column for name in ("p50", "평균", "p95")},
                                         "비중": st.column_config.NumberColumn(format="%.1f%%")})

with size_tab:
    if variant_note:
        st.caption(f"{variant_note} (사이드바에서 변경)")
    show(figures["efficiency"], "efficiency")
    for model in MODEL_ORDER:
        report = selected.get(("latency", model))
        note = (report.data.get("model_stats") or {}).get("flops_method") if report else None
        if note:
            st.caption(f"{model_name(model)} 연산량 측정 방법: {note}")

with data_tab:
    st.subheader("요약 표")
    table(rows, column_config=SUMMARY_COLUMN_CONFIG)
    st.subheader("사용한 결과 파일")
    source_rows = []
    chosen = {id(report) for report in selected.values()}
    for report in sorted(reports,
                         key=lambda r: (r.kind, MODEL_ORDER.index(r.model), r.created_at)):
        source_rows.append({"사용": id(report) in chosen, "종류": report.kind,
                            "모델": report.model_name,
                            "체크포인트": checkpoint_of.get(id(report)) or model_label(report),
                            **({VARIANT_LABEL: variant_names[report.variant]} if show_variant else {}),
                            "측정 시각": report.created_at,
                            "환경": describe_environment(report.environment),
                            "커밋": (report.data.get("git_commit") or "")[:12],
                            "파일": report.source, "메모": " ".join(report.notes)})
    table(source_rows, column_config={"사용": st.column_config.CheckboxColumn(disabled=True)})
    report_scope = " · ".join(part for part in (model_labels.get(CHECKPOINT_MODEL),
                                                show_variant and variant_label, device) if part)
    report_title = "Light R-CNN 성능 비교 보고서" + (f" ({report_scope})" if report_scope else "")
    # The report's before/after chart carries the same mismatch warnings as the '개선 전/후' tab.
    report_warnings = [*warnings, *improvement_issues]
    st.download_button(
        "HTML 보고서 내려받기",
        data=viz.html_report(viz.build_all(selected, targets, "light", baseline_selected, variant_label,
                                           **names),
                             rows, report_warnings, selected, "light", report_title, **names),
        file_name="light_rcnn_report.html", mime="text/html", icon=":material/download:")
    with st.expander("선택된 결과 원본 JSON"):
        for (kind, model), report in sorted(selected.items()):
            raw = dict(report.data)
            latency = raw.get("latency")
            if isinstance(latency, dict) and isinstance(latency.get("samples_ms"), list):
                raw["latency"] = {**latency, "samples_ms": f"{len(latency['samples_ms'])}개 생략"}
            st.markdown(f"**{model_name(model)} · {kind}** — `{report.source}`")
            st.code(json.dumps(raw, indent=2, ensure_ascii=False)[:6000], language="json")
