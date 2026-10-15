"""Interactive comparison dashboard for Faster R-CNN, Light R-CNN and YOLO26."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


MODEL_ORDER = ["Vanilla Faster R-CNN", "Light R-CNN", "YOLO26"]
MODEL_KEYS = {
    "Vanilla Faster R-CNN": "faster_rcnn",
    "Light R-CNN": "light_rcnn",
    "YOLO26": "yolo26",
}


def identify_model(report: dict) -> str | None:
    name = " ".join(str(report.get(key, "")) for key in ("model_name", "model", "model_family", "architecture")).lower()
    compact = re.sub(r"[^a-z0-9]+", "", name)
    if "yolo26" in compact or compact.startswith("yolo"):
        return "YOLO26"
    if "light" in compact:
        return "Light R-CNN"
    if "fasterrcnn" in compact or compact in {"fasterrcnn", "fasterrcnnresnet50fpn"}:
        return "Vanilla Faster R-CNN"
    return None


def first_value(report: dict, *keys: str):
    for key in keys:
        value = report.get(key)
        if value is not None:
            return value
    return None


def merge_report(records: dict, report: dict, source: str) -> None:
    model_name = identify_model(report)
    if model_name is None:
        return
    row = records.setdefault(model_name, {"Model": model_name, "Sources": []})
    accuracy = report.get("accuracy", {}) or {}
    latency = report.get("latency", {}) or {}
    mapping = {
        "AP@[.50:.95]": first_value(accuracy, "AP_50_95", "map", "mAP")
        if accuracy else first_value(report, "AP_50_95", "map", "mAP"),
        "AP50": first_value(accuracy, "AP_50", "map50") if accuracy else first_value(report, "AP_50", "map50"),
        "AP75": first_value(accuracy, "AP_75", "map75") if accuracy else first_value(report, "AP_75", "map75"),
        "p50 (ms)": first_value(latency, "p50_ms") if latency else first_value(report, "p50_ms", "median_ms"),
        "p95 (ms)": first_value(latency, "p95_ms") if latency else first_value(report, "p95_ms"),
        "Parameters (M)": first_value(report, "parameters_m", "params_m"),
        "FLOPs (G)": first_value(report, "flops_g", "gflops"),
        "Precision": report.get("precision"),
        "Device": report.get("device"),
        "Image size": first_value(report, "image_size", "imgsz"),
        "Timed scope": report.get("timed_scope"),
    }
    for key, value in mapping.items():
        if value is not None:
            row[key] = value
    row["Sources"].append(source)


def parse_reports(payloads):
    records = {}
    errors = []
    for name, payload in payloads:
        try:
            data = json.loads(payload)
            reports = data if isinstance(data, list) else [data]
            for report in reports:
                if isinstance(report, dict):
                    merge_report(records, report, name)
        except Exception as exc:  # show malformed user reports in the UI instead of crashing
            errors.append(f"{name}: {exc}")
    rows = []
    for model in MODEL_ORDER:
        row = records.get(model, {"Model": model, "Sources": []})
        rows.append(row)
    return pd.DataFrame(rows), errors


def percent_ap(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.where(values > 1, values * 100)


st.set_page_config(page_title="Light R-CNN 성능 비교", page_icon="📊", layout="wide")
st.title("3개 객체 탐지 모델 성능 비교")
st.caption("바닐라 Faster R-CNN · Light R-CNN · YOLO26 — COCO 정확도와 추론 지연 시간")

with st.sidebar:
    st.header("결과 불러오기")
    output_dir = Path(st.text_input("결과 폴더", value=os.environ.get("LIGHT_RCNN_RESULTS_DIR", "outputs")))
    uploaded = st.file_uploader("JSON 결과 파일 추가", type="json", accept_multiple_files=True)
    st.caption("평가/벤치마크 스크립트가 outputs 폴더에 저장한 JSON을 자동으로 읽습니다.")

payloads = []
if output_dir.exists():
    for path in sorted(output_dir.rglob("*.json"), key=lambda item: item.stat().st_mtime):
        try:
            payloads.append((str(path), path.read_text(encoding="utf-8")))
        except OSError as exc:
            st.warning(f"파일을 읽을 수 없습니다: {path} ({exc})")
if uploaded:
    payloads.extend((file.name, file.getvalue().decode("utf-8")) for file in uploaded)

table, errors = parse_reports(payloads)
if errors:
    with st.expander("읽지 못한 결과 파일"):
        for error in errors:
            st.write(error)

has_accuracy = table["AP@[.50:.95]"].notna().any() if "AP@[.50:.95]" in table else False
has_latency = table["p50 (ms)"].notna().any() if "p50 (ms)" in table else False
if not payloads or not (has_accuracy or has_latency):
    st.info("아직 비교 결과가 없습니다. 아래 순서로 평가/벤치마크를 실행하면 그래프가 채워집니다.")
    st.code(
        "python scripts/evaluate_coco.py --model faster_rcnn\n"
        "python scripts/evaluate_coco.py --model light_rcnn --weights outputs/coco/last.pth\n"
        "python scripts/evaluate_yolo26.py\n"
        "python scripts/benchmark.py --model faster_rcnn --source data/coco/images/val2017\n"
        "python scripts/benchmark.py --model light_rcnn --weights outputs/coco/last.pth --source data/coco/images/val2017\n"
        "python scripts/benchmark_yolo26.py --source data/coco/images/val2017 --device 0"
    )

accuracy = table.copy()
if "AP@[.50:.95]" in accuracy:
    accuracy["AP@[.50:.95] (%)"] = percent_ap(accuracy["AP@[.50:.95]"])
if "AP50" in accuracy:
    accuracy["AP50 (%)"] = percent_ap(accuracy["AP50"])
if "AP75" in accuracy:
    accuracy["AP75 (%)"] = percent_ap(accuracy["AP75"])

numeric_ap = pd.to_numeric(table.get("AP@[.50:.95]", pd.Series(dtype=float)), errors="coerce")
numeric_latency = pd.to_numeric(table.get("p50 (ms)", pd.Series(dtype=float)), errors="coerce")
k1, k2, k3 = st.columns(3)
if numeric_ap.notna().any():
    ap_values = percent_ap(numeric_ap)
    best_index = ap_values.idxmax()
    k1.metric("최고 COCO AP", f"{ap_values.loc[best_index]:.2f}%", str(table.loc[best_index, "Model"]))
else:
    k1.metric("최고 COCO AP", "데이터 없음")
if numeric_latency.notna().any():
    valid = numeric_latency.dropna()
    best_index = valid.idxmin()
    k2.metric("최저 p50 지연", f"{valid.loc[best_index]:.2f} ms", str(table.loc[best_index, "Model"]))
else:
    k2.metric("최저 p50 지연", "데이터 없음")
k3.metric("불러온 결과 파일", str(len(payloads)))

left, right = st.columns([1.2, 1])
with left:
    st.subheader("정확도와 속도 트레이드오프")
    scatter = table.dropna(subset=["p50 (ms)", "AP@[.50:.95]"]).copy() if has_accuracy and has_latency else pd.DataFrame()
    if not scatter.empty:
        scatter["AP (%)"] = percent_ap(scatter["AP@[.50:.95]"])
        hover_columns = [column for column in ("AP50", "AP75", "p95 (ms)", "Precision", "Device")
                         if column in scatter]
        fig = px.scatter(scatter, x="p50 (ms)", y="AP (%)", color="Model", text="Model",
                         hover_data=hover_columns,
                         title="낮은 지연 시간 · 높은 COCO AP가 좋은 결과")
        fig.update_traces(textposition="top center", marker_size=14)
        fig.update_layout(height=430, xaxis_title="p50 지연 시간 (ms) ↓", yaxis_title="COCO AP@[.50:.95] (%) ↑")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("같은 모델의 정확도 JSON과 지연 시간 JSON을 함께 불러오면 산점도를 표시합니다.")

with right:
    st.subheader("지연 시간")
    latency_columns = [column for column in ("p50 (ms)", "p95 (ms)") if column in table]
    latency = (table.melt(id_vars="Model", value_vars=latency_columns,
                          var_name="Percentile", value_name="Latency (ms)").dropna()
               if latency_columns else pd.DataFrame())
    if not latency.empty:
        fig = px.bar(latency, x="Model", y="Latency (ms)", color="Percentile", barmode="group")
        fig.update_layout(height=430, yaxis_title="밀리초 ↓")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("지연 시간 JSON이 아직 없습니다.")

st.subheader("COCO 정확도")
ap_columns = [column for column in ("AP@[.50:.95] (%)", "AP50 (%)", "AP75 (%)") if column in accuracy]
ap_table = accuracy[["Model", *ap_columns]].dropna(how="all", subset=ap_columns) if ap_columns else pd.DataFrame()
if not ap_table.empty:
    plot_data = ap_table.melt(id_vars="Model", var_name="Metric", value_name="AP (%)").dropna()
    fig = px.bar(plot_data, x="Model", y="AP (%)", color="Metric", barmode="group")
    fig.update_layout(height=360, yaxis_title="COCO AP (%) ↑")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("COCO AP 평가 JSON이 아직 없습니다.")

st.subheader("비교 데이터")
display_columns = [column for column in (
    "Model", "AP@[.50:.95]", "AP50", "AP75", "p50 (ms)", "p95 (ms)",
    "Parameters (M)", "FLOPs (G)", "Precision", "Device", "Image size", "Timed scope", "Sources"
) if column in table]
st.dataframe(table[display_columns], hide_index=True, use_container_width=True)
