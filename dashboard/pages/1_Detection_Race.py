"""탐지 레이스: 같은 사진을 여러 탐지 모델이 같은 조건으로 처리하는 시간을 겨루는 라이브 데모.

레이스 구성은 두 가지입니다. 'v2 · v3 · YOLO26'은 Lite R-CNN v2·v3와 YOLO26n의 대결이고, '개선 전 vs
개선 후'는 같은 Lite R-CNN v2 가중치를 기본 설정(개선 전)과 추론 설정(개선 후, 기본은 빠른 설정)으로
나란히 달리게 하고 YOLO26n을 참고로 더합니다. YOLO 가중치는 LIGHT_RCNN_YOLO_WEIGHTS, 프로젝트 루트의
yolo26n.pt 순으로 찾고, 둘 다 없으면 Ultralytics가 이름으로 내려받습니다.
"""

from __future__ import annotations

import contextlib
import csv
import dataclasses
import html
import importlib.util
import io
import json
import os
import platform
import sys
import threading
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(page_title="탐지 레이스", page_icon=":material/sprint:", layout="wide")

RCNN_WEIGHTS = {
    "light_rcnn_v2": ROOT / "outputs" / "coco_4090_v2" / "best.pth",
    "light_rcnn_v3": ROOT / "outputs" / "coco_4090_v3" / "best.pth",
}
SAMPLE = ROOT / "147753808597_20161028.jpeg"
EVALUATION_RUNS = ROOT / "outputs" / "evaluation_runs"
ANIMATION_SECONDS = 2.5


def theme() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except AttributeError:
        return "light"


def yolo_weights() -> str:
    local = ROOT / "yolo26n.pt"
    return os.environ.get("LIGHT_RCNN_YOLO_WEIGHTS") or (str(local) if local.is_file() else "yolo26n.pt")


if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torchvision") is None:
    st.title("탐지 레이스")
    st.error("이 페이지는 모델을 직접 실행하므로 PyTorch(torch·torchvision)가 필요합니다. "
             "YOLO26n 레인에는 Ultralytics도 필요합니다.")
    st.code("python -m pip install -r dashboard/requirements.txt", language="bash")
    st.stop()
HAS_ULTRALYTICS = importlib.util.find_spec("ultralytics") is not None
HAS_SEGNO = importlib.util.find_spec("segno") is not None

import torch
import yaml

from light_rcnn.bench import comparison_setting
from light_rcnn.inference_opts import PRESET_LABELS
from light_rcnn.race import (
    BEFORE_PRESET,
    IMPROVEMENT_PRESET,
    PRESET_NOTES,
    RACES,
    compare_results,
    draw_detections,
    improvement_presets,
    load_rcnn_racer,
    load_yolo_racer,
    prepare_image,
    race_lanes,
    run_race,
    summarize,
    warm_up,
)
from light_rcnn.reporting import cuda_graph_note

CONFIG = yaml.safe_load((ROOT / "configs" / "coco.yaml").read_text(encoding="utf-8"))
SCORE_FLOOR = float(comparison_setting(CONFIG, "latency_score_threshold"))
MAX_DETECTIONS = int(comparison_setting(CONFIG, "max_detections"))


@st.cache_resource(show_spinner="모델을 불러오고 예열하는 중입니다 (모델·설정마다 처음 한 번만)…")
def load_racer(model: str, preset: str | None, device: str, weights: str, score: float, cap: int):
    """One warmed-up racer per model, preset, device and rules (a handful of entries, so none is
    evicted). Lanes take copies with their own key, label and color, so a racer that both race
    configurations use loads once."""
    if model == "yolo26":
        racer = load_yolo_racer(model, "YOLO26n", "", weights, device, score, cap,
                                int(CONFIG.get("image_size", 640)))
    else:
        racer = load_rcnn_racer(model, model, "", weights, CONFIG, device, score, cap,
                                options=preset)
    warm_up([racer], prepare_image(_sample_image()))
    return racer


@st.cache_resource
def race_lock() -> threading.Lock:
    # One race or model load at a time across all visitors, so no race times another's work.
    return threading.Lock()


@st.cache_resource
def loaded_racers() -> set[tuple]:
    """load_racer arguments already cached; reruns that only need those skip the race lock."""
    return set()


@contextlib.contextmanager
def exclusive(waiting_message: str):
    """Hold the race lock, so no model load or race overlaps another visitor's timing."""
    lock = race_lock()
    waiting = st.empty()
    if not lock.acquire(blocking=False):
        waiting.info(waiting_message)
        lock.acquire()
    try:
        waiting.empty()
        yield
    finally:
        lock.release()


def lane_racers(lanes, device: str):
    """Racers of the lanes that can run, and a note for every lane that cannot."""
    planned, missing = [], []
    for lane in lanes:
        if lane.model == "yolo26":
            weights = yolo_weights()
            if not HAS_ULTRALYTICS:
                missing.append(f"{lane.label} (Ultralytics 미설치)")
                continue
        else:
            weights = str(RCNN_WEIGHTS[lane.model])
            if not Path(weights).is_file():
                missing.append(f"{lane.label} ({Path(weights).relative_to(ROOT).as_posix()} 없음)")
                continue
        planned.append((lane, (lane.model, lane.preset, device, weights, SCORE_FLOOR, MAX_DETECTIONS)))
    loaded = loaded_racers()
    racers = []
    with (contextlib.nullcontext() if all(args in loaded for _, args in planned)
          else exclusive("다른 참가자의 레이스가 끝나면 모델을 준비합니다…")):
        for lane, args in planned:
            try:
                racer = load_racer(*args)
            except Exception as exc:  # noqa: BLE001 - e.g. YOLO weights that cannot be downloaded; race the rest
                missing.append(f"{lane.label} (불러오기 실패: {exc})")
                continue
            loaded.add(args)
            racers.append(dataclasses.replace(racer, key=lane.key, label=lane.label, color=lane.color()))
    return racers, missing


def _sample_image():
    from PIL import Image

    if SAMPLE.is_file():
        return Image.open(SAMPLE)
    return Image.new("RGB", (640, 480), "#7f7f7f")


def qr_png(url: str) -> bytes:
    import segno

    buffer = io.BytesIO()
    segno.make(url, error="m").save(buffer, kind="png", scale=8, border=2, dark="#0b0b0b", light="#ffffff")
    return buffer.getvalue()


def device_name(device: str) -> str:
    if device.startswith("cuda"):
        return torch.cuda.get_device_name(0)
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "CPU"


def official_gpu_record(folder: Path = EVALUATION_RUNS) -> tuple[str, list[dict]]:
    """GPU name and table of the newest full-val2017 RTX 4090 evaluation recorded by the team
    (outputs/evaluation_runs/*/), chosen from each run's evaluation.json, not its folder name."""
    runs = []
    for path in folder.glob("*/summary.csv"):
        try:
            meta = json.loads((path.parent / "evaluation.json").read_text(encoding="utf-8"))["metadata"]
            environment = meta.get("environment") or {}
            gpu = str(environment.get("device_name") or "")
            full_val = int(meta.get("images_evaluated") or 0) >= 5000
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
        # The 2026-11-01 4090 runs predate environment records; every newer run (Colab too) has one.
        legacy = not environment and "colab" not in path.parent.name.lower()
        if ("4090" in gpu or legacy) and full_val and str(meta.get("device", "")).startswith("cuda"):
            runs.append((str(meta.get("evaluation_timestamp") or ""), path, gpu))
    for _, path, gpu in sorted(runs, reverse=True):
        try:
            with path.open(encoding="utf-8-sig") as file:
                rows = [{"모델": row["model"], "AP (%)": round(float(row["AP"]) * 100, 2),
                         "p50 (ms)": round(float(row["latency_p50_ms"]), 2),
                         "p95 (ms)": round(float(row["latency_p95_ms"]), 2), "기록": path.parent.name}
                        for row in csv.DictReader(file) if str(row.get("device") or "").startswith("cuda")]
        except (OSError, KeyError, TypeError, ValueError):
            continue
        if rows:
            return gpu.removeprefix("NVIDIA ").removeprefix("GeForce ") or "RTX 4090", rows
    return "", []


def setting_label(lane) -> str:
    if lane.preset is None:
        return "Ultralytics 기본 (Conv+BN 융합)"
    if lane.preset == BEFORE_PRESET:  # the lane label already says 개선 전
        return "기본 설정"
    return PRESET_LABELS.get(lane.preset, lane.preset)


def lane_detail(lane, race: str) -> str:
    """Setting named next to an improvement-race lane's label."""
    return setting_label(lane) if race == "improvement" and lane.preset is not None else ""


def lane_html(label: str, detail: str, color: str, fraction: float, text: str) -> str:
    width = max(0.0, min(fraction, 1.0)) * 100
    detail = f" <span style='font-size:13px;opacity:0.72'>{html.escape(detail)}</span>" if detail else ""
    return (f"<div style='margin:6px 0 10px'><div style='display:flex;justify-content:space-between;"
            f"font-size:15px'><span><b>{html.escape(label)}</b>{detail}</span>"
            f"<span style='font-variant-numeric:tabular-nums'>{html.escape(text)}</span></div>"
            f"<div style='height:14px;border-radius:7px;background:rgba(127,127,127,0.18);overflow:hidden'>"
            f"<div style='width:{width:.1f}%;height:100%;background:{color};border-radius:7px'></div>"
            f"</div></div>")


device = "cuda:0" if torch.cuda.is_available() else "cpu"

with st.sidebar:
    st.header("청중 참여")
    url = st.text_input("이 페이지 주소", value=getattr(st.context, "url", "") or "",
                        help="Streamlit 클라우드 주소가 자동으로 들어갑니다. 비어 있으면 직접 붙여 넣으세요.")
    if url and HAS_SEGNO:
        st.image(qr_png(url), caption="휴대폰 카메라로 찍으면 바로 참여", width=220)
    elif url:
        st.caption("QR 코드를 그리려면 segno 패키지가 필요합니다.")
    st.header("설정")
    repeats = st.slider("반복 측정 횟수", 1, 7, 3, help="같은 사진을 여러 번 처리해 가운데 값(중앙값)을 씁니다.")
    min_score = st.slider("표시할 최소 신뢰도", 0.3, 0.9, 0.5, 0.05)

st.title("탐지 레이스")
race_column, preset_column = st.columns([3, 2], vertical_alignment="bottom")
with race_column:
    race = st.segmented_control(
        "레이스 구성", list(RACES), default="models", required=True, format_func=RACES.get,
        help="v2 · v3 · YOLO26: 세 모델의 대결. 개선 전 vs 개선 후: 같은 Lite R-CNN v2를 기본 설정과 "
             "추론 설정으로 나란히 실행하고 YOLO26n을 참고로 더합니다.")
after_preset = IMPROVEMENT_PRESET
if race == "improvement":
    choices = improvement_presets(device)
    with preset_column:
        after_preset = st.selectbox(
            "개선 후 설정", choices,
            index=choices.index(IMPROVEMENT_PRESET) if IMPROVEMENT_PRESET in choices else 0,
            format_func=lambda name: PRESET_LABELS.get(name, name),
            help="개선 후 레인에 적용할 추론 설정입니다. 다시 학습하지 않고 같은 가중치에 설정만 바꿉니다."
                 + (" CUDA 전용 설정은 GPU에서만 고를 수 있습니다." if device == "cpu" else ""))

lanes = race_lanes(race, after_preset)
with st.container():  # fixed slot: load spinners and waits never shift the tabs below
    racers, missing = lane_racers(lanes, device)

if race == "improvement":
    st.caption("같은 Lite R-CNN v2 가중치를 개선 전(기본 설정)과 개선 후(추론 설정)로 나란히 달리게 합니다. "
               "다시 학습하지 않고 설정만 바꿨을 때 얼마나 빨라지는지 보세요. YOLO26n은 참고용입니다.")
else:
    st.caption("같은 사진을 Lite R-CNN v2·v3와 YOLO26n이 같은 규칙으로 처리합니다. "
               "누가 먼저 끝내고, 무엇을 찾았을까요?")

with st.container():  # fixed slot: showing the big QR code never shifts the tabs below
    if url and HAS_SEGNO and st.toggle("QR 코드 크게 보기 (발표 화면용)"):
        st.image(qr_png(url), width=420)

if missing:
    st.warning("레이스에서 빠진 모델: " + ", ".join(missing))
if not racers:
    st.error("레이스를 할 모델이 없습니다. 가중치 파일과 패키지 설치를 확인하세요.")
    st.stop()

camera_tab, upload_tab, sample_tab = st.tabs(["카메라로 찍기", "사진 올리기", "예시 사진"])
with camera_tab:
    shot = st.camera_input("사진 찍기", label_visibility="collapsed")
with upload_tab:
    uploaded = st.file_uploader("사진 파일 (JPG, PNG)", type=["jpg", "jpeg", "png", "webp"])
with sample_tab:
    use_sample = st.button("예시 사진으로 레이스", icon=":material/photo:")

source = shot or uploaded
if source is None and not use_sample:
    st.info("카메라로 찍거나 사진을 올리면 레이스가 바로 시작됩니다. 예시 사진으로 먼저 해 볼 수도 있어요.")
    st.stop()

from PIL import Image

image = prepare_image(Image.open(source) if source is not None else _sample_image())
with exclusive("다른 참가자의 레이스가 끝나기를 기다리는 중입니다…"), st.spinner("측정 중…"):
    warm_up(racers, image, rounds=1)  # this photo's input shapes, so no lane times its kernel setup
    results = run_race(racers, image, repeats=repeats)

lane_of = {lane.key: lane for lane in lanes}
colors = {key: lane.color(theme()) for key, lane in lane_of.items()}
ranks = {result.racer.key: rank for rank, result in enumerate(results, start=1)}
by_key = {result.racer.key: result for result in results}
in_lanes = [by_key[lane.key] for lane in lanes if lane.key in by_key]
# The apply-time summary names a CUDA graph; only its status says whether the lane replayed it.
graph_notes = {result.racer.key: note for result in in_lanes
               if (note := cuda_graph_note(result.racer.inference))}

st.subheader("출발!")
slowest = max(result.median_ms for result in results)
bars = {result.racer.key: st.empty() for result in in_lanes}
started = time.perf_counter()
finished = set()
while len(finished) < len(results):
    elapsed = (time.perf_counter() - started) / ANIMATION_SECONDS * slowest  # ms on the measured scale
    for result in in_lanes:
        key = result.racer.key
        fraction = min(elapsed / result.median_ms, 1.0) if result.median_ms > 0 else 1.0
        if fraction >= 1.0:
            finished.add(key)
            text = f"{ranks[key]}위 · {result.median_ms:.1f} ms"
        else:
            text = f"{min(elapsed, result.median_ms):.1f} ms"
        bars[key].markdown(lane_html(result.racer.label, lane_detail(lane_of[key], race), colors[key],
                                     fraction, text), unsafe_allow_html=True)
    time.sleep(0.03)

if race == "improvement" and {"before", "after"} <= by_key.keys():
    change = compare_results(by_key["before"], by_key["after"], min_score)
    st.subheader("개선 효과")
    speed, timing, found = st.columns(3)
    # Deltas start with their sign: Streamlit reads the arrow direction from it.
    speed.metric("속도 향상 (개선 전 ÷ 개선 후)", f"{change.speedup:.2f}×",
                 f"{change.time_change_pct:+.1f}% 시간", delta_color="inverse",
                 help="1×보다 크면 개선 후가 빠릅니다. 같은 사진을 번갈아 잰 중앙값끼리 비교합니다.")
    timing.metric("개선 후 처리 시간 (중앙값)", f"{change.after_ms:.1f} ms",
                  f"{change.after_ms - change.before_ms:+.1f} ms (개선 전 {change.before_ms:.1f} ms)",
                  delta_color="inverse")
    found.metric(f"찾은 물체 (신뢰도 {min_score:g} 이상)", f"{change.before_count} → {change.after_count}개",
                 f"{change.count_change:+d}개" if change.count_change else None, delta_color="off")
    record = by_key["after"].racer.inference or {}
    graph_note = graph_notes.pop("after", None)
    st.caption(f"개선 후 설정 **{PRESET_LABELS.get(after_preset, after_preset)}**: "
               f"{PRESET_NOTES.get(after_preset, '')} 적용된 설정: {record.get('summary', '-')}. "
               + (f"**{graph_note}**. " if graph_note else "")
               + (f"{record['note']} " if record.get("note") else "")
               + f"개선 전은 같은 가중치를 모델 기본 설정 그대로 실행합니다. 다시 학습하지 않았고, 두 레인 모두 "
               f"점수 임계값 {SCORE_FLOOR:g}·최대 검출 {MAX_DETECTIONS}개로 같습니다. "
               f"클래스별 물체 수 변화: {change.class_change_text() or '없음'}.")

columns = st.columns(len(in_lanes))
for column, result in zip(columns, in_lanes):
    key = result.racer.key
    count, _ = summarize(result.detections, min_score)
    with column:
        st.markdown(f"**{ranks[key]}위 · {result.racer.label}**")
        st.image(draw_detections(image, result.detections, colors[key], min_score),
                 caption=f"{result.median_ms:.1f} ms · 물체 {count}개", width="stretch")

rows = []
for rank, result in enumerate(results, start=1):
    count, names = summarize(result.detections, min_score)
    rows.append({"순위": rank, "모델": result.racer.label, "시간 (ms, 중앙값)": round(result.median_ms, 1),
                 "찾은 물체 수": count, "찾은 물체": names or "-",
                 "추론 설정": setting_label(lane_of[result.racer.key]),
                 "측정값 (ms)": ", ".join(f"{t:.1f}" for t in result.times_ms)})
st.dataframe(rows, hide_index=True, width="stretch")

rule = ("R-CNN 레인은 Ultralytics가 YOLO에 적용하는 것과 같은 Conv+BN 융합·메모리 형식 규칙(배포 기본)으로 "
        "실행합니다. " if race == "models" else "")
st.caption(
    f"측정 장치: {device_name(device)} · torch 스레드 {torch.get_num_threads()}개 · 예열 1회 뒤 반복 {repeats}회 중앙값 · "
    f"점수 임계값 {SCORE_FLOOR:g}(측정)·{min_score:g}(표시) · 최대 검출 {MAX_DETECTIONS}개 · "
    f"범위: 디코드된 사진 → 최종 검출. {rule}막대 애니메이션은 실제 측정 시간의 비율대로 늘려서 보여 줍니다. "
    "여러 명이 동시에 쓰면 차례로 측정합니다."
    + "".join(f" {by_key[key].racer.label}: {note}." for key, note in graph_notes.items()))

gpu, gpu_rows = official_gpu_record()
if gpu_rows:
    with st.expander(f"팀의 {gpu} 공식 측정 기록 (COCO val2017 5,000장)"):
        st.dataframe(gpu_rows, hide_index=True, width="stretch")
        st.caption("이 페이지의 시간은 지금 이 서버에서 잰 값이라 GPU 기록과 다릅니다.")
