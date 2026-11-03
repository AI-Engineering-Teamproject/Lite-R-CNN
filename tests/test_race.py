import ast
import csv
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from PIL import Image

from light_rcnn import race
from light_rcnn.race import (
    BEFORE_PRESET,
    COCO_KO,
    IMPROVEMENT_PRESET,
    PRESET_NOTES,
    Detection,
    Racer,
    RaceResult,
    compare_results,
    draw_detections,
    improvement_presets,
    load_rcnn_racer,
    prepare_image,
    race_lanes,
    run_race,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]
PAGE = str(ROOT / "dashboard" / "pages" / "1_Detection_Race.py")
PHOTO = ROOT / "147753808597_20161028.jpeg"
WEIGHTS = {version: ROOT / f"outputs/coco_4090_{version}/best.pth" for version in ("v2", "v3")}
SPEED_LABEL = "속도 향상 (개선 전 ÷ 개선 후)"
GPU_RECORD_LABEL = "팀의 RTX 4090 공식 측정 기록 (COCO val2017 5,000장)"
T4 = {"device": "cuda", "device_type": "cuda", "device_name": "Tesla T4"}
RTX_4090 = {"device": "cuda:0", "device_type": "cuda", "device_name": "NVIDIA GeForce RTX 4090"}


def race_rules() -> tuple[float, int]:
    """The page's score floor and detection cap, from configs/coco.yaml."""
    from light_rcnn.bench import comparison_setting

    config = yaml.safe_load((ROOT / "configs/coco.yaml").read_text(encoding="utf-8"))
    return (float(comparison_setting(config, "latency_score_threshold")),
            int(comparison_setting(config, "max_detections")))


def fake_racer(key, cost_ms, detections=()):
    def predict(image):
        fake_racer.clock[0] += cost_ms / 1000
        return list(detections)

    return Racer(key, key.upper(), "#2a78d6", predict)


fake_racer.clock = [0.0]


def test_run_race_ranks_by_median_and_interleaves_rounds():
    fake_racer.clock[0] = 0.0
    slow, fast = fake_racer("slow", 30.0), fake_racer("fast", 10.0)
    results = run_race([slow, fast], np.zeros((4, 4, 3), np.uint8), repeats=3, clock=lambda: fake_racer.clock[0])
    assert [r.racer.key for r in results] == ["fast", "slow"]
    assert results[0].times_ms == pytest.approx([10.0] * 3)
    assert results[1].median_ms == pytest.approx(30.0)
    with pytest.raises(ValueError):
        run_race([fast], np.zeros((4, 4, 3), np.uint8), repeats=0)
    assert run_race([], np.zeros((4, 4, 3), np.uint8)) == []


def test_run_race_rotates_which_racer_starts_each_round():
    order = []
    racers = [Racer(key, key, "#2a78d6", lambda image, key=key: order.append(key) or []) for key in "abc"]
    run_race(racers, np.zeros((4, 4, 3), np.uint8), repeats=4)
    assert ["".join(order[i:i + 3]) for i in range(0, 12, 3)] == ["abc", "bca", "cab", "abc"]


def test_prepare_image_downscales_and_converts_to_rgb():
    image = prepare_image(Image.new("L", (2560, 1280), 128), max_side=1280)
    assert image.shape == (640, 1280, 3) and image.dtype == np.uint8
    assert prepare_image(Image.new("RGB", (100, 50))).shape == (50, 100, 3)


def test_draw_and_summarize_respect_the_score_floor():
    detections = [Detection((1, 1, 20, 20), 0.9, "person"), Detection((5, 5, 30, 30), 0.8, "person"),
                  Detection((2, 2, 10, 10), 0.2, "dog")]
    canvas = draw_detections(np.zeros((64, 64, 3), np.uint8), detections, "#eda100", min_score=0.5)
    assert canvas.size == (64, 64) and np.asarray(canvas).any()
    assert summarize(detections, min_score=0.5) == (2, "사람 2")
    assert summarize([], min_score=0.5) == (0, "")


def test_korean_names_cover_all_80_coco_classes():
    assert len(COCO_KO) == 80
    assert Detection((0, 0, 1, 1), 1.0, "unknown thing").name_ko == "unknown thing"


def test_race_lanes_keep_colors_distinct_and_attached_to_the_model():
    models, improvement = race_lanes("models"), race_lanes("improvement")
    assert [lane.key for lane in models] == ["light_rcnn_v2", "light_rcnn_v3", "yolo26"]
    assert [lane.preset for lane in models] == ["deploy", "deploy", None]
    assert [lane.key for lane in improvement] == ["before", "after", "yolo26"]
    assert [lane.model for lane in improvement] == ["light_rcnn_v2", "light_rcnn_v2", "yolo26"]
    assert [lane.preset for lane in improvement] == [BEFORE_PRESET, IMPROVEMENT_PRESET, None]
    assert race_lanes("improvement", "accurate")[1].preset == "accurate"
    for lanes in (models, improvement):
        for theme in ("light", "dark"):
            colors = [lane.color(theme) for lane in lanes]
            assert len(set(colors)) == len(colors)
    assert models[2].colors == improvement[2].colors  # YOLO26n keeps its color in both races
    assert improvement[1].colors == models[0].colors  # and the improved v2 keeps v2's
    assert models[0].color("dark") != models[0].color("light")
    with pytest.raises(ValueError):
        race_lanes("relay")


def test_improvement_presets_offer_only_presets_that_run_on_the_device():
    from light_rcnn.inference_opts import PRESETS

    cpu, cuda = improvement_presets("cpu"), improvement_presets("cuda:0")
    assert BEFORE_PRESET not in cpu + cuda
    assert IMPROVEMENT_PRESET in cpu and IMPROVEMENT_PRESET in cuda
    assert not any(PRESETS[name].fp16 or PRESETS[name].cuda_graph for name in cpu)
    assert set(cpu) < set(cuda) and "gpu_graph" in cuda
    assert set(PRESET_NOTES) <= set(PRESETS) and IMPROVEMENT_PRESET in PRESET_NOTES


def test_compare_results_reports_speedup_and_detection_changes():
    person, dog, cup = (Detection((0, 0, 9, 9), 0.9, name) for name in ("person", "dog", "cup"))
    unsure = Detection((0, 0, 9, 9), 0.3, "dog")
    before = RaceResult(Racer("before", "B", "#eb6834", list), [40.0, 30.0, 50.0], [person, dog, unsure])
    after = RaceResult(Racer("after", "A", "#2a78d6", list), [15.0, 20.0, 25.0], [person, person, cup])
    change = compare_results(before, after, min_score=0.5)
    assert change.speedup == pytest.approx(2.0) and change.time_change_pct == pytest.approx(-50.0)
    assert (change.before_count, change.after_count, change.count_change) == (2, 3, 1)
    assert change.class_changes == {"사람": 1, "개": -1, "컵": 1}
    assert change.class_change_text() == "개 -1, 사람 +1, 컵 +1"
    same = compare_results(after, after, min_score=0.5)
    assert same.speedup == pytest.approx(1.0) and same.count_change == 0
    assert same.class_change_text() == ""
    slower = compare_results(after, before, min_score=0.5)
    assert slower.speedup == pytest.approx(0.5) and slower.time_change_pct == pytest.approx(100.0)


@pytest.mark.skipif(not WEIGHTS["v2"].is_file(), reason="needs outputs/coco_4090_v2/best.pth")
def test_load_rcnn_racer_applies_the_preset_under_the_race_rules():
    from light_rcnn.inference_opts import resolve_options

    config = yaml.safe_load((ROOT / "configs/coco.yaml").read_text(encoding="utf-8"))
    rules = {"score_threshold": 0.3, "max_detections": 50}
    # A preset may not loosen or tighten the race's own score floor and cap.
    custom = resolve_options("fast", box_score_thresh=0.9, detections_per_img=3)
    after = load_rcnn_racer("after", "A", "#2a78d6", WEIGHTS["v2"], config, **rules, options=custom)
    before = load_rcnn_racer("before", "B", "#eb6834", WEIGHTS["v2"], config, **rules,
                             options=BEFORE_PRESET)
    default = load_rcnn_racer("v2", "V2", "#2a78d6", WEIGHTS["v2"], config, **rules)
    assert [r.inference["preset"] for r in (after, before, default)] == ["fast", "baseline", "deploy"]
    for racer in (after, before, default):
        assert racer.inference["applied"]["box_score_thresh"] == pytest.approx(0.3)
        assert racer.inference["applied"]["detections_per_img"] == 50
        json.dumps(racer.inference)
    assert after.inference["applied"]["rpn_pre_nms_top_n_test"] == 500
    assert after.inference["applied"]["rpn_score_thresh"] == pytest.approx(0.05)
    assert after.inference["applied"]["conv_bn_fused"] and default.inference["applied"]["conv_bn_fused"]
    assert not before.inference["applied"]["conv_bn_fused"]
    assert not any("note" in racer.inference for racer in (after, before, default))  # no Soft-NMS

    image = prepare_image(Image.open(PHOTO)) if PHOTO.is_file() else np.zeros((480, 640, 3), np.uint8)
    found = {key: racer.predict(image) for key, racer in (("after", after), ("before", before))}
    for detections in found.values():
        assert len(detections) <= 50 and all(d.score >= 0.3 for d in detections)
    if PHOTO.is_file():
        counts = [summarize(found[key])[0] for key in ("before", "after")]
        assert counts[0] > 0 and abs(counts[0] - counts[1]) <= 2


@pytest.mark.skipif(not WEIGHTS["v2"].is_file(), reason="needs outputs/coco_4090_v2/best.pth")
def test_soft_nms_lane_never_returns_boxes_below_the_race_floor():
    import torch

    from light_rcnn.bench import load_comparison_model, set_detection_limits
    from light_rcnn.inference_opts import (
        CUSTOM_LABEL,
        PRESET_LABELS,
        apply_inference_options,
        resolve_options,
    )

    config = yaml.safe_load((ROOT / "configs/coco.yaml").read_text(encoding="utf-8"))
    rules = {"score_threshold": 0.3, "max_detections": 50}
    racer = load_rcnn_racer("after", "A", "#2a78d6", WEIGHTS["v2"], config, **rules, options="accurate")
    record = racer.inference
    # Still read as the chosen preset, with the race's change spelled out.
    assert (record["preset"], record["label"]) == ("accurate", PRESET_LABELS["accurate"])
    assert record["applied"]["soft_nms_min_score"] == pytest.approx(0.3)
    assert "감쇠 하한 0.3" in record["summary"]
    assert "설정 기본값 0.001 대신 점수 하한과 같은 값(0.3)" in record["note"]
    json.dumps(record)
    image = prepare_image(Image.open(PHOTO)) if PHOTO.is_file() else np.zeros((480, 640, 3), np.uint8)
    assert all(detection.score > 0.3 for detection in racer.predict(image))
    if PHOTO.is_file():
        # The preset's own drop floor (0.001) keeps decayed boxes under the race floor.
        model = load_comparison_model("light_rcnn", config, str(WEIGHTS["v2"]))
        set_detection_limits(model, **rules)
        apply_inference_options(model, "accurate", torch.device("cpu"))
        with torch.inference_mode():
            scores = model([torch.from_numpy(image).permute(2, 0, 1).float().div_(255)])[0]["scores"]
        assert float(scores.min()) < 0.3
    # Options that match no preset keep their custom label; the note still says what changed.
    custom = load_rcnn_racer("after", "A", "#2a78d6", WEIGHTS["v2"], config, **rules,
                             options=resolve_options("accurate", soft_nms_sigma=0.3))
    assert (custom.inference["preset"], custom.inference["label"]) == (None, CUSTOM_LABEL)
    assert custom.inference["applied"]["soft_nms_min_score"] == pytest.approx(0.3)
    assert "note" in custom.inference


# --------------------------------------------------------------------------- the Streamlit page


@pytest.fixture
def fresh_page():
    """Empty Streamlit resource caches, so no racer leaks between page tests."""
    st = pytest.importorskip("streamlit")
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


@pytest.fixture
def fake_models(monkeypatch, fresh_page):
    """Instant stand-ins for the model loaders: the baseline takes 30 ms, every preset 10 ms and
    finds one more object (a dog). ``records`` adds entries to a preset's apply record."""
    models = SimpleNamespace(calls=[], broken=set(), records={})

    def load_rcnn(key, label, color, weights, config, device="cpu", score_threshold=0.25,
                  max_detections=100, note="", options="deploy"):
        models.calls.append(("rcnn", key, options, score_threshold, max_detections))
        if options in models.broken:
            raise RuntimeError("broken checkpoint")
        found = [Detection((5.0, 5.0, 60.0, 90.0), 0.9, "person")]
        if options != BEFORE_PRESET:
            found.append(Detection((70.0, 20.0, 120.0, 60.0), 0.8, "dog"))

        def predict(image):
            time.sleep(0.03 if options == BEFORE_PRESET else 0.01)
            return list(found)

        record = {"preset": options, "summary": f"{options} 요약", **models.records.get(options, {})}
        return Racer(key, label, color, predict, inference=record)

    def load_yolo(key, label, color, weights, device="cpu", score_threshold=0.25, max_detections=100,
                  image_size=640, note=""):
        models.calls.append(("yolo", key, str(weights), score_threshold, max_detections))
        return Racer(key, label, color, lambda image: [Detection((1.0, 1.0, 50.0, 50.0), 0.7, "person")])

    monkeypatch.setattr(race, "load_rcnn_racer", load_rcnn)
    monkeypatch.setattr(race, "load_yolo_racer", load_yolo)
    return models


def open_page(timeout: float = 60):
    from streamlit.testing.v1 import AppTest

    page = AppTest.from_file(PAGE, default_timeout=timeout)
    page.run()
    assert not page.exception, [e.value for e in page.exception]
    return page


def start_race(page):
    page.button[0].click().run()
    assert not page.exception, [e.value for e in page.exception]
    return page.dataframe[0].value


def missing_spec(monkeypatch, *names):
    real = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        return None if name in names else real(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)


def page_function(name: str, **names):
    """One top-level function of the page, defined on its own (running the page runs Streamlit)."""
    source = Path(PAGE).read_text(encoding="utf-8")
    node = next(node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"csv": csv, "json": json, "Path": Path, **names}
    code = compile(ast.Module(body=[node], type_ignores=[]), PAGE, "exec")
    exec(code, namespace)  # noqa: S102 - the page's own code
    return namespace[name]


def write_evaluation_run(folder: Path, timestamp: str, device: str = "cuda", images: int = 5000,
                         environment: dict | None = None, ap: str = "0.28") -> Path:
    """summary.csv and evaluation.json as scripts/evaluate_three_models.py writes them."""
    folder.mkdir(parents=True)
    metadata = {"images_evaluated": images, "device": device, "evaluation_timestamp": timestamp}
    if environment is not None:
        metadata["environment"] = environment
    (folder / "evaluation.json").write_text(json.dumps({"metadata": metadata, "results": []}),
                                            encoding="utf-8")
    with (folder / "summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, ["model", "AP", "latency_p50_ms", "latency_p95_ms", "device"])
        writer.writeheader()
        writer.writerow({"model": "Faster R-CNN", "AP": "0.32", "latency_p50_ms": "7.114",
                         "latency_p95_ms": "8.304", "device": device})
        writer.writerow({"model": "Lite R-CNN v2", "AP": ap, "latency_p50_ms": "5.342",
                         "latency_p95_ms": "14.284", "device": device})
    return folder


def test_race_page_switches_between_the_model_race_and_before_after(fake_models):
    import torch

    page = open_page()
    assert page.segmented_control[0].value == "models" and not page.selectbox
    assert page.segmented_control[0].proto.required  # clicking the chosen race cannot clear it
    page.sidebar.text_input[0].set_value("https://example.org/race").run()  # QR code for the audience
    assert not page.exception
    table = start_race(page)
    assert table["모델"][0] == "YOLO26n"  # the instant stand-in finishes first
    assert set(table["모델"]) == {"YOLO26n", "Lite R-CNN v2", "Lite R-CNN v3"}
    assert set(table["추론 설정"]) == {"배포 기본 (Conv+BN 융합·메모리 형식)", "Ultralytics 기본 (Conv+BN 융합)"}
    assert not page.metric
    if (ROOT / "outputs/evaluation_runs/20261101_053245/evaluation.json").is_file():
        assert [expander.label for expander in page.expander] == [GPU_RECORD_LABEL]

    page.segmented_control[0].set_value("improvement").run()
    assert page.selectbox[0].value == IMPROVEMENT_PRESET
    if not torch.cuda.is_available():
        assert "GPU CUDA 그래프 (CUDA 전용)" not in page.selectbox[0].options
    table = start_race(page)
    assert set(table["모델"]) == {"Lite R-CNN v2 개선 전", "Lite R-CNN v2 개선 후", "YOLO26n (참고)"}
    assert set(table["추론 설정"]) == {"기본 설정", "빠른 설정", "Ultralytics 기본 (Conv+BN 융합)"}
    metrics = {metric.label: metric for metric in page.metric}
    assert float(metrics[SPEED_LABEL].value.rstrip("×")) > 1.2  # 30 ms vs 10 ms
    assert metrics[SPEED_LABEL].delta.startswith("-") and metrics[SPEED_LABEL].delta.endswith("% 시간")
    assert metrics["개선 후 처리 시간 (중앙값)"].delta.startswith("-")  # down arrow = faster
    assert metrics["찾은 물체 (신뢰도 0.5 이상)"].value == "1 → 2개"
    assert metrics["찾은 물체 (신뢰도 0.5 이상)"].delta == "+1개"
    caption = next(c.value for c in page.caption if c.value.startswith("개선 후 설정"))
    assert "**빠른 설정**" in caption and PRESET_NOTES["fast"] in caption and "fast 요약" in caption
    score, cap = race_rules()
    assert f"점수 임계값 {score:g}·최대 검출 {cap}개" in caption and "개 +1" in caption
    assert any("빠른 설정" in md.value and "<div" in md.value for md in page.markdown)  # lane bar

    page.selectbox[0].set_value("accurate").run()
    table = start_race(page)
    assert "정확도 우선" in set(table["추론 설정"])
    assert any(c.value.startswith("개선 후 설정 **정확도 우선**") for c in page.caption)

    rcnn = [call for call in fake_models.calls if call[0] == "rcnn"]
    assert [(key, preset) for _, key, preset, *_ in rcnn] == [
        ("light_rcnn_v2", "deploy"), ("light_rcnn_v3", "deploy"), ("light_rcnn_v2", "baseline"),
        ("light_rcnn_v2", "fast"), ("light_rcnn_v2", "accurate")]  # each racer loads once
    assert sum(call[0] == "yolo" for call in fake_models.calls) == 1  # shared by both races
    assert {call[3:] for call in fake_models.calls} == {race_rules()}  # one score floor and cap


def test_race_page_captions_carry_each_lane_record_note(fake_models):
    """A CUDA graph the self-check turned off (the summary still names it) and the Soft-NMS floor
    note are told next to the lane's settings; other lanes' graph notes go to the closing caption."""
    disabled = {"status": "disabled (graph replay differs from eager)", "replays": 0}
    soft_note = "레이스 규칙에 맞춰 Soft-NMS 감쇠 하한을 설정 기본값 0.001 대신 점수 하한과 같은 값(0.25)으로 씁니다."
    fake_models.records.update(fast={"cuda_graph_status": disabled},
                               accurate={"note": soft_note},
                               deploy={"cuda_graph_status": {"status": "pending (captured on the first CUDA call)"}})
    page = open_page()
    start_race(page)
    closing = next(c.value for c in page.caption if c.value.startswith("측정 장치:"))
    assert closing.endswith(" Lite R-CNN v2: CUDA 그래프 상태: pending (captured on the first CUDA call)."
                            " Lite R-CNN v3: CUDA 그래프 상태: pending (captured on the first CUDA call).")

    page.segmented_control[0].set_value("improvement").run()
    start_race(page)
    caption = next(c.value for c in page.caption if c.value.startswith("개선 후 설정"))
    assert ("적용된 설정: fast 요약. **CUDA 그래프 사용 안 됨 (자체 검사에서 재생 결과가 eager와 달라 "
            "백본을 eager로 실행)**. 개선 전은") in caption
    closing = next(c.value for c in page.caption if c.value.startswith("측정 장치:"))
    assert "CUDA 그래프" not in closing  # told once, beside the after lane's settings

    page.selectbox[0].set_value("accurate").run()
    start_race(page)
    caption = next(c.value for c in page.caption if c.value.startswith("개선 후 설정"))
    assert f"적용된 설정: accurate 요약. {soft_note} 개선 전은" in caption
    assert "CUDA 그래프" not in caption


def test_race_page_keeps_racing_when_a_lane_cannot_load(fake_models):
    fake_models.broken.add("accurate")
    page = open_page()
    page.segmented_control[0].set_value("improvement").run()
    page.selectbox[0].set_value("accurate").run()
    assert "Lite R-CNN v2 개선 후 (불러오기 실패: broken checkpoint)" in page.warning[0].value
    table = start_race(page)
    assert set(table["모델"]) == {"Lite R-CNN v2 개선 전", "YOLO26n (참고)"}
    assert not page.metric


def test_race_page_races_without_ultralytics(monkeypatch, fake_models):
    missing_spec(monkeypatch, "ultralytics")
    page = open_page()
    assert "YOLO26n (Ultralytics 미설치)" in page.warning[0].value
    table = start_race(page)
    assert set(table["모델"]) == {"Lite R-CNN v2", "Lite R-CNN v3"}
    assert not any(call[0] == "yolo" for call in fake_models.calls)


def test_race_page_explains_a_missing_pytorch(monkeypatch, fresh_page):
    missing_spec(monkeypatch, "torch")
    page = open_page()
    assert "PyTorch" in page.error[0].value and not page.tabs and not page.button


def test_official_gpu_record_is_the_newest_full_val_4090_run(tmp_path):
    official_gpu_record = page_function("official_gpu_record", EVALUATION_RUNS=tmp_path)
    assert official_gpu_record() == ("", [])
    # The 2026-11-01 runs (no environment record): full val2017 4090, 1000-image 4090, CPU.
    write_evaluation_run(tmp_path / "20261101_053245", "2026-11-01T05:32:45")
    write_evaluation_run(tmp_path / "20261101_052008", "2026-11-01T05:20:08", images=1000)
    write_evaluation_run(tmp_path / "20261101_054559", "2026-11-01T05:45:59", device="cpu",
                         images=1000)
    write_evaluation_run(tmp_path / "rtx4090_first_try", "2026-10-30T12:00:00")  # older, named by hand
    # Notebook step C on Colab: newer, and its folder name sorts ahead of the 4090 runs.
    write_evaluation_run(tmp_path / "colab_20261105_101010", "2026-11-05T10:10:10", environment=T4)
    # Step C with an older evaluate_three_models.py: no environment record, only the folder name.
    write_evaluation_run(tmp_path / "colab_20261105_090000", "2026-11-05T09:00:00")
    write_evaluation_run(tmp_path / "20261105_120000_colab_t4", "2026-11-05T12:00:00")
    write_evaluation_run(tmp_path / "20261106_080000", "2026-11-06T08:00:00", images=50,
                         environment=RTX_4090)  # QUICK_TEST
    (tmp_path / "unfinished").mkdir()
    (tmp_path / "unfinished" / "summary.csv").write_text("model,AP,device\n", encoding="utf-8")
    write_evaluation_run(tmp_path / "hand_edited", "2026-11-07T00:00:00").joinpath(
        "evaluation.json").write_text("{", encoding="utf-8")

    gpu, rows = official_gpu_record()
    assert gpu == "RTX 4090"
    assert rows == [
        {"모델": "Faster R-CNN", "AP (%)": 32.0, "p50 (ms)": 7.11, "p95 (ms)": 8.3,
         "기록": "20261101_053245"},
        {"모델": "Lite R-CNN v2", "AP (%)": 28.0, "p50 (ms)": 5.34, "p95 (ms)": 14.28,
         "기록": "20261101_053245"}]

    # A later 4090 run wins whatever its folder is called; one with a broken table is passed over.
    write_evaluation_run(tmp_path / "4090_rerun", "2026-11-20T08:00:00", device="cuda:0",
                         environment=RTX_4090)
    write_evaluation_run(tmp_path / "4090_rerun_blank_ap", "2026-11-21T08:00:00", ap="",
                         environment=RTX_4090)
    gpu, rows = official_gpu_record()
    assert gpu == "RTX 4090" and {row["기록"] for row in rows} == {"4090_rerun"}


@pytest.mark.skipif(not all(path.is_file() for path in WEIGHTS.values()),
                    reason="needs the trained v2/v3 checkpoints in outputs/")
def test_race_page_with_the_trained_models(monkeypatch, fresh_page):
    """Both races end to end on the real checkpoints; without yolo26n.pt in the project root the
    YOLO lane is an untrained yolo26n.yaml graph (Ultralytics builds it without a download)."""
    if not (ROOT / "yolo26n.pt").is_file():
        monkeypatch.setenv("LIGHT_RCNN_YOLO_WEIGHTS", "yolo26n.yaml")
    page = open_page(timeout=600)
    page.sidebar.slider[0].set_value(1).run()  # one timed round per racer keeps the test short
    table = start_race(page)
    assert not page.warning and len(table) == 3

    page.segmented_control[0].set_value("improvement").run()
    table = start_race(page)
    assert not page.warning and len(table) == 3
    metrics = {metric.label: metric for metric in page.metric}
    assert metrics[SPEED_LABEL].value.endswith("×")
    caption = next(c.value for c in page.caption if c.value.startswith("개선 후 설정"))
    assert "RPN 점수 하한 0.05" in caption  # apply_inference_options' summary of the fast preset
