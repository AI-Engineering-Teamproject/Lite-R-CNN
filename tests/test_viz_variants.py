import json
from pathlib import Path

import pytest

from light_rcnn import reporting, viz
from light_rcnn.reporting import (
    BASELINE_VARIANT,
    BASELINE_VARIANT_LABEL,
    build_report,
    latency_summary,
    normalize,
    select_reports,
    write_report,
)

TARGETS = {"latency_ms": 15, "latency_percentile": "p50", "ap_metric": "AP_50_95", "ap": 0.6}
CPU = {"device": "cpu", "device_type": "cpu", "device_name": "Xeon", "torch_num_threads": 4}
T4 = {"device": "cuda:0", "device_type": "cuda", "device_name": "Tesla T4"}
SUMMARY = "Conv+BN 융합 · RPN NMS 전 후보 레벨당 500개 · RPN 점수 하한 0.05"
FAST = {"variant": "fast", "variant_label": "빠른 설정",  # as apply_inference_options records it
        "inference_options": {"preset": "fast", "label": "빠른 설정", "summary": SUMMARY,
                              "device": "cpu", "skipped": {},
                              "options": {"fuse_conv_bn": True, "rpn_pre_nms_top_n": 500,
                                          "rpn_score_thresh": 0.05}}}
# (variant, model) -> (AP@[.50:.95], AP75, p50 ms)
NUMBERS = {
    (BASELINE_VARIANT, "faster_rcnn"): (0.370, 0.400, 1450.0),
    ("fast", "faster_rcnn"): (0.369, 0.398, 1310.0),
    (BASELINE_VARIANT, "light_rcnn"): (0.213, 0.220, 41.0),
    ("fast", "light_rcnn"): (0.223, 0.231, 33.0),
    (BASELINE_VARIANT, "yolo26"): (0.401, 0.430, 30.0),  # never measured with the preset
}


def raw_reports(numbers=NUMBERS, environment=CPU, created_at=None):
    found = []
    for (variant, model), (ap, ap75, p50) in numbers.items():
        extra = FAST if variant == "fast" else {}
        accuracy = build_report("accuracy", model, precision="fp32", image_size=640, environment={},
                                accuracy={"evaluator": "pycocotools", "images_evaluated": 128,
                                          "images_total": 128, "AP_50_95": ap, "AP_50": ap + 0.2,
                                          "AP_75": ap75}, **extra)
        latency = build_report("latency", model, precision="fp32", image_size=640,
                               environment=dict(environment), **extra,
                               latency={"scope_id": reporting.SCOPE_HOST_TO_DETECTIONS,
                                        "memory_format": "contiguous",
                                        **latency_summary([p50 + i % 3 - 1 for i in range(150)])})
        for report in (accuracy, latency):
            if created_at:
                report["created_at"] = created_at
            found.append(report)
    return found


def selections(numbers=NUMBERS):
    found = [r for index, raw in enumerate(raw_reports(numbers)) for r in normalize(raw, f"r{index}")]
    return select_reports(found, variant=BASELINE_VARIANT), select_reports(found, variant="fast")


def texts(fig) -> list[str]:
    return [text for trace in fig.data for text in (trace.text or [])]


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_improvement_figure_is_two_dumbbell_panels_with_signed_deltas(theme):
    before, after = selections()
    fig = viz.improvement_figure(before, after, theme, "빠른 설정")
    fig.to_json()
    assert not viz.is_empty(fig)
    # Two small multiples sharing the model axis, never a second y-axis on one panel.
    assert fig.layout.xaxis2 is not None and fig.layout.yaxis2.matches == "y"
    assert fig.layout.yaxis.overlaying is None and fig.layout.yaxis2.overlaying is None
    assert "yaxis3" not in fig.layout.to_plotly_json()
    assert list(fig.layout.yaxis.categoryarray) == ["Light R-CNN", "Vanilla Faster R-CNN"]  # no YOLO26
    assert set(texts(fig)) == {"+1.0%p", "-0.1%p", "-8.0 ms (-20%)", "-140 ms (-9.7%)"}
    for model in ("faster_rcnn", "light_rcnn"):
        marks = [t for t in fig.data if t.y == (viz.MODELS[model],) and t.marker.symbol]
        symbols = {t.marker.symbol for t in marks}
        assert symbols == {viz.MODEL_SYMBOLS[model], f"{viz.MODEL_SYMBOLS[model]}-open"}
        assert {t.marker.color for t in marks} == {viz.model_color(model, theme)}
    legend = [t.name for t in fig.data if t.showlegend is not False]
    assert legend == [BASELINE_VARIANT_LABEL, "빠른 설정"]
    # 1450 ms vs 33 ms: a log axis keeps both models' changes readable.
    assert fig.layout.xaxis2.type == "log" and "로그 눈금" in fig.layout.annotations[1].text
    assert all(10 ** fig.layout.xaxis2.range[0] <= v <= 10 ** fig.layout.xaxis2.range[1]
               for v in fig.layout.xaxis2.tickvals)
    assert fig.layout.annotations[0].xanchor == "left"


def test_linear_axes_and_delta_label_sits_right_of_each_dumbbell():
    numbers = {key: value for key, value in NUMBERS.items() if key[1] != "faster_rcnn"}
    before, after = selections(numbers)
    fig = viz.improvement_figure(before, after)  # label comes from the reports
    assert fig.layout.xaxis2.type in (None, "linear") and fig.layout.xaxis2.range[0] >= 0
    assert "빠른 설정" in fig.layout.title.subtitle.text
    labeled = {t.text[0]: t.x[0] for t in fig.data if t.text and t.mode == "markers+text"}
    assert labeled == {"+1.0%p": pytest.approx(22.3), "-8.0 ms (-20%)": 41.0}  # right end
    for name, axis in (("x", fig.layout.xaxis), ("x2", fig.layout.xaxis2)):
        xs = [x for t in fig.data if (t.xaxis or "x") == name for x in (t.x or []) if x is not None]
        assert axis.range[0] < min(xs) and axis.range[1] > max(xs)


def test_rows_without_a_before_after_pair_say_so():
    before, after = selections()
    for chosen in (before, after):
        del chosen[("latency", "faster_rcnn")]
    fig = viz.improvement_figure(before, after)
    [note] = [t for t in fig.data if t.mode == "text"]
    assert note.xaxis == "x2" and list(note.y) == ["Vanilla Faster R-CNN"]
    assert note.text == ("비교할 결과 없음",)
    assert fig.layout.xaxis2.type in (None, "linear")  # only Light R-CNN left in that panel


def test_empty_states():
    before, after = selections()
    assert viz.is_empty(viz.improvement_figure({}, {}))
    assert viz.is_empty(viz.improvement_figure(before, {}))
    only_yolo = {key: report for key, report in before.items() if key[1] == "yolo26"}
    assert viz.is_empty(viz.improvement_figure(only_yolo, after))


def test_improvement_rows_are_the_table_twin():
    before, after = selections()
    rows = viz.improvement_rows(before, after)
    assert {row["모델"] for row in rows} == {"Vanilla Faster R-CNN", "Light R-CNN"}
    light = {row["지표"]: row for row in rows if row["모델"] == "Light R-CNN"}
    assert list(light) == ["AP@[.50:.95] (%)", "AP75 (%)", "p50 (ms)", "p95 (ms)", "FPS"]
    assert light["AP@[.50:.95] (%)"]["변화"] == pytest.approx(1.0)
    assert light["p50 (ms)"]["변화"] == pytest.approx(-8.0)
    assert light["p50 (ms)"]["상대 변화 (%)"] == pytest.approx(-8 / 41 * 100)
    assert {row["판정"] for row in light.values()} == {"좋아짐"}
    faster = {row["지표"]: row for row in rows if row["모델"] == "Vanilla Faster R-CNN"}
    assert faster["AP@[.50:.95] (%)"]["판정"] == "나빠짐" and faster["FPS"]["판정"] == "좋아짐"
    assert viz.improvement_rows(before, before)[0]["판정"] == "변화 없음"


def band_rows(ap: tuple[float, float], p50: tuple[float, float]) -> dict[str, dict]:
    numbers = {(BASELINE_VARIANT, "light_rcnn"): (ap[0], 0.220, p50[0]),
               ("fast", "light_rcnn"): (ap[1], 0.220, p50[1])}
    return {row["지표"]: row for row in viz.improvement_rows(*selections(numbers))}


def test_changes_inside_the_noise_band_are_neither_better_nor_worse():
    # PERFORMANCE_PLAN 5절: |ΔAP| < 0.1%p and |Δp50| < 3% count as the same.
    rows = band_rows(ap=(0.2130, 0.2125), p50=(40.0, 40.4))
    assert rows["AP@[.50:.95] (%)"]["변화"] == pytest.approx(-0.05)
    assert rows["p50 (ms)"]["상대 변화 (%)"] == pytest.approx(1.0)
    assert rows["AP@[.50:.95] (%)"]["판정"] == "변화 없음 (오차 범위)"
    assert rows["p50 (ms)"]["판정"] == "변화 없음 (오차 범위)"
    assert rows["AP75 (%)"]["판정"] == "변화 없음"  # identical numbers keep the plain label


@pytest.mark.parametrize(("ap", "p50", "verdict"), [((0.213, 0.214), (15.0, 14.55), "좋아짐"),
                                                    ((0.214, 0.213), (15.0, 15.45), "나빠짐")])
def test_an_exact_band_edge_stays_outside_the_band(ap, p50, verdict):
    # In floats these are ±0.09999...%p and ±2.99999...%: still a change, not noise.
    rows = band_rows(ap, p50)
    assert abs(rows["AP@[.50:.95] (%)"]["변화"]) < viz.NOISE_AP_PP
    assert abs(rows["p50 (ms)"]["상대 변화 (%)"]) < viz.NOISE_RELATIVE_PCT
    assert rows["AP@[.50:.95] (%)"]["판정"] == verdict and rows["p50 (ms)"]["판정"] == verdict


def test_the_ap_band_reads_the_raw_change_and_only_ap_can_be_unchanged():
    # 0.0996%p shows as +0.10 but is inside the 0.1%p band; a timing never reads plain '변화 없음'.
    rows = band_rows(ap=(0.35, 0.350996), p50=(100.0, 100.003))
    assert round(rows["AP@[.50:.95] (%)"]["변화"], 2) == viz.NOISE_AP_PP
    assert rows["AP@[.50:.95] (%)"]["판정"] == "변화 없음 (오차 범위)"
    assert rows["AP75 (%)"]["판정"] == "변화 없음"  # identical AP keeps the plain label
    assert {rows[name]["판정"] for name in ("p50 (ms)", "p95 (ms)", "FPS")} == {"변화 없음 (오차 범위)"}
    same = band_rows(ap=(0.35, 0.35), p50=(100.0, 100.0))
    assert {row["판정"] for name, row in same.items() if "AP" not in name} == {"변화 없음 (오차 범위)"}


def test_signed_labels_never_show_minus_zero():
    assert viz._signed(-0.04) == "±0.0" and viz._signed(0.04) == "±0.0"
    assert viz._signed(1.26) == "+1.3" and viz._signed(-140.2, 0) == "-140"
    assert viz._delta_text("latency", 30.4, 30.5) == "+0.1 ms (+0.3%)"
    assert viz._delta_text("accuracy", 40.1, 40.1) == "±0.0%p"


def test_build_all_adds_improvement_only_for_a_variant_selection():
    before, after = selections()
    assert set(viz.build_all(after, TARGETS)) == set(viz.ALL_FIGURES)  # old call unchanged
    figures = viz.build_all(after, TARGETS, "dark", before, "빠른 설정")
    assert set(figures) == set(viz.ALL_FIGURES) | set(viz.VARIANT_FIGURES)
    assert list(figures)[:2] == ["tradeoff", "improvement"]
    assert not viz.is_empty(figures["improvement"])
    assert "improvement" not in viz.build_all(before, TARGETS, baseline_selected=before)
    p95 = viz.build_all(after, {**TARGETS, "latency_percentile": "p95"}, baseline_selected=before)
    assert p95["improvement"].layout.annotations[1].text.startswith("p95 지연 시간")
    cards = [viz.html_report(built, [], [], after, "light", "t").count("<div class='card'>")
             for built in (figures, viz.build_all(after, TARGETS, "dark"))]
    assert cards[0] == cards[1] + 1  # the HTML report carries the before/after chart too


# --------------------------------------------------------------------------- dashboard


def write_results(root: Path) -> None:
    for index, report in enumerate(raw_reports()):
        variant = report.get("variant", BASELINE_VARIANT)
        write_report(report, root / variant / f"{report['model']}_{report['kind']}_{index}.json")


def write_t4_run(root: Path) -> None:
    # A newer GPU run of the preset for Light R-CNN only.
    numbers = {("fast", "light_rcnn"): (0.223, 0.231, 5.0)}
    for report in raw_reports(numbers, environment=T4, created_at="2999-01-01T00:00:00+00:00"):
        if report["kind"] == "latency":
            write_report(report, root / "fast_t4" / "light_rcnn_latency.json")


@pytest.fixture
def app_test(monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("LIGHT_RCNN_RESULTS_DIR", str(tmp_path))
    app_path = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")

    def run():
        app = AppTest.from_file(app_path, default_timeout=120)
        app.run()
        assert not app.exception, [e.value for e in app.exception]
        return app
    return run


def sidebar_box(app, label):
    return next((box for box in app.sidebar.selectbox if box.label == label), None)


def tab(app, label):
    return next(t for t in app.tabs if t.label == label)


def metric_values(app, suffix):
    return [m.value for m in app.metric if m.label.endswith(suffix)]


def chart_titles(block) -> list[str]:
    return [json.loads(chart.proto.spec)["layout"]["title"]["text"] for chart in block.get("plotly_chart")]


def test_dashboard_switches_between_baseline_and_preset(app_test, tmp_path):
    write_results(tmp_path)
    app = app_test()
    box = sidebar_box(app, "추론 설정")
    assert box.options == [BASELINE_VARIANT_LABEL, "빠른 설정"] and box.value == BASELINE_VARIANT
    assert metric_values(app, "AP@[.50:.95]") == ["37.0%", "21.3%", "40.1%"]
    assert metric_values(app, "지연") == ["1450 ms", "41.0 ms", "30.0 ms"]
    improvement = tab(app, "개선 전/후")
    assert any("빠른 설정" in info.value for info in improvement.info)
    overview = next(df.value for df in improvement.dataframe if "AP 변화 (%p)" in df.value.columns)
    assert list(overview["추론 설정"]) == ["빠른 설정", "빠른 설정"]
    assert list(overview["p50 변화 (ms)"]) == pytest.approx([-140.0, -8.0])
    assert list(overview["비교"]) == ["가능", "가능"]
    assert not any("'비교 불가'" in c.value for c in improvement.caption)
    assert not any("체크포인트가" in w.value for w in app.warning)

    box.set_value("fast").run()
    assert not app.exception, [e.value for e in app.exception]
    assert metric_values(app, "AP@[.50:.95]") == ["36.9%", "22.3%"]
    assert metric_values(app, "지연") == ["1310 ms", "33.0 ms"]
    captions = [c.value for c in app.caption]
    assert any(c.startswith("추론 설정: **빠른 설정**") for c in captions)  # above the KPI cards
    assert ("이 추론 설정으로 측정한 결과 없음 · 사이드바에서 다른 설정 선택: "
            f"{BASELINE_VARIANT_LABEL}") in captions  # YOLO26 only ran the defaults
    assert "추론 설정: 빠른 설정 · 측정 장치: Xeon (사이드바에서 변경)" in captions
    improvement = tab(app, "개선 전/후")
    assert chart_titles(improvement) == ["<b>개선 전/후 비교</b>"]
    assert f"비교: {BASELINE_VARIANT_LABEL} → 빠른 설정 · 측정 장치: Xeon" in [
        c.value for c in improvement.caption]
    deltas = next(df.value for df in improvement.dataframe if "판정" in df.value.columns)
    assert list(deltas.columns) == ["모델", "지표", "개선 전", "개선 후", "변화", "상대 변화 (%)", "판정"]
    light_p50 = deltas[(deltas["모델"] == "Light R-CNN") & (deltas["지표"] == "p50 (ms)")]
    assert light_p50["변화"].tolist() == pytest.approx([-8.0])
    assert "적용된 추론 옵션" in [e.label for e in improvement.expander]
    assert any('"rpn_pre_nms_top_n": 500' in c.value for c in improvement.code)
    assert SUMMARY in [c.value for c in improvement.caption]
    assert not any("개선 전/후 측정 조건이 다릅니다" in e.value for e in improvement.error)
    sources = next(df.value for df in tab(app, "데이터·출처").dataframe if "파일" in df.value.columns)
    assert set(sources["추론 설정"]) == {BASELINE_VARIANT_LABEL, "빠른 설정"}
    assert sources[sources["사용"]]["추론 설정"].eq("빠른 설정").all()


def test_variant_and_device_filters_combine_in_the_dashboard(app_test, tmp_path):
    write_results(tmp_path)
    write_t4_run(tmp_path)
    app = app_test()
    device = sidebar_box(app, "지연 시간 측정 장치")
    assert device.options == ["모델 3개 · Xeon"]  # the baseline was only timed on the CPU

    sidebar_box(app, "추론 설정").set_value("fast").run()
    device = sidebar_box(app, "지연 시간 측정 장치")
    assert device.options == ["모델 1개 · Tesla T4", "모델 2개 · Xeon"] and device.value == "Tesla T4"
    assert metric_values(app, "지연") == ["–", "5.0 ms"]
    assert ("이 장치에서 측정한 결과 없음 · 사이드바에서 다른 장치 선택: Xeon"
            not in [c.value for c in app.caption])  # Faster R-CNN still has its accuracy card
    # No baseline timing on the T4: the latency panel has nothing to pair, AP still compares.
    improvement = tab(app, "개선 전/후")
    assert chart_titles(improvement) == ["<b>개선 전/후 비교</b>"]
    deltas = next(df.value for df in improvement.dataframe if "판정" in df.value.columns)
    assert set(deltas["지표"]) == {"AP@[.50:.95] (%)", "AP75 (%)"}

    device.set_value("Xeon").run()
    assert not app.exception, [e.value for e in app.exception]
    assert metric_values(app, "지연") == ["1310 ms", "33.0 ms"]
    assert sidebar_box(app, "추론 설정").value == "fast"


def capture_html_reports(monkeypatch) -> list[str]:
    """Every HTML report the dashboard builds for its download button, newest last."""
    pages, original = [], viz.html_report

    def capture(*args, **kwargs):
        pages.append(original(*args, **kwargs))
        return pages[-1]
    monkeypatch.setattr(viz, "html_report", capture)
    return pages


def test_overview_and_html_report_flag_a_mismatched_before_after(app_test, tmp_path, monkeypatch):
    # Light R-CNN's preset was scored on another image subset than its baseline. (A preset run on
    # another checkpoint is never paired: the checkpoint selector shows one checkpoint at a time.)
    for index, report in enumerate(raw_reports()):
        if report["model"] == "light_rcnn" and report["kind"] == "accuracy":
            report["accuracy"]["images_evaluated"] = 64 if report.get("variant") == "fast" else 128
        variant = report.get("variant", BASELINE_VARIANT)
        write_report(report, tmp_path / variant / f"{report['model']}_{report['kind']}_{index}.json")
    pages = capture_html_reports(monkeypatch)
    app = app_test()
    improvement = tab(app, "개선 전/후")
    overview = next(df.value for df in improvement.dataframe if "AP 변화 (%p)" in df.value.columns)
    assert dict(zip(overview["모델"], overview["비교"])) == {"Vanilla Faster R-CNN": "가능",
                                                            "Light R-CNN": "비교 불가"}
    assert any(c.value.startswith("'비교 불가'는") for c in improvement.caption)
    assert "측정 조건이 다릅니다" not in pages[-1]  # no before/after chart in the baseline report

    sidebar_box(app, "추론 설정").set_value("fast").run()
    assert not app.exception, [e.value for e in app.exception]
    message = "Light R-CNN 정확도의 개선 전/후 측정 조건이 다릅니다 (평가 이미지 수 128 → 64)"
    assert any(message in e.value for e in tab(app, "개선 전/후").error)
    assert message in pages[-1]


V3_LIGHT = {(BASELINE_VARIANT, "light_rcnn"): (0.263, 0.270, 51.0),
            ("fast", "light_rcnn"): (0.268, 0.275, 37.0)}


def write_versions(root: Path, v2_complete: bool = False) -> None:
    """compare_improvements.py's layout: reports/v2 and reports/v3, each with Faster R-CNN, both
    under the 'light_rcnn' model key."""
    for version, created_at in (("v2", "2026-11-01T00:00:00+00:00"),
                                ("v3", "2026-11-02T00:00:00+00:00")):
        numbers = {key: value for key, value in NUMBERS.items() if key[1] != "yolo26"}
        if version == "v3":
            numbers.update(V3_LIGHT)
        for index, report in enumerate(raw_reports(numbers, created_at=created_at)):
            light = report["model"] == "light_rcnn"
            report.update(model_tag=version if light else "faster_rcnn",
                          model_label=f"Light R-CNN {version}" if light else "Vanilla Faster R-CNN",
                          weights=(f"outputs/coco_4090_{version}/best.pth" if light
                                   else "torchvision_coco_v1"),
                          weights_sha256=version[-1] * 64 if light else None)
            if v2_complete and light and version == "v2" and report["kind"] == "accuracy":
                report["accuracy"]["is_partial"] = False  # outranks v3's accuracy
            folder = root / "reports" / version
            write_report(report, folder / f"{report['model']}_{report['kind']}_{index}.json")


def card_titles(app) -> list[str]:
    return [m.value.rsplit("</span>", 1)[-1] for m in app.markdown if m.value.endswith("**")]


CHECKPOINT_LABEL = "Light R-CNN 체크포인트"


def test_mixed_light_rcnn_checkpoints_get_a_selector_not_a_warning(app_test, tmp_path, monkeypatch):
    write_versions(tmp_path)
    pages = capture_html_reports(monkeypatch)
    app = app_test()
    box = sidebar_box(app, CHECKPOINT_LABEL)
    assert box.options == ["Light R-CNN v3", "Light R-CNN v2"] and box.value == "3" * 64  # newest
    assert not any("체크포인트" in w.value for w in app.warning)  # outputs/ mixes v2 and v3 anyway
    assert any("Light R-CNN 체크포인트 2개" in c.value for c in app.sidebar.caption)
    assert card_titles(app) == ["**Vanilla Faster R-CNN**", "**Light R-CNN v3**", "**YOLO26**"]
    assert metric_values(app, "AP@[.50:.95]") == ["37.0%", "26.3%"]
    assert metric_values(app, "지연") == ["1450 ms", "51.0 ms"]
    assert ("<title>Light R-CNN 성능 비교 보고서 (Light R-CNN v3 · 개선 전 (기본 설정) · Xeon)</title>"
            in pages[-1])
    sources = next(df.value for df in tab(app, "데이터·출처").dataframe if "파일" in df.value.columns)
    assert set(sources["체크포인트"]) == {"torchvision_coco_v1", "Light R-CNN v2", "Light R-CNN v3"}
    assert set(sources[sources["사용"]]["체크포인트"]) == {"torchvision_coco_v1", "Light R-CNN v3"}

    sidebar_box(app, "추론 설정").set_value("fast").run()
    assert not app.exception, [e.value for e in app.exception]
    improvement = tab(app, "개선 전/후")
    assert (f"비교: {BASELINE_VARIANT_LABEL} → 빠른 설정 · 체크포인트: Light R-CNN v3 · 측정 장치: Xeon"
            in [c.value for c in improvement.caption])
    deltas = next(df.value for df in improvement.dataframe if "판정" in df.value.columns)
    assert set(deltas["모델"]) == {"Vanilla Faster R-CNN", "Light R-CNN v3"}
    figure = json.loads(improvement.get("plotly_chart")[0].proto.spec)
    assert "Light R-CNN v3" in figure["layout"]["yaxis"]["categoryarray"]
    assert "체크포인트: Light R-CNN v3 · 추론 설정: 빠른 설정 · 측정 장치: Xeon (사이드바에서 변경)" in [
        c.value for c in app.caption]

    # The filters combine: another checkpoint keeps the inference setting and the device.
    sidebar_box(app, CHECKPOINT_LABEL).set_value("2" * 64).run()
    assert not app.exception, [e.value for e in app.exception]
    assert sidebar_box(app, "추론 설정").value == "fast"
    assert card_titles(app)[1] == "**Light R-CNN v2**"
    assert metric_values(app, "AP@[.50:.95]") == ["36.9%", "22.3%"]
    assert metric_values(app, "지연") == ["1310 ms", "33.0 ms"]
    tradeoff = next(spec for spec in map(json.loads, (c.proto.spec for c in app.get("plotly_chart")))
                    if spec["layout"]["title"]["text"] == "<b>정확도 대 속도</b>")
    assert [trace["name"] for trace in tradeoff["data"]] == ["Vanilla Faster R-CNN", "Light R-CNN v2"]

    # One version's folder holds one checkpoint, even with a v1 file naming it by another path.
    v2 = tmp_path / "reports" / "v2"
    (v2 / "light_rcnn_latency_v1.json").write_text(json.dumps({
        "model": "light_rcnn", "device": "cpu", "iterations": 10, "p50_ms": 50.0,
        "weights": "C:\\Users\\me\\Lite-R-CNN\\outputs\\coco_4090_v2\\best.pth"}), encoding="utf-8")
    app.text_input[0].set_value(str(v2)).run()
    assert not app.exception, [e.value for e in app.exception]
    assert sidebar_box(app, CHECKPOINT_LABEL) is None
    assert not any("체크포인트가" in w.value for w in app.warning)
    assert "**Light R-CNN v2**" in card_titles(app)
    sources = next(df.value for df in tab(app, "데이터·출처").dataframe if "파일" in df.value.columns)
    assert set(sources["체크포인트"]) == {"torchvision_coco_v1", "Light R-CNN v2"}


def test_the_selector_never_mixes_two_checkpoints_on_one_card(app_test, tmp_path):
    # Unfiltered, v2's complete accuracy would outrank v3's partial one next to v3's latency.
    write_versions(tmp_path, v2_complete=True)
    app = app_test()
    assert card_titles(app)[1] == "**Light R-CNN v3**"
    assert metric_values(app, "AP@[.50:.95]")[1] == "26.3%"  # v3's own accuracy
    assert not any("서로 다른 가중치(체크포인트)" in e.value for e in app.error)
    sidebar_box(app, CHECKPOINT_LABEL).set_value("2" * 64).run()
    assert not app.exception, [e.value for e in app.exception]
    assert card_titles(app)[1] == "**Light R-CNN v2**"
    assert metric_values(app, "AP@[.50:.95]")[1] == "21.3%"
    assert not any("서로 다른 가중치(체크포인트)" in e.value for e in app.error)


def test_baseline_only_folder_keeps_the_old_sidebar(app_test, tmp_path):
    numbers = {key: value for key, value in NUMBERS.items() if key[0] == BASELINE_VARIANT}
    for index, report in enumerate(raw_reports(numbers)):
        write_report(report, tmp_path / f"{report['model']}_{report['kind']}_{index}.json")
    app = app_test()
    assert sidebar_box(app, "추론 설정") is None
    assert not any(c.value.startswith("추론 설정") for c in app.caption)
    assert any("다른 추론 설정으로 측정한 결과가 아직 없습니다" in i.value
               for i in tab(app, "개선 전/후").info)
    sources = next(df.value for df in tab(app, "데이터·출처").dataframe if "파일" in df.value.columns)
    assert "추론 설정" not in sources.columns
