"""Paired 95% CIs (scripts/compare_improvements.py) in the before/after views, the verdict rules
and the dashboard's Light R-CNN checkpoint selector on a combined v2 + v3 run."""

import json
from pathlib import Path

import pytest

from light_rcnn import reporting, viz
from light_rcnn.reporting import (
    BASELINE_VARIANT_LABEL,
    build_report,
    latency_summary,
    normalize,
    select_reports,
    write_report,
)

CPU = {"device": "cpu", "device_type": "cpu", "device_name": "Intel(R) Xeon(R) Processor @ 2.10GHz",
       "torch_num_threads": 4}
T4 = {"device": "cuda:0", "device_type": "cuda", "device_name": "Tesla T4"}
LABELS = {"baseline": BASELINE_VARIANT_LABEL, "deploy": "배포 기본 (Conv+BN 융합·메모리 형식)",
          "fast": "빠른 설정"}
# (version, preset) -> (AP@[.50:.95], p50 ms, paired vs_baseline 95% CI in %), from the final CPU run
RUN = {
    ("v2", "baseline"): (0.4222, 113.33, None),
    ("v2", "deploy"): (0.4222, 73.52, (-38.98, -30.82)),
    ("v2", "fast"): (0.4224, 54.87, (-54.58, -48.06)),
    ("v3", "baseline"): (0.3490, 127.92, None),
    ("v3", "deploy"): (0.3490, 79.73, (-42.80, -34.12)),
    ("v3", "fast"): (0.3501, 50.76, (-61.61, -58.46)),
}
CHECKPOINT_LABEL = "Light R-CNN 체크포인트"
SHA = {"v2": "a" * 64, "v3": "c" * 64}


def run_reports(numbers=RUN, run="run1-4b1c3447", environment=CPU, extra=None):
    """compare_improvements.py's reports: Light R-CNN only (no Faster R-CNN weights), one
    comparison_run, interleaved latency with paired CIs against the baseline."""
    found = []
    for (version, preset), (ap, p50, ci) in numbers.items():
        common = {"variant": preset, "variant_label": LABELS.get(preset, preset),
                  "model_tag": version, "model_label": f"Light R-CNN {version}",
                  "weights": f"outputs/coco_4090_{version}/best.pth", "weights_sha256": SHA[version],
                  "comparison_run": run if isinstance(run, str) else run[(version, preset)],
                  **((extra or {}).get(preset) or {})}
        stamp = "2026-11-02T17:40:05+00:00" if version == "v2" else "2026-11-02T17:43:44+00:00"
        paired = None
        if ci is not None:
            paired = {"vs_baseline": {"variant": "baseline", "delta_p50_pct": sum(ci) / 2,
                                      "ci95_pct": list(ci), "resamples": 2000,
                                      "method": "paired bootstrap over rounds"},
                      "vs_deploy": None}
        accuracy = build_report("accuracy", "light_rcnn", precision="fp32", image_size=640,
                                environment=dict(environment), **common,
                                accuracy={"evaluator": "pycocotools", "images_evaluated": 128,
                                          "images_total": 128, "AP_50_95": ap, "AP_50": ap + 0.2,
                                          "AP_75": ap + 0.03})
        latency = build_report("latency", "light_rcnn", precision="fp32", image_size=640,
                               environment=dict(environment), **common,
                               latency={"scope_id": reporting.SCOPE_HOST_TO_DETECTIONS,
                                        "memory_format": "contiguous", "design": "interleaved_blocks",
                                        "rounds": 20, "block_size": 16, "paired": paired,
                                        **latency_summary([p50 + i % 3 - 1 for i in range(320)])})
        for report in (accuracy, latency):
            report["created_at"] = stamp
            found.append(report)
    return found


def write_run(root: Path, **kwargs) -> Path:
    for raw in run_reports(**kwargs):
        name = f"{raw['model']}_{raw['kind']}_{raw['variant']}.json"
        write_report(raw, root / "reports" / raw["model_tag"] / name)
    return root / "reports"


def pick(found, version="v2", preset="fast"):
    reports = [r for index, raw in enumerate(found) for r in normalize(raw, f"r{index}")]
    key = SHA[version]
    return (select_reports(reports, variant="baseline", checkpoint=key),
            select_reports(reports, variant=preset, checkpoint=key))


def light_rows(before, after) -> dict[str, dict]:
    return {row["지표"]: row for row in viz.improvement_rows(before, after)}


# --------------------------------------------------------------------------- viz


def test_paired_ci_needs_one_comparison_run_one_checkpoint_and_the_before_preset():
    before, after = pick(run_reports())
    key = ("latency", "light_rcnn")
    assert viz.paired_ci(before[key], after[key]) == (-54.58, -48.06)
    assert viz.paired_ci(before[key], before[key]) is None  # the baseline has no paired entry
    assert viz.paired_ci(None, after[key]) is None
    assert viz.paired_ci(before[("accuracy", "light_rcnn")], after[("accuracy", "light_rcnn")]) is None
    runs = {key: "run1" for key in RUN} | {("v2", "fast"): "run2"}  # fast was measured another day
    assert viz.paired_ci(*(s[key] for s in pick(run_reports(run=runs)))) is None
    unrecorded = {key: "" for key in RUN}  # older reports carry no comparison_run
    assert viz.paired_ci(*(s[key] for s in pick(run_reports(run=unrecorded)))) is None
    # v3's fast (same run) never lends its CI to a v2 baseline.
    found = [r for index, raw in enumerate(run_reports()) for r in normalize(raw, f"r{index}")]
    v2 = select_reports(found, variant="baseline", checkpoint=SHA["v2"])
    v3 = select_reports(found, variant="fast", checkpoint=SHA["v3"])
    assert viz.paired_ci(v2[key], v3[key]) is None
    # A deploy 'before' takes the vs_deploy entry, not vs_baseline.
    deploy, fast = pick(run_reports(), preset="deploy")[1], after
    assert viz.paired_ci(deploy[key], fast[key]) is None  # this run wrote vs_deploy = null
    fast[key].section["paired"]["vs_deploy"] = {"variant": "deploy", "ci95_pct": [-31.3, -19.4]}
    assert viz.paired_ci(deploy[key], fast[key]) == (-31.3, -19.4)
    for broken in ([-31.3], ["x", 1], None, "[-1, 1]"):
        fast[key].section["paired"]["vs_deploy"]["ci95_pct"] = broken
        assert viz.paired_ci(deploy[key], fast[key]) is None


def test_rows_add_a_ci_cell_for_p50_only_within_one_comparison_run():
    rows = viz.improvement_rows(*pick(run_reports()))
    assert list(rows[0]) == ["모델", "지표", "개선 전", "개선 후", "변화", "상대 변화 (%)",
                             viz.CI_COLUMN, "판정"]
    cells = {row["지표"]: row[viz.CI_COLUMN] for row in rows}
    assert cells == {"AP@[.50:.95] (%)": None, "AP75 (%)": None, "p50 (ms)": "-54.6 ~ -48.1%",
                     "p95 (ms)": None, "FPS": None}
    runs = {key: "run1" for key in RUN} | {("v2", "fast"): "run2"}
    plain = viz.improvement_rows(*pick(run_reports(run=runs)))
    assert all(viz.CI_COLUMN not in row for row in plain)  # exactly the table without CIs
    assert [row["판정"] for row in plain] == [row["판정"] for row in rows]


@pytest.mark.parametrize(("p50", "ci", "verdict"), [
    ((100.0, 95.0), (-9.0, 1.0), "변화 없음 (오차 범위)"),  # -5%, but the CI holds 0
    ((100.0, 98.0), (-3.5, -0.5), "좋아짐"),  # -2%, inside the 3% band, but the CI excludes 0
    ((100.0, 104.0), (1.2, 6.8), "나빠짐"),
    ((100.0, 97.0), (-5.04, -0.04), "좋아짐"),  # misses 0 by 0.04%: the raw bounds decide
    ((100.0, 103.0), (0.04, 6.0), "나빠짐"),
    ((100.0, 95.0), (-6.0, 0.0), "변화 없음 (오차 범위)"),  # a bound at 0 holds it
])
def test_a_paired_ci_decides_the_p50_verdict(p50, ci, verdict):
    numbers = {("v2", "baseline"): (0.4222, p50[0], None), ("v2", "fast"): (0.4222, p50[1], ci)}
    rows = light_rows(*pick(run_reports(numbers)))
    assert rows["p50 (ms)"]["판정"] == verdict
    # Without a CI the same numbers fall back to the 3% band.
    runs = {("v2", "baseline"): "run1", ("v2", "fast"): "run2"}
    relative = (p50[1] - p50[0]) / p50[0] * 100
    expected = ("변화 없음 (오차 범위)" if abs(relative) < viz.NOISE_RELATIVE_PCT
                else "좋아짐" if relative < 0 else "나빠짐")
    assert light_rows(*pick(run_reports(numbers, run=runs)))["p50 (ms)"]["판정"] == expected
    assert rows["AP@[.50:.95] (%)"]["판정"] == "변화 없음"  # identical AP keeps the plain label


@pytest.mark.parametrize(("ci", "cell"), [
    ((1.2, 6.8), "+1.2 ~ +6.8%"),
    ((-5.04, -0.04), "-5.04 ~ -0.04%"),  # never '±0.0' for a bound that is not 0 (it decides)
    ((-6.0, 0.0), "-6.0 ~ ±0.0%"),  # and never '-0.0' for one that is
])
def test_ci_cell_shows_signed_bounds_and_never_minus_zero(ci, cell):
    numbers = {("v2", "baseline"): (0.4222, 100.0, None), ("v2", "fast"): (0.4222, 97.0, ci)}
    assert light_rows(*pick(run_reports(numbers)))["p50 (ms)"][viz.CI_COLUMN] == cell


def test_figure_latency_label_names_the_ci_only_for_p50_of_one_run():
    numbers = {("v2", "baseline"): (0.4222, 100.0, None), ("v2", "fast"): (0.4224, 61.0, (-41.2, -36.4))}
    before, after = pick(run_reports(numbers))
    fig = viz.improvement_figure(before, after)
    labels = {t.text[0] for t in fig.data if t.text and t.mode == "markers+text"}
    assert labels == {"±0.0%p", "-39 ms (-39%)<br>95% CI -41~-36%"}  # CI on its own line
    assert any("(-39 ms (-39%, 95% CI -41~-36%))" in (t.hovertemplate or "") for t in fig.data)
    small = {("v2", "baseline"): (0.4222, 50.0, None), ("v2", "fast"): (0.4222, 48.4, (-8.2, 4.9))}
    assert "-1.6 ms (-3.2%)<br>95% CI -8.2~+4.9%" in {
        t.text[0] for t in viz.improvement_figure(*pick(run_reports(small))).data
        if t.text and t.mode == "markers+text"}
    p95 = viz.improvement_figure(before, after, percentile="p95")
    assert not any("CI" in t.text[0] for t in p95.data if t.text and t.mode == "markers+text")
    runs = {key: f"run{index}" for index, key in enumerate(numbers)}
    plain = viz.improvement_figure(*pick(run_reports(numbers, run=runs)))
    assert {t.text[0] for t in plain.data if t.text and t.mode == "markers+text"} == {
        "±0.0%p", "-39 ms (-39%)"}
    # The labels sit in a right margin sized in pixels for their longest line (room in data
    # units shrank with the browser and cut the CI line off below ~1024 px); the axis keeps the
    # same small room with or without a CI.
    width = viz.LABEL_CHAR_PX * len("95% CI -41~-36%") + viz.LABEL_GAP_PX
    assert fig.layout.margin.r == width > plain.layout.margin.r >= viz.LABEL_CHAR_PX * len(
        "-39 ms (-39%)") + viz.LABEL_GAP_PX
    assert fig.layout.xaxis2.range == pytest.approx(plain.layout.xaxis2.range)
    assert fig.layout.xaxis2.range == pytest.approx([61 - 39 * 0.12, 100 + 39 * 0.12])
    # The narrower plot keeps the AP panel's title over its own panel: two short lines.
    titles = [annotation for annotation in fig.layout.annotations if "좋음" in annotation.text]
    assert [title.text.split("<br>")[1] for title in titles] == ["높을수록 좋음", "낮을수록 좋음"]
    assert {(title.xanchor, title.align) for title in titles} == {("left", "left")}


def test_model_labels_rename_light_rcnn_in_every_chart_and_table():
    before, after = pick(run_reports())
    names = {"light_rcnn": "Light R-CNN v2"}
    targets = {"latency_ms": 15, "latency_percentile": "p50", "ap_metric": "AP_50_95", "ap": 0.6}
    figures = viz.build_all(after, targets, "light", before, "빠른 설정", model_labels=names)
    for key, fig in figures.items():
        if viz.is_empty(fig):
            continue
        text = json.dumps(fig.to_plotly_json(), ensure_ascii=False)
        assert "Light R-CNN v2" in text, key
        assert '"Light R-CNN"' not in text, key
    assert {row["모델"] for row in viz.improvement_rows(before, after, model_labels=names)} == {
        "Light R-CNN v2"}
    assert [row["모델"] for row in viz.summary_rows(after, model_labels=names)] == ["Light R-CNN v2"]
    page = viz.html_report(figures, [], [], after, "light", "t", model_labels=names)
    assert "<td>Light R-CNN v2</td>" in page
    assert [row["모델"] for row in viz.summary_rows(after)] == ["Light R-CNN"]  # default unchanged


# --------------------------------------------------------------------------- dashboard


@pytest.fixture
def dashboard(monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app_path = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")

    def run(folder: Path):
        monkeypatch.setenv("LIGHT_RCNN_RESULTS_DIR", str(folder))
        app = AppTest.from_file(app_path, default_timeout=120)
        app.run()
        assert not app.exception, [e.value for e in app.exception]
        return app
    return run


def box(app, label):
    return next((item for item in app.sidebar.selectbox if item.label == label), None)


def choose(app, label, value):
    box(app, label).set_value(value).run()
    assert not app.exception, [e.value for e in app.exception]


def tab(app, label):
    return next(item for item in app.tabs if item.label == label)


def frame(block, column):
    return next(df.value for df in block.dataframe if column in df.value.columns)


def card_title(app) -> str:
    return next(m.value.rsplit("</span>", 1)[-1] for m in app.markdown
                if m.value.endswith("**") and "Light R-CNN" in m.value)


def metrics(app, suffix):
    return [m.value for m in app.metric if m.label.endswith(suffix)]


def test_combined_v2_v3_run_opens_on_the_newest_checkpoint_without_a_warning(dashboard, tmp_path):
    app = dashboard(write_run(tmp_path))
    selector = box(app, CHECKPOINT_LABEL)
    assert selector.options == ["Light R-CNN v3", "Light R-CNN v2"]
    assert selector.value == SHA["v3"]
    assert not any("체크포인트" in w.value for w in app.warning)
    assert not any("체크포인트" in e.value for e in app.error)
    assert "결과 폴더에 Light R-CNN 체크포인트 2개의 결과가 함께 있어 고른 체크포인트만 보여 줍니다." in [
        c.value for c in app.sidebar.caption]
    assert card_title(app) == "**Light R-CNN v3**"
    assert metrics(app, "AP@[.50:.95]") == ["34.9%"] and metrics(app, "지연") == ["128 ms"]
    assert any("- **Light R-CNN 체크포인트**:" in m.value for m in app.markdown)  # '그래프 읽는 법'

    choose(app, CHECKPOINT_LABEL, SHA["v2"])
    assert card_title(app) == "**Light R-CNN v2**"
    assert metrics(app, "AP@[.50:.95]") == ["42.2%"] and metrics(app, "지연") == ["113 ms"]
    overview = frame(tab(app, "개선 전/후"), "AP 변화 (%p)")
    assert list(overview["모델"]) == ["Light R-CNN v2", "Light R-CNN v2"]
    assert list(overview["p50 95% CI"]) == ["-39.0 ~ -30.8%", "-54.6 ~ -48.1%"]

    choose(app, "추론 설정", "fast")
    improvement = tab(app, "개선 전/후")
    deltas = frame(improvement, "판정")
    assert list(deltas.columns) == ["모델", "지표", "개선 전", "개선 후", "변화", "상대 변화 (%)",
                                    viz.CI_COLUMN, "판정"]
    p50 = deltas[deltas["지표"] == "p50 (ms)"].iloc[0]
    assert (p50["모델"], p50[viz.CI_COLUMN], p50["판정"]) == ("Light R-CNN v2", "-54.6 ~ -48.1%", "좋아짐")
    assert set(deltas[deltas["지표"] != "p50 (ms)"][viz.CI_COLUMN]) == {""}  # blank, not None
    spec = json.loads(improvement.get("plotly_chart")[0].proto.spec)
    labels = [text for trace in spec["data"] for text in trace.get("text") or []]
    assert "-58 ms (-52%)<br>95% CI -55~-48%" in labels
    assert spec["layout"]["yaxis"]["categoryarray"] == ["Light R-CNN v2"]

    choose(app, CHECKPOINT_LABEL, SHA["v3"])  # the inference setting stays
    assert box(app, "추론 설정").value == "fast" and card_title(app) == "**Light R-CNN v3**"
    assert metrics(app, "지연") == ["50.8 ms"]
    deltas = frame(tab(app, "개선 전/후"), "판정")
    p50 = deltas[deltas["지표"] == "p50 (ms)"].iloc[0]
    assert (p50["모델"], p50[viz.CI_COLUMN]) == ("Light R-CNN v3", "-61.6 ~ -58.5%")


def test_no_ci_column_when_before_and_after_come_from_different_runs(dashboard, tmp_path):
    runs = {key: "run1" for key in RUN} | {(version, "fast"): "run2" for version in ("v2", "v3")}
    app = dashboard(write_run(tmp_path, run=runs))
    overview = frame(tab(app, "개선 전/후"), "AP 변화 (%p)")
    assert "p50 95% CI" in overview.columns  # deploy still pairs with the baseline
    assert list(overview["p50 95% CI"]) == ["-42.8 ~ -34.1%", ""]  # a blank cell, not None
    choose(app, "추론 설정", "fast")
    deltas = frame(tab(app, "개선 전/후"), "판정")
    assert viz.CI_COLUMN not in deltas.columns
    assert list(deltas.columns) == ["모델", "지표", "개선 전", "개선 후", "변화", "상대 변화 (%)", "판정"]
    labels = [text for trace in json.loads(tab(app, "개선 전/후").get("plotly_chart")[0].proto.spec)["data"]
              for text in trace.get("text") or []]
    assert not any("CI" in text for text in labels)


def test_caption_states_the_verdict_rules_and_the_ci_scope(dashboard, tmp_path):
    app = dashboard(write_run(tmp_path))
    choose(app, "추론 설정", "fast")
    caption = next(c.value for c in tab(app, "개선 전/후").caption if c.value.startswith("변화 = "))
    assert f"{viz.NOISE_AP_PP:g}%p보다 작으면 '변화 없음 (오차 범위)'" in caption
    assert f"상대 변화가 {viz.NOISE_RELATIVE_PCT:g}%보다 작으면" in caption
    assert "CI가 0을 포함하면 '변화 없음 (오차 범위)'" in caption
    assert "한 번의 실행 안에서" in caption and "다른 날이나 다른 기계" in caption
    assert "128장 스모크 세트의 AP 오차는 ±0.5~1%p" in caption
    assert caption.count("~") == 1  # a second '~' would strike the text between through


def test_cuda_graph_that_did_not_replay_is_flagged_in_the_applied_options(dashboard, tmp_path):
    numbers = {("v2", "baseline"): (0.4222, 6.0, None), ("v2", "gpu_graph"): (0.4222, 5.9, None)}
    record = {"preset": "gpu_graph", "label": "GPU CUDA 그래프 (CUDA 전용)",
              "summary": "Conv+BN 융합 · 640×640 고정 캔버스 · CUDA 그래프", "device": "cuda:0",
              "cuda_graph_status": {"status": "disabled (graph replay differs from eager)",
                                    "replays": 0, "eager_calls": 40}}
    extra = {"gpu_graph": {"inference_options": record, "variant_label": record["label"]}}
    app = dashboard(write_run(tmp_path, numbers=numbers, environment=T4, extra=extra))
    choose(app, "추론 설정", "gpu_graph")
    improvement = tab(app, "개선 전/후")
    note = reporting.cuda_graph_note(record)
    assert note.startswith("CUDA 그래프 사용 안 됨")
    assert note in [w.value for w in improvement.warning]
    assert record["summary"] in [c.value for c in improvement.caption]

    record["cuda_graph_status"]["status"] = "captured"  # a replayed graph needs no warning
    write_run(tmp_path, numbers=numbers, environment=T4, extra=extra)
    app = dashboard(tmp_path / "reports")
    choose(app, "추론 설정", "gpu_graph")
    assert not any("CUDA 그래프" in w.value for w in tab(app, "개선 전/후").warning)


def test_checkpoints_measured_with_different_presets(dashboard, tmp_path):
    numbers = {key: value for key, value in RUN.items()
               if key in {("v2", "baseline"), ("v2", "fast"), ("v3", "baseline"), ("v3", "deploy")}}
    app = dashboard(write_run(tmp_path, numbers=numbers))
    assert box(app, "추론 설정").options == [BASELINE_VARIANT_LABEL, LABELS["deploy"]]  # v3's presets
    sources = frame(tab(app, "데이터·출처"), "파일")
    assert set(sources["추론 설정"]) == {BASELINE_VARIANT_LABEL, LABELS["deploy"], "빠른 설정"}
    choose(app, CHECKPOINT_LABEL, SHA["v2"])
    assert box(app, "추론 설정").options == [BASELINE_VARIANT_LABEL, "빠른 설정"]


def test_a_smoke_run_under_outputs_pairs_with_its_own_baseline_not_the_training_import(dashboard,
                                                                                       tmp_path):
    """compare_improvements.py writes to outputs/improvements/<시각>/reports, inside the
    dashboard's default folder, where the v2 training import (val2017 5000장, 배치 4) outranks the
    run's 128-image baseline: the 'before' must still come from the run itself."""
    write_run(tmp_path / "improvements" / "20261102_174005")
    imported = build_report("accuracy", "light_rcnn", weights="outputs/coco_4090_v2/best.pth",
                            weights_sha256=SHA["v2"], batch_size=4, image_size=640, environment={},
                            accuracy={"evaluator": "pycocotools", "images_evaluated": 5000,
                                      "images_total": 5000, "is_partial": False,
                                      "AP_50_95": 0.2882, "AP_50": 0.4777, "AP_75": 0.2992})
    imported["created_at"] = "2026-10-19T12:40:59+00:00"
    write_report(imported, tmp_path / "reports" / "light_rcnn_accuracy_training.json")
    app = dashboard(tmp_path)
    choose(app, CHECKPOINT_LABEL, SHA["v2"])
    assert metrics(app, "AP@[.50:.95]") == ["28.8%"]  # the baseline card still shows the import
    overview = frame(tab(app, "개선 전/후"), "AP 변화 (%p)")
    assert list(overview["비교"]) == ["가능", "가능"]
    assert list(overview["AP 변화 (%p)"]) == pytest.approx([0.0, 0.02])
    choose(app, "추론 설정", "fast")
    improvement = tab(app, "개선 전/후")
    assert not any("측정 조건이 다릅니다" in alert.value
                   for alert in [*improvement.error, *improvement.warning])
    deltas = frame(improvement, "판정")
    ap = deltas[deltas["지표"] == "AP@[.50:.95] (%)"].iloc[0]
    assert (ap["개선 전"], ap["판정"]) == (pytest.approx(42.22), "변화 없음 (오차 범위)")
    p50 = deltas[deltas["지표"] == "p50 (ms)"].iloc[0]
    assert p50[viz.CI_COLUMN] == "-54.6 ~ -48.1%"


def test_a_one_version_folder_names_the_version_in_tables_charts_and_the_html_report(
        dashboard, tmp_path, monkeypatch):
    pages, original = [], viz.html_report
    monkeypatch.setattr(viz, "html_report",
                        lambda *args, **kwargs: pages.append(original(*args, **kwargs)) or pages[-1])
    app = dashboard(write_run(tmp_path) / "v2")  # summary.md: '버전별 폴더를 열면 그 버전만 봅니다'
    assert box(app, CHECKPOINT_LABEL) is None
    choose(app, "추론 설정", "fast")
    assert card_title(app) == "**Light R-CNN v2**"
    assert set(frame(tab(app, "개선 전/후"), "판정")["모델"]) == {"Light R-CNN v2"}
    assert set(frame(tab(app, "데이터·출처"), "AP@[.50:.95] (%)")["모델"]) == {"Light R-CNN v2"}
    assert ("<title>Light R-CNN 성능 비교 보고서 (Light R-CNN v2 · 빠른 설정 · "
            "Intel(R) Xeon(R) Processor @ 2.10GHz)</title>") in pages[-1]
    assert "<td>Light R-CNN v2</td>" in pages[-1]


def test_without_a_baseline_the_caption_says_so_instead_of_a_checkpoint_change(dashboard, tmp_path):
    numbers = {key: value for key, value in RUN.items() if key in {("v2", "deploy"), ("v2", "fast")}}
    app = dashboard(write_run(tmp_path, numbers=numbers) / "v2")
    choose(app, "추론 설정", "fast")
    caption = next(c.value for c in tab(app, "개선 전/후").caption if c.value.startswith("비교:"))
    assert "체크포인트: Light R-CNN v2 (개선 전 결과 없음)" in caption
    assert "→ Light R-CNN v2" not in caption
