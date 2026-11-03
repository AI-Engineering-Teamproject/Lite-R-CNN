from test_viz import TARGETS, selection

from light_rcnn import viz
from light_rcnn.reporting import COLD_START_FIRST_PREDICT, COLD_START_LABELS


def with_cpu_latency(selected):
    for (kind, _), report in selected.items():
        if kind == "latency":
            report.data["environment"] = {"device": "cpu", "device_type": "cpu",
                                          "device_name": "Xeon"}
    return selected


def test_negative_ap_sentinel_is_missing_not_minus_100():
    selected = selection()
    selected[("accuracy", "light_rcnn")].data["accuracy"]["AP_small"] = -1
    assert viz._pct(-1) is None and viz._pct(0.0) == 0.0
    fig = viz.accuracy_figure(selected)
    trace = next(t for t in fig.data if t.name == "Light R-CNN" and "AP 소형" in t.x)
    assert trace.y[list(trace.x).index("AP 소형")] is None
    assert not any("-100" in str(text) for t in fig.data for text in (t.text or []))


def test_p95_basis_has_no_whisker_note_or_whisker():
    fig = viz.tradeoff_figure(selection(), {**TARGETS, "latency_percentile": "p95"})
    assert "수염" not in fig.layout.title.subtitle.text
    assert all(trace.error_x.visible is False for trace in fig.data)
    assert "수염" in viz.tradeoff_figure(selection(), TARGETS).layout.title.subtitle.text


def test_cpu_latency_flags_the_gpu_target():
    gpu = viz.tradeoff_figure(selection(), TARGETS)
    assert not any("CPU" in (a.text or "") for a in gpu.layout.annotations)
    cpu = with_cpu_latency(selection())
    annotations = viz.tradeoff_figure(cpu, TARGETS).layout.annotations
    assert any("CPU" in (a.text or "") for a in annotations)
    for build in (viz.latency_percentile_figure, viz.latency_distribution_figure):
        assert any("T4 GPU" in (a.text or "") for a in build(cpu, TARGETS).layout.annotations)


def test_percentile_axis_padding_and_target_label_on_top():
    fig = viz.latency_percentile_figure(selection(), TARGETS)
    assert tuple(fig.layout.yaxis.range) == (-0.6, 2.6)
    target = next(a for a in fig.layout.annotations if a.text.startswith("목표"))
    assert target.yanchor == "top"


def test_size_stats_pairs_counts_from_one_graph():
    head = {"parameters_m": 2.57, "gflops": 6.24, "deployed_parameters_m": 2.41,
            "deployed_gflops": 5.48}
    interim = {"parameters_m": 2.57, "deployed_parameters_m": 2.41, "gflops": 5.48,
               "checkpoint_gflops": 6.24}
    current = {"parameters_m": 2.41, "checkpoint_parameters_m": 2.57, "gflops": 5.48,
               "checkpoint_gflops": 6.24}
    for stats in (head, interim, current):
        assert viz._size_stats(stats) == (2.41, 5.48)
    assert viz._size_stats({"parameters_m": 3.2, "gflops": None}) == (3.2, None)
    assert viz._size_stats({}) == (None, None)


def test_null_items_class_ap_and_samples_do_not_crash():
    selected = selection()
    selected[("stages", "light_rcnn")].data["stages"]["items"] = None
    yolo_accuracy = selected[("accuracy", "yolo26")].data["accuracy"]
    yolo_accuracy["per_category_ap"] = {"person": None, "dog": 0.2}
    selected[("latency", "faster_rcnn")].data["latency"]["samples_ms"][:3] = [None, "x", None]
    selected[("latency", "yolo26")].data["latency"]["samples_ms"] = None
    figures = viz.build_all(selected, TARGETS)
    for fig in figures.values():
        fig.to_json()
    yolo = next(t for t in figures["per_class"].data if t.name == "YOLO26")
    assert None not in [x for x, y in zip(yolo.x, yolo.y) if y == "dog"]
    assert "YOLO26" not in {t.name for t in figures["latency_distribution"].data}


def test_summary_rows_show_cold_start_scope():
    selected = selection()
    selected[("latency", "yolo26")].data["latency"]["cold_start_scope"] = COLD_START_FIRST_PREDICT
    rows = {row["모델"]: row for row in viz.summary_rows(selected)}
    assert rows["YOLO26"]["첫 추론 범위"] == COLD_START_LABELS[COLD_START_FIRST_PREDICT]
    assert "첫 추론 범위" in rows["Light R-CNN"]


def test_per_class_markers_are_not_clipped():
    fig = viz.per_class_figure(selection())
    assert all(t.cliponaxis is False for t in fig.data if t.mode == "markers")
    assert fig.layout.yaxis.ticklabelstandoff == 8


def test_rolling_median_uses_full_windows_only():
    assert viz._rolling_median([1, 9, 1, 1], 3) == [None, None, 1, 1]
    assert viz._rolling_median([4, 2], 15) == [None, 3]
    selected = selection()
    for (kind, _), report in selected.items():
        if kind == "latency":
            report.data["latency"]["samples_ms"] = report.data["latency"]["samples_ms"][:10]
    assert "최근 10회" in viz.latency_stability_figure(selected).layout.title.subtitle.text


def test_stage_labels_stay_horizontal_and_bars_are_capped():
    fig = viz.stage_figure(selection())
    assert all(t.textangle == 0 for t in fig.data)
    plot_px = fig.layout.height - fig.layout.margin.t - fig.layout.margin.b - viz.AXIS_TITLE_PX
    assert plot_px / 2 * (1 - fig.layout.bargap) <= viz.BAR_MAX_PX + 1e-9
    assert all(t.width == 0.15 for t in viz.accuracy_figure(selection()).data)


def test_efficiency_missing_value_is_labeled_and_bars_have_no_outline():
    selected = selection()
    selected[("latency", "yolo26")].data["model_stats"]["gflops"] = None
    fig = viz.efficiency_figure(selected)
    gflops = fig.data[1]
    index = list(gflops.y).index("YOLO26")
    assert gflops.x[index] == 0 and gflops.text[index] == "측정 안 됨"
    assert gflops.customdata[index] == "측정 안 됨"
    assert all(t.marker.line.width == 0 for t in fig.data)
