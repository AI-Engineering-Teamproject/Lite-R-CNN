import pytest

from light_rcnn import viz
from light_rcnn.reporting import build_report, latency_summary, normalize, select_reports

TARGETS = {"latency_ms": 15, "latency_percentile": "p50", "ap_metric": "AP_50_95", "ap": 0.6}


def selection():
    reports = []
    for index, model in enumerate(("faster_rcnn", "light_rcnn", "yolo26")):
        samples = [20.0 + index * 5 + (i % 7) for i in range(150)]
        reports += normalize(build_report(
            "latency", model, precision="fp32", image_size=640, environment={"device_name": "Tesla T4"},
            latency={"scope_id": "host_uint8_to_detections", **latency_summary(samples)},
            model_stats={"parameters_m": 3.0 + index, "gflops": 4.0 + index}), f"{model}_lat")
        reports += normalize(build_report(
            "accuracy", model, precision="fp32", image_size=640, environment={},
            accuracy={"evaluator": "pycocotools", "images_evaluated": 5000, "images_total": 5000,
                      "is_partial": False, "AP_50_95": 0.3 + index / 10, "AP_50": 0.5, "AP_75": 0.3,
                      "AP_small": 0.1, "AP_medium": 0.3, "AP_large": 0.5,
                      "per_category_ap": {"person": 0.5 + index / 10, "dog": 0.2, "cat": 0.4 - index / 10}}),
            f"{model}_acc")
        if model != "yolo26":
            items = [{"stage": stage, "mean_ms": 1.0 + i, "p50_ms": 1.0 + i}
                     for i, stage in enumerate(("input", "preprocess", "backbone", "rpn", "roi_heads", "postprocess"))]
            reports += normalize(build_report("stages", model, environment={}, stages={"items": items}),
                                 f"{model}_stages")
    return select_reports(reports)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_every_figure_renders_with_full_data(theme):
    figures = viz.build_all(selection(), TARGETS, theme)
    assert set(figures) == set(viz.ALL_FIGURES)
    assert not any(viz.is_empty(fig) for fig in figures.values())
    for fig in figures.values():
        fig.to_json()  # serializes without invalid properties


def test_empty_selection_yields_placeholders():
    figures = viz.build_all({}, TARGETS, "light")
    assert all(viz.is_empty(fig) for fig in figures.values())


def test_model_colors_are_fixed_per_model_not_rank():
    full = selection()
    only_yolo = {key: value for key, value in full.items() if key[1] == "yolo26"}
    color = viz.model_color("yolo26")
    fig = viz.tradeoff_figure(only_yolo, TARGETS)
    assert fig.data[0].marker.color == color


def test_stage_chart_folds_input_into_preprocess():
    fig = viz.stage_figure(selection())
    preprocess = next(trace for trace in fig.data if trace.name == viz.CHART_STAGE_LABELS["preprocess"])
    assert list(preprocess.x) == [3.0, 3.0]  # input 1 ms + preprocess 2 ms


def test_summary_rows_and_html_report():
    selected = selection()
    rows = viz.summary_rows(selected)
    assert [row["모델"] for row in rows] == ["Vanilla Faster R-CNN", "Light R-CNN", "YOLO26"]
    page = viz.html_report(viz.build_all(selected, TARGETS), rows, [], selected, "light", "<script>t</script>")
    assert "<script>t</script>" not in page and "&lt;script&gt;" in page
