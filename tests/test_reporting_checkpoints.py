import json
import os
import subprocess
import sys
from pathlib import Path

from light_rcnn import reporting
from light_rcnn.reporting import (
    BASELINE_VARIANT,
    MODELS,
    build_report,
    checkpoint_groups,
    checkpoint_reports,
    checkpoint_warnings,
    checkpoints,
    comparability_warnings,
    improvement_warnings,
    latency_summary,
    load_reports,
    model_label,
    normalize,
    own_folder,
    same_file,
    select_baseline,
    select_reports,
    write_report,
)

CPU = {"device": "cpu", "device_type": "cpu", "device_name": "Xeon", "torch_num_threads": 4}
T4 = {"device": "cuda:0", "device_type": "cuda", "device_name": "Tesla T4"}
SHA = {"v2": "2" * 64, "v3": "3" * 64}
CREATED = {"v2": "2026-11-01T00:00:00+00:00", "v3": "2026-11-02T00:00:00+00:00"}
RUN = "colab_20261105_101010"  # one compare_improvements.py run
WINDOWS_V2 = "C:\\Users\\me\\Lite-R-CNN\\outputs\\coco_4090_v2\\best.pth"


def raw(kind, model="light_rcnn", version="v2", variant=BASELINE_VARIANT, environment=CPU,
        p50=40.0, **top):
    """A compare_improvements.py report: Light R-CNN names its checkpoint, Faster R-CNN does not."""
    light = model == "light_rcnn"
    common = {"precision": "fp32", "image_size": 640, "environment": dict(environment),
              "created_at": CREATED[version], "comparison_run": RUN, "variant": variant,
              "model_tag": version if light else model,
              "model_label": f"Light R-CNN {version}" if light else MODELS[model],
              "weights": f"outputs/coco_4090_{version}/best.pth" if light else "torchvision_coco_v1",
              "weights_sha256": SHA[version] if light else None, **top}
    if kind == "accuracy":
        section = {"evaluator": "pycocotools", "images_evaluated": 5000, "images_total": 5000,
                   "is_partial": False, "AP_50_95": 0.3}
    else:
        section = {"scope_id": reporting.SCOPE_HOST_TO_DETECTIONS, "memory_format": "contiguous",
                   **latency_summary([p50 + i % 3 - 1 for i in range(120)])}
    return build_report(kind, model, **common, **{kind: section})


def load(*items):
    return [report for index, item in enumerate(items) for report in normalize(item, f"r{index}.json")]


def write_run(root: Path, presets=(BASELINE_VARIANT,)) -> None:
    """compare_improvements.py's layout: reports/v2 and reports/v3, each holding the same Faster
    R-CNN files, both versions under the 'light_rcnn' model key."""
    for preset in presets:
        for kind in ("accuracy", "latency"):
            faster = raw(kind, "faster_rcnn", variant=preset, p50=900.0)
            for version in ("v2", "v3"):
                folder = root / "reports" / version
                write_report(faster, folder / f"faster_rcnn_{kind}_{preset}.json")
                write_report(raw(kind, version=version, variant=preset),
                             folder / f"light_rcnn_{kind}_{preset}.json")


def light_sources(selection) -> dict[str, str]:
    return {kind: Path(report.source).parent.name for (kind, model), report in selection.items()
            if model == "light_rcnn"}


def test_two_checkpoints_are_listed_newest_first(tmp_path):
    write_run(tmp_path)
    found, errors = load_reports(tmp_path / "reports")
    assert not errors
    assert checkpoints(found) == [(SHA["v3"], "Light R-CNN v3"), (SHA["v2"], "Light R-CNN v2")]
    assert checkpoints(iter(found)) == checkpoints(found)  # any iterable
    groups = checkpoint_groups(found)["light_rcnn"]
    assert {label: len(members) for label, members in groups.items()} == {
        "Light R-CNN v2": 2, "Light R-CNN v3": 2}
    assert checkpoints([report for report in found if report.model != "light_rcnn"]) == []

    v2 = select_reports(found, checkpoint=SHA["v2"])
    assert light_sources(v2) == {"accuracy": "v2", "latency": "v2"}
    assert select_reports(found, checkpoint="Light R-CNN v2") == v2  # a label works too
    assert light_sources(select_reports(found, checkpoint=SHA["v3"])) == {
        "accuracy": "v3", "latency": "v3"}
    # An unknown checkpoint leaves no Light R-CNN report; the other models stay.
    unknown = select_reports(found, checkpoint="Light R-CNN v9")
    assert set(unknown) == {("accuracy", "faster_rcnn"), ("latency", "faster_rcnn")}


def test_unnamed_checkpoints_take_the_file_stem_and_enough_folders_to_differ():
    def light(version, weights, sha=None, created_at=None):
        report = raw("latency", version=version, weights=weights, weights_sha256=sha)
        report.pop("model_label")
        if created_at:
            report["created_at"] = created_at
        return report

    found = load(light("v2", "outputs/coco_4090_v2/best.pth", "a" * 64),
                 light("v3", "/home/me/Lite-R-CNN/outputs/coco_4090_v3/best.pth"),
                 light("v3", "outputs/coco_4090_v3/best.pth"),  # the same file, relative
                 light("v2", "runs/ablation/epoch12.pth", "b" * 64, "2026-10-01T00:00:00+00:00"))
    assert checkpoints(found) == [
        ("/home/me/Lite-R-CNN/outputs/coco_4090_v3/best.pth", "coco_4090_v3/best"),
        ("a" * 64, "coco_4090_v2/best"),
        ("b" * 64, "epoch12")]
    # Same file name and folder, different contents (retrained in place): the hash tells them apart.
    retrained = load(light("v2", "outputs/coco_4090_v2/best.pth", "c" * 64),
                     light("v3", "outputs/coco_4090_v2/best.pth", "d" * 64))
    assert [label for _, label in checkpoints(retrained)] == [
        f"outputs/coco_4090_v2/best ({'d' * 12})", f"outputs/coco_4090_v2/best ({'c' * 12})"]
    hash_only = light("v2", None, "e" * 64)
    assert checkpoints(load(hash_only)) == [("e" * 64, f"sha256 {'e' * 12}")]


def test_schema_v2_results_rank_ahead_of_newer_old_format_files():
    old = {"model": "light_rcnn", "device": "cpu", "p50_ms": 50.0,
           "weights": "outputs/coco_4090_v3/best.pth"}  # v1: no created_at, so the file time
    found = [*normalize(old, "old.json", mtime=4102444800.0),  # 2100-01-01
             *load(raw("latency", version="v2"))]
    assert [label for _, label in checkpoints(found)] == ["Light R-CNN v2", "best"]


def test_an_old_format_report_naming_the_weights_by_a_windows_path_joins_its_checkpoint(tmp_path):
    write_run(tmp_path)
    (tmp_path / "reports/v2/light_rcnn_latency_baseline.json").unlink()
    (tmp_path / "reports/v2/light_rcnn_latency_v1.json").write_text(json.dumps({
        "model": "light_rcnn", "device": "cpu", "iterations": 10, "p50_ms": 50.0,
        "weights": WINDOWS_V2}), encoding="utf-8")
    found, errors = load_reports(tmp_path / "reports")
    assert not errors
    assert [label for _, label in checkpoints(found)] == ["Light R-CNN v3", "Light R-CNN v2"]
    [v1] = [report for report in found if report.legacy]
    groups = checkpoint_groups(found)["light_rcnn"]
    assert any(member is v1 for member in groups["Light R-CNN v2"])
    v2 = select_reports(found, checkpoint=SHA["v2"])
    assert v2["latency", "light_rcnn"] is v1 and v2["accuracy", "light_rcnn"].data["weights_sha256"]
    assert not select_reports(found, checkpoint=SHA["v3"])["latency", "light_rcnn"].legacy
    assert same_file(WINDOWS_V2.replace("\\", "/"), "outputs/coco_4090_v2/best.pth")
    assert not same_file(WINDOWS_V2.replace("\\", "/"), "outputs/coco_4090_v3/best.pth")


def test_identical_faster_rcnn_copies_of_one_compare_run_count_once(tmp_path):
    write_run(tmp_path, presets=(BASELINE_VARIANT, "fast"))
    found, _ = load_reports(tmp_path / "reports")
    copies = [report for report in found if report.model == "faster_rcnn"]
    assert len(copies) == 8 and {r.data["comparison_run"] for r in copies} == {RUN}
    assert {label: len(members) for label, members in checkpoint_groups(found)["faster_rcnn"].items()
            } == {"torchvision_coco_v1": 8}
    assert [label for _, label in checkpoints(found)] == ["Light R-CNN v3", "Light R-CNN v2"]
    [note] = checkpoint_warnings(found, select_reports(found, variant="fast"))
    assert note.message.startswith("결과 폴더에 Light R-CNN 체크포인트가 2개(Light R-CNN v2, Light R-CNN v3)")
    assert "Faster R-CNN" not in note.message
    # A local Faster R-CNN weights file is hashed; its copies still form one checkpoint.
    hashed = [{**report.data, "weights_sha256": "f" * 64, "weights": "weights/frcnn.pth"}
              for report in copies]
    assert len(checkpoint_groups(load(*hashed))["faster_rcnn"]) == 1
    # Narrowing Light R-CNN to one checkpoint leaves Faster R-CNN's copies as they are.
    for key, _ in checkpoints(found):
        narrowed = checkpoint_reports(found, key)
        assert [r for r in narrowed if r.model == "faster_rcnn"] == copies
        assert select_reports(narrowed, variant="fast") == select_reports(found, variant="fast",
                                                                          checkpoint=key)


def test_variant_device_and_checkpoint_filters_combine():
    found = load(
        raw("accuracy", version="v2"), raw("accuracy", version="v3"),
        raw("accuracy", version="v2", variant="fast"), raw("accuracy", version="v3", variant="fast"),
        raw("latency", version="v2", variant="fast", p50=30.0),
        raw("latency", version="v2", variant="fast", environment=T4, p50=5.0),
        raw("latency", version="v3", variant="fast", environment=T4, p50=4.0),
        raw("latency", version="v3", environment=T4, p50=6.0),
        raw("latency", "faster_rcnn", variant="fast", environment=T4, p50=50.0),
    )
    picked = select_reports(found, device="Tesla T4", variant="fast", checkpoint=SHA["v2"])
    assert set(picked) == {("accuracy", "light_rcnn"), ("latency", "light_rcnn"),
                           ("latency", "faster_rcnn")}
    assert picked["latency", "light_rcnn"].section["p50_ms"] == 5.0
    assert picked["accuracy", "light_rcnn"].data["weights_sha256"] == SHA["v2"]
    assert all(report.variant == "fast" for report in picked.values())
    assert picked["latency", "faster_rcnn"].section["p50_ms"] == 50.0  # untouched by the checkpoint
    assert select_reports(found, device="Xeon", variant="fast",
                          checkpoint="Light R-CNN v2")["latency", "light_rcnn"].section["p50_ms"] == 30.0
    # v3 was never timed with 'fast' on the CPU, and v2 never with the baseline on the T4.
    assert ("latency", "light_rcnn") not in select_reports(found, device="Xeon", variant="fast",
                                                           checkpoint=SHA["v3"])
    assert ("latency", "light_rcnn") not in select_reports(found, device="Tesla T4",
                                                           variant=BASELINE_VARIANT,
                                                           checkpoint=SHA["v2"])
    # Without a checkpoint the newest report still wins, whichever checkpoint it is.
    assert select_reports(found, device="Tesla T4", variant="fast")[
        "latency", "light_rcnn"].data["weights_sha256"] == SHA["v3"]


def test_the_before_of_a_preset_comes_from_its_own_comparison_run():
    """outputs/ holds the v2 training import (5000 images, batch 4, no comparison_run), which
    outranks a smoke compare_improvements.py run's baseline; the run still pairs with its own."""
    imported = raw("accuracy", comparison_run=None, batch_size=4,
                   created_at="2026-10-19T00:00:00+00:00")
    smoke = [raw(kind, variant=variant, batch_size=1)
             for kind in ("accuracy", "latency") for variant in (BASELINE_VARIANT, "fast")]
    for report in smoke:
        if report["kind"] == "accuracy":
            report["accuracy"].update(images_evaluated=128, images_total=128, is_partial=True)
    found = load(imported, *smoke)
    after = select_reports(found, variant="fast", checkpoint=SHA["v2"])
    plain = select_reports(found, variant=BASELINE_VARIANT, checkpoint=SHA["v2"])
    assert plain["accuracy", "light_rcnn"].data["comparison_run"] is None  # the larger one
    assert any("평가 이미지 수 5000 → 128" in w.message for w in improvement_warnings(plain, after))
    before = select_baseline(found, after, checkpoint=SHA["v2"])
    assert {key: report.data["comparison_run"] for key, report in before.items()} == {
        ("accuracy", "light_rcnn"): RUN, ("latency", "light_rcnn"): RUN}
    assert all(report.variant == BASELINE_VARIANT for report in before.values())
    assert improvement_warnings(before, after) == []
    # A run without its baseline, or a preset without a comparison_run, keeps the plain pick.
    orphan = load(imported, *(raw(kind, variant="fast") for kind in ("accuracy", "latency")))
    assert select_baseline(orphan, select_reports(orphan, variant="fast")) == select_reports(
        orphan, variant=BASELINE_VARIANT)
    for report in after.values():
        report.data["comparison_run"] = ""
    assert select_baseline(found, after, checkpoint=SHA["v2"]) == plain


def test_one_file_recorded_absolute_and_as_a_windows_relative_path_is_no_weights_mismatch():
    """The repo's outputs/ records v3 both ways; the selector already shows one checkpoint."""
    accuracy = {"model": "light_rcnn", "AP_50_95": 0.3, "images_evaluated": 5000,
                "weights": "/home/user/Desktop/Lite-R-CNN/outputs/coco_4090_v3/best.pth"}
    latency = {"model": "light_rcnn", "device": "cpu", "p50_ms": 50.0,
               "weights": "outputs\\coco_4090_v3\\best.pth"}

    def weights_warnings():
        found = [*normalize(accuracy, "a.json"), *normalize(latency, "l.json")]
        return found, [w for w in comparability_warnings(select_reports(found)) if "가중치" in w.message]

    found, warnings = weights_warnings()
    assert len(checkpoints(found)) == 1 and warnings == []
    latency["weights"] = "outputs\\coco_4090_v2\\best.pth"  # another file: still flagged
    found, [warning] = weights_warnings()
    assert len(checkpoints(found)) == 2
    assert warning.level == "warning" and "기록된 가중치 경로가 다릅니다" in warning.message


def test_checkpoint_warnings_say_which_checkpoint_is_shown_and_where_each_one_is(tmp_path):
    write_run(tmp_path)
    found, _ = load_reports(tmp_path / "reports")
    v2, v3 = tmp_path / "reports" / "v2", tmp_path / "reports" / "v3"
    [note] = checkpoint_warnings(found, select_reports(found))
    assert note.level == "warning"
    assert ("결과 폴더에 Light R-CNN 체크포인트가 2개(Light R-CNN v2, Light R-CNN v3) 섞여 있어 카드와 "
            "그래프는 그중 Light R-CNN v3만 씁니다.") in note.message
    assert f"Light R-CNN v2 → {v2} · Light R-CNN v3 → {v3}" in note.message
    mixed = {**select_reports(found, checkpoint=SHA["v3"]),
             ("accuracy", "light_rcnn"): select_reports(found, checkpoint=SHA["v2"])[
                 "accuracy", "light_rcnn"]}
    [note] = checkpoint_warnings(found, mixed)
    assert "그중 Light R-CNN v2(정확도), Light R-CNN v3(지연 시간)만 씁니다." in note.message
    [note] = checkpoint_warnings(found, {})
    assert "섞여 있습니다. 카드와 그래프는 모델마다 체크포인트 하나의 결과만 씁니다." in note.message
    single, _ = load_reports(v2)
    assert checkpoint_warnings(single, select_reports(single)) == []


def test_own_folder_and_model_label():
    def at(path, version="v2", label=None):
        item = raw("latency", version=version)
        if label is not None:
            item["model_label"] = label
        return normalize(item, path)[0]

    v2 = [at("run/reports/v2/a.json"), at("run/reports/v2/b.json"), at("업로드: c.json")]
    v3 = [at("run/reports/v3/a.json", "v3")]
    assert own_folder(v2, v3) == Path("run/reports/v2")
    assert own_folder(v2, [*v3, at("run/reports/v2/sub/x.json", "v3")]) is None
    assert own_folder([at("업로드: d.json")], v3) is None
    assert model_label(at("x.json", label=" Light R-CNN v2 ")) == "Light R-CNN v2"
    for label in ("Light R-CNN", "", "  ", 3):  # the bare model name names no checkpoint
        assert model_label(at("x.json", label=label)) is None


TORCH_FREE = """
import importlib.abc, sys

class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("torch", "torchvision"):
            raise ModuleNotFoundError(f"No module named {name!r}")

sys.meta_path.insert(0, NoTorch())
from light_rcnn import reporting

record = {"cuda_graph_status": {"status": "disabled (graph replay differs from eager)"}}
raw = reporting.build_report("latency", "light_rcnn", environment=reporting.collect_environment("cpu"),
                             weights="outputs/coco_4090_v2/best.pth", model_label="Light R-CNN v2",
                             inference_options=record, latency={"p50_ms": 5.0})
found = reporting.normalize(raw, "r.json")
selected = reporting.select_reports(found, checkpoint="Light R-CNN v2")
assert reporting.checkpoints(found) == [("outputs/coco_4090_v2/best.pth", "Light R-CNN v2")]
assert reporting.cuda_graph_note(record).startswith("CUDA 그래프 사용 안 됨")
assert any("CUDA 그래프" in w.message for w in reporting.comparability_warnings(selected))
assert reporting.checkpoint_warnings(found, selected) == []
assert "torch" not in sys.modules and "torch_num_threads" not in raw["environment"]
print("ok")
"""


def test_reporting_works_without_pytorch():
    src = str(Path(reporting.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONIOENCODING": "utf-8",
           "PYTHONPATH": os.pathsep.join([src, os.environ.get("PYTHONPATH", "")])}
    result = subprocess.run([sys.executable, "-c", TORCH_FREE], check=False, env=env,
                            capture_output=True, encoding="utf-8", timeout=120)
    assert result.returncode == 0 and result.stdout.strip() == "ok", result.stderr
