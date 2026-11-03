import json

from light_rcnn import reporting
from light_rcnn.reporting import (
    build_report,
    comparability_warnings,
    latency_summary,
    load_reports,
    normalize,
    select_reports,
)


def cpu_env(device_name="Xeon", threads=4):
    return {"device": "cpu", "device_type": "cpu", "device_name": device_name, "torch_num_threads": threads}


def latency(model="light_rcnn", device_name="Xeon", top=None, **section):
    samples = [10.0 + i * 0.1 for i in range(200)]
    return build_report("latency", model, precision="fp32", image_size=640, environment=cpu_env(device_name),
                        latency={"scope_id": reporting.SCOPE_HOST_TO_DETECTIONS, "memory_format": "contiguous",
                                 **latency_summary(samples), **section}, **(top or {}))


def accuracy(model="light_rcnn", **top):
    return build_report("accuracy", model, precision="fp32", image_size=640, environment={},
                        accuracy={"evaluator": "pycocotools", "images_evaluated": 5000, "images_total": 5000,
                                  "is_partial": False, "score_threshold": 0.001, "AP_50_95": 0.3}, **top)


def stages(model, device_name="Xeon"):
    return build_report("stages", model, precision="fp32", image_size=640, environment=cpu_env(device_name),
                        stages={"iterations": 200, "items": []})


def warnings_for(*raw):
    reports = [report for index, item in enumerate(raw) for report in normalize(item, f"r{index}")]
    return comparability_warnings(select_reports(reports))


def texts(warnings, level=None):
    return [w.message for w in warnings if level is None or w.level == level]


def test_memory_format_mismatch_is_an_error_and_missing_format_is_flagged():
    mixed = warnings_for(latency("light_rcnn"), latency("yolo26", memory_format="channels_last"))
    assert any("메모리 형식이 모델마다" in m for m in texts(mixed, "error"))
    same = warnings_for(latency("light_rcnn"), latency("yolo26"))
    assert not any("메모리 형식" in m for m in texts(same))
    fused = warnings_for(latency("light_rcnn", conv_bn_fused=False), latency("yolo26", conv_bn_fused=True))
    assert any("Conv+BN" in m for m in texts(fused, "warning"))
    unrecorded = warnings_for(latency("light_rcnn"), latency("yolo26", memory_format=None))
    assert any("기록되지 않았습니다 (YOLO26)" in m for m in texts(unrecorded, "warning"))


def test_latency_run_settings_are_compared():
    found = warnings_for(latency("light_rcnn", score_threshold=0.001, image_count=8),
                         latency("yolo26", score_threshold=0.25, image_count=16))
    warned = texts(found, "warning")
    assert any("점수 임계값" in m for m in warned)
    assert any("이미지 수가 다릅니다 (8, 16장)" in m for m in warned)
    same = warnings_for(latency("light_rcnn", score_threshold=0.25), latency("yolo26", score_threshold=0.25))
    assert not any("점수 임계값" in m or "이미지 수" in m for m in texts(same))


def test_accuracy_and_latency_from_different_checkpoints_are_an_error():
    found = warnings_for(accuracy(weights_sha256="old"), latency(top={"weights_sha256": "new"}))
    errors = [m for m in texts(found, "error") if "가중치" in m]
    assert len(errors) == 1 and "Light R-CNN" in errors[0]
    same = warnings_for(accuracy(weights_sha256="abc"), latency(top={"weights_sha256": "abc"}))
    assert not any("가중치" in m for m in texts(same))
    # v1 files carry only a path; a path is not compared against a v2 hash.
    legacy = warnings_for({"model": "light_rcnn", "weights": "old.pth", "map": 0.3},
                          latency(top={"weights": "new.pth", "weights_sha256": "abc"}))
    assert not any("가중치" in m for m in texts(legacy))
    paths_only = warnings_for({"model": "light_rcnn", "weights": "a.pth", "map": 0.3},
                              {"model": "light_rcnn", "weights": "b.pth", "p50_ms": 80})
    assert any("가중치 경로가 다릅니다 (a.pth, b.pth)" in m for m in texts(paths_only, "warning"))
    assert not any("가중치" in m for m in texts(paths_only, "error"))


def test_cpu_note_is_a_warning_and_uses_the_given_target():
    found = comparability_warnings(select_reports(normalize(latency(), "a")), latency_target_ms=30)
    [note] = [w for w in found if "CPU에서 측정한" in w.message]
    assert note.level == "warning" and "30 ms" in note.message and "15 ms" not in note.message
    plain = comparability_warnings(select_reports(normalize(latency(), "a")))
    assert not any("ms)" in m for m in texts(plain) if "CPU에서 측정한" in m)


def test_malformed_files_are_listed_instead_of_crashing(tmp_path):
    reporting.write_report(accuracy(), tmp_path / "valid.json")
    bad = {
        "str_images.json": {"model": "yolo26", "map": 0.4, "images_evaluated": "128"},
        "int_device.json": {"model": "yolo26", "device": 0, "p50_ms": 5},
        "cpu_device.json": {"model": "light_rcnn", "device": "cpu", "p50_ms": 80},
        "list_size.json": {"model": "faster_rcnn", "map": 0.2, "imgsz": [640, 640]},
        "bad_section.json": {"schema_version": 2, "kind": "latency", "model": "faster_rcnn", "latency": "fast"},
        "bad_env.json": {**latency("faster_rcnn"), "environment": "cpu"},
    }
    for name, payload in bad.items():
        (tmp_path / name).write_text(json.dumps(payload))
    upload = json.dumps({"schema_version": 2, "kind": "stages", "model": "light_rcnn", "stages": [1]})
    reports, errors = load_reports(tmp_path, [("upload.json", upload.encode())])
    assert sorted(e.split(":")[0].rsplit("/", 1)[-1] for e in errors) == [
        "bad_env.json", "bad_section.json", "upload.json"]
    assert ("accuracy", "light_rcnn") in {(r.kind, r.model) for r in reports}
    selected = select_reports(reports)
    found = comparability_warnings(selected)
    assert selected[("accuracy", "yolo26")].section["images_evaluated"] == 128
    assert any("서로 다른 장치" in m and "cpu / cuda" in m for m in texts(found, "error"))
    assert any("입력 크기" in m for m in texts(found))


def test_string_image_counts_do_not_break_selection():
    first = accuracy("yolo26")
    first["accuracy"]["images_evaluated"] = "128"
    reports = [*normalize(first, "a"), *normalize(accuracy("yolo26"), "b")]
    assert select_reports(reports)[("accuracy", "yolo26")].source == "b"


def test_stage_reports_are_checked_for_device():
    base = [latency("faster_rcnn"), latency("light_rcnn")]
    baseline = texts(warnings_for(*base))
    assert texts(warnings_for(*base, stages("faster_rcnn"), stages("light_rcnn"))) == baseline
    mixed = warnings_for(*base, stages("faster_rcnn", "Xeon"), stages("light_rcnn", "Tesla T4"))
    assert any(m.startswith("단계별 처리 시간을 서로 다른 장치") for m in texts(mixed, "error"))


def test_v1_device_strings_are_not_read_as_hardware_names():
    v2 = [latency("light_rcnn"), latency("yolo26")]
    found = warnings_for({"model": "faster_rcnn", "device": "cpu", "p50_ms": 900}, *v2)
    assert not any("서로 다른 장치" in m for m in texts(found))
    assert any("확인할 수 없습니다" in m for m in texts(found, "warning"))
    gpu = warnings_for({"model": "faster_rcnn", "device": "cuda", "p50_ms": 9}, *v2)
    assert any("서로 다른 장치" in m for m in texts(gpu, "error"))


def test_zero_detection_latency_is_flagged():
    found = warnings_for(latency("light_rcnn", mean_detections=0.0), latency("yolo26", mean_detections=2.7))
    assert any("검출이 0개" in m and "Light R-CNN" in m for m in texts(found, "warning"))
    fine = warnings_for(latency("light_rcnn", mean_detections=1.0), latency("yolo26", mean_detections=2.7))
    assert not any("검출이 0개" in m for m in texts(fine))


def test_utf8_bom_files_and_uploads_are_read(tmp_path):
    payload = b"\xef\xbb\xbf" + json.dumps(latency(), ensure_ascii=False).encode("utf-8")
    (tmp_path / "bom.json").write_bytes(payload)
    reports, errors = load_reports(tmp_path)
    assert errors == [] and len(reports) == 1
    reports, errors = load_reports(tmp_path / "missing", [("a.json", payload),
                                                          ("b.json", payload.decode("utf-8"))])
    assert errors == [] and len(reports) == 2


def partial_accuracy(model, ids_hash=None, seed=42, batch_size=1):
    report = accuracy(model, batch_size=batch_size)
    report["accuracy"].update({"images_evaluated": 500, "is_partial": True, "image_ids_sha256": ids_hash,
                               "subset_selection": "seeded_random", "subset_seed": seed})
    return report


def test_partial_runs_on_different_images_are_an_error():
    errors = texts(warnings_for(partial_accuracy("light_rcnn", "aaa"), partial_accuracy("yolo26", "bbb")), "error")
    assert any("평가한 이미지가 다릅니다" in text for text in errors)
    same = texts(warnings_for(partial_accuracy("light_rcnn", "aaa"), partial_accuracy("yolo26", "aaa")), "error")
    assert not any("이미지" in text for text in same)


def test_subset_seed_is_compared_when_a_hash_is_missing():
    errors = texts(warnings_for(partial_accuracy("light_rcnn", seed=42), partial_accuracy("yolo26", seed=7)), "error")
    assert any("seed" in text for text in errors)


def test_accuracy_batch_size_mismatch_is_flagged():
    found = texts(warnings_for(accuracy("light_rcnn", batch_size=4), accuracy("yolo26", batch_size=1)), "warning")
    assert any("배치 크기" in text for text in found)


def test_only_t4_timings_count_as_on_target():
    gpu_env = {"device": "cuda:0", "device_type": "cuda", "device_name": "NVIDIA GeForce RTX 4090"}
    rtx = latency("light_rcnn")
    rtx["environment"] = gpu_env
    t4 = latency("light_rcnn")
    t4["environment"] = {**gpu_env, "device_name": "Tesla T4"}
    [rtx_report] = normalize(rtx, "rtx")
    [t4_report] = normalize(t4, "t4")
    assert not reporting.on_target_device(rtx_report)
    assert reporting.on_target_device(t4_report)
    assert any("T4가 아닌 장치" in text for text in texts(warnings_for(rtx), "warning"))
    assert not any("T4 GPU 기준" in text for text in texts(warnings_for(t4)))


def test_report_files_are_ascii(tmp_path):
    path = reporting.write_report(accuracy("light_rcnn", note="한글 메모"), tmp_path / "r.json")
    raw = path.read_bytes()
    assert raw.isascii()
    assert json.loads(raw.decode("ascii"))["note"] == "한글 메모"


def test_cpu_runs_do_not_show_a_cuda_runtime():
    text = reporting.describe_environment({"device_type": "cpu", "device_name": "Xeon", "cuda_runtime": "13.0"})
    assert "CUDA" not in text
