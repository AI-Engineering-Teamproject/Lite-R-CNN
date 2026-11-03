from pycocotools.coco import COCO

from light_rcnn.coco_eval import STAT_NAMES, summarize_detections


def make_coco(boxes_per_image: dict[int, list[list[float]]]) -> COCO:
    coco = COCO()
    pairs = [(image_id, box) for image_id, boxes in boxes_per_image.items() for box in boxes]
    annotations = [
        {"id": number, "image_id": image_id, "category_id": 1, "bbox": box,
         "area": box[2] * box[3], "iscrowd": 0}
        for number, (image_id, box) in enumerate(pairs, start=1)
    ]
    coco.dataset = {
        "images": [{"id": image_id, "width": 640, "height": 640} for image_id in boxes_per_image],
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "dog"}],
        "annotations": annotations,
    }
    coco.createIndex()
    return coco


def test_empty_detections_follow_cocoeval_path():
    coco = make_coco({1: [[10, 10, 200, 200]], 2: [[50, 50, 300, 250]]})
    summary = summarize_detections(coco, [], [1, 2])
    assert summary["AP_50_95"] == 0.0
    assert summary["AR_100"] == 0.0
    assert summary["AP_small"] is None and summary["AP_medium"] is None
    assert summary["AP_large"] == 0.0
    assert summary["per_category_ap"] == {"person": 0.0}
    assert summary["detections"] == 0
    assert summary["coco_max_dets"] == 100


def test_no_ground_truth_sentinel_is_none():
    coco = make_coco({1: [[10, 10, 200, 200]]})
    detections = [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 200, 200], "score": 0.9}]
    summary = summarize_detections(coco, detections, [1])
    assert summary["AP_50_95"] > 0.99
    assert summary["AP_small"] is None and summary["AR_small"] is None
    assert all(summary[name] is None or summary[name] >= 0 for name in STAT_NAMES)
    assert summary["coco_max_dets"] == 100
