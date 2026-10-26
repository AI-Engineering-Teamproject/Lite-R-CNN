import contextlib
import io

import pytest
from pycocotools.coco import COCO

from light_rcnn.coco_eval import STAT_NAMES, summarize_detections

# image 1: two persons + a dog, image 2: two dogs, image 3: a cat only
BOXES = [(1, 1, [10, 10, 40, 60]), (1, 1, [100, 20, 40, 60]), (1, 2, [60, 120, 50, 40]),
         (2, 2, [10, 10, 50, 40]), (2, 2, [120, 120, 50, 40]), (3, 3, [30, 30, 60, 60])]


@pytest.fixture
def coco_gt():
    coco = COCO()
    coco.dataset = {
        "images": [{"id": i, "width": 200, "height": 200} for i in (1, 2, 3)],
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "dog"},
                       {"id": 3, "name": "cat"}],
        "annotations": [{"id": n, "image_id": img, "category_id": cat, "bbox": box,
                         "area": box[2] * box[3], "iscrowd": 0}
                        for n, (img, cat, box) in enumerate(BOXES, 1)],
    }
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def perfect(image_ids):
    return [{"image_id": img, "category_id": cat, "bbox": box, "score": 0.9}
            for img, cat, box in BOXES if img in image_ids]


def score(coco_gt, detections, image_ids):
    with contextlib.redirect_stdout(io.StringIO()):
        return summarize_detections(coco_gt, detections, image_ids)


def test_perfect_detections_score_one_per_class_on_image_subset(coco_gt):
    summary = score(coco_gt, perfect({1, 2}), [1, 2])
    assert summary["AP_50_95"] == pytest.approx(1.0)
    # two boxes per class per image: a maxDets=1 slice would drop these well below 1.0
    assert summary["per_category_ap"] == pytest.approx({"person": 1.0, "dog": 1.0})
    assert summary["detections"] == 5


def test_image_subset_decides_which_classes_count(coco_gt):
    summary = score(coco_gt, perfect({1, 2}), [1, 2, 3])
    assert summary["per_category_ap"] == pytest.approx({"person": 1.0, "dog": 1.0, "cat": 0.0})
    assert summary["AP_50_95"] == pytest.approx(2 / 3)


def test_empty_detections_keep_the_full_summary_shape(coco_gt):
    summary = score(coco_gt, [], [1, 2])
    keys = set(STAT_NAMES) | {"per_category_ap", "detections", "coco_max_dets"}
    assert set(summary) == set(score(coco_gt, perfect({1, 2}), [1, 2])) == keys
    # every box is medium-sized, so the small/large stats have no ground truth (None)
    assert all(summary[name] in (0.0, None) for name in STAT_NAMES)
    assert summary["per_category_ap"] == {"person": 0.0, "dog": 0.0}
    assert summary["detections"] == 0
