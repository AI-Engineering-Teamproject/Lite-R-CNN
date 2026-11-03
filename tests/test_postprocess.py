import copy
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from torchvision.models.detection._utils import BoxCoder
from torchvision.ops import boxes as box_ops

from light_rcnn.bench import load_comparison_model
from light_rcnn.model.detector import build_detector
from light_rcnn.postprocess import (
    NMS_METHODS,
    SOFT_NMS_MIN_SCORE,
    PostprocessDetections,
    install_postprocess,
    matrix_nms,
    soft_nms,
)

ROOT = Path(__file__).resolve().parents[1]
PHOTO = ROOT / "147753808597_20161028.jpeg"
V2_WEIGHTS = ROOT / "outputs/coco_4090_v2/best.pth"


def soft_nms_one_class(boxes, scores, method, param, min_score=0.001):
    """Verbatim from the investigation's perf/research/postproc_sweep.py."""
    kept_boxes, kept_scores = [], []
    while scores.numel():
        top = int(torch.argmax(scores))
        best_box = boxes[top]
        kept_boxes.append(best_box)
        kept_scores.append(scores[top])
        rest = torch.ones(scores.numel(), dtype=torch.bool)
        rest[top] = False
        boxes, scores = boxes[rest], scores[rest]
        if not scores.numel():
            break
        iou = box_ops.box_iou(best_box[None], boxes)[0]
        if method == "gaussian":
            scores = scores * torch.exp(-(iou * iou) / param)
        else:  # linear, Nt = param
            scores = scores * torch.where(iou > param, 1.0 - iou, torch.ones_like(iou))
        alive = scores > min_score
        boxes, scores = boxes[alive], scores[alive]
    if not kept_boxes:
        return torch.zeros((0, 4)), torch.zeros((0,))
    return torch.stack(kept_boxes), torch.stack(kept_scores)


def reference_soft_nms(boxes, scores, labels, sigma, detections, min_score=0.001):
    """postproc_sweep.py's Gaussian Soft-NMS branch: soft_nms_one_class per label, best first.

    The sweep fed it candidates pre-filtered at > 0.05 and kept soft_nms_one_class's default
    min_score=0.001; the random-set tests below also pass other drop floors.
    """
    out_b, out_s, out_l = [], [], []
    for label in labels.unique():
        sel = labels == label
        b, s = soft_nms_one_class(boxes[sel], scores[sel], "gaussian", sigma, min_score)
        out_b.append(b)
        out_s.append(s)
        out_l.append(torch.full((len(s),), int(label), dtype=torch.int64))
    if out_b:
        boxes, scores, labels = torch.cat(out_b), torch.cat(out_s), torch.cat(out_l)
        order = scores.argsort(descending=True)[:detections]
        boxes, scores, labels = boxes[order], scores[order], labels[order]
    return boxes, scores, labels


def sweep_candidates(heads, class_logits, box_regression, proposals, image_shapes):
    """postproc_sweep.py's steps before Soft-NMS for one image, pre-filtered at the head's floor."""
    boxes = heads.box_coder.decode(box_regression, proposals)
    boxes = box_ops.clip_boxes_to_image(boxes, image_shapes[0])[:, 1:].reshape(-1, 4)
    scores = torch.softmax(class_logits, -1)[:, 1:].reshape(-1)
    labels = torch.arange(1, class_logits.shape[1]).repeat(len(class_logits))
    keep = torch.where(scores > heads.score_thresh)[0]
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
    return boxes[keep], scores[keep], labels[keep]


def random_detections(seed: int, floor: float, clusters: int = 60, copies: int = 5,
                      classes: int = 4):
    """Clusters of jittered near-duplicates (heavy overlaps) plus scattered boxes."""
    generator = torch.Generator().manual_seed(seed)
    corner = torch.rand(clusters, 2, generator=generator) * 560
    size = 16 + torch.rand(clusters, 2, generator=generator) * 180
    base = torch.cat([corner, corner + size], dim=1)
    boxes = base.repeat_interleave(copies, dim=0)
    boxes = boxes + (torch.rand(boxes.shape, generator=generator) - 0.5) * 12
    boxes[:, 2:] = torch.maximum(boxes[:, 2:], boxes[:, :2] + 1)
    labels = torch.randint(1, classes + 1, (clusters,), generator=generator)
    labels = labels.repeat_interleave(copies)
    flip = torch.rand(len(labels), generator=generator) < 0.15  # some same-place other-class boxes
    labels = torch.where(flip, labels % classes + 1, labels)
    scores = floor + (1 - floor) * torch.rand(len(labels), generator=generator)
    return boxes, scores, labels


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("sigma,floor,detections", [
    (0.5, 0.05, 100), (0.5, 0.001, 100), (0.3, 0.05, 25), (0.5, 0.001, 10),
])
def test_soft_nms_matches_the_reference_loop(seed, sigma, floor, detections):
    boxes, scores, labels = random_detections(seed, floor)
    expected_boxes, expected_scores, expected_labels = reference_soft_nms(
        boxes, scores, labels, sigma, detections, min_score=floor)
    keep, kept_scores = soft_nms(boxes, scores, labels, sigma, floor, detections)
    assert len(keep) == len(expected_scores) > 0
    assert torch.equal(boxes[keep], expected_boxes)
    assert torch.equal(labels[keep], expected_labels)
    torch.testing.assert_close(kept_scores, expected_scores, rtol=0, atol=1e-7)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("detections", [100, 10_000])
def test_soft_nms_after_the_roi_prefilter_matches_the_sweep(seed, detections):
    # The swept setting: candidates above the 0.05 RoI floor, decayed scores pruned at 0.001.
    boxes, scores, labels = random_detections(seed, 0.001)  # scores span 0.001..1
    keep = torch.where(scores > 0.05)[0]
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    expected_boxes, expected_scores, expected_labels = reference_soft_nms(
        boxes, scores, labels, 0.5, detections)
    keep, kept_scores = soft_nms(boxes, scores, labels, 0.5, SOFT_NMS_MIN_SCORE, detections)
    assert len(keep) == len(expected_scores) > 0
    assert torch.equal(boxes[keep], expected_boxes)
    assert torch.equal(labels[keep], expected_labels)
    torch.testing.assert_close(kept_scores, expected_scores, rtol=0, atol=1e-7)
    if detections > len(boxes):  # the whole list: duplicates decayed under the RoI floor stay
        assert bool((kept_scores <= 0.05).any())


def test_soft_nms_early_stop_keeps_the_exact_top_list():
    boxes, scores, labels = random_detections(3, 0.001, clusters=120)
    full_keep, full_scores = soft_nms(boxes, scores, labels, 0.5, 0.001, 10_000)
    for detections in (1, 7, 40):
        keep, kept_scores = soft_nms(boxes, scores, labels, 0.5, 0.001, detections)
        assert torch.equal(keep, full_keep[:detections])
        assert torch.equal(kept_scores, full_scores[:detections])


def test_soft_nms_decays_duplicates_and_handles_empty_input():
    boxes = torch.tensor([[0.0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 10, 10]])
    keep, kept = soft_nms(boxes, torch.tensor([0.9, 0.8, 0.7]), torch.tensor([1, 1, 2]), 0.5, 0.05)
    assert keep.tolist() == [0, 2, 1]  # the other label is untouched; the duplicate decays
    torch.testing.assert_close(kept, torch.tensor([0.9, 0.7, 0.8 * math.exp(-2)]))
    empty = torch.zeros((0, 4)), torch.zeros(0), torch.zeros(0, dtype=torch.int64)
    for nms in (soft_nms, matrix_nms):
        keep, kept = nms(*empty)
        assert keep.shape == (0,) and keep.dtype == torch.int64 and kept.shape == (0,)


def test_matrix_nms_keeps_the_top_box_and_decays_same_label_overlaps_only():
    boxes = torch.tensor([
        [0.0, 0, 100, 100],     # top box, label 1
        [10, 0, 110, 100],      # overlaps it, label 1 -> decayed
        [10, 0, 110, 100],      # same place, label 2 -> untouched
        [300, 300, 400, 400],   # label 1, far away -> untouched
    ])
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
    labels = torch.tensor([1, 1, 2, 1])
    keep, kept = matrix_nms(boxes, scores, labels, sigma=0.5, score_thresh=0.05)
    iou = float(box_ops.box_iou(boxes[:1], boxes[1:2]))
    by_index = dict(zip(keep.tolist(), kept.tolist()))
    assert by_index[0] == pytest.approx(0.9)
    assert by_index[1] == pytest.approx(0.8 * math.exp(-iou * iou / 0.5))
    assert by_index[2] == pytest.approx(0.7) and by_index[3] == pytest.approx(0.6)
    assert kept.tolist() == sorted(kept.tolist(), reverse=True)


def test_matrix_nms_compensates_boxes_that_were_suppressed_themselves():
    # B is a near-duplicate of A; C overlaps B more than A. B was itself suppressed by A, so C's
    # decay comes from A alone (Matrix NMS' compensation), not from its larger IoU with B.
    boxes = torch.tensor([[0.0, 0, 100, 100], [5, 0, 105, 100], [60, 0, 160, 100]])
    keep, kept = matrix_nms(boxes, torch.tensor([0.9, 0.85, 0.8]), torch.tensor([1, 1, 1]),
                            sigma=0.5, score_thresh=0.0)
    iou = box_ops.box_iou(boxes, boxes)
    by_index = dict(zip(keep.tolist(), kept.tolist()))
    assert by_index[2] == pytest.approx(0.8 * math.exp(-float(iou[0, 2]) ** 2 / 0.5))
    assert by_index[2] > 0.8 * math.exp(-float(iou[1, 2]) ** 2 / 0.5)


def test_matrix_nms_applies_floor_cap_and_candidate_limit():
    boxes = torch.tensor([[0.0, 0, 10, 10]] * 3 + [[50, 50, 60, 60], [80, 80, 90, 90]])
    scores = torch.tensor([0.9, 0.85, 0.1, 0.5, 0.4])
    labels = torch.ones(5, dtype=torch.int64)
    # Duplicate 1 decays to 0.85 * e^-2 = 0.115 (kept), duplicate 2 to 0.1 * e^-2 (dropped).
    keep, kept = matrix_nms(boxes, scores, labels, sigma=0.5, score_thresh=0.1)
    assert keep.tolist() == [0, 3, 4, 1]
    torch.testing.assert_close(kept, torch.tensor([0.9, 0.5, 0.4, 0.85 * math.exp(-2)]))
    keep, _ = matrix_nms(boxes, scores, labels, sigma=0.5, score_thresh=0.1, max_detections=2)
    assert keep.tolist() == [0, 3]
    keep, _ = matrix_nms(boxes, scores, labels, sigma=0.5, score_thresh=0.0, top_k=3)
    assert keep.tolist() == [0, 3, 1]  # only the three best scores enter the matrix


def test_unknown_method_bad_sigma_and_bad_min_score_are_rejected():
    heads = torch.nn.Module()
    with pytest.raises(ValueError, match="NMS method"):
        PostprocessDetections(heads, "greedy")
    with pytest.raises(ValueError, match="sigma"):
        PostprocessDetections(heads, "soft", sigma=0)
    for min_score in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match="min_score"):
            PostprocessDetections(heads, "soft", min_score=min_score)
    assert PostprocessDetections(heads, "soft").min_score == SOFT_NMS_MIN_SCORE == 0.001
    assert NMS_METHODS == ("hard", "soft", "matrix")


def synthetic_head(seed, classes=5, detections=300):
    """A stand-in RoI head and its postprocess_detections arguments for one 800x800 image."""
    generator = torch.Generator().manual_seed(seed)
    proposals, _, labels = random_detections(seed, 0.001, classes=classes)
    class_logits = torch.randn(len(proposals), classes + 1, generator=generator)
    class_logits[torch.arange(len(proposals)), labels] += 4  # one confident class per box
    box_regression = 0.05 * torch.randn(len(proposals), 4 * (classes + 1), generator=generator)
    heads = SimpleNamespace(box_coder=BoxCoder((10.0, 10.0, 5.0, 5.0)), score_thresh=0.05,
                            nms_thresh=0.5, detections_per_img=detections)
    return heads, (class_logits, box_regression, [proposals], [(800, 800)])


@pytest.mark.parametrize("min_score", [SOFT_NMS_MIN_SCORE, 0.05])
def test_soft_postprocess_prunes_at_min_score_after_the_roi_prefilter(min_score):
    heads, args = synthetic_head(0)
    boxes, scores, labels = PostprocessDetections(heads, "soft", 0.5, min_score)(*args)
    expected = reference_soft_nms(*sweep_candidates(heads, *args), 0.5, heads.detections_per_img,
                                  min_score)
    assert len(scores[0]) > 0
    assert torch.equal(boxes[0], expected[0])
    assert torch.equal(labels[0], expected[2])
    torch.testing.assert_close(scores[0], expected[1], rtol=0, atol=1e-7)
    assert bool((scores[0] > min_score).all())
    if min_score < heads.score_thresh:  # duplicates decayed under the RoI floor stay ranked
        assert bool((scores[0] <= heads.score_thresh).any())


def test_matrix_postprocess_keeps_the_roi_floor_and_ignores_min_score():
    heads, args = synthetic_head(1)
    boxes, scores, labels = PostprocessDetections(heads, "matrix", 0.5, 0.001)(*args)
    candidates = sweep_candidates(heads, *args)
    keep, expected = matrix_nms(*candidates, 0.5, heads.score_thresh, heads.detections_per_img)
    assert len(keep) > 0
    assert torch.equal(boxes[0], candidates[0][keep]) and torch.equal(labels[0], candidates[2][keep])
    assert torch.equal(scores[0], expected)
    assert bool((scores[0] > heads.score_thresh).all())


def load_photo() -> torch.Tensor:
    if not PHOTO.is_file():
        pytest.skip(f"{PHOTO.name} is not in the work tree")
    with Image.open(PHOTO) as image:
        array = np.asarray(image.convert("RGB"))
    return torch.from_numpy(array.copy()).permute(2, 0, 1).float().div(255)


def head_outputs(model, image):
    """The arguments the RoI head hands to postprocess_detections for one image."""
    captured = []
    stock = model.roi_heads.postprocess_detections

    def spy(*args):
        captured.append(args)
        return stock(*args)

    model.roi_heads.postprocess_detections = spy
    with torch.inference_mode():
        model([image])
    del model.roi_heads.postprocess_detections
    return captured[0]


@pytest.fixture(scope="module")
def trained_v2():
    if not V2_WEIGHTS.is_file():
        pytest.skip(f"{V2_WEIGHTS} is not in the work tree")
    config = yaml.safe_load((ROOT / "configs/coco.yaml").read_text(encoding="utf-8"))
    return load_comparison_model("light_rcnn", config, str(V2_WEIGHTS)).eval()


@pytest.fixture(scope="module")
def random_faster_rcnn():
    torch.manual_seed(0)
    model = build_detector("faster_rcnn", pretrained=False, num_classes=91, min_size=320,
                           max_size=320).eval()
    model.roi_heads.score_thresh = 0.001
    return model


@pytest.mark.parametrize("which", ["trained_v2", "random_faster_rcnn"])
def test_hard_replica_is_bit_identical_to_torchvision(which, request):
    model = copy.deepcopy(request.getfixturevalue(which))
    args = head_outputs(model, load_photo())
    heads = model.roi_heads
    with torch.inference_mode():
        stock = type(heads).postprocess_detections(heads, *args)
        replica = PostprocessDetections(heads, "hard")(*args)
    assert len(stock[0][0]) > 0
    for expected, actual in zip(stock, replica):
        assert torch.equal(expected[0], actual[0])


def test_soft_postprocess_on_trained_head_matches_the_reference(trained_v2):
    model = copy.deepcopy(trained_v2)
    heads = model.roi_heads
    args = head_outputs(model, load_photo())
    with torch.inference_mode():
        boxes, scores, labels = PostprocessDetections(heads, "soft", 0.5, 0.001)(*args)
        # As swept: RoI pre-filter at the head's 0.05, decayed scores pruned at 0.001.
        expected = reference_soft_nms(*sweep_candidates(heads, *args), 0.5,
                                      heads.detections_per_img, min_score=0.001)
    assert len(scores[0]) > 0
    assert bool((scores[0] <= heads.score_thresh).any())
    assert torch.equal(boxes[0], expected[0])
    assert torch.equal(labels[0], expected[2])
    torch.testing.assert_close(scores[0], expected[1], rtol=0, atol=1e-7)


@pytest.mark.parametrize("method", ["soft", "matrix"])
def test_installed_decay_nms_respects_the_head_limits(trained_v2, method):
    model = copy.deepcopy(trained_v2)
    install_postprocess(model.roi_heads, method, 0.5)
    model.roi_heads.detections_per_img = 20  # read at call time, after installation
    with torch.inference_mode():
        output = model([load_photo()])[0]
    scores = output["scores"]
    floor = SOFT_NMS_MIN_SCORE if method == "soft" else model.roi_heads.score_thresh
    assert 0 < len(scores) <= 20
    assert bool((scores > floor).all())
    assert torch.equal(scores, scores.sort(descending=True).values)
    install_postprocess(model.roi_heads, "hard")
    assert "postprocess_detections" not in vars(model.roi_heads)


def test_installed_postprocess_follows_a_deep_copied_model(trained_v2):
    model = copy.deepcopy(trained_v2)
    install_postprocess(model.roi_heads, "soft", 0.3, 0.01)
    clone = copy.deepcopy(model)
    installed = clone.roi_heads.postprocess_detections
    assert installed.roi_heads is clone.roi_heads
    assert (installed.method, installed.sigma, installed.min_score) == ("soft", 0.3, 0.01)
