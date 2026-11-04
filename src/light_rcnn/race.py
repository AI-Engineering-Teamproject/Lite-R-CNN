"""Detection race: time several detectors on one photo under the same rules, for the live demo page.

Every racer receives the same decoded RGB image and is timed from that host array to final
detections (the scope the benchmark scripts use), several times, reporting the median. Every
racer uses the same score floor (Soft-NMS's drop floor included) and detection cap, so the race
compares models and inference settings rather than thresholds.

Two race configurations (``RACES``): the model race runs Light R-CNN v2 and v3 with the Conv+BN
fusion and memory-format rule Ultralytics applies to YOLO (the 'deploy' preset) against YOLO26n;
the improvement race runs the same Light R-CNN v2 weights before (model defaults) and after an
inference preset of light_rcnn.inference_opts, with YOLO26n as the reference.
"""

from __future__ import annotations

import dataclasses
import math
import os
import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from light_rcnn.inference_opts import InferenceOptions

    Options = str | InferenceOptions | Mapping[str, Any] | None

COCO_KO = {
    "person": "사람", "bicycle": "자전거", "car": "자동차", "motorcycle": "오토바이", "airplane": "비행기",
    "bus": "버스", "train": "기차", "truck": "트럭", "boat": "보트", "traffic light": "신호등",
    "fire hydrant": "소화전", "stop sign": "정지 표지판", "parking meter": "주차 미터기", "bench": "벤치",
    "bird": "새", "cat": "고양이", "dog": "개", "horse": "말", "sheep": "양", "cow": "소",
    "elephant": "코끼리", "bear": "곰", "zebra": "얼룩말", "giraffe": "기린", "backpack": "배낭",
    "umbrella": "우산", "handbag": "핸드백", "tie": "넥타이", "suitcase": "여행 가방", "frisbee": "프리스비",
    "skis": "스키", "snowboard": "스노보드", "sports ball": "공", "kite": "연", "baseball bat": "야구 방망이",
    "baseball glove": "야구 글러브", "skateboard": "스케이트보드", "surfboard": "서핑보드",
    "tennis racket": "테니스 라켓", "bottle": "병", "wine glass": "와인잔", "cup": "컵", "fork": "포크",
    "knife": "칼", "spoon": "숟가락", "bowl": "그릇", "banana": "바나나", "apple": "사과",
    "sandwich": "샌드위치", "orange": "오렌지", "broccoli": "브로콜리", "carrot": "당근", "hot dog": "핫도그",
    "pizza": "피자", "donut": "도넛", "cake": "케이크", "chair": "의자", "couch": "소파",
    "potted plant": "화분", "bed": "침대", "dining table": "식탁", "toilet": "변기", "tv": "TV",
    "laptop": "노트북", "mouse": "마우스", "remote": "리모컨", "keyboard": "키보드", "cell phone": "휴대폰",
    "microwave": "전자레인지", "oven": "오븐", "toaster": "토스터", "sink": "싱크대",
    "refrigerator": "냉장고", "book": "책", "clock": "시계", "vase": "꽃병", "scissors": "가위",
    "teddy bear": "곰 인형", "hair drier": "헤어드라이어", "toothbrush": "칫솔",
}

RACES = {  # race configuration -> selector label
    "models": "v2 · v3 · YOLO26",
    "improvement": "개선 전 vs 개선 후",
}
MODELS_PRESET = "deploy"  # R-CNN lanes of the model race: Ultralytics' fusion and memory-format rule
BEFORE_PRESET = "baseline"
IMPROVEMENT_PRESET = "fast"  # default 'after' lane: the CPU-validated speed preset
# Plain-language notes for the audience; the exact settings come from the apply record's summary.
PRESET_NOTES = {
    "deploy": "학습 때 따로 계산하던 BatchNorm을 앞의 Conv 가중치에 미리 합치고(결과는 그대로), CPU에서는 "
              "Ultralytics와 같은 규칙으로 channels_last 메모리 배치를 씁니다.",
    "fast": "배포 기본에 더해, RPN이 NMS 전에 살펴보는 후보 상자를 레벨마다 500개로 줄이고 물체일 확률이 "
            "0.05 미만인 후보를 미리 버립니다. 팀의 소규모 검증에서 AP는 거의 그대로였습니다.",
    "accurate": "배포 기본에 더해, 모든 사진을 640×640 고정 캔버스에 넣어 처리하고(RoI 배율 고정) 겹친 상자를 "
                "지우는 대신 점수를 낮추는 Soft-NMS를 씁니다. 정확도(AP)를 올리는 설정이라 시간은 늘 수 "
                "있습니다.",
    "accurate_fast": "정확도 우선과 같지만, 상자를 하나씩 골라 가며 점수를 낮추는 Soft-NMS 대신 한 번의 행렬 "
                     "계산으로 끝내는 Matrix NMS를 씁니다.",
    "gpu_fp16": "배포 기본에 더해, GPU에서 백본과 헤드를 FP16(반정밀도)으로 계산합니다. CUDA 전용입니다.",
    "gpu_graph": "Conv+BN을 합치고, 640×640 고정 캔버스에서 백본 계산을 CUDA 그래프로 녹화해 한 번에 "
                 "재생하므로 GPU 커널 실행 부담이 줄어듭니다. CUDA 전용입니다.",
}

# Lane colors (light theme, dark theme) from the dataviz reference palette. Each race's three
# lanes pass validate_palette.js --pairs all in both themes (v2/red/aqua sits in the CVD 6-8 band,
# legal because every lane is labelled); YOLO26n keeps its color in both races and the improved v2
# keeps v2's. v3 is red, not yellow: the dashboard draws MobileNetV3 Faster R-CNN in yellow.
BLUE = ("#2a78d6", "#3987e5")
ORANGE = ("#eb6834", "#d95926")
AQUA = ("#1baf7a", "#199e70")
RED = ("#e34948", "#e66767")


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in input-image pixels
    score: float
    name: str  # English COCO class name

    @property
    def name_ko(self) -> str:
        return COCO_KO.get(self.name, self.name)


@dataclass
class Racer:
    key: str
    label: str
    color: str
    predict: Callable[[np.ndarray], list[Detection]]  # RGB uint8 HxWx3 -> detections
    sync: Callable[[], None] = lambda: None
    note: str = ""
    inference: dict[str, Any] | None = None  # apply_inference_options record of an R-CNN racer


@dataclass
class RaceResult:
    racer: Racer
    times_ms: list[float]
    detections: list[Detection] = field(default_factory=list)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.times_ms)


@dataclass(frozen=True)
class Lane:
    key: str
    model: str  # 'light_rcnn_v2', 'light_rcnn_v3' or 'yolo26'
    label: str
    colors: tuple[str, str]  # light theme, dark theme
    preset: str | None = None  # inference preset of an R-CNN lane

    def color(self, theme: str = "light") -> str:
        return self.colors[1] if theme == "dark" else self.colors[0]


def race_lanes(race: str, preset: str = IMPROVEMENT_PRESET) -> list[Lane]:
    """Lanes of a race configuration in display order; ``preset`` is the improved lane's."""
    if race == "models":
        return [Lane("light_rcnn_v2", "light_rcnn_v2", "Lite R-CNN v2", BLUE, MODELS_PRESET),
                Lane("light_rcnn_v3", "light_rcnn_v3", "Lite R-CNN v3", RED, MODELS_PRESET),
                Lane("yolo26", "yolo26", "YOLO26n", AQUA)]
    if race == "improvement":
        return [Lane("before", "light_rcnn_v2", "Lite R-CNN v2 개선 전", ORANGE, BEFORE_PRESET),
                Lane("after", "light_rcnn_v2", "Lite R-CNN v2 개선 후", BLUE, preset),
                Lane("yolo26", "yolo26", "YOLO26n (참고)", AQUA)]
    raise ValueError(f"unknown race {race!r}; choose from {', '.join(RACES)}")


def improvement_presets(device: str = "cpu") -> list[str]:
    """Presets offered for the improved lane: all but the baseline, without the CUDA-only ones
    off CUDA (their GPU options would be skipped there, so the lane would not show them)."""
    from light_rcnn.inference_opts import PRESETS

    cuda = str(device).startswith("cuda")
    return [name for name, options in PRESETS.items()
            if name != BEFORE_PRESET and (cuda or not (options.fp16 or options.cuda_graph))]


def coco_category_names() -> list[str]:
    """Index = COCO category id (0..90, 'N/A' for unused ids), as torchvision's R-CNN labels use."""
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

    return list(FasterRCNN_ResNet50_FPN_Weights.COCO_V1.meta["categories"])


def load_rcnn_racer(key: str, label: str, color: str, weights: str | Path, config: dict,
                    device: str = "cpu", score_threshold: float = 0.25, max_detections: int = 100,
                    note: str = "", options: Options = MODELS_PRESET) -> Racer:
    """Light R-CNN (v2 or v3, architecture read from the checkpoint) with an inference preset.

    ``options`` is a preset name, InferenceOptions or mapping (light_rcnn.inference_opts); the
    default 'deploy' is the Conv+BN fusion and memory-format rule Ultralytics applies to YOLO.
    The race's score floor and detection cap replace the options' own, so every racer is scored
    alike. Soft-NMS lowers scores after that floor, so its drop floor becomes the race floor too
    and no box below it comes back; the record still names the preset and says so in ``note``.
    The apply record is kept on ``Racer.inference``.
    """
    import torch

    from light_rcnn.bench import load_comparison_model, set_detection_limits
    from light_rcnn.inference_opts import (
        PRESET_LABELS,
        PRESETS,
        apply_inference_options,
        resolve_options,
    )

    options = dataclasses.replace(resolve_options(options), box_score_thresh=None,
                                  detections_per_img=None)
    preset = next((name for name, value in PRESETS.items() if value == options), None)
    preset_floor = options.soft_nms_min_score
    if options.nms == "soft":
        options = dataclasses.replace(options, soft_nms_min_score=score_threshold)
    torch_device = torch.device(device)
    model = load_comparison_model("light_rcnn", config, str(weights))
    set_detection_limits(model, score_threshold, max_detections)
    record = apply_inference_options(model, options, torch_device)
    if options.soft_nms_min_score != preset_floor:
        if preset is not None:
            record.update(preset=preset, label=PRESET_LABELS[preset])
        record["note"] = (f"레이스 규칙에 맞춰 Soft-NMS 감쇠 하한을 설정 기본값 {preset_floor:g} 대신 "
                          f"점수 하한과 같은 값({score_threshold:g})으로 씁니다. 점수가 깎인 상자도 이 값 "
                          "이하이면 버려, 모든 레인이 같은 기준으로 상자를 남깁니다.")
    names = coco_category_names()

    def predict(image: np.ndarray) -> list[Detection]:
        with torch.inference_mode():
            x = torch.from_numpy(image).permute(2, 0, 1).to(torch_device).float().div_(255)
            output = model([x])[0]
        boxes = output["boxes"].tolist()
        scores = output["scores"].tolist()
        labels = output["labels"].tolist()
        return [Detection(tuple(box), score, names[label] if label < len(names) else str(label))
                for box, score, label in zip(boxes, scores, labels)]

    def sync() -> None:
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)

    return Racer(key, label, color, predict, sync, note, record)


def load_yolo_racer(key: str, label: str, color: str, weights: str | Path, device: str = "cpu",
                    score_threshold: float = 0.25, max_detections: int = 100,
                    image_size: int = 640, note: str = "") -> Racer:
    """YOLO26 through Ultralytics predict, keeping torch's thread count (Ultralytics lowers it)."""
    import torch

    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[2] / "outputs" / ".ultralytics"))
    from ultralytics import YOLO

    threads = torch.get_num_threads()
    model = YOLO(str(weights))
    yolo_device = "cpu" if device == "cpu" else device.replace("cuda:", "").replace("cuda", "0")
    kwargs = {"imgsz": image_size, "device": yolo_device, "conf": score_threshold,
              "max_det": max_detections, "verbose": False}

    def predict(image: np.ndarray) -> list[Detection]:
        result = model.predict(source=image[:, :, ::-1], **kwargs)[0]  # Ultralytics expects BGR arrays
        if torch.get_num_threads() != threads:  # the first predict runs select_device()
            torch.set_num_threads(threads)
        boxes = result.boxes
        return [Detection(tuple(box), score, model.names[int(cls)])
                for box, score, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist())]

    def sync() -> None:
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()

    return Racer(key, label, color, predict, sync, note)


def warm_up(racers: list[Racer], image: np.ndarray, rounds: int = 2) -> None:
    """Build lazily created state (predictors, kernels) so the race does not time the first call."""
    for _ in range(rounds):
        for racer in racers:
            racer.predict(image)
            racer.sync()


def run_race(racers: list[Racer], image: np.ndarray, repeats: int = 3,
             clock: Callable[[], float] = time.perf_counter) -> list[RaceResult]:
    """Time every racer ``repeats`` times on ``image`` in interleaved rounds. Each round starts
    with the next racer, so drift and a slow first call after a pause hit all lanes alike."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    results = {racer.key: RaceResult(racer, []) for racer in racers}
    for index in range(repeats):
        first = index % len(racers) if racers else 0
        for racer in racers[first:] + racers[:first]:
            racer.sync()
            start = clock()
            detections = racer.predict(image)
            racer.sync()
            results[racer.key].times_ms.append((clock() - start) * 1000)
            results[racer.key].detections = detections
    return sorted(results.values(), key=lambda result: result.median_ms)


def prepare_image(image: Image.Image, max_side: int = 1280) -> np.ndarray:
    """RGB uint8 array, downscaled so phone photos do not dominate memory and time."""
    image = image.convert("RGB")
    scale = max_side / max(image.size)
    if scale < 1:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.BILINEAR)
    return np.asarray(image).copy()


def _font(size: int) -> Any:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def draw_detections(image: np.ndarray, detections: list[Detection], color: str,
                    min_score: float = 0.5) -> Image.Image:
    """Boxes in the racer's color with an English label on a dark tab (photos vary too much for
    colored text; Hangul is left to the table because default fonts lack its glyphs)."""
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    width = max(2, round(max(canvas.size) / 320))
    font = _font(max(12, round(max(canvas.size) / 55)))
    for detection in sorted(detections, key=lambda d: d.score):
        if detection.score < min_score:
            continue
        x1, y1, x2, y2 = detection.box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        text = f"{detection.name} {detection.score:.2f}"
        left, top, right, bottom = draw.textbbox((x1, y1), text, font=font)
        tab_top = max(0, y1 - (bottom - top) - 2 * width)
        draw.rectangle([x1, tab_top, x1 + (right - left) + 2 * width, tab_top + (bottom - top) + 2 * width],
                       fill="#0b0b0b")
        draw.text((x1 + width, tab_top + width - (top - y1)), text, fill="#ffffff", font=font)
    return canvas


def _class_counts(detections: list[Detection], min_score: float) -> dict[str, int]:
    counts: dict[str, int] = {}
    for detection in detections:
        if detection.score >= min_score:
            counts[detection.name_ko] = counts.get(detection.name_ko, 0) + 1
    return counts


def summarize(detections: list[Detection], min_score: float = 0.5, top: int = 5) -> tuple[int, str]:
    """(count above min_score, 'Korean name xN' summary of the most frequent classes)."""
    counts = _class_counts(detections, min_score)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top]
    return sum(counts.values()), ", ".join(f"{name} {count}" if count > 1 else name for name, count in ranked)


@dataclass(frozen=True)
class Improvement:
    """Before -> after of one model in the same race; counts are detections above the display floor."""

    before_ms: float
    after_ms: float
    before_count: int
    after_count: int
    class_changes: dict[str, int] = field(default_factory=dict)  # Korean name -> after - before

    @property
    def speedup(self) -> float:
        """How many times faster the improved lane ran (below 1 = slower)."""
        return self.before_ms / self.after_ms if self.after_ms > 0 else math.inf

    @property
    def time_change_pct(self) -> float:
        """Change of the median time in percent; negative = faster."""
        return (self.after_ms - self.before_ms) / self.before_ms * 100 if self.before_ms > 0 else 0.0

    @property
    def count_change(self) -> int:
        return self.after_count - self.before_count

    def class_change_text(self) -> str:
        """'사람 +2, 컵 -1', largest changes first; empty when every class count is unchanged."""
        ranked = sorted(self.class_changes.items(), key=lambda item: (-abs(item[1]), item[0]))
        return ", ".join(f"{name} {change:+d}" for name, change in ranked)


def compare_results(before: RaceResult, after: RaceResult, min_score: float = 0.5) -> Improvement:
    """Speedup and detection-count difference between two lanes of one race."""
    old, new = _class_counts(before.detections, min_score), _class_counts(after.detections, min_score)
    changes = {name: new.get(name, 0) - old.get(name, 0) for name in old.keys() | new.keys()}
    return Improvement(before.median_ms, after.median_ms, sum(old.values()), sum(new.values()),
                       {name: change for name, change in changes.items() if change})
