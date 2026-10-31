"""Draw separate, source-verified Lite R-CNN v3 training and inference diagrams."""

from __future__ import annotations

import argparse
from pathlib import Path


BG = "#111316"
GRAY = "#D8D8D8"
STUDENT = "#B9E5A9"
STUDENT_HI = "#C9EDBA"
TEACHER = "#B9D9EF"
LOSS = "#F4D88E"
INK = "#15191B"
WHITE = "#F2F4F5"
MUTED = "#C4CBD0"


def setup(title: str, subtitle: str, height: float):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(16, height), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, height)
    ax.axis("off")
    ax.text(0.45, height - 0.45, title, fontsize=20, fontweight="bold", color=WHITE, va="center")
    ax.text(0.48, height - 0.88, subtitle, fontsize=9.8, color=MUTED, va="center")

    def box(x, y, w, h, text, color=STUDENT, *, fs=9, bold=False, radius=0.11):
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0.04,rounding_size={radius}",
            facecolor=color, edgecolor=color, linewidth=1.0, zorder=2,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=INK,
                fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.25, zorder=3)

    def arrow(start, end, color=WHITE, width=1.6, scale=12):
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=scale,
            linewidth=width, color=color, shrinkA=0, shrinkB=0, zorder=1,
        ))

    return fig, ax, box, arrow


def draw_training(output_dir: Path) -> list[Path]:
    height = 13.5
    fig, ax, box, arrow = setup(
        "Lite R-CNN v3  |  학습 구조 (지식 증류)",
        "초록: 학습되는 학생 모델  ·  파랑: 고정 교사  ·  노랑: 손실 계산  |  코드: DistillationTrainer",
        height,
    )

    # Shared image/target transform and fork into student and frozen-teacher branches.
    box(4.6, 11.55, 6.8, 0.66, "COCO 입력 이미지 + 정답 상자/클래스", GRAY, fs=10, bold=True)
    box(3.7, 10.48, 8.6, 0.72,
        "student GeneralizedRCNNTransform  |  이미지/상자 크기 변환\n정규화·패딩 후 같은 이미지 텐서를 학생과 교사에 전달",
        GRAY, fs=8.8)
    arrow((8.0, 11.52), (8.0, 11.22))

    box(0.55, 9.28, 7.25, 0.47, "학생 v3  |  학습되는 Lite R-CNN", GRAY, fs=10, bold=True)
    box(8.2, 9.28, 7.25, 0.47, "교사  |  고정 COCO Faster R-CNN ResNet50-FPN", GRAY, fs=9.7, bold=True)
    arrow((6.3, 10.45), (4.15, 9.8))
    arrow((9.7, 10.45), (11.8, 9.8), color="#8EC1E0")

    # Student forward: features split to supervised proposal/RoI work and feature KD.
    box(0.55, 8.12, 7.25, 0.82,
        "MobileNetV3-Large C2–C5  →  Lite FPN P2–P7\nP2–P5: RoI/feature KD  ·  P2–P7: RPN", STUDENT, fs=8.8)
    arrow((4.18, 9.25), (4.18, 8.97))
    box(0.55, 6.96, 7.25, 0.82,
        "RPN + 정답 상자  →  앵커 점수/박스 손실 + 후보 상자\n후보와 정답 상자를 매칭해 학습 RoI 샘플 선택", STUDENT, fs=8.5)
    arrow((4.18, 8.09), (4.18, 7.81))
    box(0.55, 5.78, 7.25, 0.84,
        "선택된 RoI  →  RoIAlign 7×7  →  LightRoIHead\nFastRCNNPredictor  →  학생 분류 logits / 박스 보정값", STUDENT, fs=8.6)
    arrow((4.18, 6.93), (4.18, 6.65))

    # Teacher forward has no RPN call: it produces feature maps and logits on student RoIs.
    box(8.2, 8.12, 7.25, 0.82,
        "교사 ResNet50-FPN backbone  |  eval + no_grad\n교사 RPN은 실행하지 않음", TEACHER, fs=8.8)
    arrow((11.82, 9.25), (11.82, 8.97), color="#8EC1E0")
    box(8.2, 6.96, 7.25, 0.82,
        "교사 feature maps  |  v3의 P2–P5와 대응하는 4개 레벨\n교사 출력은 gradient를 받지 않음", TEACHER, fs=8.5)
    arrow((11.82, 8.09), (11.82, 7.81), color="#8EC1E0")
    box(8.2, 5.78, 7.25, 0.84,
        "학생이 선택한 같은 학습 RoI를 교사 RoI head에 입력\n교사 class logits 계산  |  교사 box 출력은 증류에 사용하지 않음", TEACHER, fs=8.3)
    arrow((7.83, 6.18), (8.15, 6.18), color="#F3C968", width=1.8)

    # The two branches provide outputs to one clearly grouped loss calculation.
    box(0.55, 4.62, 7.25, 0.66,
        "학생 출력  |  지도 손실 + P2–P5 feature maps + RoI class logits", STUDENT_HI, fs=8.3)
    box(8.2, 4.62, 7.25, 0.66,
        "교사 출력  |  4개 대응 feature maps + 같은 학생 RoI의 class logits", TEACHER, fs=8.3)
    arrow((4.18, 5.76), (4.18, 5.32), color="#8CCB76")
    arrow((11.82, 5.76), (11.82, 5.32), color="#8EC1E0")

    loss_y, loss_h = 2.76, 1.25
    box(1.05, loss_y, 13.9, loss_h,
        "손실 합산  |  train.py는 losses.values()를 합산\n"
        "지도: RPN objectness/box + RoI classification/box     ·     "
        "Feature KD: 정규화 Smooth L1 × 0.5 (4개 레벨)\n"
        "RoI KD: 학생/교사 class-logit KL × 1.0 (temperature 2)",
        LOSS, fs=8.7, bold=True)
    arrow((4.18, 4.60), (6.1, loss_y + loss_h + 0.04), color="#8CCB76")
    arrow((11.82, 4.60), (9.9, loss_y + loss_h + 0.04), color="#8EC1E0")
    box(4.25, 1.48, 7.5, 0.68,
        "역전파/AdamW  →  학생 v3 + feature adapter 갱신  |  교사 가중치는 고정",
        STUDENT_HI, fs=8.9, bold=True)
    arrow((8.0, loss_y - 0.02), (8.0, 2.2))
    box(0.55, 0.3, 14.9, 0.47,
        "v3 설정: student MobileNetV3 feature extractor만 v2 체크포인트에서 초기화  ·  그 외 student head/FPN/RPN은 v3 학습 대상",
        GRAY, fs=8.2)

    return save(fig, output_dir, "lite_rcnn_v3_training")


def draw_inference(output_dir: Path) -> list[Path]:
    height = 12.8
    fig, ax, box, arrow = setup(
        "Lite R-CNN v3  |  추론 구조",
        "학습된 v3 student 체크포인트만 사용  ·  정답 라벨, 손실 계산, teacher branch 없음",
        height,
    )
    cx = 8.0
    box(cx - 2.6, 11.1, 5.2, 0.62, "입력 이미지  |  640 × 640 설정", GRAY, fs=9.8, bold=True)
    stages = [
        ("GeneralizedRCNNTransform\nresize / normalize / batch padding", 10.0, 0.72),
        ("MobileNetV3-Large  |  C2–C5 특징 추출\nstride 4 / 8 / 16 / 32", 8.9, 0.78),
        ("Lite FPN  |  P2–P5 64채널 + P6/P7 max-pool\nP2–P5는 RoI용  ·  P2–P7은 RPN용", 7.72, 0.8),
    ]
    arrow((cx, 11.07), (cx, 10.75))
    for idx, (text, y, h) in enumerate(stages):
        box(cx - 5.15, y, 10.3, h, text, STUDENT, fs=9.2)
        if idx < len(stages) - 1:
            next_y, next_h = stages[idx + 1][1], stages[idx + 1][2]
            arrow((cx, y - 0.02), (cx, next_y + next_h + 0.03))

    # Split features into RPN proposals and the RoI feature path.
    branch_header_y = 6.6
    arrow((cx, 7.69), (4.1, branch_header_y + 0.51))
    arrow((cx, 7.69), (11.9, branch_header_y + 0.51), color="#8CCB76")
    box(0.65, branch_header_y, 6.9, 0.48, "RPN 경로  |  후보 상자 계산", GRAY, fs=9.6, bold=True)
    box(8.45, branch_header_y, 6.9, 0.48, "RoI 경로  |  특징 지도 전달", GRAY, fs=9.6, bold=True)
    box(0.65, 5.15, 6.9, 1.05,
        "AnchorGenerator P2–P7  |  크기 8/16/32/64/128/256\n각 위치 3개 앵커 → RPN objectness/box 예측\n추론: 사전 1,000 → 사후 NMS 최대 300 proposals",
        STUDENT, fs=8.2)
    box(8.45, 5.15, 6.9, 1.05,
        "P2–P5 feature maps 전달 (각 64채널)\nP6/P7은 RoIAlign에는 사용하지 않음",
        STUDENT, fs=8.7)
    arrow((4.1, branch_header_y - 0.02), (4.1, 6.22))
    arrow((11.9, branch_header_y - 0.02), (11.9, 6.22), color="#8CCB76")

    roi_y, roi_h = 4.0, 0.7
    arrow((4.1, 5.13), (6.2, roi_y + roi_h + 0.04))
    arrow((11.9, 5.13), (9.8, roi_y + roi_h + 0.04), color="#8CCB76")
    box(cx - 4.5, roi_y, 9.0, roi_h,
        "두 경로 합류  |  MultiScaleRoIAlign · RoIAlign 7×7 · sampling ratio 2",
        STUDENT_HI, fs=9.0)
    box(cx - 5.15, 2.82, 10.3, 0.78,
        "LightRoIHead  |  1×1 Conv 64→32  →  AdaptiveAvgPool 7×7\nflatten 1,568  →  Linear 1,568→256 + ReLU",
        STUDENT, fs=8.9)
    arrow((cx, roi_y - 0.02), (cx, 3.64))
    box(cx - 5.15, 1.7, 10.3, 0.7,
        "FastRCNNPredictor  |  91 class logits + 364 box deltas",
        STUDENT, fs=9.2)
    arrow((cx, 2.8), (cx, 2.44))
    box(cx - 5.15, 0.82, 10.3, 0.7,
        "score threshold 0.05  →  class-wise NMS IoU 0.50\n최대 100 detections/image  →  원본 이미지 크기로 box 변환",
        STUDENT_HI, fs=8.9)
    arrow((cx, 1.68), (cx, 1.56))
    box(cx - 3.3, 0.08, 6.6, 0.48, "최종 출력  |  boxes · labels · scores", GRAY, fs=9.2, bold=True)
    arrow((cx, 0.8), (cx, 0.59))

    return save(fig, output_dir, "lite_rcnn_v3_inference")


def save(fig, output_dir: Path, stem: str) -> list[Path]:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "svg"):
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=190, bbox_inches="tight", facecolor=BG)
        paths.append(path)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="docs/figures", help="Directory for PNG/SVG diagrams")
    args = parser.parse_args()
    for path in draw_training(Path(args.output_dir)) + draw_inference(Path(args.output_dir)):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
