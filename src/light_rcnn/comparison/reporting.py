"""Create the requested single comparison table plus inspectable machine-readable results."""

from __future__ import annotations

import csv
from pathlib import Path

from light_rcnn.comparison.protocol import write_json

COLUMNS = (
    "Model",
    "Status",
    "AP",
    "AP50",
    "AP75",
    "AP-small",
    "AP-medium",
    "AP-large",
    "AR@100",
    "Parameters (M)",
    "p50 (ms)",
    "p95 (ms)",
    "Eval images",
    "Latency images",
    "Device",
    "Precision",
    "Input size",
    "Geometry",
)


def comparison_rows(reports: list[dict], config: dict) -> list[dict]:
    rows = []
    for report in reports:
        accuracy, latency = report.get("accuracy", {}), report.get("latency", {})
        row = dict.fromkeys(COLUMNS)
        row.update(
            {
                "Model": report["metadata"]["name"],
                "Status": report["status"],
                "Parameters (M)": (
                    report["metadata"]["parameters"] / 1_000_000
                    if "parameters" in report["metadata"]
                    else None
                ),
                "p50 (ms)": latency.get("p50_ms"),
                "p95 (ms)": latency.get("p95_ms"),
                "Eval images": accuracy.get("images_evaluated"),
                "Latency images": latency.get("image_count"),
                "Device": config["device"],
                "Precision": "fp32",
                "Input size": config["image_size"],
                "Geometry": config["geometry"],
            }
        )
        for column, key in zip(
            ("AP", "AP50", "AP75", "AP-small", "AP-medium", "AP-large", "AR@100"),
            ("AP_50_95", "AP_50", "AP_75", "AP_small", "AP_medium", "AP_large", "AR_100"),
        ):
            value = accuracy.get(key)
            row[column] = 100 * value if value is not None else None
        rows.append(row)
    return rows


def format_cell(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value).replace("|", "\\|")


def save_comparison(output: Path, reports: list[dict], config: dict, manifest: dict) -> None:
    rows = comparison_rows(reports, config)
    write_json(output / "comparison.json", {"manifest": manifest, "models": reports, "table": rows})
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# 내 버전 객체 검출 비교 결과",
        "",
        f"실행 상태: **{manifest['status']}** / 모드: `{config['mode']}`",
        "",
        "AP와 AR은 0–100 포인트, latency는 ms입니다. —는 미측정 또는 해당 GT 없음입니다.",
        "부분 COCO 평가와 미완료 실행은 최종 성능으로 보고하지 마십시오.",
        "",
        "| " + " | ".join(COLUMNS) + " |",
        "| " + " | ".join("---" for _ in COLUMNS) + " |",
    ]
    lines.extend(
        "| " + " | ".join(format_cell(row[column]) for column in COLUMNS) + " |" for row in rows
    )
    lines += [
        "",
        "## 측정 조건",
        "",
        "- 네 모델 모두 batch=1, FP32, 동일한 원본 COCO 이미지 ID와 annotation을 사용합니다.",
        "- CPU RGB uint8 입력부터 원본 좌표의 CPU 검출 결과까지 측정합니다. 파일 디코딩은 제외합니다.",
        "- 평가: 원본 crowd annotation을 보존한 공식 COCOeval, maxDets=[1,10,100].",
        (
            f"- 전처리: `{config['geometry']}`. square는 공통 PIL bilinear resize와 114 패딩을 "
            "포함합니다. native는 프레임워크별 실제 텐서 크기가 다를 수 있습니다."
        ),
        "- YOLO의 native end-to-end/NMS 모드와 각 모델의 proposal 설정은 개별 JSON에 기록합니다.",
        "- 기본 threshold/전처리가 기존 학습 검증과 다를 수 있어 기존 AP 수치와 직접 같지 않을 수 있습니다.",
        (
            "- MobileNet 비교군은 기존 사전학습 모델입니다. 학습 방식과 설정이 다르므로 "
            "구조 하나의 인과 효과를 분리한 ablation은 아닙니다."
        ),
        "- FLOPs, GPU tensor-only latency, 단계별 NMS 시간은 이 표에서 측정하지 않습니다.",
        "- manifest.json: 평가/벤치마크 ID, annotation 해시, 실행 환경 및 설정.",
        "- 각 모델 JSON: 가중치 해시, 파라미터 수, 실제 입력 크기, 개별 지연 시간 표본.",
        "- 각 predictions JSON: 원래 COCO category ID와 원본 이미지 좌표의 예측.",
    ]
    for report in reports:
        if report.get("error"):
            lines += ["", f"실패 `{report['model']}`: {report['error']}"]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", "utf-8")
