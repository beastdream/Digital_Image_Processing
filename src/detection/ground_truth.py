"""Load and draw real YOLO ground-truth annotations for static images."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.data.dataset_utils import CLASS_NAMES, parse_yolo_line, validate_yolo_box

GROUND_TRUTH_COLOR = (0, 200, 0)  # BGR green
PREDICTION_COLOR = (0, 0, 255)  # BGR red


def find_yolo_label(image_path: Path) -> Path | None:
    """Infer ``labels/<split>/<stem>.txt`` from ``images/<split>/<image>``."""
    image_path = Path(image_path)
    parts = list(image_path.parts)
    image_indexes = [index for index, part in enumerate(parts) if part.lower() == "images"]
    if not image_indexes:
        return None
    index = image_indexes[-1]
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def load_yolo_ground_truth(image_path: Path, label_path: Path) -> list[dict]:
    """Read validated normalized YOLO labels and convert them to pixel xyxy."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"Invalid or unreadable image: {image_path}")
    if not label_path.is_file():
        return []

    image_height, image_width = image.shape[:2]
    annotations: list[dict] = []
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            (class_id, xc, yc, width, height), _ = validate_yolo_box(parse_yolo_line(raw_line))
        except ValueError as error:
            raise ValueError(f"Invalid YOLO annotation {label_path}:{line_number}: {error}") from error
        x1 = max(0, min(image_width, round((xc - width / 2) * image_width)))
        y1 = max(0, min(image_height, round((yc - height / 2) * image_height)))
        x2 = max(0, min(image_width, round((xc + width / 2) * image_width)))
        y2 = max(0, min(image_height, round((yc + height / 2) * image_height)))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid YOLO annotation {label_path}:{line_number}: empty pixel bbox")
        annotations.append({
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "xyxy": [x1, y1, x2, y2],
        })
    return annotations


def scale_annotations(annotations: list[dict], source_size: tuple[int, int],
                      target_size: tuple[int, int]) -> list[dict]:
    """Scale annotation xyxy values between (width, height) canvases."""
    source_width, source_height = source_size
    target_width, target_height = target_size
    x_scale, y_scale = target_width / source_width, target_height / source_height
    return [
        {**annotation, "xyxy": [
            round(annotation["xyxy"][0] * x_scale), round(annotation["xyxy"][1] * y_scale),
            round(annotation["xyxy"][2] * x_scale), round(annotation["xyxy"][3] * y_scale),
        ]}
        for annotation in annotations
    ]


def draw_labeled_boxes(image, annotations: list[dict], prefix: str, color: tuple[int, int, int]) -> None:
    """Draw boxes with an explicit source prefix; predictions include confidence."""
    for annotation in annotations:
        x1, y1, x2, y2 = annotation["xyxy"]
        confidence = annotation.get("confidence")
        label = f"{prefix}: {annotation['class_name']}"
        if confidence is not None:
            label += f" {confidence:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_top = max(0, y1 - text_height - 6)
        cv2.rectangle(image, (x1, label_top), (x1 + text_width + 5, y1), color, -1)
        cv2.putText(image, label, (x1 + 2, max(text_height, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_legend(image, include_ground_truth: bool = True, include_predictions: bool = True) -> None:
    entries = []
    if include_ground_truth:
        entries.append(("Green = Ground Truth", GROUND_TRUTH_COLOR))
    if include_predictions:
        entries.append(("Red = Prediction", PREDICTION_COLOR))
    y = 22
    for text, color in entries:
        cv2.rectangle(image, (8, y - 13), (25, y + 2), color, -1)
        cv2.putText(image, text, (31, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 22


def box_iou(first: list[int], second: list[int]) -> float:
    """Return intersection-over-union for two xyxy boxes."""
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def compare_detections(ground_truth: list[dict], predictions: list[dict],
                       iou_threshold: float = 0.5) -> dict[str, dict[str, int]]:
    """Greedily match same-class boxes by descending IoU and report TP/FP/FN."""
    comparison: dict[str, dict[str, int]] = {}
    for class_id, class_name in CLASS_NAMES.items():
        gt_indexes = [index for index, item in enumerate(ground_truth) if item["class_id"] == class_id]
        pred_indexes = [index for index, item in enumerate(predictions) if item["class_id"] == class_id]
        candidates = sorted(
            (
                (box_iou(ground_truth[gt_index]["xyxy"], predictions[pred_index]["xyxy"]),
                 gt_index, pred_index)
                for gt_index in gt_indexes for pred_index in pred_indexes
            ),
            reverse=True,
        )
        matched_gt: set[int] = set()
        matched_predictions: set[int] = set()
        for iou, gt_index, pred_index in candidates:
            if iou < iou_threshold:
                break
            if gt_index not in matched_gt and pred_index not in matched_predictions:
                matched_gt.add(gt_index)
                matched_predictions.add(pred_index)
        comparison[class_name] = {
            "tp": len(matched_gt),
            "fp": len(pred_indexes) - len(matched_predictions),
            "fn": len(gt_indexes) - len(matched_gt),
        }
    return comparison


def render_side_by_side(base_image, ground_truth: list[dict], predictions: list[dict]):
    """Render independent GT and prediction panels with explicit headings."""
    left = base_image.copy()
    right = base_image.copy()
    draw_labeled_boxes(left, ground_truth, "GT", GROUND_TRUTH_COLOR)
    draw_legend(left, include_ground_truth=True, include_predictions=False)
    draw_labeled_boxes(right, predictions, "Pred", PREDICTION_COLOR)
    draw_legend(right, include_ground_truth=False, include_predictions=True)
    if not predictions:
        cv2.putText(right, "Predictions: none", (10, right.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, PREDICTION_COLOR, 2, cv2.LINE_AA)

    header_height = 38
    height, width = base_image.shape[:2]
    canvas = np.zeros((height + header_height, width * 2, 3), dtype=np.uint8)
    canvas[header_height:, :width] = left
    canvas[header_height:, width:] = right
    cv2.putText(canvas, "Ground Truth", (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, GROUND_TRUTH_COLOR, 2, cv2.LINE_AA)
    cv2.putText(canvas, "Model Prediction", (width + 10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, PREDICTION_COLOR, 2, cv2.LINE_AA)
    cv2.line(canvas, (width, 0), (width, height + header_height), (255, 255, 255), 1)
    return canvas
