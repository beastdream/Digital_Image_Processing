"""Side-by-side review sheets for the manual visual inspection of a suite.

For a fixed, seeded sample of raw test images, each sheet shows the ground
truth followed by every experiment's prediction drawn on that experiment's own
preprocessed input, so reviewers judge box alignment and texture preservation
together. Per-panel TP/FP/FN (IoU 0.5) are also written to ``panels.csv``.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.data.dataset_utils import IMAGE_SUFFIXES, project_root
from src.detection.experiment_suite import DEFAULT_SUITE_CONFIG, load_experiment_suite, suite_experiments
from src.detection.ground_truth import (GROUND_TRUTH_COLOR, PREDICTION_COLOR, compare_detections,
                                        draw_labeled_boxes, find_yolo_label, load_yolo_ground_truth)
from src.detection.model_contract import load_experiment_config, preprocess_for_inference
from src.detection.video_predict import detections_from_result
from src.utils.result_paths import reset_directory

PANEL_SIZE = 360
HEADER = 30


def sample_images(image_dir: Path, count: int, seed: int) -> list[Path]:
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if len(images) < count:
        raise ValueError(f"Only {len(images)} images in {image_dir}; need {count}")
    return sorted(random.Random(seed).sample(images, count))


def _panel(image: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    scale = PANEL_SIZE / max(image.shape[:2])
    resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
    canvas = np.full((PANEL_SIZE + HEADER, PANEL_SIZE, 3), 30, np.uint8)
    canvas[HEADER:HEADER + resized.shape[0], :resized.shape[1]] = resized
    cv2.putText(canvas, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if subtitle:
        (width, _), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(canvas, subtitle, (PANEL_SIZE - width - 6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (180, 220, 255), 1, cv2.LINE_AA)
    return canvas


def _grid(panels: list[np.ndarray], columns: int = 4) -> np.ndarray:
    blank = np.full_like(panels[0], 30)
    rows = [panels[index:index + columns] for index in range(0, len(panels), columns)]
    rows[-1] = rows[-1] + [blank] * (columns - len(rows[-1]))
    return np.vstack([np.hstack(row) for row in rows])


def build_review_sheets(suite_path: str = DEFAULT_SUITE_CONFIG, count: int = 10, conf: float = 0.25,
                        split: str = "test") -> Path:
    root = project_root()
    suite = load_experiment_suite(suite_path, root=root)
    seed = int(suite["project"].get("seed", 42))
    project_dir = root / suite["project"]["output"]
    output = reset_directory(root / suite["project"].get("results", "results/experiments") / "visual_inspection",
                             root / "results")
    image_dir = (root / suite["dataset"]["source_yaml"]).parent / "images" / split
    images = sample_images(image_dir, count, seed)

    models = []
    for name, _ in suite_experiments(suite):
        weights = project_dir / name / "weights" / "best.pt"
        if not weights.is_file():
            raise FileNotFoundError(f"{name}: missing {weights}")
        models.append((name, YOLO(str(weights)), load_experiment_config(weights)))

    rows = []
    for number, image_path in enumerate(images, 1):
        raw = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        ground_truth = load_yolo_ground_truth(image_path, find_yolo_label(image_path))
        reference = raw.copy()
        draw_labeled_boxes(reference, ground_truth, "GT", GROUND_TRUTH_COLOR)
        panels = [_panel(reference, f"Ground truth: {image_path.stem}", f"{len(ground_truth)} box")]
        for name, model, contract in models:
            prepared, _ = preprocess_for_inference(raw, contract)
            if prepared.shape[:2] != raw.shape[:2]:
                raise ValueError(f"{name}: preprocessing changed image size; boxes would not align")
            detections = detections_from_result(model.predict(prepared, conf=conf, verbose=False)[0])
            counts = compare_detections(ground_truth, detections)["pothole"]
            canvas = prepared.copy()
            draw_labeled_boxes(canvas, ground_truth, "GT", GROUND_TRUTH_COLOR)
            draw_labeled_boxes(canvas, detections, "Pred", PREDICTION_COLOR)
            panels.append(_panel(canvas, name, f"TP {counts['tp']} FP {counts['fp']} FN {counts['fn']}"))
            rows.append({"sheet": number, "image": image_path.name, "experiment": name, **counts})
        sheet = output / f"sheet_{number:02d}_{image_path.stem}.jpg"
        if not cv2.imwrite(str(sheet), _grid(panels)):
            raise RuntimeError(f"Could not write {sheet}")
    with (output / "panels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sheet", "image", "experiment", "tp", "fp", "fn"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(images)} review sheets to {output}")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create GT-vs-prediction review sheets for every experiment")
    parser.add_argument("--suite", default=DEFAULT_SUITE_CONFIG)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--split", default="test")
    args = parser.parse_args(argv)
    build_review_sheets(args.suite, args.count, args.conf, args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
