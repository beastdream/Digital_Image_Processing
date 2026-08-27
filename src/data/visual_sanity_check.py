"""Render reproducible post-processing sanity samples with transformed boxes."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from src.data.dataset_utils import CLASS_NAMES, SPLITS, project_root
from src.preprocessing.pipeline import ImagePreprocessingPipeline

COLORS = {0: (0, 0, 255), 1: (255, 0, 0), 2: (0, 255, 0)}

def _load_boxes(path: Path) -> list[list[float]]:
    return [[int(parts[0]), *map(float, parts[1:])] for line in path.read_text(encoding="utf-8").splitlines() if (parts := line.split())]

def _draw(image: np.ndarray, boxes: list[list[float]] | None, title: str) -> np.ndarray:
    canvas = image.copy(); height, width = canvas.shape[:2]
    for class_id, xc, yc, box_width, box_height in boxes or []:
        x1, y1 = int((xc - box_width / 2) * width), int((yc - box_height / 2) * height)
        x2, y2 = int((xc + box_width / 2) * width), int((yc + box_height / 2) * height)
        color = COLORS[int(class_id)]
        cv2.rectangle(canvas, (max(0, x1), max(0, y1)), (min(width - 1, x2), min(height - 1, y2)), color, 2)
        cv2.putText(canvas, CLASS_NAMES[int(class_id)], (max(0, x1), max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
    canvas = cv2.resize(canvas, (320, 320), interpolation=cv2.INTER_LINEAR)
    banner = np.zeros((28, 320, 3), dtype=np.uint8); cv2.putText(banner, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack((banner, canvas))

def _select(images: list[Path], label_dir: Path, count: int, seed: int) -> list[Path]:
    edge, multi, remaining = [], [], []
    for image in images:
        boxes = _load_boxes(label_dir / f"{image.stem}.txt")
        classes = {int(box[0]) for box in boxes}
        at_edge = any(min(box[1] - box[3] / 2, box[2] - box[4] / 2, 1 - (box[1] + box[3] / 2), 1 - (box[2] + box[4] / 2)) < .03 for box in boxes)
        if at_edge: edge.append(image)
        elif len(classes) > 1: multi.append(image)
        else: remaining.append(image)
    random.Random(seed).shuffle(remaining)
    return (edge + multi + remaining)[:count]

def create_visual_sanity_samples(dataset_dir: str | Path | None = None, per_split: int = 10) -> int:
    dataset_dir = Path(dataset_dir) if dataset_dir else project_root() / "data/processed/road_damage_detection"
    output_dir = dataset_dir / "reports/visual_samples"; output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = ImagePreprocessingPipeline(str(project_root() / "configs/experiments/full_preprocessing.yaml"))
    written = 0
    for split in SPLITS:
        image_dir, label_dir = dataset_dir / "images" / split, dataset_dir / "labels" / split
        chosen = _select(sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}), label_dir, per_split, 42 + len(split))
        for image_path in chosen:
            image, boxes = cv2.imread(str(image_path)), _load_boxes(label_dir / f"{image_path.stem}.txt")
            _, _, metadata = pipeline.process(image, boxes, split="train" if split == "train" else split, return_intermediates=True)
            images, stage_boxes = metadata["intermediates"], metadata["bbox_intermediates"]
            panels = [_draw(images["original"], stage_boxes["original"], "processed input")]
            if "augmentation" in images:
                panels.append(_draw(images["augmentation"], stage_boxes["augmentation"], "train augmentation"))
            else:
                panels.append(_draw(images["original"], stage_boxes["original"], "augmentation: not applied"))
            panels.append(_draw(images["resize"], stage_boxes["resize"], "after resize"))
            panels.append(_draw(images["final_uint8"], stage_boxes["final_uint8"], "after preprocessing"))
            cv2.imwrite(str(output_dir / f"{split}_{image_path.stem}.jpg"), np.hstack(panels)); written += 1
    return written

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", default=None); parser.add_argument("--per-split", type=int, default=10)
    args = parser.parse_args(); print(f"Wrote {create_visual_sanity_samples(args.dataset, args.per_split)} visual samples")
