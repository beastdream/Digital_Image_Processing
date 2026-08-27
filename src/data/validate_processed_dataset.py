"""Independent validation gate for the generated YOLO detection dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from src.data.dataset_utils import CLASS_NAMES, IMAGE_SUFFIXES, SPLITS, parse_yolo_line, project_root, validate_yolo_box


def validate_processed_dataset(dataset_dir: str | Path | None = None) -> tuple[dict, bool]:
    dataset_dir = Path(dataset_dir) if dataset_dir else project_root() / "data/processed/road_damage_detection"
    reports_dir = dataset_dir / "reports"; reports_dir.mkdir(parents=True, exist_ok=True)
    report = {"dataset": str(dataset_dir), "splits": {}, "errors": [], "status": "PASSED"}
    for split in SPLITS:
        image_dir, label_dir = dataset_dir / "images" / split, dataset_dir / "labels" / split
        split_errors, images = [], {}
        if not image_dir.is_dir() or not label_dir.is_dir():
            split_errors.append({"reason": "missing image or label directory"})
        else:
            images = {path.stem: path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES}
            labels = {path.stem: path for path in label_dir.glob("*.txt")}
            for stem in sorted(set(images) - set(labels)):
                split_errors.append({"image": images[stem].name, "reason": "missing label file"})
            for stem in sorted(set(labels) - set(images)):
                split_errors.append({"label_file": labels[stem].name, "reason": "orphan label file"})
            for stem, image_path in images.items():
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None or image.size == 0 or image.shape[0] <= 0 or image.shape[1] <= 0:
                    split_errors.append({"image": image_path.name, "reason": "unreadable processed image"})
                    continue
                label_path = labels.get(stem)
                if label_path is None:
                    continue
                for line_number, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                    if not raw.strip():
                        continue
                    try:
                        box = parse_yolo_line(raw)
                        corrected, status = validate_yolo_box(box)
                        if status != "valid" or corrected != box:
                            raise ValueError("processed bbox requires correction")
                    except ValueError as exc:
                        split_errors.append({"image": image_path.name, "label_file": label_path.name,
                                             "line": line_number, "content": raw, "reason": str(exc)})
        report["splits"][split] = {"images": len(images), "errors": len(split_errors), "details": split_errors}
        report["errors"].extend({"split": split, **error} for error in split_errors)
    if report["errors"]:
        report["status"] = "FAILED"
    (reports_dir / "post_processing_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report, report["status"] == "PASSED"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", default=None)
    args = parser.parse_args()
    result, valid = validate_processed_dataset(args.dataset)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if valid else 1)
