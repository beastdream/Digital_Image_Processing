"""Analyze class imbalance and object sizes before changing sampling policy."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.data.dataset_utils import CLASS_NAMES, IMAGE_SUFFIXES, SPLITS, parse_yolo_line, project_root
from src.utils.result_paths import class_balance_dir

SMALL_AREA = 32 ** 2
MEDIUM_AREA = 96 ** 2


def size_category(area: float) -> str:
    return "small" if area < SMALL_AREA else "medium" if area < MEDIUM_AREA else "large"


def summarize_sizes(boxes: list[dict]) -> dict:
    if not boxes:
        return {"count": 0, "width_px": {}, "height_px": {}, "area_px2": {},
                "size_counts": {key: 0 for key in ("small", "medium", "large")},
                "size_frequency": {key: 0.0 for key in ("small", "medium", "large")}}
    summary = {"count": len(boxes)}
    for output_key, input_key in (("width_px", "width"), ("height_px", "height"), ("area_px2", "area")):
        values = np.asarray([box[input_key] for box in boxes], dtype=float)
        summary[output_key] = {
            "min": round(float(values.min()), 3), "p25": round(float(np.percentile(values, 25)), 3),
            "median": round(float(np.median(values)), 3), "p75": round(float(np.percentile(values, 75)), 3),
            "max": round(float(values.max()), 3), "mean": round(float(values.mean()), 3),
        }
    counts = Counter(size_category(box["area"]) for box in boxes)
    summary["size_counts"] = {key: counts[key] for key in ("small", "medium", "large")}
    summary["size_frequency"] = {key: round(counts[key] / len(boxes), 6) for key in ("small", "medium", "large")}
    return summary


def collect_projected_boxes(dataset_dir: Path, imgsz: int) -> tuple[list[dict], dict]:
    """Project normalized labels through YOLO-style square letterboxing."""
    records, split_counts = [], {split: Counter() for split in SPLITS}
    for split in SPLITS:
        image_dir, label_dir = dataset_dir / "images" / split, dataset_dir / "labels" / split
        for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Unreadable image during bbox analysis: {image_path}")
            height, width = image.shape[:2]
            scale = min(imgsz / width, imgsz / height)
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label during bbox analysis: {label_path}")
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    class_id, _, _, norm_width, norm_height = parse_yolo_line(line)
                except ValueError as error:
                    raise ValueError(f"{label_path}:{line_number}: {error}") from error
                width_px, height_px = norm_width * width * scale, norm_height * height * scale
                records.append({"class_id": class_id, "split": split, "width": width_px,
                                "height": height_px, "area": width_px * height_px})
                split_counts[split][class_id] += 1
    return records, {split: {CLASS_NAMES[cid]: split_counts[split][cid] for cid in CLASS_NAMES} for split in SPLITS}


def load_per_class_metrics(metrics_path: Path | None) -> tuple[dict, str | None]:
    if metrics_path is None:
        return {}, None
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    by_class = {}
    for row in payload.get("per_class", []):
        class_id = int(row["class_id"])
        by_class[CLASS_NAMES[class_id]] = {key: row.get(key) for key in ("precision", "recall", "mAP50", "mAP50_95")}
    return by_class, str(metrics_path.resolve())


def build_analysis(dataset_dir: Path, configured_imgsz: int, metrics_path: Path | None = None) -> dict:
    sizes = sorted({configured_imgsz, 320})
    analyses, configured_records, split_counts = {}, None, None
    for input_size in sizes:
        records, current_split_counts = collect_projected_boxes(dataset_dir, input_size)
        if input_size == configured_imgsz:
            configured_records, split_counts = records, current_split_counts
        by_class = {CLASS_NAMES[cid]: summarize_sizes([box for box in records if box["class_id"] == cid]) for cid in CLASS_NAMES}
        analyses[str(input_size)] = {"overall": summarize_sizes(records), "per_class": by_class}
    counts = Counter(box["class_id"] for box in configured_records)
    total = sum(counts.values())
    total_nonzero = [counts[cid] for cid in CLASS_NAMES if counts[cid] > 0]
    overall_ratio = max(total_nonzero) / min(total_nonzero) if total_nonzero else 0.0
    train_counts = {cid: split_counts["train"][CLASS_NAMES[cid]] for cid in CLASS_NAMES}
    train_total = sum(train_counts.values())
    train_nonzero = [value for value in train_counts.values() if value > 0]
    ratio = max(train_nonzero) / min(train_nonzero) if train_nonzero else 0.0
    imbalance_level = "HIGH" if ratio >= 2.0 else "MODERATE" if ratio >= 1.5 else "LOW"
    performance, metrics_source = load_per_class_metrics(metrics_path)
    class_frequency = {}
    for class_id, name in CLASS_NAMES.items():
        class_frequency[name] = {
            "class_id": class_id, "objects": counts[class_id],
            "frequency": round(counts[class_id] / total, 6) if total else 0.0,
            "train_objects": train_counts[class_id],
            "train_frequency": round(train_counts[class_id] / train_total, 6) if train_total else 0.0,
            "evaluation": performance.get(name),
        }
    small_320 = analyses["320"]["overall"]["size_frequency"]["small"]
    warnings = []
    if small_320 >= 0.25:
        warnings.append({"code": "MANY_SMALL_OBJECTS_AT_320", "severity": "HIGH",
                         "message": f"{small_320:.1%} of objects are small at imgsz=320; avoid 320 for final training without a targeted recall study."})
    for name, stats in analyses["320"]["per_class"].items():
        fraction = stats["size_frequency"]["small"]
        if fraction >= 0.5:
            warnings.append({"code": f"MANY_SMALL_{name.upper()}_AT_320", "severity": "HIGH",
                             "message": f"{fraction:.1%} of {name} objects are small at imgsz=320."})
    if imbalance_level == "HIGH":
        warnings.append({"code": "CLASS_IMBALANCE", "severity": "HIGH",
                         "message": f"Largest/smallest class object-count ratio is {ratio:.2f}x."})
    return {
        "dataset": str(dataset_dir.resolve()), "configured_imgsz": configured_imgsz,
        "size_definition": {"standard": "COCO area thresholds projected at model input",
                            "small": "area < 32^2 px", "medium": "32^2 <= area < 96^2 px", "large": "area >= 96^2 px"},
        "class_imbalance": {"level": imbalance_level, "basis": "training split object counts",
                            "largest_to_smallest_ratio": round(ratio, 4),
                            "overall_largest_to_smallest_ratio": round(overall_ratio, 4),
                            "total_objects": total, "per_split_objects": split_counts,
                            "class_frequency_and_performance": class_frequency,
                            "sampling_decision": "Do not enable weighted sampling, oversampling, or class-specific augmentation automatically. Reassess only after class frequency and validated per-class precision, recall, and AP show a persistent class-specific failure."},
        "bbox_statistics_by_imgsz": analyses, "evaluation_metrics_source": metrics_source,
        "evaluation_metrics_note": "Metrics are copied from the supplied artifact. Confirm its experiment_config.yaml and final-run training args before using them to justify sampling changes." if metrics_source else "No evaluation artifact supplied; sampling changes are deferred.",
        "warnings": warnings,
    }


def write_analysis(report: dict, output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path, csv_path, markdown_path = (output_dir / "class_imbalance_and_bbox_analysis.json",
                                           output_dir / "bbox_size_statistics.csv",
                                           output_dir / "class_imbalance_and_bbox_analysis.md")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["imgsz", "class", "objects", "median_width_px", "median_height_px", "median_area_px2",
                  "small", "medium", "large", "small_frequency"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for imgsz, analysis in report["bbox_statistics_by_imgsz"].items():
            for name, stats in analysis["per_class"].items():
                writer.writerow({"imgsz": imgsz, "class": name, "objects": stats["count"],
                    "median_width_px": stats["width_px"].get("median"), "median_height_px": stats["height_px"].get("median"),
                    "median_area_px2": stats["area_px2"].get("median"), **stats["size_counts"],
                    "small_frequency": stats["size_frequency"]["small"]})
    imbalance = report["class_imbalance"]
    lines = ["# Class imbalance and bbox-size analysis", "", f"Imbalance: **{imbalance['level']}** ({imbalance['largest_to_smallest_ratio']:.2f}× largest/smallest).", "",
             "No weighted sampling, oversampling, or class-specific augmentation is enabled by this analysis.", "", "## Class evidence", ""]
    for name, item in imbalance["class_frequency_and_performance"].items():
        lines.append(f"- {name}: {item['objects']} objects ({item['frequency']:.1%}); metrics={item['evaluation']}")
    lines.extend(["", "## Warnings", ""] + [f"- [{warning['severity']}] {warning['message']}" for warning in report["warnings"]])
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, csv_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/road_damage_detection")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--metrics", default=None, help="Optional metrics.json with per-class precision/recall/AP")
    parser.add_argument("--output", default=str(class_balance_dir(project_root())))
    args = parser.parse_args(); root = project_root()
    dataset = Path(args.dataset); dataset = dataset if dataset.is_absolute() else root / dataset
    if args.imgsz is None:
        training = yaml.safe_load((root / "configs/yolo_training.yaml").read_text(encoding="utf-8"))
        imgsz = int(training["imgsz"])
    else:
        imgsz = args.imgsz
    metrics = Path(args.metrics) if args.metrics else None
    if metrics is not None and not metrics.is_absolute(): metrics = root / metrics
    output = Path(args.output); output = output if output.is_absolute() else root / output
    paths = write_analysis(build_analysis(dataset, imgsz, metrics), output)
    print("Saved:", *(str(path) for path in paths), sep="\n")


if __name__ == "__main__":
    main()
