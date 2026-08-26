import os
import csv
import json
import numpy as np
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
import yaml
from src.data.dataset_utils import CLASS_NAMES, project_root

def evaluate_yolo(
    model_path: str,
    data_yaml: str = "data/processed/road_damage_detection/dataset.yaml",
    output_dir: str = "results/yolo/baseline",
    split: str = "test"
) -> dict:
    base_dir = project_root()
    data_path = Path(data_yaml)
    if not data_path.is_absolute(): data_path = base_dir / data_path
    with data_path.open(encoding="utf-8") as handle: dataset = yaml.safe_load(handle)
    names = dataset.get("names", {}); names = {int(k): v for k, v in names.items()} if isinstance(names, dict) else dict(enumerate(names))
    if dataset.get("nc") != 3 or names != CLASS_NAMES:
        raise ValueError(f"Dataset must preserve {CLASS_NAMES}; got nc={dataset.get('nc')}, names={names}")
    full_output_dir = base_dir / output_dir if not os.path.isabs(output_dir) else Path(output_dir)
    os.makedirs(full_output_dir, exist_ok=True)

    print(f"Evaluating YOLO Model [{model_path}] on split '{split}' using data [{data_yaml}]...")

    model = YOLO(model_path)
    model_names = {int(key): value for key, value in model.names.items()}
    if model_names != CLASS_NAMES:
        raise ValueError(f"Weights must preserve {CLASS_NAMES}; got {model_names}")

    # Run validation strictly on 'test' split
    metrics = model.val(
        data=str(data_path),
        split=split,
        imgsz=640,
        batch=16,
        save_json=True,
        plots=True,
        project=full_output_dir,
        name="val_run",
        exist_ok=True
    )

    # Extract overall metrics
    p_overall = float(metrics.results_dict.get("metrics/precision(B)", 0.0))
    r_overall = float(metrics.results_dict.get("metrics/recall(B)", 0.0))
    map50_overall = float(metrics.results_dict.get("metrics/mAP50(B)", 0.0))
    map50_95_overall = float(metrics.results_dict.get("metrics/mAP50-95(B)", 0.0))

    class_names = CLASS_NAMES
    class_metrics_rows = []

    # Per-class metrics extraction
    p_per_class = np.asarray(metrics.box.p).reshape(-1)
    r_per_class = np.asarray(metrics.box.r).reshape(-1)
    maps_per_class = np.asarray(getattr(metrics.box, "maps", [])).reshape(-1)
    class_indices = np.asarray(getattr(metrics.box, "ap_class_index", range(len(p_per_class)))).reshape(-1)
    metric_index = {int(class_id): index for index, class_id in enumerate(class_indices)}

    all_ap = getattr(metrics.box, "all_ap", None)

    for cid in [0, 1, 2]:
        cname = class_names[cid]
        index = metric_index.get(cid)
        p_c = float(p_per_class[index]) if index is not None and index < len(p_per_class) else 0.0
        r_c = float(r_per_class[index]) if index is not None and index < len(r_per_class) else 0.0

        if all_ap is not None and index is not None and index < len(all_ap):
            m50_c = float(all_ap[index][0])
        else:
            m50_c = float(maps_per_class[index]) if index is not None and index < len(maps_per_class) else 0.0

        m95_c = float(maps_per_class[index]) if index is not None and index < len(maps_per_class) else 0.0

        class_metrics_rows.append({
            "class_id": cid,
            "class_name": cname,
            "precision": round(p_c, 4),
            "recall": round(r_c, 4),
            "mAP50": round(m50_c, 4),
            "mAP50_95": round(m95_c, 4)
        })

    # Save class_metrics.csv
    csv_path = os.path.join(full_output_dir, "class_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "class_name", "precision", "recall", "mAP50", "mAP50_95"])
        writer.writeheader()
        writer.writerows(class_metrics_rows)

    print(f"Saved class-wise metrics CSV to {csv_path}")

    # Summary metrics dictionary
    summary_metrics = {
        "overall": {
            "precision": round(p_overall, 4),
            "recall": round(r_overall, 4),
            "mAP50": round(map50_overall, 4),
            "mAP50_95": round(map50_95_overall, 4)
        },
        "per_class": class_metrics_rows
    }

    metrics_json_path = os.path.join(full_output_dir, "metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    print(f"Evaluation finished! Overall mAP50: {map50_overall:.4f}, mAP50-95: {map50_95_overall:.4f}")
    return summary_metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to best.pt weights")
    parser.add_argument("--data", type=str, default="data/processed/road_damage_detection/dataset.yaml")
    parser.add_argument("--output", type=str, default="results/yolo/baseline")
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    evaluate_yolo(args.model, args.data, args.output, args.split)
