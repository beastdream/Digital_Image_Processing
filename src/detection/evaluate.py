import os
import csv
import json
import numpy as np
import pandas as pd
from ultralytics import YOLO

def evaluate_yolo(
    model_path: str,
    data_yaml: str = "d:/Digital_Image_Processing/data/processed/detection/dataset.yaml",
    output_dir: str = "results/yolo/baseline",
    split: str = "test"
) -> dict:
    base_dir = r"d:\Digital_Image_Processing"
    full_output_dir = os.path.join(base_dir, output_dir) if not os.path.isabs(output_dir) else output_dir
    os.makedirs(full_output_dir, exist_ok=True)

    print(f"Evaluating YOLO Model [{model_path}] on split '{split}' using data [{data_yaml}]...")

    model = YOLO(model_path)

    # Run validation strictly on 'test' split
    metrics = model.val(
        data=data_yaml,
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

    class_names = {0: "Pothole", 1: "Crack", 2: "Manhole"}
    class_metrics_rows = []

    # Per-class metrics extraction
    p_per_class = metrics.box.p if isinstance(metrics.box.p, (list, np.ndarray)) else [metrics.box.p]
    r_per_class = metrics.box.r if isinstance(metrics.box.r, (list, np.ndarray)) else [metrics.box.r]
    maps_per_class = metrics.box.maps if hasattr(metrics.box, "maps") and isinstance(metrics.box.maps, (list, np.ndarray)) else [0.0, 0.0, 0.0]

    all_ap = getattr(metrics.box, "all_ap", None)

    for cid in [0, 1, 2]:
        cname = class_names[cid]
        p_c = float(p_per_class[cid]) if len(p_per_class) > cid else 0.0
        r_c = float(r_per_class[cid]) if len(r_per_class) > cid else 0.0

        if all_ap is not None and len(all_ap) > cid:
            m50_c = float(all_ap[cid][0])
        else:
            m50_c = float(maps_per_class[cid]) if len(maps_per_class) > cid else 0.0

        m95_c = float(maps_per_class[cid]) if len(maps_per_class) > cid else 0.0

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
    parser.add_argument("--data", type=str, default="d:/Digital_Image_Processing/data/processed/detection/dataset.yaml")
    parser.add_argument("--output", type=str, default="results/yolo/baseline")
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    evaluate_yolo(args.model, args.data, args.output, args.split)
