import os
import csv
import json
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import yaml
from src.data.dataset_utils import CLASS_NAMES, project_root
from src.detection.model_contract import load_experiment_config, validate_evaluation_dataset
from src.utils.result_paths import evaluations_root, latest_evaluation_dir, reset_directory

MATRIX_LABELS = [CLASS_NAMES[index] for index in sorted(CLASS_NAMES)] + ["background"]


def prepare_evaluation_output(
    output_dir: str | Path | None,
    clean_output: bool = False,
    root: Path | None = None,
) -> Path:
    """Resolve evaluation output, cleaning only default/latest or an approved child."""
    root = root or project_root()
    if output_dir is None:
        return reset_directory(latest_evaluation_dir(root), evaluations_root(root))
    resolved = Path(output_dir)
    if not resolved.is_absolute():
        resolved = root / resolved
    if clean_output:
        return reset_directory(resolved, evaluations_root(root))
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def log_evaluation_output(output_dir: Path, replacing: bool) -> None:
    print(f"Evaluation output:\n{output_dir}")
    if replacing:
        print(f"Replacing previous evaluation artifacts for {output_dir.name}.")


def extract_detection_metrics(metrics) -> dict:
    """Extract overall and all three class metrics without dropping absent classes."""
    overall = {
        "precision": round(float(metrics.results_dict.get("metrics/precision(B)", 0.0)), 4),
        "recall": round(float(metrics.results_dict.get("metrics/recall(B)", 0.0)), 4),
        "mAP50": round(float(metrics.results_dict.get("metrics/mAP50(B)", 0.0)), 4),
        "mAP50_95": round(float(metrics.results_dict.get("metrics/mAP50-95(B)", 0.0)), 4),
    }
    precision = np.asarray(metrics.box.p).reshape(-1)
    recall = np.asarray(metrics.box.r).reshape(-1)
    map50_95 = np.asarray(getattr(metrics.box, "maps", [])).reshape(-1)
    class_indices = np.asarray(getattr(metrics.box, "ap_class_index", range(len(precision)))).reshape(-1)
    metric_index = {int(class_id): index for index, class_id in enumerate(class_indices)}
    all_ap = np.asarray(getattr(metrics.box, "all_ap", []))
    rows = []
    for class_id, class_name in CLASS_NAMES.items():
        index = metric_index.get(class_id)
        valid = index is not None
        rows.append({
            "class_id": class_id,
            "class_name": class_name,
            "precision": round(float(precision[index]), 4) if valid and index < len(precision) else 0.0,
            "recall": round(float(recall[index]), 4) if valid and index < len(recall) else 0.0,
            "mAP50": round(float(all_ap[index, 0]), 4) if valid and all_ap.ndim == 2 and index < len(all_ap) else 0.0,
            "mAP50_95": round(float(map50_95[index]), 4) if valid and index < len(map50_95) else 0.0,
        })
    return {"overall": overall, "per_class": rows}


def extract_confusion_matrix(metrics) -> np.ndarray:
    matrix = np.asarray(metrics.confusion_matrix.matrix, dtype=float)
    expected_shape = (len(MATRIX_LABELS), len(MATRIX_LABELS))
    if matrix.shape != expected_shape:
        raise ValueError(f"Expected {expected_shape} detection confusion matrix, got {matrix.shape}")
    return matrix


def analyze_confusions(matrix: np.ndarray) -> dict:
    """Rows are predicted classes and columns are actual classes in Ultralytics."""
    checks = (("pothole_to_manhole", 0, 2), ("manhole_to_pothole", 2, 0),
              ("crack_to_background", 1, 3), ("pothole_to_background", 0, 3))
    analysis = {}
    for key, actual, predicted in checks:
        total_actual = float(matrix[:, actual].sum())
        count = float(matrix[predicted, actual])
        analysis[key] = {
            "actual": MATRIX_LABELS[actual], "predicted": MATRIX_LABELS[predicted],
            "count": int(count) if count.is_integer() else count,
            "rate_over_actual": round(count / total_actual, 6) if total_actual else 0.0,
            "actual_instances": int(total_actual) if total_actual.is_integer() else total_actual,
        }
    return analysis


def _write_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["predicted\\actual", *MATRIX_LABELS])
        for label, row in zip(MATRIX_LABELS, matrix):
            writer.writerow([label, *[int(value) if float(value).is_integer() else round(float(value), 6) for value in row]])


def _plot_matrix(path: Path, matrix: np.ndarray, title: str, value_format: str) -> None:
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(4), MATRIX_LABELS, rotation=30, ha="right")
    axis.set_yticks(range(4), MATRIX_LABELS)
    axis.set_xlabel("Actual class")
    axis.set_ylabel("Predicted class")
    axis.set_title(title)
    threshold = float(np.nanmax(matrix)) / 2 if matrix.size else 0
    for row in range(4):
        for column in range(4):
            axis.text(column, row, format(matrix[row, column], value_format), ha="center", va="center",
                      color="white" if matrix[row, column] > threshold else "black")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_evaluation_artifacts(output_dir: Path, summary: dict, matrix: np.ndarray) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = [{"scope": "overall", "class_id": "", "class_name": "overall", **summary["overall"]}]
    metric_rows.extend({"scope": "class", **row} for row in summary["per_class"])
    with (output_dir / "metrics_by_scope.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scope", "class_id", "class_name", "precision", "recall", "mAP50", "mAP50_95"])
        writer.writeheader(); writer.writerows(metric_rows)
    _write_matrix_csv(output_dir / "confusion_matrix.csv", matrix)
    column_totals = matrix.sum(axis=0, keepdims=True)
    normalized = np.divide(matrix, column_totals, out=np.zeros_like(matrix), where=column_totals != 0)
    _write_matrix_csv(output_dir / "confusion_matrix_normalized.csv", normalized)
    _plot_matrix(output_dir / "confusion_matrix.png", matrix, "Confusion matrix (counts)", ".0f")
    _plot_matrix(output_dir / "confusion_matrix_normalized.png", normalized, "Confusion matrix (normalized by actual class)", ".2f")
    analysis = analyze_confusions(matrix)
    (output_dir / "confusion_analysis.json").write_text(json.dumps({
        "orientation": "rows=predicted, columns=actual", "labels": MATRIX_LABELS,
        "important_confusions": analysis,
    }, indent=2), encoding="utf-8")
    lines = ["# Important confusion analysis", "", "Matrix orientation: rows = predicted, columns = actual.", ""]
    for item in analysis.values():
        lines.append(f"- {item['actual']} → {item['predicted']}: {item['count']} / {item['actual_instances']} ({item['rate_over_actual']:.2%})")
    (output_dir / "confusion_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return analysis

def evaluate_yolo(
    model_path: str,
    data_yaml: str = "data/processed/road_damage_detection/dataset.yaml",
    output_dir: str | None = None,
    split: str = "test",
    experiment_config_path: str | None = None,
    clean_output: bool = False,
) -> dict:
    base_dir = project_root()
    data_path = Path(data_yaml)
    if not data_path.is_absolute(): data_path = base_dir / data_path
    with data_path.open(encoding="utf-8") as handle: dataset = yaml.safe_load(handle)
    names = dataset.get("names", {}); names = {int(k): v for k, v in names.items()} if isinstance(names, dict) else dict(enumerate(names))
    if dataset.get("nc") != 3 or names != CLASS_NAMES:
        raise ValueError(f"Dataset must preserve {CLASS_NAMES}; got nc={dataset.get('nc')}, names={names}")
    experiment_config = load_experiment_config(model_path, experiment_config_path)
    validate_evaluation_dataset(data_path, experiment_config)
    full_output_dir = prepare_evaluation_output(output_dir, clean_output, base_dir)
    log_evaluation_output(full_output_dir, output_dir is None or clean_output)

    print(f"Evaluating YOLO Model [{model_path}] on split '{split}' using data [{data_yaml}]...")

    model = YOLO(model_path)
    model_names = {int(key): value for key, value in model.names.items()}
    if model_names != CLASS_NAMES:
        raise ValueError(f"Weights must preserve {CLASS_NAMES}; got {model_names}")

    # Run validation strictly on 'test' split
    metrics = model.val(
        data=str(data_path),
        split=split,
        imgsz=int(experiment_config["training"].get("imgsz", 640)),
        batch=int(experiment_config["training"].get("batch", 16)),
        save_json=True,
        plots=True,
        project=full_output_dir,
        name="val_run",
        exist_ok=True
    )

    summary_metrics = extract_detection_metrics(metrics)
    class_metrics_rows = summary_metrics["per_class"]

    # Save class_metrics.csv
    csv_path = os.path.join(full_output_dir, "class_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "class_name", "precision", "recall", "mAP50", "mAP50_95"])
        writer.writeheader()
        writer.writerows(class_metrics_rows)

    print(f"Saved class-wise metrics CSV to {csv_path}")

    matrix = extract_confusion_matrix(metrics)
    summary_metrics["confusion_matrix"] = {
        "orientation": "rows=predicted, columns=actual", "labels": MATRIX_LABELS,
        "counts": matrix.tolist(), "important_confusions": save_evaluation_artifacts(full_output_dir, summary_metrics, matrix),
    }

    metrics_json_path = os.path.join(full_output_dir, "metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    print(f"Evaluation finished! Overall mAP50: {summary_metrics['overall']['mAP50']:.4f}, "
          f"mAP50-95: {summary_metrics['overall']['mAP50_95']:.4f}")
    for row in class_metrics_rows:
        print(f"  {row['class_name']}: P={row['precision']:.4f}, R={row['recall']:.4f}, "
              f"mAP50={row['mAP50']:.4f}, mAP50-95={row['mAP50_95']:.4f}")
    return summary_metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to best.pt weights")
    parser.add_argument("--data", type=str, default="data/processed/road_damage_detection/dataset.yaml")
    parser.add_argument("--output", type=str, default=None,
                        help="Custom output directory (default: reset results/evaluations/latest)")
    parser.add_argument("--clean-output", action="store_true",
                        help="Reset a custom --output below results/evaluations before writing")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--experiment-config", type=str, default=None)
    args = parser.parse_args()

    evaluate_yolo(args.model, args.data, args.output, args.split, args.experiment_config,
                  args.clean_output)
