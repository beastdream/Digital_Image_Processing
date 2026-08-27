"""Latency benchmarking and automatic experiment result-table generation."""
from __future__ import annotations

import csv
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.data.dataset_utils import project_root
from src.detection.model_contract import load_experiment_config, preprocess_for_inference

RESULT_COLUMNS = ["experiment", "preprocessing", "epochs", "batch", "imgsz", "fraction", "seed",
                  "precision", "recall", "mAP50", "mAP50_95", "pothole_mAP50", "crack_mAP50",
                  "manhole_mAP50", "preprocessing_ms", "inference_ms", "total_ms"]


def describe_preprocessing(config: dict) -> str:
    stages = []
    denoise = config.get("denoise", {})
    if denoise.get("enabled", False): stages.append(str(denoise.get("method")))
    contrast = config.get("contrast", {})
    if contrast.get("enabled", False): stages.append(str(contrast.get("method")))
    if config.get("brightness", {}).get("enabled", False): stages.append("brightness")
    return "+".join(stages) if stages else "raw"


def benchmark_loaded_model(model, experiment_config: dict, image_paths: list[Path], imgsz: int,
                           warmup: int = 5, clock=time.perf_counter, image_reader=cv2.imread) -> dict[str, float]:
    samples = []
    for image_path in image_paths:
        total_started = clock(); raw = image_reader(str(image_path)); read_finished = clock()
        if raw is None or raw.size == 0: raise ValueError(f"Unreadable benchmark image: {image_path}")
        prepared, _ = preprocess_for_inference(raw, experiment_config); preprocessing_finished = clock()
        model.predict(prepared, imgsz=imgsz, verbose=False); inference_finished = clock()
        samples.append({"read_image_ms": (read_finished - total_started) * 1000,
            "preprocessing_ms": (preprocessing_finished - read_finished) * 1000,
            "inference_ms": (inference_finished - preprocessing_finished) * 1000,
            "total_ms": (inference_finished - total_started) * 1000})
    measured = samples[warmup:] or samples
    if not measured: raise ValueError("No images available for latency benchmark")
    return {key: float(np.mean([sample[key] for sample in measured])) for key in measured[0]}


def measure_end_to_end_latency(model_path: str, imgsz: int, max_images: int = 50, warmup: int = 5) -> dict[str, float]:
    model, contract = YOLO(model_path), load_experiment_config(model_path)
    image_dir = project_root() / "data/processed/road_damage_detection/images/test"
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})[:max_images]
    return benchmark_loaded_model(model, contract, images, imgsz, warmup)


def flatten_evaluation_metrics(metrics: dict) -> dict:
    flattened = {f"overall_{key}": value for key, value in metrics["overall"].items()}
    for row in metrics["per_class"]:
        for key in ("precision", "recall", "mAP50", "mAP50_95"):
            flattened[f"{row['class_name']}_{key}"] = row[key]
    for key, item in metrics.get("confusion_matrix", {}).get("important_confusions", {}).items():
        flattened[f"{key}_count"], flattened[f"{key}_rate"] = item["count"], item["rate_over_actual"]
    return flattened


def build_experiment_result_row(experiment: str, experiment_config: dict, actual_args: dict,
                                metrics: dict, latency: dict) -> dict:
    per_class, overall = {row["class_name"]: row for row in metrics["per_class"]}, metrics["overall"]
    return {"experiment": experiment, "preprocessing": describe_preprocessing(experiment_config["preprocessing"]),
        "epochs": actual_args["epochs"], "batch": actual_args["batch"], "imgsz": actual_args["imgsz"],
        "fraction": actual_args["fraction"], "seed": actual_args["seed"], "precision": overall["precision"],
        "recall": overall["recall"], "mAP50": overall["mAP50"], "mAP50_95": overall["mAP50_95"],
        "pothole_mAP50": per_class["pothole"]["mAP50"], "crack_mAP50": per_class["crack"]["mAP50"],
        "manhole_mAP50": per_class["manhole"]["mAP50"], "preprocessing_ms": round(latency["preprocessing_ms"], 3),
        "inference_ms": round(latency["inference_ms"], 3), "total_ms": round(latency["total_ms"], 3)}


def write_experiment_results(rows: list[dict], output: Path) -> Path:
    if not rows: raise ValueError("Cannot write experiment_results.csv before experiments finish")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS); writer.writeheader(); writer.writerows(rows)
    return output
