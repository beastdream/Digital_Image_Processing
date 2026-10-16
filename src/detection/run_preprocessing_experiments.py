"""Controlled preprocessing experiment runner using versioned materialized datasets."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from src.data.materialize_preprocessed_dataset import materialize_preprocessed_dataset
from src.data.dataset_utils import project_root
from src.detection.evaluate import evaluate_yolo
from src.detection.train import train_yolo
from src.detection.model_contract import load_experiment_config
from src.detection.experiment_reporting import (build_experiment_result_row, describe_preprocessing,
    measure_end_to_end_latency, write_experiment_results)
from src.detection.analyze_experiment_results import (analyze_results, create_visual_inspection_template,
    write_analysis)
from src.detection.checkpoint_validator import is_completed_run, resumable_checkpoint, validate_training_checkpoints
from src.detection.experiment_suite import (DEFAULT_SUITE_CONFIG, load_experiment_suite, suite_experiments,
    suite_training_config)
from src.utils.result_paths import experiment_evaluation_dir


def generate_preprocessed_dataset(exp_name: str, config_file: str, source_yaml: str, seed: int = 42) -> str:
    root = project_root(); config_path = root / config_file
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    offline_transform = any(config.get(section, {}).get("enabled", False)
                            for section in ("resize", "denoise", "contrast", "brightness"))
    if not offline_transform:
        return str(root / source_yaml)
    return str(materialize_preprocessed_dataset(config_file, exp_name, seed=seed, source_yaml=source_yaml))


EXPERIMENTS = suite_experiments(load_experiment_suite())

FAIR_KEYS = ("model_weights", "model_size", "seed", "epochs", "batch", "imgsz", "optimizer",
             "learning_rate", "fraction", "patience", "device", "workers", "cache", "degrees",
             "translate", "scale", "fliplr", "hsv_h", "hsv_s", "hsv_v", "mosaic", "mixup",
             "copy_paste", "debug_only")
ACTUAL_FAIR_KEYS = ("model", "seed", "epochs", "batch", "imgsz", "optimizer", "lr0", "fraction",
                    "patience", "device", "workers", "cache", "degrees", "translate", "scale",
                    "fliplr", "hsv_h", "hsv_s", "hsv_v", "mosaic", "mixup", "copy_paste")

def validate_experiment_configs(root: Path) -> None:
    """Validate suite/config name consistency without encoding experiment methods in Python."""
    load_experiment_suite(root=root)


def read_actual_training_args(root: Path, experiment: str, project_dir: str | Path = "experiments/yolo") -> dict:
    """Read Ultralytics' persisted runtime arguments; reports never invent dataset usage."""
    project_dir = Path(project_dir)
    args_path = (project_dir if project_dir.is_absolute() else root / project_dir) / experiment / "args.yaml"
    if not args_path.exists():
        raise FileNotFoundError(f"YOLO did not persist runtime args: {args_path}")
    return yaml.safe_load(args_path.read_text(encoding="utf-8")) or {}


def build_fair_training_configs(base_config: dict, datasets: dict[str, str], experiments=None) -> dict[str, dict]:
    """Create configs whose only experiment-dependent fields are data path and run name."""
    if float(base_config.get("fraction", 1.0)) != 1.0 or bool(base_config.get("debug_only", False)):
        raise ValueError("Final preprocessing comparison requires fraction: 1.0 and debug_only: false")
    experiments = experiments or EXPERIMENTS
    configs = {}
    for name, _ in experiments:
        cfg = copy.deepcopy(base_config)
        preprocessing_file = dict(experiments)[name]
        cfg.update({"data_yaml": datasets[name], "name": name,
                    "preprocessing_config": preprocessing_file})
        configs[name] = cfg
    reference_name = experiments[0][0]
    reference = {key: configs[reference_name].get(key) for key in FAIR_KEYS}
    for name, cfg in configs.items():
        actual = {key: cfg.get(key) for key in FAIR_KEYS}
        if actual != reference:
            raise ValueError(f"Unfair training configuration for {name}")
    return configs


def completed_weights(project_dir: Path, experiment: str) -> str | None:
    """Return best.pt of a finished run so an interrupted suite resumes instead of retraining."""
    exp_dir = project_dir / experiment
    return str(validate_training_checkpoints(exp_dir)["best"]) if is_completed_run(exp_dir) else None


def run_preprocessing_experiments(suite_path: str | Path = DEFAULT_SUITE_CONFIG) -> None:
    root, rows, actual_reference = project_root(), [], None
    suite = load_experiment_suite(suite_path, root=root); experiments = suite_experiments(suite)
    source_yaml = suite["dataset"]["source_yaml"]
    seed = int(suite["project"].get("seed", 42))
    project_dir = root / suite["project"]["output"]
    results_dir = root / suite["project"].get("results", "results/experiments")
    evaluations_dir = suite["project"].get("evaluations")
    raw_test_images = (root / source_yaml).parent / "images" / "test"
    datasets = {name: generate_preprocessed_dataset(name, preprocessing, source_yaml, seed)
                for name, preprocessing in experiments}
    base_config = suite_training_config(suite)
    training_configs = build_fair_training_configs(base_config, datasets, experiments)
    generated_dir = project_dir / "generated_configs"; generated_dir.mkdir(parents=True, exist_ok=True)
    for label, _ in experiments:
        data_yaml = datasets[label]
        config_path = generated_dir / f"{label}.yaml"
        config_path.write_text(yaml.safe_dump(training_configs[label], sort_keys=False), encoding="utf-8")
        weights = completed_weights(project_dir, label)
        if weights:
            print(f"=== {label}: completed run found, reusing {weights} ===")
        elif resumable_checkpoint(project_dir / label):
            print(f"=== {label}: interrupted run found, resuming from last.pt ===")
            weights = train_yolo(str(config_path), label, resume=True)
        else:
            weights = train_yolo(str(config_path), label, overwrite_incomplete=True)
        actual_args = read_actual_training_args(root, label, project_dir)
        actual_fair = {key: actual_args.get(key) for key in ACTUAL_FAIR_KEYS}
        if actual_reference is None:
            actual_reference = actual_fair
        elif actual_fair != actual_reference:
            differences = {key: (actual_reference[key], actual_fair[key]) for key in ACTUAL_FAIR_KEYS
                           if actual_reference[key] != actual_fair[key]}
            raise RuntimeError(f"Actual YOLO args are unfair for {label}: {differences}")
        actual_fraction = float(actual_args.get("fraction", 1.0))
        if actual_fraction < 1.0:
            raise RuntimeError(f"{label} used fraction={actual_fraction}; final comparison aborted (DEBUG ONLY)")
        evaluation_dir = root / evaluations_dir / label if evaluations_dir else experiment_evaluation_dir(label, root)
        metrics = evaluate_yolo(weights, data_yaml, str(evaluation_dir), "test",
                                clean_output=True)
        experiment_config = load_experiment_config(weights)
        latency = measure_end_to_end_latency(weights, int(actual_args["imgsz"]), image_dir=raw_test_images)
        latency_path = evaluation_dir / "latency_breakdown.json"
        latency_path.parent.mkdir(parents=True, exist_ok=True)
        latency_path.write_text(json.dumps({"experiment": label,
            "preprocessing": describe_preprocessing(experiment_config["preprocessing"]),
            "timing_scope": "read image + preprocessing + model inference",
            **{key: round(value, 3) for key, value in latency.items()}}, indent=2), encoding="utf-8")
        rows.append(build_experiment_result_row(label, experiment_config, actual_args, metrics, latency))
    results_path = write_experiment_results(rows, results_dir / "experiment_results.csv")
    visual_path = create_visual_inspection_template(results_dir / "visual_inspection.csv", [name for name, _ in experiments])
    write_analysis(analyze_results(results_path, visual_path, [name for name, _ in experiments]), results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, evaluate, and compare the preprocessing experiment suite")
    parser.add_argument("--suite", default=DEFAULT_SUITE_CONFIG,
                        help="Suite YAML, e.g. configs/video_preprocessing_experiments.yaml")
    run_preprocessing_experiments(parser.parse_args().suite)
