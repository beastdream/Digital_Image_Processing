"""Load and validate the config-driven preprocessing experiment suite."""
from __future__ import annotations

from pathlib import Path

import yaml

from src.data.dataset_utils import CLASS_NAMES, project_root

DEFAULT_SUITE_CONFIG = "configs/preprocessing_experiments.yaml"


def load_experiment_suite(path: str | Path = DEFAULT_SUITE_CONFIG, root: Path | None = None) -> dict:
    root = root or project_root(); path = Path(path)
    if not path.is_absolute(): path = root / path
    suite = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    classes = {int(key): value for key, value in suite.get("dataset", {}).get("classes", {}).items()}
    if classes != CLASS_NAMES:
        raise ValueError(f"Suite classes must be {CLASS_NAMES}; got {classes}")
    experiments = suite.get("experiments", [])
    names = [item.get("name") for item in experiments]
    if not experiments or None in names or len(names) != len(set(names)):
        raise ValueError("Suite experiments must have unique non-empty names")
    for item in experiments:
        config_path = Path(item.get("preprocessing_config", ""))
        if not config_path.is_absolute(): config_path = root / config_path
        if not config_path.is_file(): raise FileNotFoundError(f"Missing preprocessing config: {config_path}")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if config.get("experiment_name") != item["name"]:
            raise ValueError(f"{config_path.name}: experiment_name must be {item['name']}")
    training = suite.get("training", {})
    required = ("model", "epochs", "batch", "imgsz", "fraction", "optimizer", "learning_rate", "patience")
    missing = [key for key in required if key not in training]
    if missing: raise ValueError(f"Suite training config missing: {', '.join(missing)}")
    return suite


def suite_experiments(suite: dict) -> list[tuple[str, str]]:
    return [(item["name"], item["preprocessing_config"]) for item in suite["experiments"]]


def suite_training_config(suite: dict) -> dict:
    project, dataset, training = suite["project"], suite["dataset"], dict(suite["training"])
    training["model_weights"] = training.pop("model")
    training["seed"] = project["seed"]
    training["project"] = project["output"]
    training["data_yaml"] = dataset["source_yaml"]
    training.setdefault("exist_ok", True); training.setdefault("save", True)
    training.setdefault("save_period", -1); training.setdefault("plots", True); training.setdefault("verbose", True)
    return training
