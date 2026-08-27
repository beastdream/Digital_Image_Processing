"""Persist and enforce the preprocessing contract shipped with each trained model."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import yaml

from src.data.dataset_utils import CLASS_NAMES, project_root
from src.preprocessing.pipeline import ImagePreprocessingPipeline


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def load_training_preprocessing(training_cfg: dict, data_yaml: Path, root: Path | None = None) -> dict:
    """Load the exact preprocessing used to create the training images."""
    root = root or project_root()
    manifest_path = data_yaml.parent / "build_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        preprocessing = manifest.get("config")
        if not isinstance(preprocessing, dict):
            raise ValueError(f"Missing preprocessing config in {manifest_path}")
        return preprocessing
    config_path = training_cfg.get("preprocessing_config")
    if not config_path:
        raise ValueError("Training config must specify preprocessing_config when dataset has no build_manifest.json")
    return yaml.safe_load(_resolve(root, config_path).read_text(encoding="utf-8")) or {}


def write_experiment_config(exp_dir: Path, training_cfg: dict, data_yaml: Path,
                            runtime_args: dict | None = None, root: Path | None = None) -> Path:
    root = root or project_root()
    preprocessing = load_training_preprocessing(training_cfg, data_yaml, root)
    metadata = {
        "schema_version": 1,
        "name": training_cfg.get("name"),
        "classes": CLASS_NAMES,
        "preprocessing": preprocessing,
        "training": runtime_args or {
            "model": training_cfg.get("model_weights"), "imgsz": training_cfg.get("imgsz"),
            "epochs": training_cfg.get("epochs"), "batch": training_cfg.get("batch"),
            "seed": training_cfg.get("seed"), "fraction": training_cfg.get("fraction", 1.0),
        },
        "training_data_yaml": str(data_yaml.resolve()),
    }
    path = exp_dir / "experiment_config.yaml"
    path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    weights_dir = exp_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    (weights_dir / "experiment_config.yaml").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return path


def find_experiment_config(weights_path: str | Path, metadata_path: str | Path | None = None) -> Path:
    if metadata_path:
        path = Path(metadata_path)
        if not path.is_absolute():
            path = project_root() / path
        if not path.exists():
            raise FileNotFoundError(f"Experiment metadata not found: {path}")
        return path
    weights = Path(weights_path).resolve()
    candidates = (weights.parent / "experiment_config.yaml", weights.parent.parent / "experiment_config.yaml")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No experiment_config.yaml shipped with {weights}. Pass --experiment-config explicitly."
    )


def load_experiment_config(weights_path: str | Path, metadata_path: str | Path | None = None) -> dict:
    config = yaml.safe_load(find_experiment_config(weights_path, metadata_path).read_text(encoding="utf-8")) or {}
    classes = {int(key): value for key, value in config.get("classes", {}).items()}
    if classes != CLASS_NAMES or not isinstance(config.get("preprocessing"), dict):
        raise ValueError("Invalid experiment_config.yaml: classes/preprocessing contract is missing or incompatible")
    return config


def preprocess_for_inference(image: np.ndarray, experiment_config: dict) -> tuple[np.ndarray, dict]:
    """Apply the stored offline preprocessing and return uint8 for Ultralytics."""
    preprocessing = copy.deepcopy(experiment_config["preprocessing"])
    preprocessing["augmentation"] = {"enabled": False}
    pipeline = ImagePreprocessingPipeline(preprocessing)
    _, _, metadata = pipeline.process(image, bboxes=None, split="test", return_intermediates=True)
    return metadata["intermediates"]["final_uint8"], metadata


def validate_evaluation_dataset(data_yaml: Path, experiment_config: dict) -> None:
    """Reject evaluation data whose preprocessing differs from the model contract."""
    expected = experiment_config["preprocessing"]
    manifest_path = data_yaml.parent / "build_manifest.json"
    has_transform = any(expected.get(section, {}).get("enabled", False)
                        for section in ("resize", "denoise", "contrast", "brightness"))
    if has_transform:
        if not manifest_path.exists():
            raise ValueError("Processed model cannot be evaluated on raw images: build_manifest.json is missing")
        actual = json.loads(manifest_path.read_text(encoding="utf-8")).get("config")
        if actual != expected:
            raise ValueError("Evaluation preprocessing does not match experiment_config.yaml")
