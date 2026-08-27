"""Canonical output paths and guarded cleanup for generated result artifacts."""
from __future__ import annotations

import shutil
from pathlib import Path

from src.data.dataset_utils import project_root


def results_root(root: Path | None = None) -> Path:
    return (root or project_root()) / "results"


def predictions_root(root: Path | None = None) -> Path:
    return results_root(root) / "predictions"


def latest_predictions_dir(root: Path | None = None) -> Path:
    return predictions_root(root) / "latest"


def ground_truth_root(root: Path | None = None) -> Path:
    return results_root(root) / "ground_truth"


def latest_ground_truth_dir(root: Path | None = None) -> Path:
    return ground_truth_root(root) / "latest"


def evaluations_root(root: Path | None = None) -> Path:
    return results_root(root) / "evaluations"


def latest_evaluation_dir(root: Path | None = None) -> Path:
    return evaluations_root(root) / "latest"


def experiment_evaluation_dir(experiment: str, root: Path | None = None) -> Path:
    if not experiment or Path(experiment).name != experiment or experiment in {".", ".."}:
        raise ValueError(f"Invalid experiment directory name: {experiment!r}")
    return evaluations_root(root) / experiment


def experiment_results_dir(root: Path | None = None) -> Path:
    return results_root(root) / "experiments"


def preprocessing_visualization_dir(root: Path | None = None) -> Path:
    return results_root(root) / "preprocessing" / "visualizations"


def class_balance_dir(root: Path | None = None) -> Path:
    return results_root(root) / "analysis" / "class_balance"


def legacy_dir(root: Path | None = None) -> Path:
    return results_root(root) / "legacy"


def reset_directory(path: Path, allowed_parent: Path) -> Path:
    """Replace one result subdirectory after proving it is below its allowed parent."""
    resolved_path = path.resolve()
    resolved_parent = allowed_parent.resolve()
    root = project_root().resolve()
    protected = {
        root,
        (root / "data").resolve(),
        (root / "experiments").resolve(),
        (root / "experiments/yolo").resolve(),
        (root / "results").resolve(),
    }
    if resolved_path in protected:
        raise ValueError(f"Refusing to delete a protected project directory: {resolved_path}")
    if resolved_path == resolved_parent:
        raise ValueError(f"Refusing to delete the protected result root: {resolved_path}")
    if resolved_parent not in resolved_path.parents:
        raise ValueError(f"Refusing to delete path outside {resolved_parent}: {resolved_path}")
    if resolved_path.exists():
        if not resolved_path.is_dir():
            raise ValueError(f"Refusing to replace a non-directory path: {resolved_path}")
        shutil.rmtree(resolved_path)
    resolved_path.mkdir(parents=True, exist_ok=True)
    return resolved_path
