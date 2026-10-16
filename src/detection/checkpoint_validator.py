"""Shared validation for the standard checkpoints of a YOLO experiment."""
from __future__ import annotations

from pathlib import Path


def validate_training_checkpoints(experiment_dir: str | Path) -> dict:
    """Return resolved checkpoint paths and their actual filesystem state."""
    experiment = Path(experiment_dir).resolve()
    best = experiment / "weights" / "best.pt"
    last = experiment / "weights" / "last.pt"
    return {
        "best": best,
        "last": last,
        "best_exists": best.is_file(),
        "last_exists": last.is_file(),
    }


def is_completed_run(experiment_dir: str | Path) -> bool:
    """A run is complete only after train.py wrote its preprocessing contract."""
    state = validate_training_checkpoints(experiment_dir)
    return state["best_exists"] and state["last_exists"] and (Path(experiment_dir) / "experiment_config.yaml").is_file()


def resumable_checkpoint(experiment_dir: str | Path) -> Path | None:
    """Return last.pt of an interrupted run, or None when there is nothing to resume."""
    state = validate_training_checkpoints(experiment_dir)
    if state["last_exists"] and not is_completed_run(experiment_dir):
        return state["last"]
    return None
