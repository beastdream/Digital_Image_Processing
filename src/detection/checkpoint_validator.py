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
