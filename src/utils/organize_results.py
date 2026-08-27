"""One-time, conservative migration from legacy result layouts."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.data.dataset_utils import project_root
from src.utils.result_paths import (
    class_balance_dir,
    evaluations_root,
    experiment_results_dir,
    latest_predictions_dir,
    legacy_dir,
    predictions_root,
    preprocessing_visualization_dir,
    results_root,
)

CLASS_BALANCE_ARTIFACTS = (
    "class_imbalance_and_bbox_analysis.json",
    "class_imbalance_and_bbox_analysis.md",
    "bbox_size_statistics.csv",
)


def _ensure_within_results(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    result_root = results_root(root).resolve()
    if result_root.parent != root.resolve():
        raise ValueError(f"Refusing to use results directory outside project root: {result_root}")
    if resolved != result_root and result_root not in resolved.parents:
        raise ValueError(f"Refusing to migrate path outside {result_root}: {resolved}")
    return resolved


def _collision_path(path: Path) -> Path:
    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}.legacy_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def _move_preserving_all(source: Path, destination: Path, root: Path, actions: list[str]) -> None:
    """Move a file/tree into results without overwriting any existing artifact."""
    source = _ensure_within_results(source, root)
    destination = _ensure_within_results(destination, root)
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.move(str(source), str(destination))
        actions.append(f"Moved {source} -> {destination}")
        return
    if source.is_dir() and destination.is_dir():
        for child in list(source.iterdir()):
            _move_preserving_all(child, destination / child.name, root, actions)
        source.rmdir()
        return
    collision = _collision_path(destination)
    shutil.move(str(source), str(collision))
    actions.append(f"Preserved collision {source} -> {collision}")


def _has_content(path: Path) -> bool:
    return path.is_dir() and next(path.iterdir(), None) is not None


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def preview_migration(root: Path | None = None) -> list[str]:
    """Plan the migration using read-only checks; never modify the filesystem."""
    root = (root or project_root()).resolve()
    result_root = results_root(root)
    if result_root.exists() and result_root.resolve().parent != root:
        raise ValueError(f"Refusing to use results directory outside project root: {result_root.resolve()}")
    actions: list[str] = []

    for filename in CLASS_BALANCE_ARTIFACTS:
        source = root / "data/reports" / filename
        if source.is_file():
            destination = class_balance_dir(root) / filename
            if destination.exists():
                destination = legacy_dir(root) / "class_balance_previous" / filename
            actions.append(f"MOVE {_relative(source, root)} -> {_relative(destination, root)}")

    old_preprocessing = result_root / "preprocessing_visualizations"
    if old_preprocessing.exists():
        destination = preprocessing_visualization_dir(root)
        if destination.exists() and _has_content(destination):
            destination = legacy_dir(root) / "preprocessing_visualizations"
        actions.append(f"MOVE {_relative(old_preprocessing, root)} -> {_relative(destination, root)}")

    prediction_root = predictions_root(root)
    old_predictions = []
    if prediction_root.is_dir():
        old_predictions = sorted(
            path for path in prediction_root.iterdir()
            if path.is_file()
            and path.name.lower().startswith("pred_")
            and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
    for image in old_predictions:
        destination = legacy_dir(root) / "predictions_previous" / image.name
        actions.append(f"MOVE {_relative(image, root)} -> {_relative(destination, root)}")
    if old_predictions and not latest_predictions_dir(root).exists():
        actions.append(f"CREATE {_relative(latest_predictions_dir(root), root)}")

    old_yolo = result_root / "yolo"
    if old_yolo.exists():
        actions.append(f"MOVE {_relative(old_yolo, root)} -> {_relative(legacy_dir(root) / 'yolo', root)}")
    return actions


def initialize_results(root: Path | None = None) -> list[str]:
    """Create the canonical directory skeleton without deleting existing content."""
    root = (root or project_root()).resolve()
    result_root = results_root(root)
    if result_root.exists() and result_root.resolve().parent != root:
        raise ValueError(f"Refusing to use results directory outside project root: {result_root.resolve()}")
    targets = (
        result_root,
        latest_predictions_dir(root),
        evaluations_root(root),
        experiment_results_dir(root),
        preprocessing_visualization_dir(root),
        class_balance_dir(root),
        legacy_dir(root),
    )
    actions = []
    for target in targets:
        if target.exists():
            if not target.is_dir():
                raise ValueError(f"Cannot initialize result directory over a file: {target}")
            continue
        _ensure_within_results(target, root)
        target.mkdir(parents=True, exist_ok=True)
        actions.append(f"CREATE {_relative(target, root)}")
    return actions


def migrate_results(root: Path | None = None) -> list[str]:
    """Migrate known legacy result locations and return a human-readable action log."""
    root = (root or project_root()).resolve()
    result_root = results_root(root)
    if result_root.exists() and result_root.resolve().parent != root:
        raise ValueError(f"Refusing to use results directory outside project root: {result_root.resolve()}")
    result_root.mkdir(parents=True, exist_ok=True)
    actions: list[str] = []

    # These three legacy analysis files are not dataset-integrity reports. Only
    # exact known names under data/reports are eligible; data/raw and
    # data/processed are never traversed or modified.
    legacy_analysis_root = root / "data/reports"
    for filename in CLASS_BALANCE_ARTIFACTS:
        source = legacy_analysis_root / filename
        if not source.is_file():
            continue
        destination = class_balance_dir(root) / filename
        if destination.exists():
            destination = legacy_dir(root) / "class_balance_previous" / filename
        destination = _ensure_within_results(destination, root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination = _collision_path(destination)
        shutil.move(str(source), str(destination))
        actions.append(f"Moved legacy class-balance artifact {source} -> {destination}")

    old_preprocessing = result_root / "preprocessing_visualizations"
    new_preprocessing = preprocessing_visualization_dir(root)
    if old_preprocessing.exists():
        if not new_preprocessing.exists() or not _has_content(new_preprocessing):
            if new_preprocessing.exists():
                new_preprocessing.rmdir()
            _move_preserving_all(old_preprocessing, new_preprocessing, root, actions)
        else:
            _move_preserving_all(
                old_preprocessing,
                legacy_dir(root) / "preprocessing_visualizations",
                root,
                actions,
            )

    prediction_root = predictions_root(root)
    old_predictions = []
    if prediction_root.is_dir():
        old_predictions = sorted(
            path for path in prediction_root.iterdir()
            if path.is_file()
            and path.name.lower().startswith("pred_")
            and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
    if old_predictions:
        previous = legacy_dir(root) / "predictions_previous"
        for image in old_predictions:
            _move_preserving_all(image, previous / image.name, root, actions)
        latest_predictions_dir(root).mkdir(parents=True, exist_ok=True)
        actions.append(f"Ensured current prediction directory: {latest_predictions_dir(root)}")

    old_yolo = result_root / "yolo"
    if old_yolo.exists():
        _move_preserving_all(old_yolo, legacy_dir(root) / "yolo", root, actions)

    return actions


def main(argv: list[str] | None = None, root: Path | None = None) -> int:
    parser = argparse.ArgumentParser(description="Safely migrate legacy generated result layouts")
    command = parser.add_mutually_exclusive_group(required=True)
    command.add_argument("--migrate", action="store_true", help="Perform the one-time migration")
    command.add_argument("--init", action="store_true", help="Create the canonical result directories")
    parser.add_argument("--dry-run", action="store_true", help="Preview migration operations without changes")
    args = parser.parse_args(argv)
    if args.dry_run and not args.migrate:
        parser.error("--dry-run is supported only with --migrate")
    if args.init:
        actions = initialize_results(root)
        if not actions:
            print("Results directory structure is already initialized.")
        else:
            print("Results directory initialized:")
            print(*actions, sep="\n")
        return 0

    actions = preview_migration(root) if args.dry_run else migrate_results(root)
    if not actions:
        print("Results layout is already organized. No migration required.")
    elif args.dry_run:
        print("Results migration preview (no changes made):")
        print(*actions, sep="\n")
    else:
        print("Results migration completed:")
        for action in actions:
            print(f"- {action}")
    return 0


if __name__ == "__main__":
    main()
