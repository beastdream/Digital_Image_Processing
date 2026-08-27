from pathlib import Path
from unittest.mock import patch

import pytest

from src.utils.result_paths import (
    experiment_evaluation_dir,
    latest_predictions_dir,
    preprocessing_visualization_dir,
    reset_directory,
    results_root,
)
from src.preprocessing.visualize import visualize_preprocessing_samples


def test_canonical_result_paths(tmp_path: Path) -> None:
    assert latest_predictions_dir(tmp_path) == tmp_path / "results/predictions/latest"
    assert experiment_evaluation_dir("A_raw", tmp_path) == tmp_path / "results/evaluations/A_raw"
    assert preprocessing_visualization_dir(tmp_path) == tmp_path / "results/preprocessing/visualizations"


def test_reset_directory_replaces_only_guarded_child(tmp_path: Path) -> None:
    root = results_root(tmp_path)
    target = root / "predictions/latest"
    target.mkdir(parents=True)
    (target / "stale.jpg").write_bytes(b"old")

    assert reset_directory(target, root) == target.resolve()
    assert target.is_dir()
    assert list(target.iterdir()) == []


def test_reset_directory_refuses_root_and_outside_path(tmp_path: Path) -> None:
    root = results_root(tmp_path)
    root.mkdir()
    with pytest.raises(ValueError, match="protected result root"):
        reset_directory(root, root)
    with pytest.raises(ValueError, match="outside"):
        reset_directory(tmp_path / "experiments/yolo", root)


def test_reset_directory_refuses_real_project_roots() -> None:
    from src.data.dataset_utils import project_root

    root = project_root()
    for protected in (root, root / "data", root / "experiments", root / "experiments/yolo", root / "results"):
        with pytest.raises(ValueError, match="protected project directory"):
            reset_directory(protected, root)


@pytest.mark.parametrize("name", ["", ".", "..", "nested/name", "nested\\name"])
def test_experiment_evaluation_dir_rejects_unsafe_names(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError):
        experiment_evaluation_dir(name, tmp_path)


def test_preprocessing_visualization_run_resets_generated_samples(tmp_path: Path) -> None:
    output = preprocessing_visualization_dir(tmp_path)
    output.mkdir(parents=True)
    (output / "stale.jpg").write_bytes(b"old")
    with patch("src.preprocessing.visualize.project_root", return_value=tmp_path), \
            patch("src.preprocessing.visualize.ImagePreprocessingPipeline"):
        visualize_preprocessing_samples()
    assert output.is_dir()
    assert not (output / "stale.jpg").exists()
