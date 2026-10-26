from pathlib import Path

import cv2
import numpy as np
import yaml

from src.data.dataset_utils import dataset_yaml_text
from src.data.materialize_preprocessed_dataset import materialize_preprocessed_dataset
from src.detection.experiment_suite import load_experiment_suite, suite_experiments, suite_training_config
from src.detection.run_preprocessing_experiments import completed_weights, read_actual_training_args


def test_video_suite_matches_image_suite_except_dataset_and_outputs() -> None:
    image_suite = load_experiment_suite("configs/preprocessing_experiments.yaml")
    video_suite = load_experiment_suite("configs/video_preprocessing_experiments.yaml")
    assert suite_experiments(video_suite) == suite_experiments(image_suite)
    assert video_suite["training"] == image_suite["training"]
    assert video_suite["dataset"]["source_yaml"] == "data/processed/pothole_video_frames/dataset.yaml"
    assert video_suite["project"]["output"] != image_suite["project"]["output"]
    assert suite_training_config(video_suite)["project"] == "experiments/yolo_video"


def test_materialize_uses_the_given_source_dataset(tmp_path: Path) -> None:
    source = tmp_path / "data/processed/pothole_video_frames"
    for split in ("train", "val", "test"):
        (source / "images" / split).mkdir(parents=True)
        (source / "labels" / split).mkdir(parents=True)
        cv2.imwrite(str(source / "images" / split / f"{split}_0001_f0000.jpg"), np.full((16, 16, 3), 90, np.uint8))
        (source / "labels" / split / f"{split}_0001_f0000.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    (source / "dataset.yaml").write_text(dataset_yaml_text(source), encoding="utf-8")
    config = tmp_path / "blur.yaml"
    config.write_text(yaml.safe_dump({"denoise": {"enabled": True, "method": "median", "kernel_size": 3},
                                      "augmentation": {"enabled": False}}), encoding="utf-8")
    output = materialize_preprocessed_dataset(config, "B_test", root_dir=tmp_path,
                                              source_yaml="data/processed/pothole_video_frames/dataset.yaml")
    assert output.parent.parent == source / "preprocessed"
    assert (output.parent / "images/val/val_0001_f0000.jpg").is_file()


def test_completed_runs_are_reused_and_incomplete_runs_are_not(tmp_path: Path) -> None:
    done = tmp_path / "A_raw"
    (done / "weights").mkdir(parents=True)
    for name in ("best.pt", "last.pt"):
        (done / "weights" / name).write_bytes(b"w")
    (done / "experiment_config.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    (done / "args.yaml").write_text("epochs: 10\n", encoding="utf-8")
    (tmp_path / "B_gaussian/weights").mkdir(parents=True)
    (tmp_path / "B_gaussian/weights/last.pt").write_bytes(b"w")
    assert completed_weights(tmp_path, "A_raw") == str(done / "weights/best.pt")
    assert completed_weights(tmp_path, "B_gaussian") is None
    assert completed_weights(tmp_path, "C_median") is None
    assert read_actual_training_args(tmp_path, "A_raw", tmp_path) == {"epochs": 10}


def test_interrupted_run_with_best_pt_is_resumable_not_completed(tmp_path: Path) -> None:
    from src.detection.checkpoint_validator import is_completed_run, resumable_checkpoint
    run = tmp_path / "A_raw"
    (run / "weights").mkdir(parents=True)
    assert resumable_checkpoint(run) is None
    for name in ("best.pt", "last.pt"):
        (run / "weights" / name).write_bytes(b"w")
    assert not is_completed_run(run)
    assert resumable_checkpoint(run) == (run / "weights/last.pt").resolve()
    assert completed_weights(tmp_path, "A_raw") is None
    (run / "experiment_config.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    assert is_completed_run(run) and resumable_checkpoint(run) is None


def test_resume_rejects_conflicting_flags() -> None:
    import pytest
    from src.detection.train import train_yolo
    with pytest.raises(ValueError, match="--resume cannot be combined"):
        train_yolo("configs/yolo_training.yaml", resume=True, smoke_test=True)


def test_ultralytics_label_cache_and_machine_path_do_not_change_dataset_version(tmp_path: Path) -> None:
    from src.data.materialize_preprocessed_dataset import _source_split_fingerprint
    source = tmp_path / "frames"
    (source / "images/train").mkdir(parents=True)
    (source / "labels/train").mkdir(parents=True)
    cv2.imwrite(str(source / "images/train/a.jpg"), np.zeros((8, 8, 3), np.uint8))
    (source / "labels/train/a.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    (source / "dataset.yaml").write_text(dataset_yaml_text(source), encoding="utf-8")
    before = _source_split_fingerprint(source)
    (source / "labels/train.cache").write_bytes(b"ultralytics cache")
    (source / "dataset.yaml").write_text(dataset_yaml_text(Path("/content/elsewhere")), encoding="utf-8")
    assert _source_split_fingerprint(source) == before
    (source / "labels/train/a.txt").write_text("0 0.4 0.5 0.5 0.5\n", encoding="utf-8")
    assert _source_split_fingerprint(source) != before
<<<<<<< HEAD


def test_review_sample_is_seeded_and_grid_keeps_panel_size(tmp_path: Path) -> None:
    from src.detection.visual_inspection_grid import HEADER, PANEL_SIZE, _grid, _panel, sample_images
    for index in range(15):
        cv2.imwrite(str(tmp_path / f"f{index:02d}.jpg"), np.zeros((8, 8, 3), np.uint8))
    first = sample_images(tmp_path, 10, seed=42)
    assert first == sample_images(tmp_path, 10, seed=42) and len(set(first)) == 10
    panels = [_panel(np.zeros((100, 50, 3), np.uint8), f"p{index}") for index in range(7)]
    assert _grid(panels).shape == (2 * (PANEL_SIZE + HEADER), 4 * PANEL_SIZE, 3)
=======
>>>>>>> da7e1cd4 (Cap nhat code)
