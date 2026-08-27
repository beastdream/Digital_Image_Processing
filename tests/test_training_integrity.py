import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from src.data.dataset_utils import dataset_yaml_text
from src.data.training_integrity import DatasetIntegrityError, run_dataset_integrity_check
from src.detection.experiment_suite import load_experiment_suite, suite_experiments, suite_training_config
from src.detection.train import train_yolo


class TestTrainingIntegrityGate(unittest.TestCase):
    def _dataset(self, root: Path, duplicate=False, bad_class=False, missing_label=False,
                 missing_image=False, invalid_bbox=False) -> Path:
        dataset = root / "dataset"
        shared = np.full((12, 20, 3), 90, np.uint8)
        for index, split in enumerate(("train", "val", "test")):
            (dataset / "images" / split).mkdir(parents=True)
            (dataset / "labels" / split).mkdir(parents=True)
            image = shared if duplicate and split in ("train", "val") else np.full((12, 20, 3), 30 + index * 50, np.uint8)
            if not (missing_image and split == "test"):
                cv2.imwrite(str(dataset / "images" / split / f"unique_{split}.png"), image)
            if not (missing_label and split == "val"):
                class_id = 9 if bad_class and split == "train" else index
                box = "0.5 0.5 1.2 0.4" if invalid_bbox and split == "train" else "0.5 0.5 0.4 0.4"
                (dataset / "labels" / split / f"unique_{split}.txt").write_text(f"{class_id} {box}", encoding="utf-8")
        yaml_path = dataset / "dataset.yaml"
        yaml_path.write_text(dataset_yaml_text(dataset), encoding="utf-8")
        return yaml_path

    def test_valid_dataset_prints_and_persists_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            yaml_path = self._dataset(Path(directory))
            report = run_dataset_integrity_check(yaml_path)
            self.assertEqual(report["status"], "PASSED")
            self.assertEqual(report["statistics"]["total_images"], 3)
            self.assertEqual(report["statistics"]["total_objects"], 3)
            self.assertTrue((yaml_path.parent / "reports/training_integrity_report.json").exists())

    def test_invalid_class_or_missing_label_fails_fast_with_clear_error(self):
        for option, expected in (({"bad_class": True}, "unsupported class_id 9"),
                                 ({"missing_label": True}, "missing label file"),
                                 ({"missing_image": True}, "orphan label file"),
                                 ({"invalid_bbox": True}, "width and height must be")):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as directory:
                yaml_path = self._dataset(Path(directory), **option)
                with self.assertRaisesRegex(DatasetIntegrityError, expected):
                    run_dataset_integrity_check(yaml_path)

    def test_exact_duplicate_cross_split_fails_fast(self):
        with tempfile.TemporaryDirectory() as directory:
            yaml_path = self._dataset(Path(directory), duplicate=True)
            with self.assertRaisesRegex(DatasetIntegrityError, "exact duplicate image leakage across splits"):
                run_dataset_integrity_check(yaml_path)

    def test_train_does_not_construct_model_when_integrity_gate_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "train.yaml"
            config.write_text(yaml.safe_dump({"data_yaml": "broken.yaml", "fraction": 1.0, "seed": 42}), encoding="utf-8")
            with patch("src.detection.train.run_dataset_integrity_check", side_effect=DatasetIntegrityError("broken dataset")), \
                 patch("src.detection.train.YOLO") as yolo:
                with self.assertRaisesRegex(DatasetIntegrityError, "broken dataset"):
                    train_yolo(str(config))
                yolo.assert_not_called()


class TestConfigDrivenExperimentSuite(unittest.TestCase):
    def test_repository_suite_drives_names_classes_and_training(self):
        suite = load_experiment_suite()
        self.assertEqual([name for name, _ in suite_experiments(suite)],
                         ["A_raw", "B_gaussian", "C_median", "D_clahe", "E_brightness", "F_combined"])
        training = suite_training_config(suite)
        self.assertEqual(training["model_weights"], "yolov8n.pt")
        self.assertEqual(training["seed"], 42)
        self.assertEqual(training["fraction"], 1.0)

    def test_custom_experiment_names_are_loaded_without_python_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "configs").mkdir()
            for name in ("custom_raw", "custom_filter"):
                (root / "configs" / f"{name}.yaml").write_text(f"experiment_name: {name}\n", encoding="utf-8")
            suite = {"project": {"seed": 7, "output": "runs", "results": "results"},
                "dataset": {"source_yaml": "dataset.yaml", "classes": {0: "pothole", 1: "crack", 2: "manhole"}},
                "training": {"model": "model.pt", "epochs": 1, "batch": 2, "imgsz": 320,
                             "fraction": 1.0, "optimizer": "Auto", "learning_rate": 0.01, "patience": 1},
                "experiments": [{"name": name, "preprocessing_config": f"configs/{name}.yaml"}
                                for name in ("custom_raw", "custom_filter")]}
            suite_path = root / "suite.yaml"; suite_path.write_text(yaml.safe_dump(suite), encoding="utf-8")
            loaded = load_experiment_suite(suite_path, root)
            self.assertEqual([name for name, _ in suite_experiments(loaded)], ["custom_raw", "custom_filter"])


if __name__ == "__main__":
    unittest.main()
