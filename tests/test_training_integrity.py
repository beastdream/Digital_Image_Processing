import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from src.data.dataset_utils import dataset_yaml_text
from src.data.training_integrity import DatasetIntegrityError, run_dataset_integrity_check
from src.detection.checkpoint_validator import validate_training_checkpoints
from src.detection.experiment_suite import load_experiment_suite, suite_experiments, suite_training_config
from src.detection.train import _checkpoint_paths, _prepare_experiment_directory, main, train_yolo


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


class TestTrainingLifecycle(unittest.TestCase):
    def _config(self, root: Path, **updates) -> Path:
        config = {
            "data_yaml": "dataset.yaml", "project": "experiments/yolo", "name": "run",
            "model_weights": "model.pt", "epochs": 10, "batch": 16, "imgsz": 640,
            "fraction": 1.0, "debug_only": False, "device": "cpu", "seed": 42,
        }
        config.update(updates)
        path = root / "train.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return path

    def _train_patches(self, root: Path, fake_yolo):
        return (
            patch("src.detection.train.project_root", return_value=root),
            patch("src.detection.train.run_dataset_integrity_check", return_value={"status": "PASSED"}),
            patch("src.detection.train.YOLO", fake_yolo),
            patch("src.detection.train.write_experiment_config"),
            patch("src.detection.train.torch.set_num_threads"),
        )

    def test_checkpoint_gate_requires_trainer_best_and_last(self):
        with tempfile.TemporaryDirectory() as directory:
            save_dir = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "trainer was not created"):
                _checkpoint_paths(SimpleNamespace(trainer=None))
            model = SimpleNamespace(trainer=SimpleNamespace(
                save_dir=save_dir, best=save_dir / "weights/best.pt",
                last=save_dir / "weights/last.pt"))
            with self.assertRaisesRegex(RuntimeError, "without producing best.pt"):
                _checkpoint_paths(model)
            (save_dir / "weights").mkdir()
            (save_dir / "weights/best.pt").write_bytes(b"best")
            with self.assertRaisesRegex(RuntimeError, "without producing last.pt"):
                _checkpoint_paths(model)
            (save_dir / "weights/last.pt").write_bytes(b"last")
            self.assertEqual(_checkpoint_paths(model)[0], save_dir.resolve())
            state = validate_training_checkpoints(save_dir)
            self.assertTrue(state["best_exists"])
            self.assertTrue(state["last_exists"])

    def test_incomplete_policy_archives_only_with_flag_and_refuses_valid_run(self):
        with tempfile.TemporaryDirectory() as directory:
            exp_dir = Path(directory) / "experiments/yolo/A_raw"
            exp_dir.mkdir(parents=True)
            (exp_dir / "partial.txt").write_text("partial", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "--overwrite-incomplete"):
                _prepare_experiment_directory(exp_dir, False)
            archived = _prepare_experiment_directory(exp_dir, True)
            self.assertFalse(exp_dir.exists())
            self.assertTrue((archived / "partial.txt").is_file())

            (exp_dir / "weights").mkdir(parents=True)
            (exp_dir / "weights/best.pt").write_bytes(b"valid")
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                _prepare_experiment_directory(exp_dir, True)

    def test_train_exception_is_reraised_and_failed_status_is_written(self):
        class FailingYOLO:
            def __init__(self, _weights):
                pass

            def train(self, **_kwargs):
                raise RuntimeError("trainer exploded")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config = self._config(root)
            patches = self._train_patches(root, FailingYOLO)
            stderr = io.StringIO()
            with patches[0], patches[1], patches[2], patches[3], patches[4], \
                    redirect_stderr(stderr), self.assertRaisesRegex(RuntimeError, "trainer exploded"):
                train_yolo(str(config))
            status = json.loads((root / "experiments/yolo/run/training_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "FAILED")
            self.assertFalse(status["checkpoint_created"])
            self.assertIn("Traceback", stderr.getvalue())
            self.assertFalse((root / "experiments/yolo/run/experiment_config.yaml").exists())

    def test_keyboard_interrupt_is_reraised_and_marked_interrupted_without_success(self):
        class InterruptedYOLO:
            def __init__(self, _weights):
                pass

            def train(self, **_kwargs):
                raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config = self._config(root)
            patches = self._train_patches(root, InterruptedYOLO)
            stdout, stderr = io.StringIO(), io.StringIO()
            with patches[0], patches[1], patches[2], patches[3], patches[4], \
                    redirect_stdout(stdout), redirect_stderr(stderr), self.assertRaises(KeyboardInterrupt):
                train_yolo(str(config))
            status = json.loads((root / "experiments/yolo/run/training_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "INTERRUPTED")
            self.assertFalse(status["checkpoint_created"])
            self.assertIn("Training was interrupted before completion.", stderr.getvalue())
            self.assertIn("A usable best.pt may not exist.", stderr.getvalue())
            self.assertNotIn("TRAINING COMPLETED SUCCESSFULLY", stdout.getvalue())

    def test_success_uses_actual_trainer_directory_and_writes_final_metadata(self):
        class SuccessfulYOLO:
            def __init__(self, _weights):
                self.trainer = None

            def train(self, **kwargs):
                save_dir = Path(kwargs["project"]) / kwargs["name"]
                (save_dir / "weights").mkdir(parents=True, exist_ok=True)
                (save_dir / "weights/best.pt").write_bytes(b"best")
                (save_dir / "weights/last.pt").write_bytes(b"last")
                self.trainer = SimpleNamespace(save_dir=save_dir,
                    best=save_dir / "weights/best.pt", last=save_dir / "weights/last.pt")
                return SimpleNamespace(args={"epochs": kwargs["epochs"], "batch": kwargs["batch"],
                    "imgsz": kwargs["imgsz"], "fraction": kwargs["fraction"], "seed": kwargs["seed"]})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config = self._config(root)
            patches = self._train_patches(root, SuccessfulYOLO)
            stdout = io.StringIO()
            with patches[0], patches[1], patches[2], patches[3] as contract_writer, patches[4], redirect_stdout(stdout):
                best = train_yolo(str(config))
            exp_dir = root / "experiments/yolo/run"
            self.assertEqual(Path(best), (exp_dir / "weights/best.pt").resolve())
            contract_writer.assert_called_once()
            self.assertEqual(contract_writer.call_args.args[0], exp_dir.resolve())
            status = json.loads((exp_dir / "training_status.json").read_text(encoding="utf-8"))
            metadata = json.loads((exp_dir / "reproducibility_info.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "COMPLETED")
            self.assertTrue(status["checkpoint_created"])
            self.assertEqual(metadata["status"], "COMPLETED")
            effective = json.loads((exp_dir / "effective_training_args.json").read_text(encoding="utf-8"))
            self.assertTrue(effective["save"])
            self.assertEqual(effective["save_period"], -1)
            self.assertEqual(effective["project"], str(root / "experiments/yolo"))
            self.assertEqual(effective["name"], "run")
            self.assertIn("=== YOLO TRAINING RETURNED ===", stdout.getvalue())
            self.assertIn("best.pt: FOUND", stdout.getvalue())
            self.assertIn("=== TRAINING COMPLETED SUCCESSFULLY ===", stdout.getvalue())

    def test_smoke_test_overrides_are_in_memory_and_generate_both_checkpoints(self):
        captured = {}

        class SmokeYOLO:
            def __init__(self, _weights):
                self.trainer = None

            def train(self, **kwargs):
                captured.update(kwargs)
                save_dir = Path(kwargs["project"]) / kwargs["name"]
                (save_dir / "weights").mkdir(parents=True, exist_ok=True)
                for name in ("best.pt", "last.pt"):
                    (save_dir / "weights" / name).write_bytes(name.encode())
                self.trainer = SimpleNamespace(save_dir=save_dir,
                    best=save_dir / "weights/best.pt", last=save_dir / "weights/last.pt")
                return SimpleNamespace(args=kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config = self._config(root, name="smoke")
            original = config.read_bytes()
            patches = self._train_patches(root, SmokeYOLO)
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                train_yolo(str(config), smoke_test=True)
            self.assertEqual(config.read_bytes(), original)
            self.assertEqual(captured["epochs"], 1)
            self.assertEqual(captured["fraction"], 0.05)
            self.assertEqual(captured["imgsz"], 320)
            self.assertLessEqual(captured["batch"], 4)
            self.assertTrue((root / "experiments/yolo/smoke/weights/best.pt").is_file())
            self.assertTrue((root / "experiments/yolo/smoke/weights/last.pt").is_file())

    def test_cli_forwards_new_flags(self):
        with patch("src.detection.train.train_yolo") as trainer:
            self.assertEqual(main(["--config", "x.yaml", "--name", "A_raw", "--overwrite-incomplete", "--smoke-test"]), 0)
        trainer.assert_called_once_with("x.yaml", "A_raw", overwrite_incomplete=True, smoke_test=True)


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
