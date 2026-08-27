import tempfile
import unittest
import io
import json
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from src.data.dataset_utils import CLASS_NAMES
from src.detection.model_contract import (load_experiment_config, preprocess_for_inference,
    validate_evaluation_dataset, write_experiment_config)
from src.detection.predict import (VIDEO_ERROR, collect_image_paths, read_valid_image,
    main, resolve_prediction_assets, run_prediction)


class _FakeResult:
    boxes = []


class _FakeYOLO:
    last_source = None
    names = CLASS_NAMES

    def __init__(self, _weights):
        pass

    def predict(self, source, **_kwargs):
        _FakeYOLO.last_source = source.copy()
        return [_FakeResult()]


class _ArrayLike:
    def __init__(self, value):
        self.value = np.asarray(value)

    def __getitem__(self, index):
        return _ArrayLike(self.value[index])

    def __int__(self):
        return int(self.value.flat[0])

    def __float__(self):
        return float(self.value.flat[0])

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _FakeDetectionResult:
    boxes = [type("Box", (), {
        "cls": _ArrayLike([0]),
        "conf": _ArrayLike([0.78]),
        "xyxy": _ArrayLike([[1, 2, 10, 12]]),
    })()]


class _FakeDetectionYOLO(_FakeYOLO):
    def predict(self, source, **_kwargs):
        _FakeYOLO.last_source = source.copy()
        return [_FakeDetectionResult()]


class TestModelPreprocessingContract(unittest.TestCase):
    def _metadata(self, root: Path, preprocessing: dict, name: str = "D_clahe") -> Path:
        data_dir = root / "dataset"
        data_dir.mkdir(parents=True)
        data_yaml = data_dir / "dataset.yaml"
        data_yaml.write_text("nc: 3\nnames: [pothole, crack, manhole]\n", encoding="utf-8")
        config_path = root / "preprocessing.yaml"
        config_path.write_text(yaml.safe_dump(preprocessing), encoding="utf-8")
        exp_dir = root / "experiment"
        exp_dir.mkdir()
        write_experiment_config(exp_dir, {
            "name": name, "preprocessing_config": str(config_path), "model_weights": "yolov8n.pt",
            "imgsz": 640, "epochs": 10, "batch": 16, "seed": 42, "fraction": 1.0,
        }, data_yaml, root=root)
        return exp_dir / "experiment_config.yaml"

    def test_prediction_loads_metadata_and_feeds_preprocessed_pixels_to_yolo(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preprocessing = {
                "resize": {"enabled": False},
                "denoise": {"enabled": True, "method": "gaussian", "gaussian_kernel": [3, 3], "gaussian_sigma": 0.8},
                "contrast": {"enabled": False}, "brightness": {"enabled": False},
                "normalization": {"enabled": True, "mode": "pixel_scale"}, "augmentation": {"enabled": False},
            }
            metadata_path = self._metadata(root, preprocessing, "B_gaussian")
            source = root / "input.jpg"
            raw = np.zeros((24, 24, 3), dtype=np.uint8); raw[12, 12] = 255
            cv2.imwrite(str(source), raw)
            decoded = cv2.imread(str(source))
            expected, _ = preprocess_for_inference(decoded, load_experiment_config("unused.pt", metadata_path))
            with patch("src.detection.predict.YOLO", _FakeYOLO):
                count = run_prediction("arbitrary_model_name.pt", str(source), str(root / "out"),
                                       max_images=1, experiment_config_path=str(metadata_path))
            self.assertEqual(count, 1)
            np.testing.assert_array_equal(_FakeYOLO.last_source, expected)
            self.assertFalse(np.array_equal(_FakeYOLO.last_source, decoded))
            summary = json.loads((root / "out/prediction_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["processed_images"], 1)
            self.assertEqual(summary["images"], [{"filename": "input.jpg", "detections": []}])

    def test_prediction_default_replaces_latest_and_writes_detection_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_path = self._metadata(root, {
                "resize": {"enabled": False}, "denoise": {"enabled": False},
                "contrast": {"enabled": False}, "brightness": {"enabled": False},
                "normalization": {"enabled": False}, "augmentation": {"enabled": False},
            }, "A_raw")
            source = root / "input.jpg"
            cv2.imwrite(str(source), np.zeros((24, 24, 3), dtype=np.uint8))
            latest = root / "results/predictions/latest"
            latest.mkdir(parents=True)
            (latest / "stale.jpg").write_bytes(b"old")
            with patch("src.detection.predict.project_root", return_value=root), \
                    patch("src.detection.predict.YOLO", _FakeDetectionYOLO):
                count = run_prediction("model.pt", str(source), max_images=1,
                                       experiment_config_path=str(metadata_path),
                                       experiment_name="A_raw")
            self.assertEqual(count, 1)
            self.assertFalse((latest / "stale.jpg").exists())
            summary = json.loads((latest / "prediction_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["experiment"], "A_raw")
            self.assertEqual(summary["classes"], {"0": "pothole", "1": "crack", "2": "manhole"})
            self.assertEqual(summary["images"][0]["detections"][0], {
                "class_id": 0, "class_name": "pothole", "confidence": 0.78,
                "xyxy": [1, 2, 10, 12],
            })

    def test_custom_prediction_output_is_preserved_unless_clean_is_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_path = self._metadata(root, {
                "resize": {"enabled": False}, "augmentation": {"enabled": False},
            }, "A_raw")
            source = root / "input.jpg"
            cv2.imwrite(str(source), np.zeros((8, 8, 3), dtype=np.uint8))
            custom = root / "results/predictions/demo"
            custom.mkdir(parents=True)
            stale = custom / "keep.txt"
            stale.write_text("keep", encoding="utf-8")
            with patch("src.detection.predict.project_root", return_value=root), \
                    patch("src.detection.predict.YOLO", _FakeYOLO):
                run_prediction("model.pt", str(source), str(custom),
                               experiment_config_path=str(metadata_path))
                self.assertTrue(stale.exists())
                run_prediction("model.pt", str(source), str(custom),
                               experiment_config_path=str(metadata_path), clean_output=True)
            self.assertFalse(stale.exists())

    def test_clean_custom_prediction_output_refuses_path_outside_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jpg"
            cv2.imwrite(str(source), np.zeros((8, 8, 3), dtype=np.uint8))
            with patch("src.detection.predict.project_root", return_value=root):
                with self.assertRaisesRegex(ValueError, "outside"):
                    run_prediction("model.pt", str(source), str(root / "elsewhere"),
                                   clean_output=True)

    def test_processed_model_rejects_raw_evaluation_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_path = self._metadata(root, {
                "resize": {"enabled": False}, "denoise": {"enabled": False},
                "contrast": {"enabled": True, "method": "clahe", "clahe_clip_limit": 2.0,
                             "clahe_tile_grid_size": [8, 8]}, "brightness": {"enabled": False},
                "normalization": {"enabled": True, "mode": "pixel_scale"}, "augmentation": {"enabled": False},
            })
            contract = load_experiment_config("not_named_after_preprocessing.pt", metadata_path)
            with self.assertRaisesRegex(ValueError, "cannot be evaluated on raw"):
                validate_evaluation_dataset(root / "dataset/dataset.yaml", contract)

    def test_contract_is_copied_next_to_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_path = self._metadata(root, {"resize": {"enabled": False}, "augmentation": {"enabled": False}}, "A_raw")
            self.assertTrue(metadata_path.exists())
            self.assertTrue((metadata_path.parent / "weights/experiment_config.yaml").exists())
            self.assertEqual(load_experiment_config("unused.pt", metadata_path)["classes"], CLASS_NAMES)

    def test_a_raw_contract_keeps_raw_pixels_unchanged(self):
        raw = np.random.default_rng(42).integers(0, 256, (18, 25, 3), dtype=np.uint8)
        prepared, metadata = preprocess_for_inference(raw, {"preprocessing": {
            "resize": {"enabled": False}, "denoise": {"enabled": False},
            "contrast": {"enabled": False}, "brightness": {"enabled": False},
            "normalization": {"enabled": False}, "augmentation": {"enabled": False},
        }})
        np.testing.assert_array_equal(prepared, raw)
        self.assertEqual(metadata["steps_executed"], [])

    def test_folder_source_collects_only_supported_static_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("b.png", "a.jpg", "ignored.txt", "ignored.mp4"):
                (root / name).write_bytes(b"placeholder")
            self.assertEqual([path.name for path in collect_image_paths(root)], ["a.jpg", "b.png"])

    def test_direct_video_source_returns_required_image_only_error(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "road.mp4"
            video.write_bytes(b"not a real video")
            with self.assertRaisesRegex(ValueError, f"^{VIDEO_ERROR.replace('.', '[.]')}$"):
                collect_image_paths(video)

    def test_experiment_resolves_weights_and_metadata_by_directory_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = Path(directory) / "any_directory_name"
            (experiment / "weights").mkdir(parents=True)
            (experiment / "weights/best.pt").write_bytes(b"weights")
            (experiment / "experiment_config.yaml").write_text("schema_version: 1", encoding="utf-8")
            weights, metadata = resolve_prediction_assets(experiment_dir=str(experiment))
            self.assertEqual(weights, experiment / "weights/best.pt")
            self.assertEqual(metadata, experiment / "experiment_config.yaml")

    def test_corrupt_image_fails_validation_instead_of_being_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "broken.jpg"
            image.write_bytes(b"not an image")
            with self.assertRaisesRegex(ValueError, "Invalid or unreadable image"):
                read_valid_image(image)

    def test_cli_reports_video_error_before_attempting_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "road.mp4"
            video.write_bytes(b"video")
            stderr = io.StringIO()
            with redirect_stderr(stderr), self.assertRaises(SystemExit):
                main(["--model", "missing.pt", "--image", str(video)])
            self.assertIn(VIDEO_ERROR, stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
