import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from src.detection.run_preprocessing_experiments import (ACTUAL_FAIR_KEYS, EXPERIMENTS, FAIR_KEYS,
    build_fair_training_configs, read_actual_training_args, validate_experiment_configs)
from src.detection.experiment_reporting import (RESULT_COLUMNS, benchmark_loaded_model,
    build_experiment_result_row, describe_preprocessing, flatten_evaluation_metrics, write_experiment_results)
from src.detection.train import validate_fraction_mode


class TestFairPreprocessingExperiments(unittest.TestCase):
    def test_required_experiment_names_and_identical_training_settings(self):
        names = [name for name, _ in EXPERIMENTS]
        self.assertEqual(names, ["A_raw", "B_gaussian", "C_median", "D_clahe", "E_brightness", "F_combined"])
        base = {key: index for index, key in enumerate(FAIR_KEYS)}
        base.update({"fraction": 1.0, "debug_only": False})
        datasets = {name: f"dataset/{name}.yaml" for name in names}
        configs = build_fair_training_configs(base, datasets)
        reference = {key: configs["A_raw"].get(key) for key in FAIR_KEYS}
        for name, config in configs.items():
            self.assertEqual({key: config.get(key) for key in FAIR_KEYS}, reference)
            self.assertEqual(config["data_yaml"], datasets[name])
            self.assertEqual(config["name"], name)

    def test_final_comparison_rejects_partial_fraction(self):
        datasets = {name: f"dataset/{name}.yaml" for name, _ in EXPERIMENTS}
        with self.assertRaisesRegex(ValueError, "fraction: 1.0"):
            build_fair_training_configs({"fraction": 0.5, "debug_only": True}, datasets)

    def test_partial_training_requires_explicit_debug_only(self):
        with self.assertRaisesRegex(ValueError, "DEBUG ONLY"):
            validate_fraction_mode({"fraction": 0.5})
        self.assertEqual(validate_fraction_mode({"fraction": 0.5, "debug_only": True}), (0.5, True))
        self.assertEqual(validate_fraction_mode({"fraction": 1.0}), (1.0, False))

    def test_actual_fair_keys_cover_fraction_model_and_optimizer(self):
        self.assertTrue({"model", "fraction", "epochs", "batch", "imgsz", "optimizer", "lr0", "seed"}
                        .issubset(ACTUAL_FAIR_KEYS))

    def test_repository_configs_have_matching_names_and_methods(self):
        validate_experiment_configs(Path(__file__).resolve().parents[1])
        median = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/experiments/median_denoise.yaml").read_text(encoding="utf-8"))
        self.assertEqual(median["experiment_name"], "C_median")
        self.assertTrue(median["denoise"]["enabled"])
        self.assertEqual(median["denoise"]["method"], "median")

    def test_report_reads_actual_ultralytics_args(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args_path = root / "experiments/yolo/A_raw/args.yaml"
            args_path.parent.mkdir(parents=True)
            args_path.write_text("fraction: 0.25\nimgsz: 320\n", encoding="utf-8")
            self.assertEqual(read_actual_training_args(root, "A_raw"), {"fraction": 0.25, "imgsz": 320})

    def test_comparison_keeps_overall_per_class_and_confusion_metrics(self):
        metrics = {
            "overall": {"precision": 0.7, "recall": 0.6, "mAP50": 0.5, "mAP50_95": 0.4},
            "per_class": [{"class_name": name, "precision": 0.1, "recall": 0.2,
                           "mAP50": 0.3, "mAP50_95": 0.4} for name in ("pothole", "crack", "manhole")],
            "confusion_matrix": {"important_confusions": {
                "pothole_to_manhole": {"count": 3, "rate_over_actual": 0.2},
            }},
        }
        flattened = flatten_evaluation_metrics(metrics)
        self.assertEqual(flattened["overall_mAP50"], 0.5)
        self.assertEqual(flattened["pothole_mAP50"], 0.3)
        self.assertEqual(flattened["crack_recall"], 0.2)
        self.assertEqual(flattened["manhole_mAP50_95"], 0.4)
        self.assertEqual(flattened["pothole_to_manhole_count"], 3)

    def test_latency_is_read_plus_contract_preprocessing_plus_inference(self):
        class Model:
            def predict(self, _image, **_kwargs):
                return []

        times = iter([0.000, 0.001, 0.003, 0.010, 0.010, 0.012, 0.015, 0.025])
        config = {"preprocessing": {
            "resize": {"enabled": False}, "denoise": {"enabled": False},
            "contrast": {"enabled": False}, "brightness": {"enabled": False},
            "normalization": {"enabled": False}, "augmentation": {"enabled": False},
        }}
        latency = benchmark_loaded_model(Model(), config, [Path("one.jpg"), Path("two.jpg")], 640,
                                         warmup=0, clock=lambda: next(times),
                                         image_reader=lambda _path: np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertAlmostEqual(latency["read_image_ms"], 1.5)
        self.assertAlmostEqual(latency["preprocessing_ms"], 2.5)
        self.assertAlmostEqual(latency["inference_ms"], 8.5)
        self.assertAlmostEqual(latency["total_ms"], 12.5)
        self.assertAlmostEqual(latency["total_ms"], latency["read_image_ms"] + latency["preprocessing_ms"] + latency["inference_ms"])

    def test_results_row_has_exact_required_schema_and_runtime_values(self):
        metrics = {
            "overall": {"precision": 0.7, "recall": 0.6, "mAP50": 0.5, "mAP50_95": 0.4},
            "per_class": [{"class_name": name, "mAP50": value} for name, value in
                          (("pothole", 0.51), ("crack", 0.52), ("manhole", 0.53))],
        }
        args = {"epochs": 50, "batch": 16, "imgsz": 640, "fraction": 1.0, "seed": 42}
        contract = {"preprocessing": {"denoise": {"enabled": False},
                    "contrast": {"enabled": True, "method": "clahe"}, "brightness": {"enabled": False}}}
        row = build_experiment_result_row("D_clahe", contract, args, metrics,
                                          {"preprocessing_ms": 4.1, "inference_ms": 35.7, "total_ms": 39.9})
        self.assertEqual(list(row), RESULT_COLUMNS)
        self.assertEqual(row["preprocessing"], "clahe")
        self.assertEqual(row["epochs"], 50)
        self.assertEqual(row["pothole_mAP50"], 0.51)
        self.assertEqual(row["total_ms"], 39.9)
        with tempfile.TemporaryDirectory() as directory:
            output = write_experiment_results([row], Path(directory) / "experiment_results.csv")
            self.assertEqual(output.read_text(encoding="utf-8").splitlines()[0].split(","), RESULT_COLUMNS)

    def test_preprocessing_label_is_derived_from_enabled_config(self):
        self.assertEqual(describe_preprocessing({"denoise": {"enabled": False}}), "raw")
        self.assertEqual(describe_preprocessing({"denoise": {"enabled": True, "method": "gaussian"},
            "contrast": {"enabled": True, "method": "clahe"}, "brightness": {"enabled": True}}),
            "gaussian+clahe+brightness")

    def test_benchmark_feeds_gaussian_processed_image_not_raw_image(self):
        class Model:
            source = None
            def predict(self, image, **_kwargs):
                self.source = image.copy()

        raw = np.zeros((15, 15, 3), dtype=np.uint8); raw[7, 7] = 255
        model = Model()
        config = {"preprocessing": {
            "resize": {"enabled": False},
            "denoise": {"enabled": True, "method": "gaussian", "gaussian_kernel": [3, 3], "gaussian_sigma": 0.8},
            "contrast": {"enabled": False}, "brightness": {"enabled": False},
            "normalization": {"enabled": False}, "augmentation": {"enabled": False},
        }}
        ticks = iter([0.0, 0.001, 0.002, 0.003])
        benchmark_loaded_model(model, config, [Path("image.jpg")], 640, warmup=0,
                               clock=lambda: next(ticks), image_reader=lambda _path: raw.copy())
        self.assertFalse(np.array_equal(model.source, raw))
        self.assertLess(model.source[7, 7, 0], 255)


if __name__ == "__main__":
    unittest.main()
