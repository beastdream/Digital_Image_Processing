import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.detection.evaluate import (MATRIX_LABELS, analyze_confusions, extract_confusion_matrix,
    extract_detection_metrics, prepare_evaluation_output, save_evaluation_artifacts)


class TestEvaluationMetrics(unittest.TestCase):
    def setUp(self):
        all_ap = np.array([
            [0.80, 0.75, 0.70],
            [0.60, 0.55, 0.50],
            [0.70, 0.65, 0.60],
        ])
        self.matrix = np.array([
            [8, 0, 2, 1],   # predicted pothole
            [0, 5, 0, 1],   # predicted crack
            [3, 0, 6, 1],   # predicted manhole
            [4, 7, 1, 0],   # predicted background (false negatives)
        ], dtype=float)
        self.metrics = SimpleNamespace(
            results_dict={
                "metrics/precision(B)": 0.71, "metrics/recall(B)": 0.62,
                "metrics/mAP50(B)": 0.70, "metrics/mAP50-95(B)": 0.51,
            },
            box=SimpleNamespace(
                p=np.array([0.8, 0.6, 0.7]), r=np.array([0.7, 0.5, 0.6]),
                maps=np.array([0.55, 0.35, 0.45]), ap_class_index=np.array([0, 1, 2]), all_ap=all_ap,
            ),
            confusion_matrix=SimpleNamespace(matrix=self.matrix),
        )

    def test_extracts_overall_and_every_named_class_metric(self):
        summary = extract_detection_metrics(self.metrics)
        self.assertEqual(summary["overall"], {
            "precision": 0.71, "recall": 0.62, "mAP50": 0.70, "mAP50_95": 0.51,
        })
        self.assertEqual([row["class_name"] for row in summary["per_class"]],
                         ["pothole", "crack", "manhole"])
        self.assertEqual(summary["per_class"][0]["mAP50"], 0.8)
        self.assertEqual(summary["per_class"][1]["mAP50_95"], 0.35)

    def test_confusion_matrix_has_three_classes_plus_background(self):
        extracted = extract_confusion_matrix(self.metrics)
        self.assertEqual(extracted.shape, (4, 4))
        self.assertEqual(MATRIX_LABELS, ["pothole", "crack", "manhole", "background"])

    def test_required_confusion_directions_use_predicted_rows_and_actual_columns(self):
        analysis = analyze_confusions(self.matrix)
        self.assertEqual(analysis["pothole_to_manhole"]["count"], 3)
        self.assertEqual(analysis["manhole_to_pothole"]["count"], 2)
        self.assertEqual(analysis["crack_to_background"]["count"], 7)
        self.assertEqual(analysis["pothole_to_background"]["count"], 4)
        self.assertAlmostEqual(analysis["pothole_to_manhole"]["rate_over_actual"], 3 / 15, places=6)

    def test_saves_metric_tables_matrices_plots_and_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = extract_detection_metrics(self.metrics)
            save_evaluation_artifacts(output, summary, self.matrix)
            expected = {
                "metrics_by_scope.csv", "confusion_matrix.csv", "confusion_matrix_normalized.csv",
                "confusion_matrix.png", "confusion_matrix_normalized.png", "confusion_analysis.json",
                "confusion_analysis.md",
            }
            self.assertTrue(all((output / name).is_file() for name in expected))
            analysis = json.loads((output / "confusion_analysis.json").read_text(encoding="utf-8"))
            self.assertEqual(analysis["orientation"], "rows=predicted, columns=actual")
            self.assertEqual(analysis["important_confusions"]["crack_to_background"]["count"], 7)
            csv_text = (output / "metrics_by_scope.csv").read_text(encoding="utf-8")
            self.assertIn("overall", csv_text)
            self.assertIn("pothole", csv_text)
            self.assertIn("crack", csv_text)
            self.assertIn("manhole", csv_text)

    def test_default_evaluation_replaces_only_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latest = root / "results/evaluations/latest"
            other = root / "results/evaluations/B_gaussian"
            latest.mkdir(parents=True)
            other.mkdir(parents=True)
            (latest / "stale.json").write_text("old", encoding="utf-8")
            (other / "metrics.json").write_text("keep", encoding="utf-8")
            self.assertEqual(prepare_evaluation_output(None, root=root), latest.resolve())
            self.assertFalse((latest / "stale.json").exists())
            self.assertTrue((other / "metrics.json").exists())

    def test_experiment_cleanup_replaces_only_requested_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected = root / "results/evaluations/A_raw"
            other = root / "results/evaluations/B_gaussian"
            selected.mkdir(parents=True)
            other.mkdir(parents=True)
            (selected / "stale.json").write_text("old", encoding="utf-8")
            (other / "metrics.json").write_text("keep", encoding="utf-8")
            prepare_evaluation_output(selected, clean_output=True, root=root)
            self.assertFalse((selected / "stale.json").exists())
            self.assertTrue((other / "metrics.json").exists())
            with self.assertRaisesRegex(ValueError, "outside"):
                prepare_evaluation_output(root / "experiments/yolo", clean_output=True, root=root)


if __name__ == "__main__":
    unittest.main()
