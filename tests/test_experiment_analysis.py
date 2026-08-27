import csv
import tempfile
import unittest
from pathlib import Path

from src.detection.analyze_experiment_results import (EXPECTED_EXPERIMENTS, VISUAL_COLUMNS,
    analyze_results, create_visual_inspection_template, write_analysis)
from src.detection.experiment_reporting import RESULT_COLUMNS


def _result(name, preprocessing, pothole, crack, manhole, overall, latency):
    return {"experiment": name, "preprocessing": preprocessing, "epochs": 50, "batch": 16,
            "imgsz": 640, "fraction": 1.0, "seed": 42, "precision": overall - 0.02,
            "recall": overall - 0.01, "mAP50": overall, "mAP50_95": overall - 0.1,
            "pothole_mAP50": pothole, "crack_mAP50": crack, "manhole_mAP50": manhole,
            "preprocessing_ms": 2.0, "inference_ms": latency - 2.5, "total_ms": latency}


class TestFairExperimentAnalysisGate(unittest.TestCase):
    def _write_results(self, path: Path, unfair: bool = False):
        rows = [
            _result("A_raw", "raw", .50, .60, .55, .55, 30),
            _result("B_gaussian", "gaussian", .62, .52, .57, .57, 34),
            _result("C_median", "median", .55, .58, .59, .56, 33),
            _result("D_clahe", "clahe", .65, .48, .60, .58, 38),
            _result("E_brightness", "brightness", .54, .66, .52, .59, 32),
            _result("F_combined", "gaussian+clahe+brightness", .63, .61, .64, .60, 42),
        ]
        if unfair: rows[-1]["epochs"] = 4
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS); writer.writeheader(); writer.writerows(rows)

    def _complete_visual(self, path: Path):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=VISUAL_COLUMNS); writer.writeheader()
            for name in EXPECTED_EXPERIMENTS:
                writer.writerow({"experiment": name, "status": "COMPLETE", "reviewed_images": 10,
                    "bbox_alignment": "PASS", "texture_preservation": "PASS",
                    "representative_failures": "reviewed false positives and misses", "reviewer": "tester"})

    def test_missing_results_and_pending_visual_review_block_all_conclusions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); visual = create_visual_inspection_template(root / "visual.csv")
            report = analyze_results(root / "missing.csv", visual)
            self.assertEqual(report["conclusion_status"], "BLOCKED")
            self.assertIsNone(report["winner"])
            self.assertIn("No claim", report["message"])

    def test_visual_template_preserves_completed_reviews_and_appends_missing_experiments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "visual_inspection.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                fields = [*VISUAL_COLUMNS, "manual_note"]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "experiment": "A_raw", "status": "COMPLETE", "reviewed_images": 20,
                    "bbox_alignment": "PASS", "texture_preservation": "PASS",
                    "representative_failures": "documented", "reviewer": "human",
                    "manual_note": "do not lose this",
                })

            create_visual_inspection_template(path, ["A_raw", "B_gaussian"])
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                self.assertIn("manual_note", reader.fieldnames)
            self.assertEqual(rows[0]["status"], "COMPLETE")
            self.assertEqual(rows[0]["reviewed_images"], "20")
            self.assertEqual(rows[0]["reviewer"], "human")
            self.assertEqual(rows[0]["manual_note"], "do not lose this")
            self.assertEqual(rows[1]["experiment"], "B_gaussian")
            self.assertEqual(rows[1]["status"], "PENDING")

    def test_visual_template_does_not_rewrite_complete_existing_table(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "visual_inspection.csv"
            create_visual_inspection_template(path, ["A_raw"])
            original = path.read_bytes()
            create_visual_inspection_template(path, ["A_raw"])
            self.assertEqual(path.read_bytes(), original)

    def test_unfair_training_blocks_conclusion_even_with_visual_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); results, visual = root / "results.csv", root / "visual.csv"
            self._write_results(results, unfair=True); self._complete_visual(visual)
            report = analyze_results(results, visual)
            self.assertEqual(report["conclusion_status"], "BLOCKED")
            self.assertTrue(any("Unfair" in reason for reason in report["blocking_reasons"]))

    def test_ready_analysis_reports_class_tradeoffs_without_universal_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); results, visual = root / "results.csv", root / "visual.csv"
            self._write_results(results); self._complete_visual(visual)
            report = analyze_results(results, visual)
            self.assertEqual(report["conclusion_status"], "READY")
            self.assertIsNone(report["winner"])
            self.assertEqual(report["metric_leaders"]["pothole_mAP50"]["experiment"], "D_clahe")
            self.assertEqual(report["metric_leaders"]["crack_mAP50"]["experiment"], "E_brightness")
            self.assertEqual(report["latency_leader"]["experiment"], "A_raw")
            clahe = next(item for item in report["per_experiment_tradeoffs"] if item["experiment"] == "D_clahe")
            self.assertGreater(clahe["mAP50_delta_vs_raw"]["pothole"], 0)
            self.assertLess(clahe["mAP50_delta_vs_raw"]["crack"], 0)
            paths = write_analysis(report, root / "report")
            self.assertTrue(all(path.is_file() for path in paths))


if __name__ == "__main__":
    unittest.main()
