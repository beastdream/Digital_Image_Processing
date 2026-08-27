import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.detection.analyze_class_balance import build_analysis, size_category, write_analysis


class TestClassBalanceAndObjectSizeAnalysis(unittest.TestCase):
    def _dataset(self, root: Path) -> Path:
        dataset = root / "dataset"
        for split in ("train", "val", "test"):
            (dataset / "images" / split).mkdir(parents=True)
            (dataset / "labels" / split).mkdir(parents=True)
            cv2.imwrite(str(dataset / "images" / split / f"{split}.jpg"), np.full((50, 100, 3), 80, np.uint8))
            # Four crack objects, one pothole, one manhole per split -> high imbalance.
            labels = ["0 0.5 0.5 0.1 0.1", *["1 0.5 0.5 0.05 0.05"] * 4, "2 0.5 0.5 0.2 0.2"]
            (dataset / "labels" / split / f"{split}.txt").write_text("\n".join(labels), encoding="utf-8")
        return dataset

    def test_coco_size_boundaries(self):
        self.assertEqual(size_category(32 ** 2 - 1), "small")
        self.assertEqual(size_category(32 ** 2), "medium")
        self.assertEqual(size_category(96 ** 2), "large")

    def test_reports_frequency_medians_size_buckets_and_320_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory))
            report = build_analysis(dataset, configured_imgsz=640)
            imbalance = report["class_imbalance"]
            self.assertEqual(imbalance["level"], "HIGH")
            self.assertEqual(imbalance["class_frequency_and_performance"]["crack"]["objects"], 12)
            self.assertEqual(imbalance["class_frequency_and_performance"]["pothole"]["objects"], 3)
            pothole_320 = report["bbox_statistics_by_imgsz"]["320"]["per_class"]["pothole"]
            self.assertEqual(pothole_320["width_px"]["median"], 32.0)
            self.assertEqual(pothole_320["height_px"]["median"], 16.0)
            self.assertEqual(pothole_320["area_px2"]["median"], 512.0)
            self.assertEqual(pothole_320["size_counts"]["small"], 3)
            self.assertTrue(any(item["code"] == "MANY_SMALL_OBJECTS_AT_320" for item in report["warnings"]))
            self.assertIn("Do not enable weighted sampling", imbalance["sampling_decision"])

    def test_writes_json_csv_and_human_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); dataset = self._dataset(root)
            paths = write_analysis(build_analysis(dataset, 640), root / "reports")
            self.assertTrue(all(path.is_file() for path in paths))
            csv_text = paths[1].read_text(encoding="utf-8")
            self.assertIn("median_width_px", csv_text)
            self.assertIn("pothole", csv_text)
            self.assertIn("320", csv_text)


if __name__ == "__main__":
    unittest.main()
