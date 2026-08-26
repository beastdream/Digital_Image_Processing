import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.data.clean_dataset import run_dataset_cleaning


class TestDatasetValidation(unittest.TestCase):
    def test_bad_files_are_reported_without_aborting_processing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images = root / "data/raw/images"
            labels = root / "data/raw/labels-YOLO"
            images.mkdir(parents=True); labels.mkdir(parents=True)
            image = np.zeros((20, 30, 3), dtype=np.uint8)
            cv2.imwrite(str(images / "20250223_valid.jpg"), image)
            (labels / "20250223_valid.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
            (images / "20250223_corrupt.jpg").write_bytes(b"not an image")
            cv2.imwrite(str(images / "20250216_no_label.jpg"), image)
            (labels / "orphan.txt").write_text("", encoding="utf-8")

            # Three independent groups keep the split requirement valid.
            cv2.imwrite(str(images / "20250219_valid.jpg"), np.full_like(image, 80))
            (labels / "20250219_valid.txt").write_text("1 0.5 0.5 0.4 0.4\n", encoding="utf-8")
            cv2.imwrite(str(images / "vlcsnap-2025-02-18-23h00m00s000.jpg"), np.full_like(image, 160))
            (labels / "vlcsnap-2025-02-18-23h00m00s000.txt").write_text("2 0.5 0.5 0.4 0.4\n", encoding="utf-8")

            run_dataset_cleaning(root)
            report_path = root / "data/processed/road_damage_detection/reports/dataset_validation.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["readable_images"], 4)
            self.assertEqual(len(report["corrupt_images"]), 1)
            self.assertEqual(report["missing_labels"], ["20250216_no_label.jpg"])
            self.assertEqual(report["orphan_labels"], ["orphan.txt"])
            stats = json.loads((root / "data/processed/road_damage_detection/reports/annotation_statistics.json").read_text(encoding="utf-8"))
            self.assertEqual(stats["all"]["total_images"], 3)
            self.assertIn("all 3 classes", stats["all"]["class_combinations"])
            self.assertEqual(set(stats["splits"]), {"train", "val", "test"})


if __name__ == "__main__":
    unittest.main()
