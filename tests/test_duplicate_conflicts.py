import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.data.clean_dataset import run_dataset_cleaning


class TestDuplicateAnnotationConflicts(unittest.TestCase):
    def test_conflicting_exact_duplicates_are_not_resolved_by_filename(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images = root / "data/raw/images"; labels = root / "data/raw/labels-YOLO"
            images.mkdir(parents=True); labels.mkdir(parents=True)
            records = [
                ("20250223_train.jpg", 10, "0 0.5 0.5 0.2 0.2"),
                ("20250219_val.jpg", 30, "1 0.5 0.5 0.2 0.2"),
                ("vlcsnap-2025-02-18-23h00m00s000.jpg", 50, "2 0.5 0.5 0.2 0.2"),
            ]
            for name, value, annotation in records:
                cv2.imwrite(str(images / name), np.full((20, 30, 3), value, dtype=np.uint8))
                (labels / f"{Path(name).stem}.txt").write_text(annotation, encoding="utf-8")
            duplicate = np.full((20, 30, 3), 90, dtype=np.uint8)
            for name, annotation in (("conflict_a.jpg", "0 0.5 0.5 0.2 0.2"), ("conflict_b.jpg", "1 0.5 0.5 0.2 0.2")):
                cv2.imwrite(str(images / name), duplicate)
                (labels / f"{Path(name).stem}.txt").write_text(annotation, encoding="utf-8")
            identical = np.full((20, 30, 3), 120, dtype=np.uint8)
            for name in ("same_a.jpg", "same_b.jpg"):
                cv2.imwrite(str(images / name), identical)
                (labels / f"{Path(name).stem}.txt").write_text("2 0.5 0.5 0.2 0.2", encoding="utf-8")

            run_dataset_cleaning(root)
            report = json.loads((root / "data/processed/road_damage_detection/reports/duplicate_annotation_conflicts.json").read_text(encoding="utf-8"))
            self.assertEqual(report["conflict_groups_count"], 1)
            self.assertEqual(report["conflicts"][0]["status"], "manual_review_required")
            self.assertEqual(set(report["conflicts"][0]["images"]), {"conflict_a.jpg", "conflict_b.jpg"})
            duplicates = json.loads((root / "data/processed/road_damage_detection/reports/duplicates.json").read_text(encoding="utf-8"))
            self.assertEqual(duplicates["duplicate_groups_count"], 1)
            self.assertEqual(duplicates["deduplicated_images"], 1)
            processed = root / "data/processed/road_damage_detection/images"
            self.assertFalse(any(processed.rglob("conflict_a.jpg")))
            self.assertFalse(any(processed.rglob("conflict_b.jpg")))
            self.assertTrue(any(processed.rglob("same_a.jpg")))
            self.assertFalse(any(processed.rglob("same_b.jpg")))


if __name__ == "__main__":
    unittest.main()
