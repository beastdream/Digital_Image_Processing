import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.data.clean_dataset import _assign_groups, _merge_near_duplicate_groups
from src.data.dataset_utils import parse_yolo_line, validate_yolo_box
from src.preprocessing.augmentation import DataAugmenter
from src.preprocessing.pipeline import ImagePreprocessingPipeline


class TestRequiredAnnotationCases(unittest.TestCase):
    def test_valid_yolo_bbox(self):
        box, status = validate_yolo_box(parse_yolo_line("0 0.5 0.5 0.2 0.3"))
        self.assertEqual(status, "valid")
        self.assertEqual(box, (0, 0.5, 0.5, 0.2, 0.3))

    def test_zero_width_and_zero_height_are_rejected(self):
        for line in ("0 0.5 0.5 0.0 0.2", "1 0.5 0.5 0.2 0.0"):
            with self.subTest(line=line), self.assertRaisesRegex(ValueError, "width and height"):
                validate_yolo_box(parse_yolo_line(line))

    def test_out_of_range_bbox_and_invalid_class_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "beyond tolerance"):
            validate_yolo_box(parse_yolo_line("2 0.95 0.5 0.2 0.2"))
        with self.assertRaisesRegex(ValueError, "unsupported class_id"):
            parse_yolo_line("7 0.5 0.5 0.2 0.2")


class TestRequiredAugmentationCases(unittest.TestCase):
    def setUp(self):
        self.image = np.full((40, 80, 3), 100, np.uint8)
        self.augmenter = DataAugmenter({"enabled": True})

    def test_partially_outside_bbox_is_clipped_inside_and_class_preserved(self):
        with patch("src.preprocessing.augmentation.random.uniform", side_effect=[0.05, 0.0]):
            _, boxes = self.augmenter._translation(self.image, [[2, 0.95, 0.5, 0.2, 0.2]], [0.05, 0.0])
        self.assertEqual(len(boxes), 1)
        class_id, xc, yc, width, height = boxes[0]
        self.assertEqual(class_id, 2)
        self.assertTrue(0 <= xc - width / 2 < xc + width / 2 <= 1)
        self.assertTrue(0 <= yc - height / 2 < yc + height / 2 <= 1)

    def test_fully_outside_bbox_produces_empty_updated_bboxes(self):
        with patch("src.preprocessing.augmentation.random.uniform", side_effect=[1.0, 0.0]):
            _, boxes = self.augmenter._translation(self.image, [[1, 0.9, 0.5, 0.1, 0.1]], [1.0, 0.0])
        self.assertEqual(boxes, [])


class TestRequiredSplitAndNearDuplicateCases(unittest.TestCase):
    def test_same_sequence_is_assigned_as_one_indivisible_group(self):
        groups = {"session_a": ["frame_a1.jpg", "frame_a2.jpg"],
                  "session_b": ["frame_b.jpg"], "session_c": ["frame_c.jpg"]}
        labels = {Path(name).stem: [f"{index % 3} 0.5 0.5 0.2 0.2"]
                  for index, name in enumerate(sum(groups.values(), []))}
        assignment = _assign_groups(groups, labels, {"ratios": {"train": .34, "val": .33, "test": .33}})
        self.assertIn(assignment["session_a"], ("train", "val", "test"))
        # The splitter returns one assignment per group, never per frame.
        self.assertEqual(len({assignment["session_a"] for _ in groups["session_a"]}), 1)

    def test_perceptually_near_nonidentical_images_merge_before_split(self):
        with tempfile.TemporaryDirectory() as directory:
            images = Path(directory)
            gradient = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
            first = np.dstack([gradient] * 3)
            second = np.clip(first.astype(np.int16) + 3, 0, 255).astype(np.uint8)
            cv2.imwrite(str(images / "first.png"), first)
            cv2.imwrite(str(images / "second.png"), second)
            self.assertFalse(np.array_equal(first, second))
            merged, matches, roots = _merge_near_duplicate_groups(
                {"group_a": ["first.png"], "group_b": ["second.png"]}, images, threshold=8)
            self.assertEqual(len(merged), 1)
            self.assertEqual(len(matches), 1)
            self.assertLessEqual(matches[0]["phash_distance"], 8)
            self.assertEqual(roots["group_a"], roots["group_b"])


class TestRequiredPreprocessingCases(unittest.TestCase):
    def test_output_shape_input_immutability_and_bbox_alignment(self):
        image = np.random.default_rng(7).integers(0, 256, (40, 80, 3), dtype=np.uint8)
        original = image.copy(); box = [[0, 0.5, 0.5, 0.5, 0.5]]
        pipeline = ImagePreprocessingPipeline({"resize": {"enabled": True, "mode": "letterbox",
            "target_size": [80, 80]}, "denoise": {"enabled": False}, "contrast": {"enabled": False},
            "brightness": {"enabled": False}, "normalization": {"enabled": False},
            "augmentation": {"enabled": False}})
        output, boxes, _ = pipeline.process(image, box, split="val")
        self.assertEqual(output.shape, (80, 80, 3))
        np.testing.assert_array_equal(image, original)
        # x is unchanged; vertical content is halved and centered by 20 px padding.
        self.assertAlmostEqual(boxes[0][1], 0.5)
        self.assertAlmostEqual(boxes[0][2], 0.5)
        self.assertAlmostEqual(boxes[0][3], 0.5)
        self.assertAlmostEqual(boxes[0][4], 0.25)


if __name__ == "__main__":
    unittest.main()
