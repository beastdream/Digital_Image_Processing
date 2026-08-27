import os
import unittest
import numpy as np
import yaml
from unittest.mock import patch

from src.preprocessing.pipeline import ImagePreprocessingPipeline
from src.preprocessing.resize import letterbox_resize
from src.preprocessing.normalize import normalize_image
from src.preprocessing.augmentation import DataAugmenter
from src.data.dataset_utils import project_root

class TestImagePreprocessingPipeline(unittest.TestCase):

    def setUp(self):
        self.base_dir = str(project_root())
        self.config_path = os.path.join(self.base_dir, "configs", "experiments", "full_preprocessing.yaml")
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.pipeline = ImagePreprocessingPipeline(self.config)

        # Create dummy synthetic image (360x640 RGB)
        np.random.seed(42)
        self.dummy_img = np.random.randint(0, 256, (360, 640, 3), dtype=np.uint8)

        # Create dummy normalized bboxes [[cls, xc, yc, w, h]]
        self.dummy_bboxes = [
            [0, 0.5, 0.5, 0.2, 0.2],  # Center pothole
            [1, 0.2, 0.3, 0.1, 0.15], # Crack
            [2, 0.8, 0.7, 0.15, 0.1]  # Manhole
        ]

    def test_image_shape(self):
        """Test that output image shape matches target size 640x640."""
        processed_img, _, _ = self.pipeline.process(self.dummy_img, bboxes=self.dummy_bboxes, split="val")
        self.assertEqual(processed_img.shape[:2], (640, 640))

    def test_dtype_and_pixel_range(self):
        """Test data type is float32 and pixel values are scaled to [0.0, 1.0]."""
        processed_img, _, _ = self.pipeline.process(self.dummy_img, bboxes=self.dummy_bboxes, split="val")
        self.assertEqual(processed_img.dtype, np.float32)
        self.assertGreaterEqual(processed_img.min(), 0.0)
        self.assertLessEqual(processed_img.max(), 1.0)

    def test_no_nan_inf(self):
        """Test that preprocessed image contains zero NaN or Inf values."""
        processed_img, _, _ = self.pipeline.process(self.dummy_img, bboxes=self.dummy_bboxes, split="train")
        self.assertFalse(np.isnan(processed_img).any(), "NaN values found in processed image")
        self.assertFalse(np.isinf(processed_img).any(), "Inf values found in processed image")

    def test_bbox_validity(self):
        """Test that transformed bounding boxes are valid in [0, 1] with w > 0 and h > 0."""
        _, updated_bboxes, _ = self.pipeline.process(self.dummy_img, bboxes=self.dummy_bboxes, split="val")
        self.assertIsNotNone(updated_bboxes)
        self.assertGreater(len(updated_bboxes), 0)

        for bbox in updated_bboxes:
            cid, xc, yc, w, h = bbox
            self.assertIn(int(cid), [0, 1, 2])
            self.assertGreater(w, 0.0)
            self.assertGreater(h, 0.0)

            xmin = xc - w / 2.0
            ymin = yc - h / 2.0
            xmax = xc + w / 2.0
            ymax = yc + h / 2.0

            self.assertGreaterEqual(xmin, -1e-4)
            self.assertGreaterEqual(ymin, -1e-4)
            self.assertLessEqual(xmax, 1.0 + 1e-4)
            self.assertLessEqual(ymax, 1.0 + 1e-4)

    def test_augmentation_train_only(self):
        """Test that Augmentation is applied ONLY for train split, and NEVER for val or test splits."""
        # For val split, pipeline should produce deterministic outputs across multiple runs
        img_val_1, bbox_val_1, _ = self.pipeline.process(self.dummy_img, bboxes=self.dummy_bboxes, split="val")
        img_val_2, bbox_val_2, _ = self.pipeline.process(self.dummy_img, bboxes=self.dummy_bboxes, split="val")

        np.testing.assert_array_almost_equal(img_val_1, img_val_2, decimal=5)
        self.assertEqual(bbox_val_1, bbox_val_2)

        # For test split, pipeline output must also be deterministic
        img_test_1, _, _ = self.pipeline.process(self.dummy_img, bboxes=self.dummy_bboxes, split="test")
        np.testing.assert_array_almost_equal(img_val_1, img_test_1, decimal=5)

    def test_letterbox_transform_accuracy(self):
        """Test math accuracy of letterbox padding bbox transformation."""
        padded_img, new_bboxes, meta = letterbox_resize(
            self.dummy_img, target_size=(640, 640), bboxes=self.dummy_bboxes
        )

        self.assertEqual(padded_img.shape, (640, 640, 3))
        self.assertAlmostEqual(meta["scale_ratio"], 1.0, places=5)
        self.assertAlmostEqual(meta["pad_w"], 0.0, places=5)
        self.assertAlmostEqual(meta["pad_h"], 140.0, places=5)

        # Center bbox (0.5, 0.5) in 360 height image -> padded height 640
        # Original pixel y = 180 -> padded y = 180 + 140 = 320 -> normalized y = 320 / 640 = 0.5
        center_bbox = new_bboxes[0]
        self.assertAlmostEqual(center_bbox[2], 0.5, places=4)

    def test_translation_can_produce_a_valid_negative_image(self):
        augmenter = DataAugmenter({"enabled": True})
        # A box close to the right edge translated one image-width right is
        # entirely out of frame.  [] must remain [], never fall back to input.
        with patch("src.preprocessing.augmentation.random.uniform", side_effect=[1.0, 0.0]):
            _, boxes = augmenter._translation(self.dummy_img, [[2, 0.95, 0.5, 0.08, 0.2]], [1.0, 0.0])
        self.assertEqual(boxes, [])

    def test_augmentation_preserves_class_and_valid_geometry(self):
        augmenter = DataAugmenter({"enabled": True, "horizontal_flip": {"prob": 1.0}})
        _, boxes = augmenter.apply(self.dummy_img, [[1, 0.2, 0.5, 0.1, 0.2]], split="train")
        self.assertEqual(boxes[0][0], 1)
        _, xc, yc, width, height = boxes[0]
        self.assertTrue(0 <= xc - width / 2 <= xc + width / 2 <= 1)
        self.assertTrue(0 <= yc - height / 2 <= yc + height / 2 <= 1)

    def test_gamma_is_photometric_and_preserves_labels(self):
        augmenter = DataAugmenter({"enabled": True, "gamma": {"prob": 1.0, "gamma_range": [1.1, 1.1]}})
        image, boxes = augmenter.apply(self.dummy_img, self.dummy_bboxes, split="train")
        self.assertFalse(np.array_equal(image, self.dummy_img))
        self.assertEqual(boxes, self.dummy_bboxes)

    def test_unsafe_large_geometric_augmentation_is_rejected(self):
        with self.assertRaises(ValueError):
            DataAugmenter({"enabled": True, "rotation": {"max_angle_deg": 30}})
        with self.assertRaises(ValueError):
            DataAugmenter({"enabled": True, "scaling": {"scale_range": [0.7, 1.3]}})

if __name__ == "__main__":
    unittest.main()
