import unittest

from src.data.dataset_utils import (CLASS_NAMES, clip_yolo_box, format_yolo_box,
                                    group_key, parse_yolo_line, validate_yolo_box)


class TestDatasetUtils(unittest.TestCase):
    def test_class_mapping_is_fixed(self):
        self.assertEqual(CLASS_NAMES, {0: "pothole", 1: "crack", 2: "manhole"})

    def test_parser_never_remaps_invalid_class(self):
        with self.assertRaisesRegex(ValueError, "unsupported class_id"):
            parse_yolo_line("3 0.5 0.5 0.2 0.2")

    def test_clip_preserves_class_id(self):
        box = parse_yolo_line("2 0.02 0.5 0.1 0.2")
        clipped = clip_yolo_box(box)
        self.assertEqual(clipped[0], 2)
        self.assertEqual(format_yolo_box(clipped).split()[0], "2")

    def test_only_epsilon_boundary_error_is_corrected(self):
        corrected, status = validate_yolo_box(parse_yolo_line("1 0.5000005 0.5 1.000001 0.2"))
        self.assertEqual(status, "clipped")
        self.assertEqual(corrected[0], 1)

    def test_real_boundary_error_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "beyond tolerance"):
            validate_yolo_box(parse_yolo_line("1 0.6 0.5 0.9 0.2"))
        with self.assertRaisesRegex(ValueError, "width and height"):
            validate_yolo_box(parse_yolo_line("1 0.5 0.5 0 0.2"))

    def test_capture_groups_are_stable(self):
        self.assertEqual(group_key("vlcsnap-2025-02-19-14h41m29s105.jpg"), "session_2025-02-19")
        self.assertEqual(group_key("vlcsnap-2025-02-19-15h00m00s000.jpg"), "session_2025-02-19")
        self.assertEqual(group_key("20250223_143826.jpg"), "session_2025-02-23")

    def test_road_hazards_augmentations_share_one_group_across_source_splits(self):
        keys = {group_key(name) for name in ("rh_train_1001.jpg", "rh_train_1001_aug_flip.jpg",
                                             "rh_valid_1001_aug_vertical_flip.jpg",
                                             "rh_train_1001_aug_brightness_contrast.jpg",
                                             "rh_train_1001_augmented.jpg")}
        self.assertEqual(keys, {"rh_1001"})
        self.assertNotEqual(group_key("rh_train_1002_aug_flip.jpg"), "rh_1001")
        self.assertEqual(group_key("train_0001_f0000.jpg"), "file_train_0001_f0000")


    def test_formatted_border_boxes_always_validate(self):
        import random
        rng = random.Random(0)
        for _ in range(20000):
            left, right = sorted(rng.random() for _ in range(2))
            box = (1, (left + 1.0) / 2, 0.5, 1.0 - left, 0.4)  # touches the right edge
            corrected, _ = validate_yolo_box(box)
            line = format_yolo_box(corrected)
            self.assertEqual(validate_yolo_box(parse_yolo_line(line))[1], "valid", line)
        self.assertEqual(format_yolo_box((0, 0.5, 0.5, 0.2, 0.3)), "0 0.50000000 0.50000000 0.20000000 0.30000000")


if __name__ == "__main__":
    unittest.main()
