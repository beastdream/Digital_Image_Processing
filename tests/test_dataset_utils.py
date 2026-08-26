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
        self.assertEqual(group_key("vlcsnap-2025-02-19-14h41m29s105.jpg"), "vlcsnap_2025-02-19_14h")
        self.assertEqual(group_key("20250223_143826.jpg"), "capture_20250223")


if __name__ == "__main__":
    unittest.main()
