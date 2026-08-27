import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.data.dataset_utils import dataset_yaml_text
from src.data.materialize_preprocessed_dataset import materialize_preprocessed_dataset


class TestMaterializedPreprocessingDataset(unittest.TestCase):
    def test_config_hash_versions_and_rebuild_removes_stale_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "data/processed/road_damage_detection"
            for split in ("train", "val", "test"):
                (source / "images" / split).mkdir(parents=True); (source / "labels" / split).mkdir(parents=True)
                image_name = f"{split}.jpg"
                cv2.imwrite(str(source / "images" / split / image_name), np.full((12, 20, 3), 40, dtype=np.uint8))
                (source / "labels" / split / f"{split}.txt").write_text("0 0.5 0.5 0.4 0.4", encoding="utf-8")
            (source / "dataset.yaml").write_text(dataset_yaml_text(source), encoding="utf-8")
            config_path = root / "preprocess.yaml"
            config = {"resize": {"enabled": True, "mode": "letterbox", "target_size": [32, 32]}, "normalization": {"enabled": True, "mode": "pixel_scale"}, "augmentation": {"enabled": False}}
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            first = materialize_preprocessed_dataset(config_path, "tiny", root_dir=root)
            stale = first.parent / "images/train/stale.jpg"; stale.write_bytes(b"stale")
            second = materialize_preprocessed_dataset(config_path, "tiny", root_dir=root)
            self.assertEqual(first, second)
            self.assertFalse(stale.exists())
            config["resize"]["target_size"] = [48, 48]
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            changed = materialize_preprocessed_dataset(config_path, "tiny", root_dir=root)
            self.assertNotEqual(first.parent.name, changed.parent.name)
            self.assertTrue(changed.exists())


if __name__ == "__main__":
    unittest.main()
