"""Import the Kaggle "Road Hazards" dataset into data/raw for clean_dataset.

Source: "potholes, cracks and openmanholes (Road Hazards)" by Sabid Rahman,
https://www.kaggle.com/datasets/sabidrahman/pothole-cracks-and-openmanhole,
licensed CC BY-NC-SA 4.0. Its classes already match this project
(0 pothole, 1 crack, 2 open manhole -> manhole).

The source ships augmented copies (``<id>_aug_flip.jpg`` ...) and its own
train/valid folders share file names and even byte-identical images. This
importer therefore:

* reads only ``train/`` and ``valid/`` (``classes/`` repeats the same images);
* keeps one copy of byte-identical images (train wins, its label is kept);
* renames files ``rh_<split>_<name>`` so names never collide, and
  ``group_key`` maps every variant of one original to one capture group, so
  ``clean_dataset`` re-splits originals and augmentations together.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from src.data.dataset_utils import IMAGE_SUFFIXES, project_root

SOURCE_SPLITS = ("train", "valid")
ATTRIBUTION = {
    "name": "potholes, cracks and openmanholes (Road Hazards)",
    "author": "Sabid Rahman",
    "url": "https://www.kaggle.com/datasets/sabidrahman/pothole-cracks-and-openmanhole",
    "license": "CC BY-NC-SA 4.0",
}


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def import_road_hazards(source: str | Path, overwrite: bool = False) -> dict:
    root = project_root()
    source = Path(source)
    raw_images, raw_labels = root / "data/raw/images", root / "data/raw/labels-YOLO"
    for split in SOURCE_SPLITS:
        if not (source / split / "images").is_dir() or not (source / split / "labels").is_dir():
            raise FileNotFoundError(f"Expected {source / split}/images and labels (the folder with train/ and valid/)")
    existing = [path for folder in (raw_images, raw_labels) if folder.is_dir() for path in folder.iterdir()]
    if existing and not overwrite:
        raise FileExistsError(f"data/raw already has {len(existing)} files; pass --overwrite to replace them")
    for folder in (raw_images, raw_labels):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)

    seen: dict[str, str] = {}
    stats: dict = {"copied": Counter(), "skipped_identical_duplicate": Counter(), "missing_label": Counter(),
                   "objects": Counter()}
    for split in SOURCE_SPLITS:
        for image in sorted((source / split / "images").iterdir()):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label = source / split / "labels" / f"{image.stem}.txt"
            if not label.is_file():
                stats["missing_label"][split] += 1
                continue
            digest = _md5(image)
            if digest in seen:
                stats["skipped_identical_duplicate"][split] += 1
                continue
            name = f"rh_{split}_{image.stem}"
            seen[digest] = name
            shutil.copy2(image, raw_images / f"{name}{image.suffix.lower()}")
            text = label.read_text(encoding="utf-8")
            (raw_labels / f"{name}.txt").write_text(text, encoding="utf-8")
            for line in text.splitlines():
                if line.strip():
                    stats["objects"][line.split()[0]] += 1
            stats["copied"][split] += 1

    manifest = {"dataset": ATTRIBUTION, "source": str(source),
                **{key: dict(value) for key, value in stats.items()},
                "images_total": sum(stats["copied"].values())}
    (root / "data/raw/import_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Copy the Road Hazards dataset into data/raw")
    parser.add_argument("--source", required=True, help="Folder containing train/ and valid/ (e.g. D:/Dataset/dataset)")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing data/raw images and labels")
    args = parser.parse_args(argv)
    import_road_hazards(args.source, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
