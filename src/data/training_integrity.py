"""Fail-fast dataset integrity gate executed immediately before model training."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from src.data.dataset_utils import CLASS_NAMES, IMAGE_SUFFIXES, SPLITS, group_key, parse_yolo_line
from src.data.validate_processed_dataset import validate_processed_dataset


class DatasetIntegrityError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_dataset_integrity_check(data_yaml: str | Path, expected_classes: dict | None = None) -> dict:
    data_yaml = Path(data_yaml).resolve(); expected_classes = expected_classes or CLASS_NAMES
    errors = []
    if not data_yaml.is_file(): raise DatasetIntegrityError(f"Dataset YAML does not exist: {data_yaml}")
    dataset_cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = dataset_cfg.get("names", {})
    names = {int(key): value for key, value in names.items()} if isinstance(names, dict) else dict(enumerate(names))
    if dataset_cfg.get("nc") != len(expected_classes) or names != expected_classes:
        errors.append(f"class mapping must be nc={len(expected_classes)}, names={expected_classes}; got nc={dataset_cfg.get('nc')}, names={names}")
    dataset_dir = data_yaml.parent
    declared_path = Path(dataset_cfg.get("path", dataset_dir))
    if not declared_path.is_absolute(): declared_path = data_yaml.parent / declared_path
    if declared_path.resolve() != dataset_dir.resolve():
        errors.append(f"dataset.yaml path resolves to {declared_path.resolve()}, expected {dataset_dir.resolve()}")
    for split in SPLITS:
        if dataset_cfg.get(split) != f"images/{split}":
            errors.append(f"dataset.yaml {split} must be images/{split}, got {dataset_cfg.get(split)!r}")

    structural_report, structurally_valid = validate_processed_dataset(dataset_dir)
    if not structurally_valid:
        for item in structural_report["errors"][:20]:
            errors.append(f"{item.get('split')}: {item.get('reason')} ({item.get('image') or item.get('label_file') or 'unknown file'})")

    split_images, hashes, groups = {}, defaultdict(list), defaultdict(set)
    object_counts = {split: Counter() for split in SPLITS}
    for split in SPLITS:
        image_dir, label_dir = dataset_dir / "images" / split, dataset_dir / "labels" / split
        images = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES) if image_dir.is_dir() else []
        split_images[split] = images
        for image in images:
            hashes[_sha256(image)].append((split, image.name))
            groups[group_key(image.name)].add(split)
            label = label_dir / f"{image.stem}.txt"
            if label.exists():
                for line in label.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        try: object_counts[split][parse_yolo_line(line)[0]] += 1
                        except ValueError: pass  # Already reported with file/line by structural validation.
    stem_splits = defaultdict(set)
    for split, images in split_images.items():
        for image in images: stem_splits[image.stem].add(split)
    duplicate_hashes = [{"sha256": digest, "files": [{"split": split, "image": name} for split, name in items]}
                        for digest, items in hashes.items() if len({split for split, _ in items}) > 1]
    duplicate_stems = {stem: sorted(splits) for stem, splits in stem_splits.items() if len(splits) > 1}
    leaked_groups = {group: sorted(splits) for group, splits in groups.items() if len(splits) > 1}
    if duplicate_hashes: errors.append(f"exact duplicate image leakage across splits: {len(duplicate_hashes)} hash group(s)")
    if duplicate_stems: errors.append(f"duplicate filename stems across splits: {len(duplicate_stems)}")
    if leaked_groups: errors.append(f"capture-group leakage across splits: {len(leaked_groups)} group(s)")
    statistics = {"images": {split: len(split_images[split]) for split in SPLITS},
                  "objects": {split: {expected_classes[cid]: object_counts[split][cid] for cid in expected_classes} for split in SPLITS},
                  "total_images": sum(len(value) for value in split_images.values()),
                  "total_objects": sum(sum(value.values()) for value in object_counts.values())}
    report = {"status": "FAILED" if errors else "PASSED", "data_yaml": str(data_yaml),
              "checks": {"dataset_yaml": "PASSED" if not errors[:1] else "SEE errors",
                         "splits_and_annotations": structural_report["status"],
                         "duplicate_cross_split": "FAILED" if duplicate_hashes or duplicate_stems else "PASSED",
                         "capture_group_cross_split": "FAILED" if leaked_groups else "PASSED"},
              "statistics": statistics, "duplicate_hash_groups": duplicate_hashes,
              "duplicate_stems": duplicate_stems, "leaked_capture_groups": leaked_groups, "errors": errors}
    reports_dir = dataset_dir / "reports"; reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "training_integrity_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Dataset integrity statistics:", json.dumps(statistics, indent=2))
    if errors:
        raise DatasetIntegrityError("Dataset integrity check failed before training:\n- " + "\n- ".join(errors))
    print("Dataset integrity check: PASSED")
    return report
