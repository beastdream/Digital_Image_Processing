"""Build a validated, group-aware YOLO dataset from raw static images."""
from __future__ import annotations
import argparse, csv, hashlib, json, shutil
from collections import Counter, defaultdict
from pathlib import Path
import cv2
from src.data.dataset_utils import (CLASS_NAMES, IMAGE_SUFFIXES, SPLITS, dataset_yaml_text,
                                    format_yolo_box, group_key, label_signature,
                                    parse_yolo_line, project_root, validate_yolo_box)

def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _read_labels(path: Path, image_name: str, report: list[dict]) -> list[str]:
    if not path.exists():
        return []  # Empty labels are valid, but a label file is still written.
    cleaned = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip(): continue
        try:
            parsed = parse_yolo_line(raw)
            corrected, status = validate_yolo_box(parsed)
            if status == "clipped":
                report.append({"image": image_name, "label_file": path.name, "line": number,
                               "class_id": parsed[0], "class": CLASS_NAMES[parsed[0]], "original_bbox": list(parsed[1:]),
                               "reason": "bbox boundary overflow within EPSILON=1e-5", "action": "clipped", "status": "clipped"})
            cleaned.append(format_yolo_box(corrected))
        except ValueError as exc:
            parts = raw.split()
            class_id = int(parts[0]) if parts and parts[0].lstrip("+-").isdigit() else None
            report.append({"image": image_name, "label_file": path.name, "line": number,
                           "class_id": class_id, "class": CLASS_NAMES.get(class_id), "original_bbox": raw,
                           "reason": str(exc), "action": "excluded", "status": "invalid"})
    return cleaned

def _assign_groups(groups: dict[str, list[str]], labels: dict[str, list[str]]) -> dict[str, str]:
    """Assign complete capture groups; known acquisition sessions use a stable split."""
    # These are capture sessions, not individual files. Keeping this reviewed map
    # avoids moving the held-out test set whenever raw data is reprocessed.
    reviewed = {
        "capture_20250216": "val", "capture_20250219": "val", "capture_20250223": "train",
        "vlcsnap_unsequenced": "train", "vlcsnap_2025-02-18_17h": "val",
        "vlcsnap_2025-02-18_18h": "train", "vlcsnap_2025-02-18_23h": "test",
        "vlcsnap_2025-02-19_13h": "test", "vlcsnap_2025-02-19_14h": "train",
        "vlcsnap_2025-02-19_15h": "val", "vlcsnap_2025-02-19_16h": "test",
        "vlcsnap_2025-02-19_17h": "train", "vlcsnap_2025-02-26_20h": "train",
    }
    fixed = {key: split for key, split in reviewed.items() if key in groups}
    unassigned = {key: files for key, files in groups.items() if key not in fixed}
    if not unassigned:
        return fixed
    # New sessions are added only to train by default; users must explicitly
    # promote an independent session to validation/test after reviewing it.
    # This prevents accidental contamination of the established held-out sets.
    return {**fixed, **{key: "train" for key in unassigned}}

def _annotation_statistics(names: list[str], labels: dict[str, list[str]]) -> dict:
    """Count objects, class presence, and mutually exclusive class combinations."""
    object_counts = Counter()
    image_counts = Counter()
    combinations = Counter({"pothole only": 0, "crack only": 0, "manhole only": 0,
                            "pothole + crack": 0, "pothole + manhole": 0,
                            "crack + manhole": 0, "all 3 classes": 0, "no object": 0})
    combination_names = {
        frozenset({0}): "pothole only", frozenset({1}): "crack only", frozenset({2}): "manhole only",
        frozenset({0, 1}): "pothole + crack", frozenset({0, 2}): "pothole + manhole",
        frozenset({1, 2}): "crack + manhole", frozenset({0, 1, 2}): "all 3 classes", frozenset(): "no object",
    }
    for name in names:
        classes = {int(line.split()[0]) for line in labels[Path(name).stem]}
        for line in labels[Path(name).stem]: object_counts[int(line.split()[0])] += 1
        for class_id in classes: image_counts[class_id] += 1
        combinations[combination_names[frozenset(classes)]] += 1
    return {"total_images": len(names), "total_objects": sum(object_counts.values()),
            "object_counts": {CLASS_NAMES[cid]: object_counts[cid] for cid in CLASS_NAMES},
            "images_containing": {CLASS_NAMES[cid]: image_counts[cid] for cid in CLASS_NAMES},
            "class_combinations": dict(combinations)}

def run_dataset_cleaning(root: str | Path | None = None) -> dict:
    root_path = Path(root) if root else project_root()
    raw_images, raw_labels = root_path / "data/raw/images", root_path / "data/raw/labels-YOLO"
    processed_dir = root_path / "data/processed/road_damage_detection"
    reports_dir = processed_dir / "reports"
    if not raw_images.is_dir() or not raw_labels.is_dir(): raise FileNotFoundError("Expected data/raw/images and data/raw/labels-YOLO")
    # Rebuild only this dataset's canonical artifacts.  In particular, do not
    # remove optional derived preprocessing experiments under this directory.
    for generated_path in (processed_dir / "images", processed_dir / "labels", reports_dir):
        if generated_path.exists():
            shutil.rmtree(generated_path)
    dataset_yaml = processed_dir / "dataset.yaml"
    if dataset_yaml.exists():
        dataset_yaml.unlink()
    reports_dir.mkdir(parents=True, exist_ok=True)
    raw_entries = sorted(path for path in raw_images.iterdir() if path.is_file())
    images = [path for path in raw_entries if path.suffix.lower() in IMAGE_SUFFIXES]
    unsupported_image_files = [path.name for path in raw_entries if path.suffix.lower() not in IMAGE_SUFFIXES]
    invalid_images, missing_label_files, readable_images = [], [], []
    valid_images = []
    for image in images:
        if not image.exists():
            invalid_images.append({"file": image.name, "reason": "file does not exist"})
            continue
        decoded = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
        channels = 1 if decoded is not None and decoded.ndim == 2 else (decoded.shape[2] if decoded is not None and decoded.ndim == 3 else 0)
        if decoded is None or decoded.size == 0 or decoded.shape[0] <= 0 or decoded.shape[1] <= 0 or channels not in (1, 3, 4):
            invalid_images.append({"file": image.name, "reason": "unreadable image, invalid dimensions, or unsupported channels"})
            continue
        readable_images.append(image)
        if not (raw_labels / f"{image.stem}.txt").exists():
            missing_label_files.append(image.name)
            continue
        valid_images.append(image)
    image_stems = {image.stem for image in images}
    orphan_labels = sorted(path.name for path in raw_labels.iterdir() if path.is_file() and path.suffix.lower() == ".txt" and path.stem not in image_stems)
    validation = {
        "total_images": len(images), "readable_images": len(readable_images),
        "corrupt_images": invalid_images, "missing_labels": missing_label_files,
        "orphan_labels": orphan_labels, "unsupported_image_files": unsupported_image_files,
        "excluded_from_processing": len(images) - len(valid_images),
        "status": "PASSED" if not invalid_images and not missing_label_files and not orphan_labels else "COMPLETED_WITH_EXCLUSIONS",
    }
    (reports_dir / "dataset_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    hashes = defaultdict(list)
    for image in valid_images: hashes[_md5(image)].append(image)
    invalid, duplicates, conflicts, labels, retained, label_cache = [], {}, [], {}, [], {}
    for digest, paths in sorted(hashes.items()):
        paths = sorted(paths)
        annotations = {}
        for path in paths:
            if path.stem not in label_cache:
                label_cache[path.stem] = _read_labels(raw_labels / f"{path.stem}.txt", path.name, invalid)
            annotations[path.name] = label_cache[path.stem]
        signatures = {name: label_signature(lines) for name, lines in annotations.items()}
        if len(set(signatures.values())) == 1:
            canonical = paths[0]
            labels[canonical.stem] = annotations[canonical.name]
            retained.append(canonical)
            if len(paths) > 1:
                duplicates[digest] = {"hash": digest, "images": [p.name for p in paths],
                                      "kept_image": canonical.name, "removed_images": [p.name for p in paths[1:]],
                                      "status": "deduplicated_same_annotation"}
        else:
            # No filename-order rule is allowed to resolve a label disagreement.
            conflicts.append({"hash": digest, "images": [p.name for p in paths],
                              "annotations": annotations, "status": "manual_review_required",
                              "action": "excluded_from_processed_dataset"})
    (reports_dir / "duplicates.json").write_text(json.dumps({"hash_algorithm": "MD5", "duplicate_groups_count": len(duplicates), "deduplicated_images": sum(len(item["removed_images"]) for item in duplicates.values()), "duplicate_groups": duplicates}, indent=2), encoding="utf-8")
    (reports_dir / "duplicate_annotation_conflicts.json").write_text(json.dumps({"hash_algorithm": "MD5", "conflict_groups_count": len(conflicts), "excluded_images": sum(len(item["images"]) for item in conflicts), "conflicts": conflicts}, indent=2), encoding="utf-8")
    annotation_summary = Counter(item["action"] for item in invalid)
    (reports_dir / "invalid_annotations.json").write_text(json.dumps({"epsilon": 1e-5, "total_flagged": len(invalid), "clipped_count": annotation_summary["clipped"], "excluded_count": annotation_summary["excluded"], "annotations": invalid}, indent=2), encoding="utf-8")
    groups = defaultdict(list)
    for image in retained: groups[group_key(image.name)].append(image.name)
    group_splits = _assign_groups(groups, labels)
    split_images = {split: sorted(name for key, names in groups.items() if group_splits[key] == split for name in names) for split in SPLITS}
    if any(not split_images[split] for split in SPLITS): raise RuntimeError("Group-aware split produced an empty split; add more independent capture groups.")
    for split in SPLITS:
        (processed_dir / "images" / split).mkdir(parents=True); (processed_dir / "labels" / split).mkdir(parents=True)
    counts = {split: Counter() for split in SPLITS}
    for split, names in split_images.items():
        for name in names:
            image = raw_images / name; shutil.copy2(image, processed_dir / "images" / split / name)
            lines = labels[image.stem]; (processed_dir / "labels" / split / f"{image.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            counts[split].update(int(line.split()[0]) for line in lines)
    (processed_dir / "dataset.yaml").write_text(dataset_yaml_text(processed_dir), encoding="utf-8")
    with (reports_dir / "dataset_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["class_id", "class_name", "train_images", "val_images", "test_images", "train_objects", "val_objects", "test_objects"])
        for cid, name in CLASS_NAMES.items(): writer.writerow([cid, name, *[sum(any(int(line.split()[0]) == cid for line in labels[Path(f).stem]) for f in split_images[s]) for s in SPLITS], *[counts[s][cid] for s in SPLITS]])
    full_statistics = _annotation_statistics([image.name for image in retained], labels)
    split_statistics = {split: _annotation_statistics(split_images[split], labels) for split in SPLITS}
    (reports_dir / "annotation_statistics.json").write_text(json.dumps({"all": full_statistics, "splits": split_statistics}, indent=2), encoding="utf-8")
    report = {"status": "PASSED", "group_aware": True, "cross_split_group_leakage": False, "group_distribution": {key: group_splits[key] for key in sorted(groups)}, "image_counts": {s: len(split_images[s]) for s in SPLITS}, "object_counts": {s: dict(counts[s]) for s in SPLITS}}
    (reports_dir / "data_leakage_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--root", default=None)
    print(json.dumps(run_dataset_cleaning(parser.parse_args().root), indent=2))
