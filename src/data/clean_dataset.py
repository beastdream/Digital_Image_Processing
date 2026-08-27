"""Build a validated, group-aware YOLO dataset from raw static images."""
from __future__ import annotations
import argparse, csv, hashlib, json, shutil
from collections import Counter, defaultdict
from pathlib import Path
from itertools import product
import cv2
import yaml
import numpy as np
from src.data.dataset_utils import (CLASS_NAMES, IMAGE_SUFFIXES, SPLITS, dataset_yaml_text,
                                    format_yolo_box, group_key, label_signature,
                                    parse_yolo_line, project_root, validate_yolo_box)
from src.utils.reproducibility import set_global_seed

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

def _assign_groups(groups: dict[str, list[str]], labels: dict[str, list[str]], split_config: dict) -> dict[str, str]:
    """Exhaustively stratify complete sessions by image and 3-class object vectors."""
    ratios = split_config.get("ratios", {"train": .70, "val": .15, "test": .15})
    if set(ratios) != set(SPLITS) or abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError("split.ratios must define train/val/test and sum to 1")
    settings = split_config.get("stratification", {})
    image_weight, class_weight = float(settings.get("image_weight", 1.0)), float(settings.get("class_weight", 1.0))
    keys = sorted(groups)
    if len(keys) > 12:
        raise ValueError("More than 12 groups requires a scalable stratifier; refusing an unsafe image-level fallback")
    total_images = sum(len(files) for files in groups.values())
    total_classes = Counter(int(line.split()[0]) for files in groups.values() for name in files for line in labels[Path(name).stem])
    targets_images = {split: total_images * ratios[split] for split in SPLITS}
    targets_classes = {split: {cid: total_classes[cid] * ratios[split] for cid in CLASS_NAMES} for split in SPLITS}
    vectors = {key: Counter(int(line.split()[0]) for name in groups[key] for line in labels[Path(name).stem]) for key in keys}
    best_score, best_assignment = float("inf"), None
    for allocation in product(SPLITS, repeat=len(keys)):
        if set(allocation) != set(SPLITS):
            continue
        image_counts = Counter(); class_counts = {split: Counter() for split in SPLITS}
        for key, split in zip(keys, allocation):
            image_counts[split] += len(groups[key]); class_counts[split].update(vectors[key])
        image_error = sum(((image_counts[split] - targets_images[split]) / max(targets_images[split], 1)) ** 2 for split in SPLITS)
        class_error = sum(((class_counts[split][cid] - targets_classes[split][cid]) / max(targets_classes[split][cid], 1)) ** 2 for split in SPLITS for cid in CLASS_NAMES)
        score = image_weight * image_error + class_weight * class_error
        if score < best_score:
            best_score, best_assignment = score, dict(zip(keys, allocation))
    if best_assignment is None:
        raise RuntimeError("Could not assign complete groups to all three splits")
    return best_assignment

def _phash(image_path: Path) -> int:
    """64-bit perceptual hash using DCT on a grayscale 32×32 thumbnail."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot pHash unreadable image: {image_path.name}")
    dct = cv2.dct(cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype("float32"))[:8, :8]
    median = float(np.median(dct[1:, :]))
    bits = dct > median
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return value

def _merge_near_duplicate_groups(groups: dict[str, list[str]], raw_images: Path, threshold: int) -> tuple[dict[str, list[str]], list[dict], dict[str, str]]:
    """Merge session groups connected by pHash-near images before split assignment."""
    image_groups = {name: key for key, names in groups.items() for name in names}
    hashes = {name: _phash(raw_images / name) for name in image_groups}
    parent = {key: key for key in groups}
    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]; key = parent[key]
        return key
    def union(left: str, right: str) -> None:
        left, right = find(left), find(right)
        if left != right: parent[right] = left
    names = sorted(hashes)
    matches = []
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            first_group, second_group = image_groups[first], image_groups[second]
            if first_group == second_group:
                continue
            distance = (hashes[first] ^ hashes[second]).bit_count()
            if distance <= threshold:
                matches.append({"image_a": first, "group_a": first_group, "image_b": second, "group_b": second_group, "phash_distance": distance})
                union(first_group, second_group)
    merged = defaultdict(list)
    for group, names_in_group in groups.items():
        merged[find(group)].extend(names_in_group)
    return dict(merged), matches, {group: find(group) for group in groups}

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

def _split_statistics(split_statistics: dict[str, dict], warning_points: float, target_ratios: dict[str, float]) -> dict:
    total = sum(item["total_objects"] for item in split_statistics.values())
    overall_counts = Counter({cid: sum(item["object_counts"][CLASS_NAMES[cid]] for item in split_statistics.values()) for cid in CLASS_NAMES})
    overall_percentages = {cid: 100 * overall_counts[cid] / max(total, 1) for cid in CLASS_NAMES}
    output, warnings = {}, []
    for split, values in split_statistics.items():
        objects, images = values["total_objects"], values["total_images"]
        class_percentages = {CLASS_NAMES[cid]: round(100 * values["object_counts"][CLASS_NAMES[cid]] / max(objects, 1), 4) for cid in CLASS_NAMES}
        positive_images = images - values["class_combinations"]["no object"]
        split_warnings = []
        for cid in CLASS_NAMES:
            delta = abs(class_percentages[CLASS_NAMES[cid]] - overall_percentages[cid])
            if delta > warning_points:
                message = f"{split} {CLASS_NAMES[cid]} class percentage differs from overall by {delta:.2f} percentage points"
                split_warnings.append(message); warnings.append(message)
        output[split] = {"images": images, "objects": objects,
                         **{CLASS_NAMES[cid]: values["object_counts"][CLASS_NAMES[cid]] for cid in CLASS_NAMES},
                         "class_percentage": class_percentages,
                         "positive_image_percentage": round(100 * positive_images / max(images, 1), 4),
                         "positive_image_percentage_by_class": {CLASS_NAMES[cid]: round(100 * values["images_containing"][CLASS_NAMES[cid]] / max(images, 1), 4) for cid in CLASS_NAMES},
                         "objects_per_image": round(objects / max(images, 1), 4), "warnings": split_warnings}
    total_images = sum(item["total_images"] for item in split_statistics.values())
    return {"target_ratios": target_ratios,
            "achieved_image_ratios": {split: split_statistics[split]["total_images"] / max(total_images, 1) for split in SPLITS},
            "warning_percentage_points": warning_points, "splits": output, "warnings": warnings}

def run_dataset_cleaning(root: str | Path | None = None, config_path: str | Path | None = None) -> dict:
    root_path = Path(root) if root else project_root()
    config_file = Path(config_path) if config_path else root_path / "configs/dataset_processing.yaml"
    config = yaml.safe_load(config_file.read_text(encoding="utf-8")) if config_file.exists() else {}
    seed = set_global_seed(config.get("seed", 42))
    near_config = config.get("near_duplicate", {})
    near_enabled = near_config.get("enabled", config_file.exists())
    if near_config.get("method", "phash") != "phash":
        raise ValueError("Only phash is currently supported for near_duplicate detection")
    near_threshold = int(near_config.get("threshold", 8))
    if near_threshold < 0 or near_threshold > 64:
        raise ValueError("near_duplicate.threshold must be between 0 and 64")
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
    if near_enabled:
        groups, pre_split_near_matches, merged_group_roots = _merge_near_duplicate_groups(dict(groups), raw_images, near_threshold)
    else:
        groups, pre_split_near_matches = dict(groups), []
        merged_group_roots = {group: group for group in groups}
    group_splits = _assign_groups(groups, labels, config.get("split", {}))
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
    warning_points = float(config.get("split", {}).get("stratification", {}).get("warning_percentage_points", 15.0))
    target_ratios = config.get("split", {}).get("ratios", {"train": .70, "val": .15, "test": .15})
    (reports_dir / "split_statistics.json").write_text(json.dumps(_split_statistics(split_statistics, warning_points, target_ratios), indent=2), encoding="utf-8")
    report = {"status": "PASSED", "seed": seed, "group_aware": True, "cross_split_group_leakage": False, "group_distribution": {key: group_splits[key] for key in sorted(groups)}, "image_counts": {s: len(split_images[s]) for s in SPLITS}, "object_counts": {s: dict(counts[s]) for s in SPLITS}}
    (reports_dir / "data_leakage_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    cross_split_near_duplicates = []
    for match in pre_split_near_matches:
        split_a, split_b = group_splits[merged_group_roots[match["group_a"]]], group_splits[merged_group_roots[match["group_b"]]]
        if split_a != split_b:
            cross_split_near_duplicates.append({"image_a": match["image_a"], "split_a": split_a, "image_b": match["image_b"], "split_b": split_b, "phash_distance": match["phash_distance"]})
    (reports_dir / "near_duplicate_cross_split.json").write_text(json.dumps({"enabled": near_enabled, "method": "phash", "threshold": near_threshold, "pre_split_group_merges": pre_split_near_matches, "cross_split_near_duplicates": cross_split_near_duplicates, "status": "PASSED" if not cross_split_near_duplicates else "FAILED"}, indent=2), encoding="utf-8")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--root", default=None); parser.add_argument("--config", default=None)
    args = parser.parse_args()
    print(json.dumps(run_dataset_cleaning(args.root, args.config), indent=2))
