import os
import json
import csv
import hashlib
import numpy as np
import yaml
from typing import Dict, List, Tuple

def compute_md5(filepath: str) -> str:
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def audit_dataset() -> Tuple[Dict, bool]:
    base_dir = r"d:\Digital_Image_Processing"
    proc_dir = os.path.join(base_dir, "data", "processed", "detection")
    output_dir = os.path.join(base_dir, "results", "yolo", "debug")
    os.makedirs(output_dir, exist_ok=True)

    dataset_yaml_path = os.path.join(proc_dir, "dataset.yaml")

    audit_result = {
        "yaml_check": {},
        "file_counts": {},
        "matching_check": {},
        "overlap_check": {},
        "label_validity": {
            "invalid_class_ids": 0,
            "out_of_bounds_coords": 0,
            "non_positive_wh": 0,
            "nan_inf_values": 0,
            "malformed_lines": 0
        },
        "class_counts_per_split": {},
        "is_valid": True,
        "errors": []
    }

    # 1. Inspect dataset.yaml
    if not os.path.exists(dataset_yaml_path):
        audit_result["errors"].append("dataset.yaml missing")
        audit_result["is_valid"] = False
    else:
        with open(dataset_yaml_path, "r", encoding="utf-8") as f:
            data_cfg = yaml.safe_load(f)
        audit_result["yaml_check"] = {
            "nc": data_cfg.get("nc"),
            "names": data_cfg.get("names"),
            "path": data_cfg.get("path"),
            "train": data_cfg.get("train"),
            "val": data_cfg.get("val"),
            "test": data_cfg.get("test")
        }
        if data_cfg.get("nc") != 3:
            audit_result["errors"].append(f"Expected nc=3, got {data_cfg.get('nc')}")
            audit_result["is_valid"] = False

    # 2. File counts & Split analysis
    splits = ["train", "val", "test"]
    split_files = {}
    split_md5s = {}
    class_counts_per_split = {s: {0: 0, 1: 0, 2: 0} for s in splits}
    total_valid_objects = 0

    for s in splits:
        img_dir = os.path.join(proc_dir, "images", s)
        lbl_dir = os.path.join(proc_dir, "labels", s)

        imgs = set(os.listdir(img_dir)) if os.path.exists(img_dir) else set()
        lbls = set(os.listdir(lbl_dir)) if os.path.exists(lbl_dir) else set()

        img_stems = {os.path.splitext(f)[0]: f for f in imgs if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
        lbl_stems = {os.path.splitext(f)[0]: f for f in lbls if f.lower().endswith('.txt')}

        audit_result["file_counts"][s] = {
            "images": len(img_stems),
            "labels": len(lbl_stems)
        }

        # Match stems
        missing_lbls = set(img_stems.keys()) - set(lbl_stems.keys())
        missing_imgs = set(lbl_stems.keys()) - set(img_stems.keys())

        if missing_lbls:
            audit_result["errors"].append(f"Split {s}: {len(missing_lbls)} images missing label files")
        if missing_imgs:
            audit_result["errors"].append(f"Split {s}: {len(missing_imgs)} labels missing image files")

        split_files[s] = img_stems

        # Compute MD5s
        md5_map = {}
        for stem, fname in img_stems.items():
            fpath = os.path.join(img_dir, fname)
            md5_map[stem] = compute_md5(fpath)
        split_md5s[s] = md5_map

        # Audit Label Content
        for stem, fname in lbl_stems.items():
            lpath = os.path.join(lbl_dir, fname)
            with open(lpath, "r", encoding="utf-8") as lf:
                lines = lf.readlines()

            for line_no, line in enumerate(lines, start=1):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) != 5:
                    audit_result["label_validity"]["malformed_lines"] += 1
                    audit_result["is_valid"] = False
                    continue

                try:
                    cid = int(parts[0])
                    xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                except ValueError:
                    audit_result["label_validity"]["nan_inf_values"] += 1
                    audit_result["is_valid"] = False
                    continue

                if np.isnan([xc, yc, w, h]).any() or np.isinf([xc, yc, w, h]).any():
                    audit_result["label_validity"]["nan_inf_values"] += 1
                    audit_result["is_valid"] = False

                if cid not in [0, 1, 2]:
                    audit_result["label_validity"]["invalid_class_ids"] += 1
                    audit_result["is_valid"] = False

                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                    audit_result["label_validity"]["out_of_bounds_coords"] += 1
                    audit_result["is_valid"] = False

                if w <= 0.0 or h <= 0.0:
                    audit_result["label_validity"]["non_positive_wh"] += 1
                    audit_result["is_valid"] = False

                if cid in [0, 1, 2]:
                    class_counts_per_split[s][cid] += 1
                    total_valid_objects += 1

    audit_result["class_counts_per_split"] = class_counts_per_split
    audit_result["total_valid_objects"] = total_valid_objects

    # 3. Filename & MD5 Overlap Check between splits
    train_stems = set(split_files["train"].keys())
    val_stems = set(split_files["val"].keys())
    test_stems = set(split_files["test"].keys())

    fn_overlap_tv = train_stems.intersection(val_stems)
    fn_overlap_tt = train_stems.intersection(test_stems)
    fn_overlap_vt = val_stems.intersection(test_stems)

    if fn_overlap_tv or fn_overlap_tt or fn_overlap_vt:
        audit_result["errors"].append(f"Filename leakage detected between splits: TV={len(fn_overlap_tv)}, TT={len(fn_overlap_tt)}, VT={len(fn_overlap_vt)}")
        audit_result["is_valid"] = False

    train_md5s = set(split_md5s["train"].values())
    val_md5s = set(split_md5s["val"].values())
    test_md5s = set(split_md5s["test"].values())

    md5_overlap_tv = train_md5s.intersection(val_md5s)
    md5_overlap_tt = train_md5s.intersection(test_md5s)
    md5_overlap_vt = val_md5s.intersection(test_md5s)

    audit_result["overlap_check"] = {
        "filename_overlap_train_val": len(fn_overlap_tv),
        "filename_overlap_train_test": len(fn_overlap_tt),
        "filename_overlap_val_test": len(fn_overlap_vt),
        "md5_overlap_train_val": len(md5_overlap_tv),
        "md5_overlap_train_test": len(md5_overlap_tt),
        "md5_overlap_val_test": len(md5_overlap_vt)
    }

    if md5_overlap_tv or md5_overlap_tt or md5_overlap_vt:
        audit_result["errors"].append(f"MD5 image content leakage detected between splits: TV={len(md5_overlap_tv)}, TT={len(md5_overlap_tt)}, VT={len(md5_overlap_vt)}")
        audit_result["is_valid"] = False

    # Save dataset_audit.json
    audit_json_path = os.path.join(output_dir, "dataset_audit.json")
    with open(audit_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_result, f, indent=2)

    # Save class_distribution.csv
    csv_path = os.path.join(output_dir, "class_distribution.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "pothole_0", "crack_1", "manhole_2", "total_objects"])
        for s in splits:
            c = class_counts_per_split[s]
            writer.writerow([s, c[0], c[1], c[2], sum(c.values())])

    print(f"Dataset Audit Complete! Is Valid: {audit_result['is_valid']}")
    print(f"Audit JSON saved to: {audit_json_path}")
    print(f"Class distribution saved to: {csv_path}")

    return audit_result, audit_result["is_valid"]

if __name__ == "__main__":
    audit_dataset()
