import os
import shutil
import json
import csv
import hashlib
from collections import defaultdict, Counter

def run_dataset_cleaning():
    base_dir = r"d:\Digital_Image_Processing"
    data_dir = os.path.join(base_dir, "data")
    raw_dir = os.path.join(data_dir, "raw")
    raw_images_dir = os.path.join(raw_dir, "images")
    raw_labels_yolo_dir = os.path.join(raw_dir, "labels-YOLO")

    reports_dir = os.path.join(data_dir, "reports")
    processed_dir = os.path.join(data_dir, "processed", "detection")

    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # 1. Protection Check
    assert os.path.exists(raw_images_dir), f"Raw images not found in {raw_images_dir}"
    assert os.path.exists(raw_labels_yolo_dir), f"Raw YOLO labels not found in {raw_labels_yolo_dir}"

    all_image_files = sorted([f for f in os.listdir(raw_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"Total raw images found: {len(all_image_files)}")

    # 2. Duplicate Check via MD5 Hash
    hashes = defaultdict(list)
    for fname in all_image_files:
        fpath = os.path.join(raw_images_dir, fname)
        with open(fpath, 'rb') as f:
            h = hashlib.md5(f.read()).hexdigest()
        hashes[h].append(fname)

    duplicate_groups = {}
    clean_image_files = []
    excluded_duplicate_images = []

    for h, file_list in hashes.items():
        kept_file = file_list[0]
        clean_image_files.append(kept_file)
        if len(file_list) > 1:
            duplicate_groups[h] = {
                "kept_image": kept_file,
                "duplicate_images": file_list[1:],
                "total_duplicates": len(file_list)
            }
            excluded_duplicate_images.extend(file_list[1:])

    # Save duplicates report
    duplicates_report_path = os.path.join(reports_dir, "duplicates.json")
    with open(duplicates_report_path, "w", encoding="utf-8") as f:
        json.dump({
            "duplicate_groups_count": len(duplicate_groups),
            "total_excluded_images": len(excluded_duplicate_images),
            "duplicate_groups": duplicate_groups
        }, f, indent=2)

    print(f"Duplicates processed: {len(duplicate_groups)} groups, {len(excluded_duplicate_images)} excluded images.")

    # 3. Read and Clean Labels
    invalid_annotations = []
    cleaned_labels = {} # stem -> list of formatted yolo lines
    clipped_coords_count = 0

    for fname in clean_image_files:
        stem = os.path.splitext(fname)[0]
        label_fname = stem + ".txt"
        label_path = os.path.join(raw_labels_yolo_dir, label_fname)

        valid_lines_for_file = []

        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            for line_idx, line in enumerate(lines):
                parts = line.split()
                if len(parts) != 5:
                    invalid_annotations.append({
                        "file": label_fname,
                        "line_index": line_idx,
                        "reason": f"Expected 5 values, got {len(parts)}",
                        "content": line
                    })
                    continue

                try:
                    cls_id = int(parts[0])
                    xc = float(parts[1])
                    yc = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])

                    if w <= 0.0 or h <= 0.0:
                        invalid_annotations.append({
                            "file": label_fname,
                            "line_index": line_idx,
                            "reason": f"Invalid dimension: width={w}, height={h}",
                            "content": line
                        })
                        continue

                    # Bbox bounds
                    xmin = xc - w / 2.0
                    ymin = yc - h / 2.0
                    xmax = xc + w / 2.0
                    ymax = yc + h / 2.0

                    was_clipped = False
                    if xmin < 0.0 or ymin < 0.0 or xmax > 1.0 or ymax > 1.0:
                        was_clipped = True
                        clipped_coords_count += 1
                        xmin = max(0.0, min(1.0, xmin))
                        ymin = max(0.0, min(1.0, ymin))
                        xmax = max(0.0, min(1.0, xmax))
                        ymax = max(0.0, min(1.0, ymax))

                        xc = (xmin + xmax) / 2.0
                        yc = (ymin + ymax) / 2.0
                        w = xmax - xmin
                        h = ymax - ymin

                    valid_lines_for_file.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

                except Exception as e:
                    invalid_annotations.append({
                        "file": label_fname,
                        "line_index": line_idx,
                        "reason": str(e),
                        "content": line
                    })

        cleaned_labels[stem] = valid_lines_for_file

    # Save invalid annotations report
    invalid_report_path = os.path.join(reports_dir, "invalid_annotations.json")
    with open(invalid_report_path, "w", encoding="utf-8") as f:
        json.dump({
            "invalid_annotations_count": len(invalid_annotations),
            "clipped_coordinates_count": clipped_coords_count,
            "invalid_annotations": invalid_annotations
        }, f, indent=2)

    print(f"Invalid annotations excluded: {len(invalid_annotations)}. Clipped coordinates: {clipped_coords_count}.")

    # 4. Sequence-based Train / Val / Test Split
    def get_sequence_key(fname):
        name = os.path.splitext(fname)[0]
        if name.startswith('vlcsnap'):
            parts = name.split('-')
            if len(parts) >= 4 and parts[1].isdigit() and len(parts[1]) == 4:
                date = parts[1] + '-' + parts[2] + '-' + parts[3]
                time = parts[4] if len(parts) >= 5 else ''
                hour = time[:3] if 'h' in time else ''
                return f'vlcsnap_{date}_{hour}'
            else:
                return 'vlcsnap_numeric'
        elif '_' in name:
            date = name.split('_')[0]
            return f'seq_{date}'
        return 'seq_other'

    seq_assignment = {
        'seq_20250216': 'val',
        'seq_20250219': 'val',
        'seq_20250223': 'train',
        'vlcsnap_numeric': 'train',
        'vlcsnap_2025-02-18_17h': 'val',
        'vlcsnap_2025-02-18_18h': 'train',
        'vlcsnap_2025-02-18_23h': 'test',
        'vlcsnap_2025-02-19_13h': 'test',
        'vlcsnap_2025-02-19_14h': 'train',
        'vlcsnap_2025-02-19_15h': 'val',
        'vlcsnap_2025-02-19_16h': 'test',
        'vlcsnap_2025-02-19_17h': 'train',
        'vlcsnap_2025-02-26_20h': 'train'
    }

    split_images = {'train': [], 'val': [], 'test': []}
    seq_split_report = defaultdict(lambda: defaultdict(int))

    for fname in clean_image_files:
        seq_key = get_sequence_key(fname)
        split = seq_assignment.get(seq_key, 'train')
        split_images[split].append(fname)
        seq_split_report[seq_key][split] += 1

    # 5. Populate Processed Dataset Folders
    splits = ['train', 'val', 'test']
    for s in splits:
        os.makedirs(os.path.join(processed_dir, "images", s), exist_ok=True)
        os.makedirs(os.path.join(processed_dir, "labels", s), exist_ok=True)

    class_counts = {s: Counter() for s in splits}
    image_counts = {s: len(split_images[s]) for s in splits}

    for s in splits:
        img_out_dir = os.path.join(processed_dir, "images", s)
        lbl_out_dir = os.path.join(processed_dir, "labels", s)

        for fname in split_images[s]:
            stem = os.path.splitext(fname)[0]

            # Copy Image
            src_img = os.path.join(raw_images_dir, fname)
            dst_img = os.path.join(img_out_dir, fname)
            shutil.copy2(src_img, dst_img)

            # Write Cleaned Label
            dst_lbl = os.path.join(lbl_out_dir, stem + ".txt")
            lines = cleaned_labels.get(stem, [])
            with open(dst_lbl, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            # Count objects
            for line in lines:
                cls_id = int(line.split()[0])
                class_counts[s][cls_id] += 1

    # 6. Create dataset.yaml
    processed_dir_posix = processed_dir.replace('\\', '/')
    yaml_content = f"""path: {processed_dir_posix}
train: images/train
val: images/val
test: images/test

nc: 3
names:
  0: pothole
  1: crack
  2: manhole
"""
    yaml_path = os.path.join(processed_dir, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"Created dataset.yaml at {yaml_path}")

    # 7. Create dataset_statistics.csv
    class_names = {0: "pothole", 1: "crack", 2: "manhole"}
    csv_path = os.path.join(reports_dir, "dataset_statistics.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "train_images", "val_images", "test_images", "train_objects", "val_objects", "test_objects"])

        for cid in [0, 1, 2]:
            cname = class_names[cid]
            # Count images containing class
            train_imgs_c = sum(1 for fname in split_images['train'] if any(int(l.split()[0]) == cid for l in cleaned_labels[os.path.splitext(fname)[0]]))
            val_imgs_c = sum(1 for fname in split_images['val'] if any(int(l.split()[0]) == cid for l in cleaned_labels[os.path.splitext(fname)[0]]))
            test_imgs_c = sum(1 for fname in split_images['test'] if any(int(l.split()[0]) == cid for l in cleaned_labels[os.path.splitext(fname)[0]]))

            tr_objs = class_counts['train'][cid]
            va_objs = class_counts['val'][cid]
            te_objs = class_counts['test'][cid]

            writer.writerow([cid, cname, train_imgs_c, val_imgs_c, test_imgs_c, tr_objs, va_objs, te_objs])

    print(f"Created dataset_statistics.csv at {csv_path}")

    # 8. Data Leakage Verification
    train_stems = set(os.path.splitext(f)[0] for f in split_images['train'])
    val_stems = set(os.path.splitext(f)[0] for f in split_images['val'])
    test_stems = set(os.path.splitext(f)[0] for f in split_images['test'])

    train_val_overlap = train_stems.intersection(val_stems)
    train_test_overlap = train_stems.intersection(test_stems)
    val_test_overlap = val_stems.intersection(test_stems)

    seq_leakage_found = False
    seq_split_summary = {}
    for seq_k, splits_dict in seq_split_report.items():
        seq_split_summary[seq_k] = dict(splits_dict)
        if len(splits_dict) > 1:
            seq_leakage_found = True

    leakage_report = {
        "status": "PASSED" if not (train_val_overlap or train_test_overlap or val_test_overlap or seq_leakage_found) else "FAILED",
        "train_val_stem_overlap": len(train_val_overlap),
        "train_test_stem_overlap": len(train_test_overlap),
        "val_test_stem_overlap": len(val_test_overlap),
        "sequence_cross_split_leakage": seq_leakage_found,
        "sequence_distribution": seq_split_summary,
        "image_counts": {
            "train": len(split_images['train']),
            "val": len(split_images['val']),
            "test": len(split_images['test']),
            "total_clean": len(clean_image_files)
        },
        "object_counts": {
            "train": dict(class_counts['train']),
            "val": dict(class_counts['val']),
            "test": dict(class_counts['test'])
        }
    }

    leakage_report_path = os.path.join(reports_dir, "data_leakage_report.json")
    with open(leakage_report_path, "w", encoding="utf-8") as f:
        json.dump(leakage_report, f, indent=2)

    print(f"Data leakage report saved to {leakage_report_path}")
    print("DATASET PREPARATION COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    run_dataset_cleaning()
