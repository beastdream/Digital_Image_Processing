import os
import csv
import json
import time
import cv2
import yaml
import shutil
import numpy as np
from ultralytics import YOLO

from src.preprocessing.pipeline import ImagePreprocessingPipeline
from src.detection.train import train_yolo
from src.detection.evaluate import evaluate_yolo

def generate_preprocessed_dataset(exp_name: str, config_name: str) -> str:
    """
    Generates uint8 preprocessed images and dataset.yaml for a given experiment
    without double normalization, saving to data/processed/preprocessed/<exp_name>/.
    """
    base_dir = r"d:\Digital_Image_Processing"
    proc_detection_dir = os.path.join(base_dir, "data", "processed", "detection")
    out_exp_dir = os.path.join(base_dir, "data", "processed", "preprocessed", exp_name)

    if exp_name == "baseline":
        return os.path.join(proc_detection_dir, "dataset.yaml")

    print(f"Generating preprocessed uint8 dataset for Experiment [{exp_name}] using config [{config_name}]...")
    os.makedirs(out_exp_dir, exist_ok=True)

    config_path = os.path.join(base_dir, "configs", "experiments", config_name)
    pipeline = ImagePreprocessingPipeline(config_path)

    splits = ['train', 'val', 'test']
    for s in splits:
        src_img_dir = os.path.join(proc_detection_dir, "images", s)
        src_lbl_dir = os.path.join(proc_detection_dir, "labels", s)

        dst_img_dir = os.path.join(out_exp_dir, "images", s)
        dst_lbl_dir = os.path.join(out_exp_dir, "labels", s)

        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)

        if not os.path.exists(src_img_dir):
            continue

        for fname in os.listdir(src_img_dir):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            stem = os.path.splitext(fname)[0]
            img_path = os.path.join(src_img_dir, fname)
            lbl_path = os.path.join(src_lbl_dir, stem + ".txt")

            img = cv2.imread(img_path)
            if img is None:
                continue

            bboxes = []
            if os.path.exists(lbl_path):
                with open(lbl_path, "r", encoding="utf-8") as f:
                    for l in f:
                        parts = l.strip().split()
                        if len(parts) == 5:
                            bboxes.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])

            # Process image up to uint8 stage (avoid float scale double-normalization)
            out_img, updated_bboxes, meta = pipeline.process(
                img, bboxes=bboxes, split=s, return_intermediates=True
            )

            uint8_img = meta.get("intermediates", {}).get("final_uint8", img)

            # Save uint8 processed image
            cv2.imwrite(os.path.join(dst_img_dir, fname), uint8_img)

            # Save labels
            lines = [f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in (updated_bboxes or bboxes)]
            with open(os.path.join(dst_lbl_dir, stem + ".txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

    # Create dataset.yaml for preprocessed experiment
    out_yaml_path = os.path.join(out_exp_dir, "dataset.yaml")
    out_exp_dir_posix = out_exp_dir.replace("\\", "/")
    yaml_content = f"""path: {out_exp_dir_posix}
train: images/train
val: images/val
test: images/test

nc: 3
names:
  0: pothole
  1: crack
  2: manhole
"""
    with open(out_yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"Dataset generated at {out_yaml_path}")
    return out_yaml_path

def run_experiment_comparison():
    base_dir = r"d:\Digital_Image_Processing"
    results_dir = os.path.join(base_dir, "results", "yolo")
    os.makedirs(results_dir, exist_ok=True)

    experiments = [
        ("baseline", "baseline.yaml"),
        ("exp_b_gaussian", "denoising.yaml"),
        ("exp_c_median", "denoising.yaml"),
        ("exp_d_clahe", "clahe.yaml"),
        ("exp_e_brightness", "brightness.yaml"),
        ("exp_f_full_preprocessing", "full_preprocessing.yaml")
    ]

    print("=== Preprocessing + YOLO Experiment Comparison Framework ===")

    # Prepare datasets for all experiments
    exp_datasets = {}
    for exp_name, cfg_name in experiments:
        exp_yaml = generate_preprocessed_dataset(exp_name, cfg_name)
        exp_datasets[exp_name] = exp_yaml

    print(f"Generated {len(exp_datasets)} experiment dataset configurations.")

    # CSV summary initialization
    csv_path = os.path.join(results_dir, "experiment_comparison.csv")

    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    print(f"Experiment framework ready. Results stored at {csv_path}")

if __name__ == "__main__":
    run_experiment_comparison()
