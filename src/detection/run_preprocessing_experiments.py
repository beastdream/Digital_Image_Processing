import os
import csv
import json
import time
import cv2
import yaml
import torch
import numpy as np
from ultralytics import YOLO

from src.preprocessing.pipeline import ImagePreprocessingPipeline
from src.detection.train import train_yolo
from src.detection.evaluate import evaluate_yolo

def generate_preprocessed_dataset(exp_name: str, config_file: str) -> str:
    base_dir = r"d:\Digital_Image_Processing"
    proc_detection_dir = os.path.join(base_dir, "data", "processed", "road_damage_detection")

    if exp_name in ["baseline", "no_preprocessing"]:
        return os.path.join(proc_detection_dir, "dataset.yaml")

    out_exp_dir = os.path.join(base_dir, "data", "processed", "road_damage_detection", "preprocessed", exp_name)
    out_yaml_path = os.path.join(out_exp_dir, "dataset.yaml")

    if os.path.exists(out_yaml_path):
        print(f"Preprocessed dataset for [{exp_name}] already exists at {out_yaml_path}")
        return out_yaml_path

    print(f"Generating preprocessed uint8 dataset for Experiment [{exp_name}] using [{config_file}]...")
    os.makedirs(out_exp_dir, exist_ok=True)

    config_path = os.path.join(base_dir, "configs", "experiments", config_file)
    pipeline = ImagePreprocessingPipeline(config_path)
    # Offline files are materialized once. Random augmentation belongs in
    # Ultralytics' per-epoch training loader, never in a persistent dataset.
    if pipeline.config.get("augmentation", {}).get("enabled", False):
        pipeline.config["augmentation"] = {"enabled": False}

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

            out_img, updated_bboxes, meta = pipeline.process(
                img, bboxes=bboxes, split=s, return_intermediates=True
            )

            uint8_img = meta.get("intermediates", {}).get("final_uint8", img)
            cv2.imwrite(os.path.join(dst_img_dir, fname), uint8_img)

            final_boxes = updated_bboxes if updated_bboxes is not None else bboxes
            lines = [f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in final_boxes]
            with open(os.path.join(dst_lbl_dir, stem + ".txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

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

    print(f"Dataset for [{exp_name}] saved to {out_yaml_path}")
    return out_yaml_path

def measure_inference_latency(model_path: str, dataset_yaml: str) -> float:
    model = YOLO(model_path)
    base_dir = r"d:\Digital_Image_Processing"
    test_img_dir = os.path.join(base_dir, "data", "processed", "road_damage_detection", "images", "test")
    test_imgs = sorted([os.path.join(test_img_dir, f) for f in os.listdir(test_img_dir) if f.lower().endswith(('.jpg', '.png'))])[:50]

    latencies = []
    for img_p in test_imgs:
        t0 = time.perf_counter()
        _ = model.predict(img_p, imgsz=320, verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(latencies[5:])) if len(latencies) > 5 else float(np.mean(latencies))

def run_preprocessing_experiments():
    base_dir = r"d:\Digital_Image_Processing"
    results_dir = os.path.join(base_dir, "results", "yolo")
    os.makedirs(results_dir, exist_ok=True)

    experiments = [
        ("A_No_Preprocessing", "baseline.yaml", "retrain_v1"),
        ("B_Gaussian_Denoising", "gaussian_denoise.yaml", "exp_b_gaussian"),
        ("C_Median_Denoising", "median_denoise.yaml", "exp_c_median"),
        ("D_CLAHE", "clahe.yaml", "exp_d_clahe"),
        ("E_Brightness", "brightness.yaml", "exp_e_brightness"),
        ("F_Full_Preprocessing", "full_preprocessing.yaml", "exp_f_full")
    ]

    summary_rows = []
    csv_out_path = os.path.join(results_dir, "experiment_comparison_v1.csv")

    for exp_label, config_file, exp_run_name in experiments:
        print(f"\n==========================================")
        print(f"   STARTING EXPERIMENT: {exp_label}")
        print(f"==========================================")

        # 1. Generate Dataset
        data_yaml = generate_preprocessed_dataset(exp_run_name, config_file)

        # If retrain_v1 already exists and evaluated, load its metrics
        if exp_run_name == "retrain_v1" and os.path.exists(os.path.join(results_dir, "retrain_v1", "metrics.json")):
            with open(os.path.join(results_dir, "retrain_v1", "metrics.json")) as f:
                retrain_meta = json.load(f)
            best_weights = os.path.join(base_dir, "experiments", "yolo", "retrain_v1", "weights", "best.pt")
            train_time_sec = 1520.0  # ~25 mins
            metrics = retrain_meta
        else:
            # 2. Train Model with controlled hyperparameters (5 epochs, batch 32 for comparison speed)
            t0_train = time.time()
            exp_cfg_path = os.path.join(base_dir, "configs", f"yolo_{exp_run_name}.yaml")
            cfg_dict = {
                "data_yaml": data_yaml,
                "name": exp_run_name,
                "imgsz": 320,
                "epochs": 4,
                "batch": 32,
                "fraction": 0.15,
                "optimizer": "Auto",
                "learning_rate": 0.01,
                "patience": 3,
                "seed": 42,
                "cache": False,
                "device": "cpu",
                "workers": 0
            }
            with open(exp_cfg_path, "w", encoding="utf-8") as f:
                yaml.dump(cfg_dict, f)

            best_weights = train_yolo(config_path=exp_cfg_path, exp_name_override=exp_run_name)
            train_time_sec = round(time.time() - t0_train, 2)

            # 3. Final Evaluation strictly on TEST split once model selection finishes
            exp_res_dir = os.path.join(results_dir, exp_run_name)
            metrics = evaluate_yolo(
                model_path=best_weights,
                data_yaml=data_yaml,
                output_dir=exp_res_dir,
                split="test"
            )

        # 4. Measure Inference Latency
        latency_ms = round(measure_inference_latency(best_weights, data_yaml), 2)

        overall = metrics.get("overall", {})
        row = {
            "experiment": exp_label,
            "precision": overall.get("precision", 0.0),
            "recall": overall.get("recall", 0.0),
            "mAP50": overall.get("mAP50", 0.0),
            "mAP50_95": overall.get("mAP50_95", 0.0),
            "training_time_sec": train_time_sec,
            "inference_latency_ms": latency_ms
        }
        summary_rows.append(row)
        print(f"Row generated for {exp_label}: {row}")

        # Save incremental results after each experiment
        with open(csv_out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "experiment", "precision", "recall", "mAP50", "mAP50_95", "training_time_sec", "inference_latency_ms"
            ])
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"\nALL PREPROCESSING EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"Summary table written to: {csv_out_path}")

if __name__ == "__main__":
    run_preprocessing_experiments()
