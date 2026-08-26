import os
import time
import json
import yaml
import cv2
import numpy as np
from typing import Dict

from src.preprocessing.pipeline import ImagePreprocessingPipeline

def run_experiment_suite():
    base_dir = r"d:\Digital_Image_Processing"
    proc_dir = os.path.join(base_dir, "data", "processed", "road_damage_detection")
    exp_configs_dir = os.path.join(base_dir, "configs", "experiments")
    experiments_output_base = os.path.join(base_dir, "experiments")

    config_files = [
        "baseline.yaml",
        "resize.yaml",
        "denoising.yaml",
        "clahe.yaml",
        "brightness.yaml",
        "augmentation.yaml",
        "full_preprocessing.yaml"
    ]

    # Pick 20 sample images across splits for experiment benchmarking
    sample_images = []
    splits = ['train', 'val', 'test']
    for s in splits:
        img_dir = os.path.join(proc_dir, "images", s)
        lbl_dir = os.path.join(proc_dir, "labels", s)
        if not os.path.exists(img_dir):
            continue
        imgs = sorted(os.listdir(img_dir))[:7 if s != 'test' else 6]
        for fname in imgs:
            stem = os.path.splitext(fname)[0]
            sample_images.append((s, fname, os.path.join(img_dir, fname), os.path.join(lbl_dir, stem + ".txt")))

    print(f"Loaded {len(sample_images)} benchmark sample images for experiment tracking.")

    suite_summary = {}

    for cfg_fname in config_files:
        cfg_path = os.path.join(exp_configs_dir, cfg_fname)
        if not os.path.exists(cfg_path):
            print(f"Skipping missing config: {cfg_path}")
            continue

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        exp_name = cfg.get("experiment_name", os.path.splitext(cfg_fname)[0])
        exp_dir = os.path.join(experiments_output_base, exp_name)
        samples_out_dir = os.path.join(exp_dir, "samples")
        os.makedirs(samples_out_dir, exist_ok=True)

        pipeline = ImagePreprocessingPipeline(cfg)

        # Save config copy
        with open(os.path.join(exp_dir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)

        # Run experiment on benchmark samples
        times = []
        means = []
        stds = []
        mins = []
        maxs = []

        for idx, (split, fname, img_path, lbl_path) in enumerate(sample_images):
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

            t0 = time.time()
            out_img, updated_bboxes, meta = pipeline.process(img, bboxes=bboxes, split=split, return_intermediates=True)
            elapsed_ms = (time.time() - t0) * 1000.0

            times.append(elapsed_ms)
            means.append(float(np.mean(out_img)))
            stds.append(float(np.std(out_img)))
            mins.append(float(np.min(out_img)))
            maxs.append(float(np.max(out_img)))

            # Save sample image output for visualization
            if idx < 5:
                vis_img = meta.get("intermediates", {}).get("final_uint8", img)
                out_sample_path = os.path.join(samples_out_dir, f"sample_{idx+1:02d}_{split}_{fname}")
                cv2.imwrite(out_sample_path, vis_img)

        exp_stats = {
            "experiment_name": exp_name,
            "description": cfg.get("description", ""),
            "processed_samples": len(times),
            "avg_latency_ms": float(np.mean(times)),
            "pixel_stats": {
                "mean_intensity": float(np.mean(means)),
                "std_intensity": float(np.mean(stds)),
                "min_pixel": float(np.min(mins)),
                "max_pixel": float(np.max(maxs))
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Save stats JSON and log
        with open(os.path.join(exp_dir, "stats.json"), "w", encoding="utf-8") as f:
            json.dump(exp_stats, f, indent=2)

        with open(os.path.join(exp_dir, "experiment.log"), "w", encoding="utf-8") as f:
            f.write(f"Experiment: {exp_name}\n")
            f.write(f"Description: {cfg.get('description', '')}\n")
            f.write(f"Avg Processing Latency: {exp_stats['avg_latency_ms']:.2f} ms/image\n")
            f.write(f"Pixel Mean: {exp_stats['pixel_stats']['mean_intensity']:.4f}\n")
            f.write(f"Pixel Std: {exp_stats['pixel_stats']['std_intensity']:.4f}\n")

        suite_summary[exp_name] = exp_stats
        print(f"Completed experiment: {exp_name} | Latency: {exp_stats['avg_latency_ms']:.2f} ms/img")

    summary_path = os.path.join(experiments_output_base, "experiment_suite_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(suite_summary, f, indent=2)

    print(f"\nExperiment Suite Execution Finished! Summary saved to {summary_path}")

if __name__ == "__main__":
    run_experiment_suite()
