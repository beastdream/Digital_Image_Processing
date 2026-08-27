import os
import random
import cv2
import numpy as np
import yaml
from typing import List
from pathlib import Path

from src.preprocessing.pipeline import ImagePreprocessingPipeline
from src.data.dataset_utils import project_root
from src.utils.result_paths import preprocessing_visualization_dir, results_root, reset_directory

def visualize_preprocessing_samples():
    base_dir = str(project_root())
    proc_dir = os.path.join(base_dir, "data", "processed", "road_damage_detection")
    output_dir = reset_directory(preprocessing_visualization_dir(Path(base_dir)), results_root(Path(base_dir)))

    config_path = os.path.join(base_dir, "configs", "experiments", "full_preprocessing.yaml")
    pipeline = ImagePreprocessingPipeline(config_path)

    class_names = {0: "pothole", 1: "crack", 2: "manhole"}
    class_colors = {
        0: (0, 0, 255),    # Red
        1: (255, 0, 0),    # Blue
        2: (0, 255, 0)     # Green
    }

    # Collect sample images from train, val, test
    splits = ['train', 'val', 'test']
    sample_items = []

    for s in splits:
        img_dir = os.path.join(proc_dir, "images", s)
        lbl_dir = os.path.join(proc_dir, "labels", s)
        if not os.path.exists(img_dir):
            continue

        imgs = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        # Pick 7 from train, 7 from val, 6 from test = 20 total
        count = 7 if s in ['train', 'val'] else 6
        random.seed(100 + len(s))
        picked = random.sample(imgs, min(count, len(imgs)))

        for fname in picked:
            stem = os.path.splitext(fname)[0]
            lbl_path = os.path.join(lbl_dir, stem + ".txt")
            sample_items.append((s, fname, os.path.join(img_dir, fname), lbl_path))

    print(f"Selected {len(sample_items)} sample images across splits for visualization.")

    for idx, (split, fname, img_path, lbl_path) in enumerate(sample_items):
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Load bboxes
        bboxes = []
        if os.path.exists(lbl_path):
            with open(lbl_path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            for l in lines:
                parts = l.split()
                if len(parts) == 5:
                    bboxes.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])

        # Run pipeline
        final_img, updated_bboxes, meta = pipeline.process(
            img, bboxes=bboxes, split=split, return_intermediates=True
        )

        intermediates = meta.get("intermediates", {})

        # Prepare stage panels for display
        stages = [
            ("1. Original", intermediates.get("original", img)),
            ("2. Resize", intermediates.get("resize", img)),
            ("3. Denoise", intermediates.get("denoise", intermediates.get("resize", img))),
            ("4. CLAHE", intermediates.get("contrast", intermediates.get("denoise", img))),
            ("5. Brightness", intermediates.get("brightness", intermediates.get("contrast", img))),
            ("6. Final (with BBoxes)", intermediates.get("final_uint8", img))
        ]

        # Draw bounding boxes on the 6th panel (Final with BBoxes)
        panel_final = stages[-1][1].copy()
        if updated_bboxes:
            h_f, w_f = panel_final.shape[:2]
            for bbox in updated_bboxes:
                cid, xc, yc, bw, bh = bbox
                x1 = int(max(0, (xc - bw/2) * w_f))
                y1 = int(max(0, (yc - bh/2) * h_f))
                x2 = int(min(w_f - 1, (xc + bw/2) * w_f))
                y2 = int(min(h_f - 1, (yc + bh/2) * h_f))

                cname = class_names.get(int(cid), str(cid))
                color = class_colors.get(int(cid), (255, 255, 255))

                cv2.rectangle(panel_final, (x1, y1), (x2, y2), color, 2)
                cv2.putText(panel_final, f"[{int(cid)}] {cname}", (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        stages[-1] = ("6. Final (with BBoxes)", panel_final)

        # Standardize stage images to equal display dimensions (320x320 per stage panel)
        stage_panels = []
        for stage_title, stage_img in stages:
            resized_stage = cv2.resize(stage_img, (320, 320), interpolation=cv2.INTER_LINEAR)
            # Add top title banner
            banner = np.zeros((30, 320, 3), dtype=np.uint8)
            cv2.putText(banner, stage_title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            panel_combined = np.vstack([banner, resized_stage])
            stage_panels.append(panel_combined)

        # Grid layout: 2 rows x 3 columns
        row1 = np.hstack(stage_panels[:3])
        row2 = np.hstack(stage_panels[3:])
        grid = np.vstack([row1, row2])

        out_fname = f"sample_{idx+1:02d}_{split}_{os.path.splitext(fname)[0]}.jpg"
        cv2.imwrite(str(output_dir / out_fname), grid)

    print(f"Saved {len(sample_items)} visualization grids to {output_dir}")

if __name__ == "__main__":
    visualize_preprocessing_samples()
