import os
import cv2
import random
import numpy as np
from src.data.dataset_utils import CLASS_NAMES, project_root

def visualize_labels():
    base_dir = str(project_root())
    proc_dir = os.path.join(base_dir, "data", "processed", "road_damage_detection")
    out_dir = os.path.join(base_dir, "results", "yolo", "debug", "label_visualization")
    os.makedirs(out_dir, exist_ok=True)

    class_names = CLASS_NAMES
    colors = {
        0: (0, 0, 255),    # Red
        1: (255, 0, 0),    # Blue
        2: (0, 255, 0)     # Green
    }

    samples_per_split = {"train": 20, "val": 10, "test": 10}
    total_rendered = 0

    for split, count in samples_per_split.items():
        img_dir = os.path.join(proc_dir, "images", split)
        lbl_dir = os.path.join(proc_dir, "labels", split)

        if not os.path.exists(img_dir):
            continue

        imgs = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        random.seed(42 + len(split))
        picked = random.sample(imgs, min(count, len(imgs)))

        for fname in picked:
            stem = os.path.splitext(fname)[0]
            img_path = os.path.join(img_dir, fname)
            lbl_path = os.path.join(lbl_dir, stem + ".txt")

            img = cv2.imread(img_path)
            if img is None:
                continue

            h_img, w_img = img.shape[:2]

            if os.path.exists(lbl_path):
                with open(lbl_path, "r", encoding="utf-8") as lf:
                    lines = [l.strip() for l in lf if l.strip()]

                for line in lines:
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    cid = int(parts[0])
                    xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                    x1 = int(max(0, (xc - bw/2) * w_img))
                    y1 = int(max(0, (yc - bh/2) * h_img))
                    x2 = int(min(w_img - 1, (xc + bw/2) * w_img))
                    y2 = int(min(h_img - 1, (yc + bh/2) * h_img))

                    cname = class_names.get(cid, f"Class {cid}")
                    color = colors.get(cid, (255, 255, 255))

                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    label_str = f"[{cid}] {cname}"
                    cv2.putText(img, label_str, (x1, max(15, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            out_path = os.path.join(out_dir, f"{split}_{fname}")
            cv2.imwrite(out_path, img)
            total_rendered += 1

    print(f"Visual Label Verification Complete! Rendered {total_rendered} images to {out_dir}")

if __name__ == "__main__":
    visualize_labels()
