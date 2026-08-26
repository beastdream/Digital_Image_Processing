import os
import cv2
import glob
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from src.data.dataset_utils import CLASS_NAMES, project_root

def run_prediction(
    weights_path: str,
    source_dir: str = "data/processed/road_damage_detection/images/test",
    output_dir: str = "results/yolo/baseline/predictions",
    conf_threshold: float = 0.25,
    max_images: int = 20
) -> int:
    base_dir = project_root()
    full_output_dir = base_dir / output_dir if not os.path.isabs(output_dir) else Path(output_dir)
    os.makedirs(full_output_dir, exist_ok=True)

    print(f"Loading YOLO Model [{weights_path}] for prediction on [{source_dir}]...")
    model = YOLO(weights_path)
    model_names = {int(key): value for key, value in model.names.items()}
    if model_names != CLASS_NAMES:
        raise ValueError(f"Weights must be a 3-class road-damage model with {CLASS_NAMES}; got {model_names}")

    # Collect source images
    source_path = Path(source_dir)
    if not source_path.is_absolute():
        source_path = base_dir / source_path
    if source_path.is_file():
        image_paths = [str(source_path)]
    elif source_path.is_dir():
        image_paths = sorted([
            str(source_path / f) for f in os.listdir(source_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])[:max_images]
    else:
        raise ValueError(f"Invalid source path: {source_path}")

    class_names = CLASS_NAMES
    colors = {
        0: (0, 0, 255),    # Bright Red for POTHOLE
        1: (255, 0, 0),    # Blue for Crack
        2: (0, 255, 0)     # Green for Manhole
    }

    processed_count = 0

    for img_path in image_paths:
        fname = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            continue

        results = model.predict(source=img, conf=conf_threshold, verbose=False)[0]

        # Draw detected bounding boxes
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int)

            x1, y1, x2, y2 = xyxy
            if cls_id not in class_names:
                raise ValueError(f"Model emitted unsupported class_id {cls_id}; expected IDs 0, 1, 2")
            cname = class_names[cls_id]
            color = colors.get(cls_id, (255, 255, 255))

            # Highlight POTHOLE distinctly with thicker border & clear label
            thickness = 3 if cls_id == 0 else 2
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            label_text = f"{cname} {conf:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(img, (x1, max(0, y1 - text_h - 4)), (x1 + text_w + 4, y1), color, -1)
            cv2.putText(img, label_text, (x1 + 2, max(text_h, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        out_path = os.path.join(full_output_dir, f"pred_{fname}")
        cv2.imwrite(out_path, img)
        processed_count += 1

    print(f"Saved {processed_count} prediction visualizations to: {full_output_dir}")
    return processed_count

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to trained YOLO weights best.pt")
    parser.add_argument("--source", type=str, default="data/processed/road_damage_detection/images/test")
    parser.add_argument("--output", type=str, default="results/yolo/baseline/predictions")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max_images", type=int, default=20)
    args = parser.parse_args()

    run_prediction(args.weights, args.source, args.output, args.conf, args.max_images)
