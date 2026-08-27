import os
import json
import torch
from ultralytics import YOLO
from src.data.dataset_utils import project_root
from src.utils.reproducibility import set_global_seed

def run_overfit_check():
    base_dir = str(project_root())
    set_global_seed(42)
    data_yaml = os.path.join(base_dir, "data", "processed", "road_damage_detection", "dataset.yaml")
    exp_dir = os.path.join(base_dir, "experiments", "yolo", "debug_overfit")
    os.makedirs(exp_dir, exist_ok=True)

    if not torch.cuda.is_available():
        torch.set_num_threads(os.cpu_count() or 4)

    print("=== Running Overfit Sanity Check on Small Train Subset (15 Images) ===")
    model = YOLO("yolov8n.pt")

    # Train for 20 epochs on 10 images with batch=8
    results = model.train(
        data=data_yaml,
        epochs=20,
        imgsz=640,
        batch=8,
        fraction=0.007,  # DEBUG ONLY: ~10 images; never use for final conclusions
        optimizer="Auto",
        lr0=0.01,
        patience=0,
        val=False,
        device="cpu",
        workers=0,
        cache=False,
        seed=42,
        project=os.path.join(base_dir, "experiments", "yolo"),
        name="debug_overfit",
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True
    )

    # Evaluate on the small training subset to verify overfitting capacity
    metrics = model.val(
        data=data_yaml,
        split="train",
        imgsz=640,
        fraction=0.011,  # DEBUG ONLY: diagnostic overfit check
        device="cpu",
        workers=0,
        verbose=False
    )

    p_val = float(metrics.results_dict.get("metrics/precision(B)", 0.0))
    r_val = float(metrics.results_dict.get("metrics/recall(B)", 0.0))
    map50_val = float(metrics.results_dict.get("metrics/mAP50(B)", 0.0))
    map50_95_val = float(metrics.results_dict.get("metrics/mAP50-95(B)", 0.0))

    overfit_summary = {
        "epochs": 50,
        "subset_images": 15,
        "precision": round(p_val, 4),
        "recall": round(r_val, 4),
        "mAP50": round(map50_val, 4),
        "mAP50_95": round(map50_95_val, 4),
        "can_overfit": map50_val > 0.5
    }

    out_json = os.path.join(exp_dir, "overfit_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(overfit_summary, f, indent=2)

    print("\n--- OVERFIT SANITY CHECK RESULT ---")
    print(json.dumps(overfit_summary, indent=2))
    return overfit_summary

if __name__ == "__main__":
    run_overfit_check()
