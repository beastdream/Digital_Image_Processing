import os
import sys
import yaml
import json
import torch
from ultralytics import YOLO

def train_yolo(config_path: str = "configs/yolo_training.yaml", exp_name_override: str = None) -> str:
    base_dir = r"d:\Digital_Image_Processing"
    full_cfg_path = os.path.join(base_dir, config_path) if not os.path.isabs(config_path) else config_path

    with open(full_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if exp_name_override:
        cfg["name"] = exp_name_override

    if not torch.cuda.is_available():
        # Maximize CPU multithreading on multi-core systems
        torch.set_num_threads(os.cpu_count() or 4)

    # Hardware & Device check
    requested_device = cfg.get("device", "auto")
    if requested_device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = requested_device

    print(f"Executing YOLO Training on Device: {device} (CUDA available: {torch.cuda.is_available()})")

    # Output experiment directory
    project_dir = os.path.join(base_dir, cfg.get("project", "experiments/yolo"))
    exp_name = cfg.get("name", "baseline")
    exp_dir = os.path.join(project_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Save Reproducibility Metadata
    reproducibility_meta = {
        "training_config": cfg,
        "seed": cfg.get("seed", 42),
        "device_used": str(device),
        "pytorch_version": torch.__version__,
        "python_version": sys.version
    }
    meta_path = os.path.join(exp_dir, "reproducibility_info.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(reproducibility_meta, f, indent=2)

    # Initialize YOLO Model
    model_weights = cfg.get("model_weights", "yolov8n.pt")
    model = YOLO(model_weights)

    # Start Training
    data_yaml = cfg.get("data_yaml", "d:/Digital_Image_Processing/data/processed/detection/dataset.yaml")

    results = model.train(
        data=data_yaml,
        epochs=cfg.get("epochs", 15),
        imgsz=cfg.get("imgsz", 640),
        batch=cfg.get("batch", 16),
        fraction=cfg.get("fraction", 1.0),
        optimizer=cfg.get("optimizer", "Auto"),
        lr0=cfg.get("learning_rate", 0.01),
        patience=cfg.get("patience", 5),
        device=device,
        workers=cfg.get("workers", 0),
        cache=cfg.get("cache", False),
        seed=cfg.get("seed", 42),
        project=project_dir,
        name=exp_name,
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True
    )

    best_weights = os.path.join(exp_dir, "weights", "best.pt")
    print(f"YOLO Training Finished! Best weights saved at: {best_weights}")
    return best_weights

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/yolo_training.yaml")
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    train_yolo(args.config, args.name)
