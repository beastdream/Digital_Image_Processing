import os
import sys
import yaml
import json
import torch
from ultralytics import YOLO
from pathlib import Path
from src.data.dataset_utils import CLASS_NAMES, project_root
from src.data.training_integrity import run_dataset_integrity_check
from src.utils.reproducibility import set_global_seed
from src.detection.model_contract import write_experiment_config


def validate_fraction_mode(cfg: dict) -> tuple[float, bool]:
    fraction = float(cfg.get("fraction", 1.0))
    debug_only = bool(cfg.get("debug_only", False))
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    if fraction < 1.0 and not debug_only:
        raise ValueError("fraction < 1 is permitted only when debug_only: true (DEBUG ONLY)")
    return fraction, debug_only

def train_yolo(config_path: str = "configs/yolo_training.yaml", exp_name_override: str = None) -> str:
    base_dir = project_root()
    full_cfg_path = base_dir / config_path if not os.path.isabs(config_path) else Path(config_path)

    with open(full_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    fraction, debug_only = validate_fraction_mode(cfg)
    if debug_only:
        print(f"DEBUG ONLY: training on fraction={fraction}; do not use this run for final conclusions.")

    if exp_name_override:
        cfg["name"] = exp_name_override
    set_global_seed(cfg.get("seed", 42))

    data_yaml = Path(cfg.get("data_yaml", base_dir / "data/processed/road_damage_detection/dataset.yaml"))
    if not data_yaml.is_absolute(): data_yaml = base_dir / data_yaml
    # Mandatory last gate before any model is constructed or training output is written.
    integrity_report = run_dataset_integrity_check(data_yaml, CLASS_NAMES)

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
    project_dir = base_dir / cfg.get("project", "experiments/yolo")
    exp_name = cfg.get("name", "baseline")
    exp_dir = project_dir / exp_name
    os.makedirs(exp_dir, exist_ok=True)

    # Save Reproducibility Metadata
    reproducibility_meta = {
        "training_config": cfg,
        "run_mode": "DEBUG ONLY" if debug_only else "FINAL COMPARISON",
        "trained_on_full_dataset": fraction == 1.0,
        "seed": cfg.get("seed", 42),
        "device_used": str(device),
        "pytorch_version": torch.__version__,
        "python_version": sys.version
    }
    reproducibility_meta["dataset_integrity"] = integrity_report
    meta_path = exp_dir / "reproducibility_info.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(reproducibility_meta, f, indent=2)
    write_experiment_config(exp_dir, cfg, data_yaml, root=base_dir)

    # Initialize YOLO Model
    model_weights = cfg.get("model_weights", "yolov8n.pt")
    model = YOLO(model_weights)

    # Start Training
    results = model.train(
        data=str(data_yaml),
        epochs=cfg.get("epochs", 15),
        imgsz=cfg.get("imgsz", 640),
        batch=cfg.get("batch", 16),
        fraction=fraction,
        optimizer=cfg.get("optimizer", "Auto"),
        lr0=cfg.get("learning_rate", 0.01),
        patience=cfg.get("patience", 5),
        device=device,
        workers=cfg.get("workers", 0),
        cache=cfg.get("cache", False),
        seed=cfg.get("seed", 42),
        project=str(project_dir),
        name=exp_name,
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True,
        degrees=cfg.get("degrees", 0.0), translate=cfg.get("translate", 0.0), scale=cfg.get("scale", 0.0),
        fliplr=cfg.get("fliplr", 0.0), hsv_h=cfg.get("hsv_h", 0.0), hsv_s=cfg.get("hsv_s", 0.0), hsv_v=cfg.get("hsv_v", 0.0),
        mosaic=cfg.get("mosaic", 0.0), mixup=cfg.get("mixup", 0.0), copy_paste=cfg.get("copy_paste", 0.0),
    )

    args_path = exp_dir / "args.yaml"
    if args_path.exists():
        runtime_args = yaml.safe_load(args_path.read_text(encoding="utf-8")) or {}
    else:
        result_args = getattr(results, "args", None)
        runtime_args = result_args if isinstance(result_args, dict) else vars(result_args) if result_args else {
            "model": model_weights, "imgsz": cfg.get("imgsz", 640), "epochs": cfg.get("epochs", 15),
            "batch": cfg.get("batch", 16), "seed": cfg.get("seed", 42), "fraction": fraction,
        }
    write_experiment_config(exp_dir, cfg, data_yaml, runtime_args=runtime_args, root=base_dir)

    best_weights = str(exp_dir / "weights" / "best.pt")
    print(f"YOLO Training Finished! Best weights saved at: {best_weights}")
    return best_weights

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/yolo_training.yaml")
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    train_yolo(args.config, args.name)
