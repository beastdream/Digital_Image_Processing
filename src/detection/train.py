"""Train YOLO models with explicit experiment lifecycle and checkpoint gates."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

from src.data.dataset_utils import CLASS_NAMES, project_root
from src.data.training_integrity import run_dataset_integrity_check
from src.detection.checkpoint_validator import validate_training_checkpoints
from src.detection.model_contract import write_experiment_config
from src.utils.reproducibility import set_global_seed


def validate_fraction_mode(cfg: dict) -> tuple[float, bool]:
    fraction = float(cfg.get("fraction", 1.0))
    debug_only = bool(cfg.get("debug_only", False))
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    if fraction < 1.0 and not debug_only:
        raise ValueError("fraction < 1 is permitted only when debug_only: true (DEBUG ONLY)")
    return fraction, debug_only


def _write_status(exp_dir: Path, **status: object) -> Path:
    """Persist a small, truthful lifecycle record even when training fails."""
    exp_dir.mkdir(parents=True, exist_ok=True)
    path = exp_dir / "training_status.json"
    path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return path


def _prepare_experiment_directory(exp_dir: Path, overwrite_incomplete: bool) -> Path | None:
    """Refuse completed runs and optionally archive an incomplete run."""
    if not exp_dir.exists():
        return None

    checkpoint_state = validate_training_checkpoints(exp_dir)
    if checkpoint_state["best_exists"]:
        raise FileExistsError(
            f"Refusing to overwrite experiment with an existing best.pt: {checkpoint_state['best']}"
        )
    if not overwrite_incomplete:
        raise FileExistsError(
            f"Incomplete experiment already exists: {exp_dir}\n"
            "No valid weights/best.pt was found. Re-run with --overwrite-incomplete "
            "to archive it and restart this experiment name."
        )

    archive_root = exp_dir.parent / "legacy_incomplete"
    archive_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archive_path = archive_root / f"{exp_dir.name}_{timestamp}"
    shutil.move(str(exp_dir), str(archive_path))
    print(f"Archived incomplete experiment:\n{exp_dir}\n-> {archive_path}")
    return archive_path


def _checkpoint_paths(model: object) -> tuple[Path, Path, Path]:
    trainer = getattr(model, "trainer", None)
    if trainer is None:
        raise RuntimeError("Ultralytics trainer was not created.")
    save_dir_value = getattr(trainer, "save_dir", None)
    if save_dir_value is None:
        raise RuntimeError("Ultralytics trainer has no save_dir.")

    actual_save_dir = Path(save_dir_value).resolve()
    checkpoint_state = validate_training_checkpoints(actual_save_dir)
    best_weights = Path(getattr(trainer, "best", None) or checkpoint_state["best"]).resolve()
    last_weights = Path(getattr(trainer, "last", None) or checkpoint_state["last"]).resolve()
    best_exists = best_weights.is_file()
    last_exists = last_weights.is_file()
    if not best_exists:
        raise RuntimeError(
            "Training ended without producing best.pt.\n"
            f"Actual save dir: {actual_save_dir}\n"
            f"Expected best.pt: {best_weights}\n"
            f"last.pt exists: {last_exists}"
        )
    if not last_exists:
        raise RuntimeError(
            "Training ended without producing last.pt.\n"
            f"Actual save dir: {actual_save_dir}\n"
            f"Expected last.pt: {last_weights}\n"
            f"best.pt exists: {best_exists}"
        )
    return actual_save_dir, best_weights, last_weights


def _runtime_args(results: object, actual_save_dir: Path, cfg: dict) -> dict:
    args_path = actual_save_dir / "args.yaml"
    if args_path.is_file():
        return yaml.safe_load(args_path.read_text(encoding="utf-8")) or {}
    result_args = getattr(results, "args", None)
    if isinstance(result_args, dict):
        return result_args
    if result_args is not None:
        return vars(result_args)
    return {
        "model": cfg.get("model_weights"), "imgsz": cfg["imgsz"],
        "epochs": cfg["epochs"], "batch": cfg["batch"],
        "seed": cfg.get("seed", 42), "fraction": cfg["fraction"],
    }


def train_yolo(
    config_path: str = "configs/yolo_training.yaml",
    exp_name_override: str | None = None,
    *,
    overwrite_incomplete: bool = False,
    smoke_test: bool = False,
) -> str:
    base_dir = project_root()
    full_cfg_path = base_dir / config_path if not os.path.isabs(config_path) else Path(config_path)
    cfg = yaml.safe_load(full_cfg_path.read_text(encoding="utf-8")) or {}

    if exp_name_override:
        cfg["name"] = exp_name_override
    if smoke_test:
        # In-memory overrides only: the source YAML remains the final-run config.
        cfg.update({"epochs": 1, "fraction": 0.05, "imgsz": 320,
                    "batch": min(int(cfg.get("batch", 16)), 4), "debug_only": True})
        print("SMOKE TEST: using temporary epochs=1, fraction=0.05, imgsz=320, and batch<=4 overrides.")

    cfg.setdefault("epochs", 15)
    cfg.setdefault("imgsz", 640)
    cfg.setdefault("batch", 16)
    fraction, debug_only = validate_fraction_mode(cfg)
    cfg["fraction"] = fraction
    if debug_only:
        print(f"DEBUG ONLY: training on fraction={fraction}; do not use this run for final conclusions.")
    set_global_seed(cfg.get("seed", 42))

    data_yaml = Path(cfg.get("data_yaml", base_dir / "data/processed/road_damage_detection/dataset.yaml"))
    if not data_yaml.is_absolute():
        data_yaml = base_dir / data_yaml
    # Mandatory last gate before model construction or experiment output.
    integrity_report = run_dataset_integrity_check(data_yaml, CLASS_NAMES)

    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        torch.set_num_threads(os.cpu_count() or 4)
    requested_device = cfg.get("device", "auto")
    device = 0 if requested_device == "auto" and cuda_available else "cpu" if requested_device == "auto" else requested_device
    cfg["device"] = device

    print(f"torch: {torch.__version__}")
    print(f"CUDA available: {cuda_available}")
    print(f"CUDA devices: {torch.cuda.device_count()}")
    print(f"Training device: {device}")
    if cuda_available and str(device).lower() == "cpu":
        print("WARNING: CUDA is available but training is configured for CPU.\n"
              f"Training imgsz={cfg['imgsz']}, batch={cfg['batch']}, fraction={fraction} may be very slow.")
    if str(device).lower() == "cpu" and int(cfg["imgsz"]) >= 640 and fraction == 1.0:
        print("WARNING: CPU training may take a long time.\n"
              "Do not close/interrupt the process before at least one epoch completes,\n"
              "otherwise no checkpoint may be available.")

    project_value = Path(cfg.get("project", "experiments/yolo"))
    project_dir = project_value if project_value.is_absolute() else base_dir / project_value
    exp_name = cfg.get("name", "baseline")
    exp_dir = project_dir / exp_name
    _prepare_experiment_directory(exp_dir, overwrite_incomplete)
    _write_status(exp_dir, status="RUNNING", checkpoint_created=False)

    print("=== YOLO TRAINING START ===")
    print(f"Experiment: {exp_name}")
    print(f"Device: {device}")
    print(f"Epochs: {cfg['epochs']}")
    print(f"Batch: {cfg['batch']}")
    print(f"Image size: {cfg['imgsz']}")
    print(f"Fraction: {fraction}")
    print(f"Dataset: {data_yaml.resolve()}")
    print(f"Save directory: {exp_dir.resolve()}")

    model_weights = cfg.get("model_weights", "yolov8n.pt")
    effective_training_args = {
        "data": str(data_yaml), "epochs": cfg["epochs"], "imgsz": cfg["imgsz"],
        "batch": cfg["batch"], "fraction": fraction,
        "optimizer": cfg.get("optimizer", "Auto"), "lr0": cfg.get("learning_rate", 0.01),
        "patience": cfg.get("patience", 5), "device": device,
        "workers": cfg.get("workers", 0), "cache": cfg.get("cache", False),
        "seed": cfg.get("seed", 42), "project": str(project_dir), "name": exp_name,
        "exist_ok": True, "save": True, "save_period": cfg.get("save_period", -1),
        "plots": cfg.get("plots", True), "verbose": cfg.get("verbose", True),
        "degrees": cfg.get("degrees", 0.0), "translate": cfg.get("translate", 0.0),
        "scale": cfg.get("scale", 0.0), "fliplr": cfg.get("fliplr", 0.0),
        "hsv_h": cfg.get("hsv_h", 0.0), "hsv_s": cfg.get("hsv_s", 0.0),
        "hsv_v": cfg.get("hsv_v", 0.0), "mosaic": cfg.get("mosaic", 0.0),
        "mixup": cfg.get("mixup", 0.0), "copy_paste": cfg.get("copy_paste", 0.0),
    }
    (exp_dir / "effective_training_args.json").write_text(
        json.dumps(effective_training_args, indent=2, default=str), encoding="utf-8"
    )
    print("Effective Ultralytics training arguments:")
    print(json.dumps(effective_training_args, indent=2, default=str))
    try:
        model = YOLO(model_weights)
        results = model.train(**effective_training_args)
        trainer = getattr(model, "trainer", None)
        returned_save_dir = getattr(trainer, "save_dir", "UNAVAILABLE")
        print("=== YOLO TRAINING RETURNED ===")
        print(f"Actual Ultralytics save directory: {returned_save_dir}")

        actual_save_dir, best_weights, last_weights = _checkpoint_paths(model)
        runtime_args = _runtime_args(results, actual_save_dir, cfg)
        write_experiment_config(actual_save_dir, cfg, data_yaml, runtime_args=runtime_args, root=base_dir)
        reproducibility_meta = {
            "status": "COMPLETED", "training_config": cfg,
            "run_mode": "DEBUG ONLY" if debug_only else "FINAL COMPARISON",
            "trained_on_full_dataset": fraction == 1.0, "seed": cfg.get("seed", 42),
            "device_used": str(device), "pytorch_version": torch.__version__,
            "python_version": sys.version, "dataset_integrity": integrity_report,
            "best_weights": str(best_weights), "last_weights": str(last_weights),
            "checkpoint_created": True,
        }
        (actual_save_dir / "reproducibility_info.json").write_text(
            json.dumps(reproducibility_meta, indent=2), encoding="utf-8"
        )
        _write_status(actual_save_dir, status="COMPLETED", best_weights=str(best_weights),
                      last_weights=str(last_weights), checkpoint_created=True)
    except KeyboardInterrupt:
        checkpoint_state = validate_training_checkpoints(exp_dir)
        _write_status(exp_dir, status="INTERRUPTED",
                      error="Training was interrupted before completion.",
                      checkpoint_created=checkpoint_state["best_exists"] and checkpoint_state["last_exists"])
        print("Training was interrupted before completion.\n"
              "A usable best.pt may not exist.", file=sys.stderr)
        raise
    except Exception as exc:
        checkpoint_state = validate_training_checkpoints(exp_dir)
        _write_status(exp_dir, status="FAILED", error=f"{type(exc).__name__}: {exc}",
                      checkpoint_created=checkpoint_state["best_exists"] and checkpoint_state["last_exists"])
        traceback.print_exc()
        raise

    print("best.pt: FOUND")
    print("last.pt: FOUND")
    print("=== TRAINING COMPLETED SUCCESSFULLY ===")
    return str(best_weights)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/yolo_training.yaml")
    parser.add_argument("--name", default=None)
    parser.add_argument("--overwrite-incomplete", action="store_true",
                        help="Archive an existing run without best.pt and restart the same name")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run one small debug epoch without modifying the YAML config")
    args = parser.parse_args(argv)
    train_yolo(args.config, args.name, overwrite_incomplete=args.overwrite_incomplete,
               smoke_test=args.smoke_test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
