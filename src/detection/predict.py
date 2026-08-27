import argparse
import json
import cv2
from ultralytics import YOLO
from pathlib import Path
from src.data.dataset_utils import CLASS_NAMES, project_root
from src.detection.model_contract import load_experiment_config, preprocess_for_inference
from src.utils.result_paths import latest_predictions_dir, predictions_root, reset_directory

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".m4v"}
VIDEO_ERROR = "Video input is not supported in the current image-only project."


def resolve_prediction_assets(model_path: str | None = None, experiment_dir: str | None = None,
                              root: Path | None = None) -> tuple[Path, Path | None]:
    """Resolve explicit CLI inputs; preprocessing is never inferred from a model name."""
    root = root or project_root()
    if bool(model_path) == bool(experiment_dir):
        raise ValueError("Specify exactly one of --model or --experiment")
    if experiment_dir:
        experiment = Path(experiment_dir)
        if not experiment.is_absolute():
            experiment = root / experiment
        weights = experiment / "weights/best.pt"
        metadata = experiment / "experiment_config.yaml"
        if not weights.is_file():
            raise FileNotFoundError(f"Model weights not found: {weights}")
        if not metadata.is_file():
            raise FileNotFoundError(f"Experiment config not found: {metadata}")
        return weights, metadata
    weights = Path(model_path)
    if not weights.is_absolute():
        weights = root / weights
    if not weights.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights}")
    return weights, None


def collect_image_paths(source: str | Path, max_images: int | None = None,
                        root: Path | None = None) -> list[Path]:
    root = root or project_root()
    path = Path(source)
    if not path.is_absolute():
        path = root / path
    if path.suffix.lower() in VIDEO_SUFFIXES:
        raise ValueError(VIDEO_ERROR)
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image type '{suffix}'. Supported types: .jpg, .jpeg, .png")
        return [path]
    if not path.is_dir():
        raise ValueError(f"Invalid image source: {path}")
    images = sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"No supported images found in folder: {path}")
    return images[:max_images] if max_images is not None else images


def read_valid_image(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Invalid or unreadable image: {path}")
    return image


def run_prediction(
    weights_path: str,
    source_dir: str,
    output_dir: str | None = None,
    conf_threshold: float = 0.25,
    max_images: int | None = None,
    experiment_config_path: str | None = None,
    clean_output: bool = False,
    experiment_name: str | None = None,
) -> int:
    base_dir = project_root()
    if not 0.0 <= conf_threshold <= 1.0:
        raise ValueError("Confidence threshold must be between 0 and 1")
    image_paths = collect_image_paths(source_dir, max_images, base_dir)
    if output_dir is None:
        full_output_dir = reset_directory(latest_predictions_dir(base_dir), predictions_root(base_dir))
    else:
        full_output_dir = Path(output_dir)
        if not full_output_dir.is_absolute():
            full_output_dir = base_dir / full_output_dir
        if clean_output:
            full_output_dir = reset_directory(full_output_dir, predictions_root(base_dir))
        else:
            full_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading YOLO Model [{weights_path}] for prediction on [{source_dir}]...")
    model = YOLO(weights_path)
    experiment_config = load_experiment_config(weights_path, experiment_config_path)
    model_names = {int(key): value for key, value in model.names.items()}
    if model_names != CLASS_NAMES:
        raise ValueError(f"Weights must be a 3-class road-damage model with {CLASS_NAMES}; got {model_names}")

    class_names = CLASS_NAMES
    colors = {
        0: (0, 0, 255),    # Bright Red for POTHOLE
        1: (255, 0, 0),    # Blue for Crack
        2: (0, 255, 0)     # Green for Manhole
    }

    image_summaries = []

    for img_path in image_paths:
        fname = img_path.name
        img = read_valid_image(img_path)

        model_input, _ = preprocess_for_inference(img, experiment_config)
        results = model.predict(source=model_input, conf=conf_threshold, verbose=False)[0]
        detections = []

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
            detections.append({
                "class_id": cls_id,
                "class_name": cname,
                "confidence": round(conf, 6),
                "xyxy": [int(x1), int(y1), int(x2), int(y2)],
            })

            # Highlight POTHOLE distinctly with thicker border & clear label
            thickness = 3 if cls_id == 0 else 2
            cv2.rectangle(model_input, (x1, y1), (x2, y2), color, thickness)

            label_text = f"{cname} {conf:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(model_input, (x1, max(0, y1 - text_h - 4)), (x1 + text_w + 4, y1), color, -1)
            cv2.putText(model_input, label_text, (x1 + 2, max(text_h, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        out_path = full_output_dir / f"pred_{fname}"
        if not cv2.imwrite(str(out_path), model_input):
            raise RuntimeError(f"Could not save prediction output: {out_path}")
        image_summaries.append({"filename": fname, "detections": detections})

    summary = {
        "model": str(Path(weights_path).resolve()),
        "experiment": experiment_name,
        "confidence_threshold": conf_threshold,
        "processed_images": len(image_summaries),
        "classes": {str(class_id): name for class_id, name in CLASS_NAMES.items()},
        "images": image_summaries,
    }
    (full_output_dir / "prediction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"Saved {len(image_summaries)} prediction visualizations to: {full_output_dir}")
    return len(image_summaries)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Road-damage prediction for static images only")
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", type=str, help="Path to best.pt (loads co-located experiment_config.yaml)")
    model_group.add_argument("--experiment", type=str, help="Experiment directory containing weights/best.pt and experiment_config.yaml")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--image", type=str, help="Single .jpg/.jpeg/.png image")
    source_group.add_argument("--source", type=str, help="Single image or folder of images")
    parser.add_argument("--output", type=str, default=None,
                        help="Custom output directory (default: reset results/predictions/latest)")
    parser.add_argument("--clean-output", action="store_true",
                        help="Reset a custom --output below results/predictions before writing")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--experiment-config", type=str, default=None)
    args = parser.parse_args(argv)

    source = args.image or args.source
    try:
        # Validate image-only input before loading a potentially expensive model.
        collect_image_paths(source, args.max_images)
        weights, experiment_config = resolve_prediction_assets(args.model, args.experiment)
        if args.experiment_config and experiment_config:
            parser.error("--experiment-config is unnecessary with --experiment")
        metadata = (args.experiment_config or str(experiment_config)) if experiment_config else args.experiment_config
        experiment_name = Path(args.experiment).name if args.experiment else None
        return run_prediction(str(weights), source, args.output, args.conf, args.max_images,
                              metadata, args.clean_output, experiment_name)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
