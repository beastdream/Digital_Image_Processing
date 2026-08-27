import argparse
import json
import cv2
from ultralytics import YOLO
from pathlib import Path
from src.data.dataset_utils import CLASS_NAMES, project_root
from src.detection.checkpoint_validator import validate_training_checkpoints
from src.detection.ground_truth import (
    GROUND_TRUTH_COLOR,
    PREDICTION_COLOR,
    compare_detections,
    draw_labeled_boxes,
    draw_legend,
    find_yolo_label,
    load_yolo_ground_truth,
    render_side_by_side,
    scale_annotations,
)
from src.detection.model_contract import load_experiment_config, preprocess_for_inference
from src.utils.result_paths import (
    ground_truth_root,
    latest_ground_truth_dir,
    latest_predictions_dir,
    predictions_root,
    reset_directory,
)

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
        checkpoint_state = validate_training_checkpoints(experiment)
        weights = checkpoint_state["best"]
        metadata = experiment / "experiment_config.yaml"
        if not checkpoint_state["best_exists"]:
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


def ground_truth_for_image(image_path: Path) -> tuple[Path | None, list[dict]]:
    label_path = find_yolo_label(image_path)
    if label_path is None or not label_path.is_file():
        return label_path, []
    return label_path, load_yolo_ground_truth(image_path, label_path)


def run_ground_truth_only(source_dir: str, max_images: int | None = None) -> int:
    """Visualize dataset annotations without loading or running a model."""
    base_dir = project_root()
    image_paths = collect_image_paths(source_dir, max_images, base_dir)
    output_dir = reset_directory(latest_ground_truth_dir(base_dir), ground_truth_root(base_dir))
    written = 0
    for image_path in image_paths:
        label_path, ground_truth = ground_truth_for_image(image_path)
        if label_path is None or not label_path.is_file():
            raise FileNotFoundError(f"Ground truth label not found for image: {image_path}")
        image = read_valid_image(image_path)
        draw_labeled_boxes(image, ground_truth, "GT", GROUND_TRUTH_COLOR)
        draw_legend(image, include_ground_truth=True, include_predictions=False)
        output_path = output_dir / f"gt_{image_path.name}"
        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"Could not save ground-truth output: {output_path}")
        written += 1
    print(f"Saved {written} ground-truth visualizations to:\n{output_dir}")
    return written


def run_prediction(
    weights_path: str,
    source_dir: str,
    output_dir: str | None = None,
    conf_threshold: float = 0.25,
    max_images: int | None = None,
    experiment_config_path: str | None = None,
    clean_output: bool = False,
    experiment_name: str | None = None,
    show_ground_truth: bool = False,
    side_by_side: bool = False,
) -> int:
    base_dir = project_root()
    if not 0.0 <= conf_threshold <= 1.0:
        raise ValueError("Confidence threshold must be between 0 and 1")
    image_paths = collect_image_paths(source_dir, max_images, base_dir)
    if output_dir is None:
        pending_output_dir = latest_predictions_dir(base_dir).resolve()
        print("Prediction output policy: latest-run replacement")
        print(f"Clearing:\n{pending_output_dir}")
        full_output_dir = reset_directory(latest_predictions_dir(base_dir), predictions_root(base_dir))
    else:
        full_output_dir = Path(output_dir)
        if not full_output_dir.is_absolute():
            full_output_dir = base_dir / full_output_dir
        if clean_output:
            print("Prediction output policy: custom-output replacement")
            print(f"Clearing:\n{full_output_dir.resolve()}")
            full_output_dir = reset_directory(full_output_dir, predictions_root(base_dir))
        else:
            print("Prediction output policy: custom-output preservation")
            full_output_dir.mkdir(parents=True, exist_ok=True)

    resolved_weights = Path(weights_path)
    if not resolved_weights.is_absolute():
        resolved_weights = base_dir / resolved_weights
    resolved_weights = resolved_weights.resolve()
    print(f"Using model:\n{resolved_weights}")
    print(f"Processing:\n{len(image_paths)} image(s)")
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
        original_height, original_width = img.shape[:2]

        model_input, _ = preprocess_for_inference(img, experiment_config)
        results = model.predict(source=model_input, conf=conf_threshold, verbose=False)[0]
        detections = []

        # Keep predictions independent from ground truth under every mode.
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int)

            x1, y1, x2, y2 = xyxy
            if cls_id not in class_names:
                raise ValueError(f"Model emitted unsupported class_id {cls_id}; expected IDs 0, 1, 2")
            cname = class_names[cls_id]
            detections.append({
                "class_id": cls_id,
                "class_name": cname,
                "confidence": round(conf, 6),
                "xyxy": [int(x1), int(y1), int(x2), int(y2)],
            })

        ground_truth = []
        ground_truth_found = False
        if show_ground_truth:
            label_path, ground_truth = ground_truth_for_image(img_path)
            if label_path is None or not label_path.is_file():
                print(f"Warning: Ground truth label not found for image: {img_path}")
            else:
                ground_truth_found = True
                target_height, target_width = model_input.shape[:2]
                ground_truth = scale_annotations(
                    ground_truth, (original_width, original_height), (target_width, target_height)
                )
            if side_by_side:
                model_input = render_side_by_side(model_input, ground_truth, detections)
            else:
                draw_labeled_boxes(model_input, ground_truth, "GT", GROUND_TRUTH_COLOR)
                draw_labeled_boxes(model_input, detections, "Pred", PREDICTION_COLOR)
                draw_legend(model_input)
                if not detections:
                    cv2.putText(model_input, "Predictions: none", (10, model_input.shape[0] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, PREDICTION_COLOR, 2, cv2.LINE_AA)
        else:
            # Preserve the existing prediction-only appearance and labels.
            for detection in detections:
                class_id = detection["class_id"]
                x1, y1, x2, y2 = detection["xyxy"]
                color = colors.get(class_id, (255, 255, 255))
                thickness = 3 if class_id == 0 else 2
                cv2.rectangle(model_input, (x1, y1), (x2, y2), color, thickness)
                label_text = f"{detection['class_name']} {detection['confidence']:.2f}"
                (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(model_input, (x1, max(0, y1 - text_h - 4)),
                              (x1 + text_w + 4, y1), color, -1)
                cv2.putText(model_input, label_text, (x1 + 2, max(text_h, y1 - 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        out_path = full_output_dir / f"{'compare' if show_ground_truth else 'pred'}_{fname}"
        if not cv2.imwrite(str(out_path), model_input):
            raise RuntimeError(f"Could not save prediction output: {out_path}")
        image_summary = {"filename": fname, "detections": detections}
        if show_ground_truth:
            image_summary["image"] = str(img_path)
            image_summary["ground_truth"] = ground_truth
            image_summary["predictions"] = detections
            if ground_truth_found:
                image_summary["comparison"] = compare_detections(ground_truth, detections)
        image_summaries.append(image_summary)

    summary = {
        "model": str(resolved_weights),
        "experiment": experiment_name,
        "confidence_threshold": conf_threshold,
        "processed_images": len(image_summaries),
        "classes": {str(class_id): name for class_id, name in CLASS_NAMES.items()},
        "mode": "prediction_with_ground_truth" if show_ground_truth else "prediction_only",
        "images": image_summaries,
    }
    if show_ground_truth and len(image_summaries) == 1:
        # Convenient single-image schema requested by the CLI contract, while
        # retaining ``images`` for folder mode and backwards compatibility.
        only_image = image_summaries[0]
        summary.update({
            "image": only_image["image"],
            "ground_truth": only_image["ground_truth"],
            "predictions": only_image["predictions"],
        })
        if "comparison" in only_image:
            summary["comparison"] = only_image["comparison"]
    (full_output_dir / "prediction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"Saved {len(image_summaries)} prediction visualizations to:\n{full_output_dir}")
    return len(image_summaries)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Road-damage prediction for static images only")
    model_group = parser.add_mutually_exclusive_group(required=False)
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
    parser.add_argument("--side-by-side", action="store_true",
                        help="With --show-ground-truth, render separate GT and prediction panels")
    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument("--show-ground-truth", action="store_true",
                               help="Draw real dataset annotations alongside model predictions")
    display_group.add_argument("--ground-truth-only", action="store_true",
                               help="Draw only real dataset annotations without loading YOLO")
    args = parser.parse_args(argv)

    source = args.image or args.source
    try:
        # Validate image-only input before loading a potentially expensive model.
        collect_image_paths(source, args.max_images)
        if args.side_by_side and not args.show_ground_truth:
            parser.error("--side-by-side requires --show-ground-truth")
        if args.ground_truth_only:
            if args.model or args.experiment or args.experiment_config:
                parser.error("--ground-truth-only does not use --model, --experiment, or --experiment-config")
            if args.output or args.clean_output:
                parser.error("--ground-truth-only writes to results/ground_truth/latest")
            return run_ground_truth_only(source, args.max_images)
        if not args.model and not args.experiment:
            parser.error("Specify exactly one of --model or --experiment unless --ground-truth-only is used")
        weights, experiment_config = resolve_prediction_assets(args.model, args.experiment)
        if args.experiment_config and experiment_config:
            parser.error("--experiment-config is unnecessary with --experiment")
        metadata = (args.experiment_config or str(experiment_config)) if experiment_config else args.experiment_config
        experiment_name = Path(args.experiment).name if args.experiment else None
        return run_prediction(str(weights), source, args.output, args.conf, args.max_images,
                              metadata, args.clean_output, experiment_name, args.show_ground_truth,
                              args.side_by_side)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
