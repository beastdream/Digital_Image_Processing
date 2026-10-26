"""Road-damage detection on video files, folders of videos, or a webcam.

Every processed frame goes through the model's stored preprocessing contract,
exactly like static-image prediction. Detections are linked across frames by
``IoUTracker`` so each physical object gets one ID and one class decided by
temporal voting, which is what distinguishes video analysis from per-image
prediction. When a video lives under ``.../rgb/`` and a same-named mask video
exists under ``.../mask/``, ground truth is found automatically.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.data.build_video_frame_dataset import mask_to_boxes, resize_longest_side
from src.data.dataset_utils import CLASS_NAMES, project_root
from src.detection.ground_truth import GROUND_TRUTH_COLOR, compare_detections, draw_labeled_boxes
from src.detection.model_contract import load_experiment_config, preprocess_for_inference
from src.detection.video_tracking import IoUTracker, summarize_tracks
from src.utils.result_paths import latest_video_predictions_dir, predictions_root, reset_directory

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".m4v"}
CLASS_COLORS = {0: (0, 0, 255), 1: (255, 0, 0), 2: (0, 255, 0)}  # BGR: pothole red, crack blue, manhole green
DEFAULT_WEBCAM_FPS = 30.0
# build_video_frame_dataset saves training frames at this longest side; inference
# frames are downscaled the same way so filters act at the scale seen in training.
DEFAULT_FRAME_SIZE = 640


def is_webcam_source(source: str | int) -> bool:
    return isinstance(source, int) or str(source).isdigit()


def is_video_source(source: str | Path, root: Path | None = None) -> bool:
    """True for a webcam index, a video file, or a folder holding videos but no images."""
    if is_webcam_source(str(source)):
        return True
    path = Path(source)
    if not path.is_absolute():
        path = (root or project_root()) / path
    if path.suffix.lower() in VIDEO_SUFFIXES:
        return True
    if path.is_dir():
        files = [item.suffix.lower() for item in path.iterdir() if item.is_file()]
        return any(suffix in VIDEO_SUFFIXES for suffix in files) and not any(
            suffix in {".jpg", ".jpeg", ".png"} for suffix in files)
    return False


def collect_video_paths(source: str | Path, max_videos: int | None = None, root: Path | None = None) -> list[Path]:
    path = Path(source)
    if not path.is_absolute():
        path = (root or project_root()) / path
    if path.is_file():
        if path.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f"Unsupported video type '{path.suffix}'. Supported: {', '.join(sorted(VIDEO_SUFFIXES))}")
        return [path]
    if not path.is_dir():
        raise ValueError(f"Invalid video source: {path}")
    videos = sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in VIDEO_SUFFIXES)
    if not videos:
        raise ValueError(f"No supported videos found in folder: {path}")
    return videos[:max_videos] if max_videos is not None else videos


def find_mask_video(video_path: Path) -> Path | None:
    """Infer ``.../mask/<clip>`` from ``.../rgb/<clip>``, mirroring images/ -> labels/."""
    parts = list(Path(video_path).parts)
    indexes = [index for index, part in enumerate(parts) if part.lower() == "rgb"]
    if not indexes:
        return None
    parts[indexes[-1]] = "mask"
    candidate = Path(*parts)
    return candidate if candidate.is_file() else None


def open_capture(source: Path | int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(source if isinstance(source, int) else str(source))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video source: {source}")
    return capture


def detections_from_result(result) -> list[dict]:
    detections = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        if class_id not in CLASS_NAMES:
            raise ValueError(f"Model emitted unsupported class_id {class_id}; expected IDs 0, 1, 2")
        x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].cpu().numpy())
        detections.append({"class_id": class_id, "class_name": CLASS_NAMES[class_id],
                           "confidence": round(float(box.conf[0]), 6), "xyxy": [x1, y1, x2, y2]})
    return detections


def draw_tracked_detections(frame, detections: list[dict]) -> None:
    for detection in detections:
        class_id = detection.get("track_class_id", detection["class_id"])
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        x1, y1, x2, y2 = detection["xyxy"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if class_id == 0 else 2)
        label = f"#{detection.get('track_id', '?')} {CLASS_NAMES[class_id]} {detection['confidence']:.2f}"
        (width, height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        top = max(0, y1 - height - 6)
        cv2.rectangle(frame, (x1, top), (x1 + width + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, max(height, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 1, cv2.LINE_AA)


def draw_hud(frame, frame_index: int, fps: float, unique_counts: dict[str, int]) -> None:
    lines = [f"frame {frame_index}  t={frame_index / fps:.2f}s"] + [
        f"{name}: {count}" for name, count in unique_counts.items()]
    y = frame.shape[0] - 12 - 22 * (len(lines) - 1)
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22


def _add_counts(total: dict, comparison: dict) -> None:
    for class_name, values in comparison.items():
        for key, value in values.items():
            total.setdefault(class_name, Counter())[key] += value


def _precision_recall(counts: dict) -> dict:
    result = {}
    for class_name, values in counts.items():
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        result[class_name] = {"tp": tp, "fp": fp, "fn": fn,
                              "precision": round(tp / (tp + fp), 4) if tp + fp else None,
                              "recall": round(tp / (tp + fn), 4) if tp + fn else None}
    return result


<<<<<<< HEAD
def _track_filtered_counts(scored_frames: list[tuple[list[dict], list[dict]]], tracker: IoUTracker,
                           min_track_frames: int) -> dict:
    """Offline re-score keeping only detections whose whole track was confirmed.

    Needs the finished video (a track's final length is known only at the end),
    so it measures how much short-lived flicker detections cost, not a live filter.
    """
    confirmed = {track.track_id for track in tracker.all_tracks() if track.hits >= min_track_frames}
    counts: dict = {}
    for ground_truth, detections in scored_frames:
        kept = [item for item in detections if item["track_id"] in confirmed]
        _add_counts(counts, compare_detections(ground_truth, kept))
    return counts


=======
>>>>>>> da7e1cd4 (Cap nhat code)
def process_video(model, source: Path | int, experiment_config: dict, output_path: Path | None, *,
                  conf: float = 0.25, frame_step: int = 1, max_frames: int | None = None,
                  mask_path: Path | None = None, min_track_frames: int = 3, iou_threshold: float = 0.3,
                  max_missed: int = 5, display: bool = False, save_frame_detections: bool = False,
                  frame_size: int | None = DEFAULT_FRAME_SIZE) -> dict:
    """Run detection + tracking on one stream and write the annotated video."""
    capture = open_capture(source)
    mask_capture = open_capture(mask_path) if mask_path else None
    fps = capture.get(cv2.CAP_PROP_FPS) or DEFAULT_WEBCAM_FPS
    tracker = IoUTracker(iou_threshold=iou_threshold, max_missed=max_missed)
    writer = None
    frame_index = processed = 0
    last_detections: list[dict] = []
    frame_records, raw_counts, voted_counts = [], {}, {}
<<<<<<< HEAD
    scored_frames: list[tuple[list[dict], list[dict]]] = []
=======
>>>>>>> da7e1cd4 (Cap nhat code)
    started = time.perf_counter()
    inference_seconds = 0.0
    try:
        while max_frames is None or frame_index < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            mask_frame = None
            if mask_capture is not None:
                ok_mask, mask_frame = mask_capture.read()
                mask_frame = mask_frame if ok_mask else None
            model_input, _ = preprocess_for_inference(resize_longest_side(frame, frame_size), experiment_config)
            if frame_index % frame_step == 0:
                tick = time.perf_counter()
                result = model.predict(source=model_input, conf=conf, verbose=False)[0]
                inference_seconds += time.perf_counter() - tick
                last_detections = tracker.update(detections_from_result(result), frame_index)
                processed += 1
                record = {"frame": frame_index, "time_s": round(frame_index / fps, 3), "detections": last_detections}
                if mask_frame is not None:
                    height, width = model_input.shape[:2]
                    if mask_frame.shape[:2] != (height, width):
                        mask_frame = cv2.resize(mask_frame, (width, height), interpolation=cv2.INTER_NEAREST)
                    ground_truth = mask_to_boxes(mask_frame)
                    voted = [{**item, "class_id": item["track_class_id"]} for item in last_detections]
                    _add_counts(raw_counts, compare_detections(ground_truth, last_detections))
                    _add_counts(voted_counts, compare_detections(ground_truth, voted))
<<<<<<< HEAD
                    scored_frames.append((ground_truth, voted))
=======
>>>>>>> da7e1cd4 (Cap nhat code)
                    record["ground_truth"] = ground_truth
                if save_frame_detections:
                    frame_records.append(record)
                if mask_frame is not None:
                    # GT is drawn only on processed frames so it matches the reported metrics.
                    draw_labeled_boxes(model_input, record["ground_truth"], "GT", GROUND_TRUTH_COLOR)
            draw_tracked_detections(model_input, last_detections)
            draw_hud(model_input, frame_index, fps,
                     summarize_tracks(tracker, min_track_frames)["unique_objects_by_class"])
            if output_path is not None:
                if writer is None:
                    height, width = model_input.shape[:2]
                    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError(f"Cannot create output video: {output_path}")
                writer.write(model_input)
            if display:
                cv2.imshow("Road damage detection (q to quit)", model_input)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_index += 1
    finally:
        capture.release()
        if mask_capture is not None:
            mask_capture.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()
    if frame_index == 0:
        raise ValueError(f"No readable frames in video source: {source}")

    elapsed = time.perf_counter() - started
    summary = {
        "source": str(source), "output_video": str(output_path) if output_path else None,
        "fps": round(fps, 3), "frames_read": frame_index, "frames_processed": processed,
        "frame_step": frame_step, "frame_size": frame_size, "duration_s": round(frame_index / fps, 3),
        "processing_fps": round(frame_index / elapsed, 2) if elapsed else None,
        "mean_inference_ms": round(1000 * inference_seconds / processed, 2) if processed else None,
        **summarize_tracks(tracker, min_track_frames, fps),
    }
    if mask_path:
        summary["ground_truth_video"] = str(mask_path)
        summary["frame_level_comparison"] = {
            "iou_threshold": 0.5, "raw_per_frame": _precision_recall(raw_counts),
<<<<<<< HEAD
            "track_voted": _precision_recall(voted_counts),
            "track_filtered": _precision_recall(_track_filtered_counts(scored_frames, tracker, min_track_frames))}
=======
            "track_voted": _precision_recall(voted_counts)}
>>>>>>> da7e1cd4 (Cap nhat code)
    if save_frame_detections:
        summary["frames"] = frame_records
    return summary


def prepare_video_output(output_dir: str | None, clean_output: bool, root: Path) -> Path:
    if output_dir is None:
        return reset_directory(latest_video_predictions_dir(root), predictions_root(root))
    path = Path(output_dir)
    if not path.is_absolute():
        path = root / path
    if clean_output:
        return reset_directory(path, predictions_root(root))
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_video_prediction(weights_path: str, source: str, output_dir: str | None = None, conf: float = 0.25,
                         experiment_config_path: str | None = None, clean_output: bool = False,
                         experiment_name: str | None = None, *, frame_step: int = 1,
                         max_frames: int | None = None, max_videos: int | None = None,
                         show_ground_truth: bool = False, min_track_frames: int = 3,
                         display: bool = False, save_frame_detections: bool = False,
<<<<<<< HEAD
                         frame_size: int | None = DEFAULT_FRAME_SIZE, save_video: bool = True) -> dict:
=======
                         frame_size: int | None = DEFAULT_FRAME_SIZE) -> dict:
>>>>>>> da7e1cd4 (Cap nhat code)
    root = project_root()
    if not 0.0 <= conf <= 1.0:
        raise ValueError("Confidence threshold must be between 0 and 1")
    if frame_step < 1:
        raise ValueError("--frame-step must be >= 1")
    webcam = is_webcam_source(source)
    if webcam and max_frames is None and not display:
        raise ValueError("Webcam input needs --max-frames or --display so the stream can end")
    sources: list[Path | int] = [int(source)] if webcam else collect_video_paths(source, max_videos, root)
    for item in sources:
        if not webcam:
            open_capture(item).release()  # Fail fast before loading the model.

    output = prepare_video_output(output_dir, clean_output, root)
    resolved_weights = Path(weights_path)
    if not resolved_weights.is_absolute():
        resolved_weights = root / resolved_weights
    print(f"Using model:\n{resolved_weights.resolve()}")
    model = YOLO(str(resolved_weights))
    experiment_config = load_experiment_config(resolved_weights, experiment_config_path)
    model_names = {int(key): value for key, value in model.names.items()}
    if model_names != CLASS_NAMES:
        raise ValueError(f"Weights must be a 3-class road-damage model with {CLASS_NAMES}; got {model_names}")

    videos, totals = [], Counter()
    for item in sources:
        name = f"webcam{item}" if webcam else item.stem
        mask = find_mask_video(item) if show_ground_truth and not webcam else None
        if show_ground_truth and not webcam and mask is None:
            print(f"Warning: no mask video found for {item}; ground-truth comparison skipped")
        print(f"Processing {item} ...")
        video_summary = process_video(
<<<<<<< HEAD
            model, item, experiment_config, output / f"pred_{name}.mp4" if save_video else None, conf=conf, frame_step=frame_step,
=======
            model, item, experiment_config, output / f"pred_{name}.mp4", conf=conf, frame_step=frame_step,
>>>>>>> da7e1cd4 (Cap nhat code)
            max_frames=max_frames, mask_path=mask, min_track_frames=min_track_frames, display=display,
            save_frame_detections=save_frame_detections, frame_size=frame_size)
        totals.update(video_summary["unique_objects_by_class"])
        counts = ", ".join(f"{key}={value}" for key, value in video_summary["unique_objects_by_class"].items())
        print(f"  {video_summary['frames_read']} frames, unique objects: {counts}")
        videos.append(video_summary)

    summary = {
        "model": str(resolved_weights.resolve()), "experiment": experiment_name, "confidence_threshold": conf,
        "classes": {str(key): value for key, value in CLASS_NAMES.items()},
        "processed_videos": len(videos), "unique_objects_by_class": {name: totals[name] for name in CLASS_NAMES.values()},
        "videos": videos,
    }
    comparisons = [video["frame_level_comparison"]["track_voted"] for video in videos if "frame_level_comparison" in video]
    if comparisons:
        pooled = {}
        for comparison in comparisons:
            _add_counts(pooled, {name: {key: values[key] for key in ("tp", "fp", "fn")}
                                 for name, values in comparison.items()})
        summary["frame_level_comparison_all_videos"] = _precision_recall(pooled)
    (output / "video_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {len(videos)} annotated video(s) and video_summary.json to:\n{output}")
    return summary
