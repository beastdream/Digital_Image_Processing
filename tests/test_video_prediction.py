import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

from src.data.build_video_frame_dataset import boxes_to_yolo_lines, build_video_frame_dataset, mask_to_boxes
from src.data.dataset_utils import CLASS_NAMES, parse_yolo_line
from src.detection import video_predict
from src.detection.predict import main
from src.detection.video_tracking import IoUTracker, summarize_tracks

SIZE = 96


def square_frame(frame_index: int, value: int = 255) -> np.ndarray:
    frame = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    left = 10 + 2 * frame_index
    frame[30:60, left:left + 30] = value
    return frame


def write_video(path: Path, frames: int = 8) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (SIZE, SIZE))
    assert writer.isOpened()
    for index in range(frames):
        writer.write(square_frame(index))
    writer.release()
    return path


def detection(class_id: int, xyxy: list[int], confidence: float = 0.9) -> dict:
    return {"class_id": class_id, "class_name": CLASS_NAMES[class_id], "confidence": confidence, "xyxy": xyxy}


def test_mask_to_boxes_finds_each_region_and_drops_specks() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:40, 10:30] = 255
    mask[60:90, 50:95] = 255
    mask[0:2, 98:100] = 255  # compression speck
    boxes = mask_to_boxes(mask, min_area=50)
    assert [box["xyxy"] for box in boxes] == [[10, 10, 30, 40], [50, 60, 95, 90]]
    assert all(box["class_name"] == "pothole" for box in boxes)
    class_id, xc, yc, width, height = parse_yolo_line(boxes_to_yolo_lines(boxes[:1], 100, 100)[0])
    assert (class_id, xc, yc, width, height) == (0, 0.2, 0.25, 0.2, 0.3)


def test_tracker_keeps_identity_and_votes_class_over_frames() -> None:
    tracker = IoUTracker(iou_threshold=0.3, max_missed=2)
    classes = [0, 0, 2, 0]  # one flickering manhole frame
    for frame, class_id in enumerate(classes):
        tracked = tracker.update([detection(class_id, [10 + frame, 10, 50 + frame, 50])], frame)
        assert tracked[0]["track_id"] == 1
    assert tracked[0]["track_class_name"] == "pothole"
    tracker.update([detection(1, [200, 200, 240, 240])], 4)  # unrelated, short-lived crack
    summary = summarize_tracks(tracker, min_frames=3, fps=10)
    assert summary["unique_objects_by_class"] == {"pothole": 1, "crack": 0, "manhole": 0}
    assert summary["unconfirmed_track_count"] == 1
    assert summary["confirmed_tracks"][0]["class_votes"]["manhole"] == pytest.approx(0.25)


def test_tracker_retires_lost_tracks_and_starts_new_ids() -> None:
    tracker = IoUTracker(iou_threshold=0.3, max_missed=1)
    tracker.update([detection(0, [0, 0, 20, 20])], 0)
    tracker.update([], 1)
    tracker.update([], 2)
    tracked = tracker.update([detection(0, [0, 0, 20, 20])], 3)
    assert tracked[0]["track_id"] == 2
    assert len(tracker.all_tracks()) == 2


def test_source_routing_and_mask_lookup(tmp_path: Path) -> None:
    video = write_video(tmp_path / "clips/rgb/0001.mp4")
    mask = write_video(tmp_path / "clips/mask/0001.mp4")
    assert video_predict.find_mask_video(video) == mask
    assert video_predict.find_mask_video(mask) is None
    assert video_predict.is_video_source(video)
    assert video_predict.is_video_source(video.parent)
    assert video_predict.is_video_source("0")
    images = tmp_path / "images"
    images.mkdir()
    cv2.imwrite(str(images / "a.jpg"), square_frame(0))
    assert not video_predict.is_video_source(images)


def test_build_video_frame_dataset_preserves_clip_splits(tmp_path: Path) -> None:
    source = tmp_path / "pothole_video"
    for split in ("train", "val", "test"):
        write_video(source / split / "rgb" / f"{split}clip.mp4")
        write_video(source / split / "mask" / f"{split}clip.mp4")
    output = tmp_path / "frames"
    manifest = build_video_frame_dataset(source, output, frame_step=4, frame_size=None, min_area=100)
    assert {split: manifest["splits"][split]["images"] for split in ("train", "val", "test")} == \
        {"train": 2, "val": 2, "test": 2}
    assert sorted(path.name for path in (output / "images/val").iterdir()) == \
        ["val_valclip_f0000.jpg", "val_valclip_f0004.jpg"]
    label = (output / "labels/train/train_trainclip_f0000.txt").read_text(encoding="utf-8").split()
    assert label[0] == "0"
    assert (output / "dataset.yaml").is_file()


class FakeBoxes(list):
    pass


class FakeBox:
    def __init__(self, class_id: int, confidence: float, xyxy: list[int]):
        self.cls = torch.tensor([class_id])
        self.conf = torch.tensor([confidence])
        self.xyxy = torch.tensor([xyxy], dtype=torch.float32)


class FakeYOLO:
    names = CLASS_NAMES
    calls = 0

    def __init__(self, weights: str):
        self.weights = weights

    def predict(self, source, conf, verbose):
        frame = FakeYOLO.calls
        FakeYOLO.calls += 1
        left = 10 + 2 * frame
        class_id = 2 if frame == 1 else 0
        result = type("Result", (), {})()
        result.boxes = FakeBoxes([FakeBox(class_id, 0.8, [left, 30, left + 30, 60])])
        return [result]


def write_model(tmp_path: Path) -> Path:
    weights = tmp_path / "exp/weights/best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"fake")
    preprocessing = yaml.safe_load(Path("configs/experiments/baseline.yaml").read_text(encoding="utf-8"))
    (weights.parent / "experiment_config.yaml").write_text(
        yaml.safe_dump({"schema_version": 1, "classes": CLASS_NAMES, "preprocessing": preprocessing,
                        "training": {"imgsz": 640}}), encoding="utf-8")
    return weights


def test_video_prediction_end_to_end_with_ground_truth(tmp_path: Path, monkeypatch) -> None:
    FakeYOLO.calls = 0
    monkeypatch.setattr(video_predict, "YOLO", FakeYOLO)
    video = write_video(tmp_path / "clips/rgb/0007.mp4", frames=6)
    write_video(tmp_path / "clips/mask/0007.mp4", frames=6)
    output = tmp_path / "out"
    summary = video_predict.run_video_prediction(
        str(write_model(tmp_path)), str(video), str(output), show_ground_truth=True, save_frame_detections=True)

    assert summary["unique_objects_by_class"] == {"pothole": 1, "crack": 0, "manhole": 0}
    clip = summary["videos"][0]
    assert clip["frames_read"] == clip["frames_processed"] == 6
    assert {frame["detections"][0]["track_id"] for frame in clip["frames"]} == {1}
    voted = clip["frame_level_comparison"]["track_voted"]["pothole"]
    raw = clip["frame_level_comparison"]["raw_per_frame"]["pothole"]
    assert voted["tp"] == 6 and voted["fn"] == 0
    assert raw["tp"] == 5  # the flickering manhole frame is fixed only by temporal voting
    capture = cv2.VideoCapture(str(output / "pred_0007.mp4"))
    assert capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 6
    capture.release()
    assert json.loads((output / "video_summary.json").read_text(encoding="utf-8"))["processed_videos"] == 1


def test_frame_step_reuses_last_detections(tmp_path: Path, monkeypatch) -> None:
    FakeYOLO.calls = 0
    monkeypatch.setattr(video_predict, "YOLO", FakeYOLO)
    video = write_video(tmp_path / "v.mp4", frames=6)
    summary = video_predict.run_video_prediction(str(write_model(tmp_path)), str(video), str(tmp_path / "o"),
                                                 frame_step=3)
    assert summary["videos"][0]["frames_processed"] == 2
    assert FakeYOLO.calls == 2


def test_cli_routes_video_and_rejects_unreadable_file_before_model_load(tmp_path: Path, capsys) -> None:
    video = tmp_path / "road.mp4"
    video.write_bytes(b"not a video")
    weights = write_model(tmp_path)
    with pytest.raises(SystemExit):
        main(["--model", str(weights), "--video", str(video), "--output", str(tmp_path / "o")])
    assert "Cannot open video source" in capsys.readouterr().err


def test_webcam_requires_an_end_condition(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Webcam input needs"):
        video_predict.run_video_prediction(str(write_model(tmp_path)), "0")


def test_border_touching_mask_boxes_encode_as_valid_labels() -> None:
    from src.data.dataset_utils import validate_yolo_box
    for xyxy in ([367, 0, 1080, 480], [0, 100, 1080, 1080], [1079, 0, 1080, 1]):
        line = boxes_to_yolo_lines([{"class_id": 0, "xyxy": xyxy}], 1080, 1080)[0]
        assert validate_yolo_box(parse_yolo_line(line))[1] == "valid"


def test_frames_are_downscaled_to_training_frame_size(tmp_path: Path, monkeypatch) -> None:
    FakeYOLO.calls = 0
    monkeypatch.setattr(video_predict, "YOLO", FakeYOLO)
    video = write_video(tmp_path / "v.mp4", frames=2)
    summary = video_predict.run_video_prediction(str(write_model(tmp_path)), str(video), str(tmp_path / "o"),
                                                 frame_size=48)
    capture = cv2.VideoCapture(summary["videos"][0]["output_video"])
    assert (capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == (48, 48)
    capture.release()
<<<<<<< HEAD


def test_track_filter_drops_only_short_lived_false_positives() -> None:
    tracker = IoUTracker(iou_threshold=0.3, max_missed=1)
    gt = [{"class_id": 0, "class_name": "pothole", "xyxy": [10, 10, 50, 50]}]
    frames = []
    for frame in range(4):
        found = [detection(0, [10, 10, 50, 50])]
        if frame == 1:
            found.append(detection(0, [200, 200, 230, 230]))  # one-frame flicker
        frames.append((gt, tracker.update(found, frame)))
    counts = video_predict._track_filtered_counts(frames, tracker, min_track_frames=3)
    assert counts["pothole"] == {"tp": 4, "fp": 0, "fn": 0}


def test_video_suite_row_pools_counts_across_videos() -> None:
    from src.detection.evaluate_video_suite import summarize_experiment

    def clip(raw, filtered):
        comparison = lambda tp, fp, fn: {"pothole": {"tp": tp, "fp": fp, "fn": fn}}
        return {"frames_processed": 10, "confirmed_tracks": [{}], "unconfirmed_track_count": 2,
                "mean_inference_ms": 20.0, "processing_fps": 9.0,
                "frame_level_comparison": {"raw_per_frame": comparison(*raw), "track_filtered": comparison(*filtered)}}

    row = summarize_experiment("A_raw", {"videos": [clip((8, 2, 2), (8, 0, 2)), clip((9, 1, 1), (9, 1, 1))]})
    assert (row["raw_tp"], row["raw_fp"], row["raw_fn"]) == (17, 3, 3)
    assert row["raw_precision"] == 0.85 and row["tracked_precision"] == round(17 / 18, 4)
    assert row["frames"] == 20 and row["unconfirmed_tracks"] == 4
=======
>>>>>>> da7e1cd4 (Cap nhat code)
