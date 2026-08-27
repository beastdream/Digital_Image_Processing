import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data.dataset_utils import CLASS_NAMES
from src.detection.ground_truth import (
    compare_detections,
    find_yolo_label,
    load_yolo_ground_truth,
    render_side_by_side,
)
from src.detection.predict import main, run_prediction


def make_dataset_image(root: Path, label_text: str | None = "2 0.5 0.5 0.5 0.4\n") -> Path:
    image_path = root / "data/processed/road_damage_detection/images/test/sample.jpg"
    image_path.parent.mkdir(parents=True)
    assert cv2.imwrite(str(image_path), np.zeros((100, 200, 3), dtype=np.uint8))
    if label_text is not None:
        label_path = root / "data/processed/road_damage_detection/labels/test/sample.txt"
        label_path.parent.mkdir(parents=True)
        label_path.write_text(label_text, encoding="utf-8")
    return image_path


def test_label_lookup_and_normalized_to_pixel_conversion(tmp_path: Path) -> None:
    image_path = make_dataset_image(tmp_path)
    label_path = find_yolo_label(image_path)
    assert label_path == tmp_path / "data/processed/road_damage_detection/labels/test/sample.txt"
    assert load_yolo_ground_truth(image_path, label_path) == [{
        "class_id": 2,
        "class_name": "manhole",
        "xyxy": [50, 30, 150, 70],
    }]


def test_three_class_mapping_is_canonical() -> None:
    assert CLASS_NAMES == {0: "pothole", 1: "crack", 2: "manhole"}


def test_same_class_iou_matching_reports_tp_fp_fn() -> None:
    ground_truth = [
        {"class_id": 0, "class_name": "pothole", "xyxy": [0, 0, 20, 20]},
        {"class_id": 0, "class_name": "pothole", "xyxy": [30, 30, 40, 40]},
    ]
    predictions = [
        {"class_id": 0, "class_name": "pothole", "confidence": 0.9, "xyxy": [0, 0, 20, 20]},
        {"class_id": 1, "class_name": "crack", "confidence": 0.8, "xyxy": [0, 0, 20, 20]},
    ]
    assert compare_detections(ground_truth, predictions) == {
        "pothole": {"tp": 1, "fp": 0, "fn": 1},
        "crack": {"tp": 0, "fp": 1, "fn": 0},
        "manhole": {"tp": 0, "fp": 0, "fn": 0},
    }


def test_side_by_side_has_two_panels_and_header() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    result = render_side_by_side(image, [], [])
    assert result.shape == (138, 400, 3)


@pytest.mark.parametrize("line", [
    "3 0.5 0.5 0.2 0.2", "0 0.5 0.5 0 0.2", "0 0.95 0.5 0.2 0.2",
])
def test_invalid_ground_truth_is_rejected(tmp_path: Path, line: str) -> None:
    image_path = make_dataset_image(tmp_path, line)
    with pytest.raises(ValueError, match="Invalid YOLO annotation"):
        load_yolo_ground_truth(image_path, find_yolo_label(image_path))


def test_ground_truth_only_does_not_load_model_and_replaces_latest(tmp_path: Path, monkeypatch) -> None:
    image_path = make_dataset_image(tmp_path)
    stale = tmp_path / "results/ground_truth/latest/stale.jpg"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")
    monkeypatch.setattr("src.detection.predict.project_root", lambda: tmp_path)
    monkeypatch.setattr("src.utils.result_paths.project_root", lambda: tmp_path)
    assert main(["--image", str(image_path), "--ground-truth-only"]) == 1
    assert not stale.exists()
    output = tmp_path / "results/ground_truth/latest/gt_sample.jpg"
    assert output.is_file()
    assert cv2.imread(str(output)).shape == (100, 200, 3)


def test_ground_truth_only_fails_clearly_when_label_missing(tmp_path: Path, monkeypatch, capsys) -> None:
    image_path = make_dataset_image(tmp_path, None)
    monkeypatch.setattr("src.detection.predict.project_root", lambda: tmp_path)
    monkeypatch.setattr("src.utils.result_paths.project_root", lambda: tmp_path)
    with pytest.raises(SystemExit):
        main(["--image", str(image_path), "--ground-truth-only"])
    assert "Ground truth label not found for image" in capsys.readouterr().err


class FakeArray:
    def __init__(self, values):
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeBox:
    def __init__(self, class_id: int, confidence: float, xyxy: list[int]):
        self.cls = [class_id]
        self.conf = [confidence]
        self.xyxy = [FakeArray(xyxy)]


class FakeModel:
    boxes: list[FakeBox] = []

    def __init__(self, _weights):
        self.names = CLASS_NAMES

    def predict(self, **_kwargs):
        return [type("Result", (), {"boxes": self.boxes})()]


def configure_fake_prediction(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("src.detection.predict.project_root", lambda: tmp_path)
    monkeypatch.setattr("src.utils.result_paths.project_root", lambda: tmp_path)
    monkeypatch.setattr("src.detection.predict.YOLO", FakeModel)
    monkeypatch.setattr("src.detection.predict.load_experiment_config", lambda *_args: {})
    monkeypatch.setattr("src.detection.predict.preprocess_for_inference", lambda image, _config: (image, {}))


def test_prediction_with_gt_writes_independent_summary_and_comparison(tmp_path: Path, monkeypatch) -> None:
    image_path = make_dataset_image(tmp_path)
    configure_fake_prediction(monkeypatch, tmp_path)
    FakeModel.boxes = [FakeBox(2, 0.82, [50, 30, 150, 70])]
    assert run_prediction("fake.pt", str(image_path), show_ground_truth=True) == 1
    summary = json.loads((tmp_path / "results/predictions/latest/prediction_summary.json").read_text())
    assert summary["mode"] == "prediction_with_ground_truth"
    assert summary["ground_truth"][0]["class_name"] == "manhole"
    assert summary["predictions"][0]["confidence"] == pytest.approx(0.82)
    assert summary["comparison"]["manhole"] == {"tp": 1, "fp": 0, "fn": 0}
    assert (tmp_path / "results/predictions/latest/compare_sample.jpg").is_file()


def test_no_prediction_preserves_gt_and_side_by_side_output(tmp_path: Path, monkeypatch) -> None:
    image_path = make_dataset_image(tmp_path)
    configure_fake_prediction(monkeypatch, tmp_path)
    FakeModel.boxes = []
    run_prediction("fake.pt", str(image_path), show_ground_truth=True, side_by_side=True)
    summary = json.loads((tmp_path / "results/predictions/latest/prediction_summary.json").read_text())
    assert summary["ground_truth"]
    assert summary["predictions"] == []
    rendered = cv2.imread(str(tmp_path / "results/predictions/latest/compare_sample.jpg"))
    assert rendered.shape == (138, 400, 3)


def test_missing_gt_warns_but_prediction_continues(tmp_path: Path, monkeypatch, capsys) -> None:
    image_path = make_dataset_image(tmp_path, None)
    configure_fake_prediction(monkeypatch, tmp_path)
    FakeModel.boxes = [FakeBox(1, 0.7, [10, 10, 30, 30])]
    assert run_prediction("fake.pt", str(image_path), show_ground_truth=True) == 1
    assert "Warning: Ground truth label not found for image" in capsys.readouterr().out
    summary = json.loads((tmp_path / "results/predictions/latest/prediction_summary.json").read_text())
    assert summary["ground_truth"] == []
    assert len(summary["predictions"]) == 1
    assert "comparison" not in summary
