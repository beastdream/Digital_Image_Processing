"""Shared invariants and safe parsers for the 3-class YOLO dataset."""
from __future__ import annotations
import math
import re
from pathlib import Path
from typing import Iterable

CLASS_NAMES = {0: "pothole", 1: "crack", 2: "manhole"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val", "test")
EPSILON = 1e-5

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def dataset_yaml_text(dataset_dir: Path) -> str:
    return (f"path: {dataset_dir.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n\n"
            "nc: 3\nnames:\n  0: pothole\n  1: crack\n  2: manhole\n")

def parse_yolo_line(line: str) -> tuple[int, float, float, float, float]:
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"expected 5 fields, got {len(parts)}")
    try:
        class_id, values = int(parts[0]), tuple(float(value) for value in parts[1:])
    except ValueError as exc:
        raise ValueError("class id and coordinates must be numeric") from exc
    if class_id not in CLASS_NAMES:
        raise ValueError(f"unsupported class_id {class_id}; expected 0, 1, or 2")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("coordinates must be finite")
    return (class_id, *values)

def validate_yolo_box(
    box: tuple[int, float, float, float, float], epsilon: float = EPSILON
) -> tuple[tuple[int, float, float, float, float], str]:
    """Validate normalized YOLO geometry and correct only epsilon-sized overflow.

    Returns the normalized box and either ``valid`` or ``clipped``. Larger
    violations are rejected instead of being silently clipped.
    """
    class_id, xc, yc, width, height = box
    if not (0 < width <= 1 + epsilon and 0 < height <= 1 + epsilon):
        raise ValueError("width and height must be in (0, 1]")
    if not (-epsilon <= xc <= 1 + epsilon and -epsilon <= yc <= 1 + epsilon):
        raise ValueError("x_center and y_center must be in [0, 1]")
    xmin, ymin = xc - width / 2, yc - height / 2
    xmax, ymax = xc + width / 2, yc + height / 2
    if xmin < -epsilon or ymin < -epsilon or xmax > 1 + epsilon or ymax > 1 + epsilon:
        raise ValueError("bbox corners exceed image bounds beyond tolerance")
    corrected = (class_id, max(0.0, min(1.0, xc)), max(0.0, min(1.0, yc)), min(1.0, width), min(1.0, height))
    # Reconstruct from clipped corners so a small boundary overflow is corrected
    # consistently even when only one edge crosses the border.
    if xmin < 0 or ymin < 0 or xmax > 1 or ymax > 1:
        left, top, right, bottom = max(0.0, xmin), max(0.0, ymin), min(1.0, xmax), min(1.0, ymax)
        corrected = (class_id, (left + right) / 2, (top + bottom) / 2, right - left, bottom - top)
        return corrected, "clipped"
    return corrected, "valid"

def clip_yolo_box(box: tuple[int, float, float, float, float]) -> tuple[int, float, float, float, float] | None:
    class_id, xc, yc, width, height = box
    left, top = max(0.0, xc - width / 2), max(0.0, yc - height / 2)
    right, bottom = min(1.0, xc + width / 2), min(1.0, yc + height / 2)
    if right <= left or bottom <= top:
        return None
    return class_id, (left + right) / 2, (top + bottom) / 2, right - left, bottom - top

def format_yolo_box(box: tuple[int, float, float, float, float]) -> str:
    class_id, xc, yc, width, height = box
    # Eight decimals prevent a tolerance-corrected edge box from becoming
    # out-of-bounds again through six-decimal rounding.
    return f"{class_id} {xc:.8f} {yc:.8f} {width:.8f} {height:.8f}"

def group_key(filename: str) -> str:
    stem = Path(filename).stem
    # Hours are not independent recordings: all captures from a dated session
    # are grouped together unless metadata later provides a finer source ID.
    match = re.match(r"vlcsnap[-_](\d{4}-\d{2}-\d{2})[-_]\d{2}h", stem, re.I)
    if match:
        return f"session_{match.group(1)}"
    match = re.match(r"(\d{8})(?:[_-]|$)", stem)
    if match:
        date = match.group(1)
        return f"session_{date[:4]}-{date[4:6]}-{date[6:]}"
    return "vlcsnap_unsequenced" if re.match(r"vlcsnap-\d+", stem, re.I) else f"file_{stem}"

def label_signature(lines: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(line.strip() for line in lines if line.strip()))
