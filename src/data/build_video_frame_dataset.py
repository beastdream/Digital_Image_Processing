"""Build a YOLO pothole dataset from paired RGB/mask road videos.

Expected source layout (the public "pothole video" segmentation dataset)::

    <source>/<split>/rgb/<clip>.mp4
    <source>/<split>/mask/<clip>.mp4

Each sampled frame becomes one image; every connected white region of the
matching mask frame becomes one ``pothole`` (class 0) box. The source split is
preserved per clip, so frames of one video never leak across splits. Frames
without a pothole are kept with an empty label file as background examples.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from src.data.dataset_utils import (CLASS_NAMES, SPLITS, dataset_yaml_text, format_yolo_box, parse_yolo_line,
                                     project_root, validate_yolo_box)
from src.data.validate_processed_dataset import validate_processed_dataset

POTHOLE_CLASS_ID = 0
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".m4v"}
DEFAULT_OUTPUT = Path("data/processed/pothole_video_frames")


def mask_to_boxes(mask_frame: np.ndarray, min_area: int = 400, threshold: int = 127,
                  class_id: int = POTHOLE_CLASS_ID) -> list[dict]:
    """Convert one (possibly compressed) mask frame into pixel xyxy boxes."""
    gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY) if mask_frame.ndim == 3 else mask_frame
    binary = (gray > threshold).astype(np.uint8)
    # Video compression leaves speckles around mask edges; opening removes them.
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = []
    for left, top, width, height, area in stats[1:count]:
        if area < min_area:
            continue
        boxes.append({"class_id": class_id, "class_name": CLASS_NAMES[class_id],
                      "xyxy": [int(left), int(top), int(left + width), int(top + height)]})
    return boxes


def boxes_to_yolo_lines(boxes: list[dict], image_width: int, image_height: int) -> list[str]:
    def floor8(value: float) -> float:
        return math.floor(value * 1e8) / 1e8

    lines = []
    for box in boxes:
        x1, y1, x2, y2 = box["xyxy"]
        xc, yc = round((x1 + x2) / 2 / image_width, 8), round((y1 + y2) / 2 / image_height, 8)
        # Boxes touching the border would overflow [0, 1] after 8-decimal rounding;
        # flooring the size against the rounded centre keeps every corner inside.
        width = floor8(min((x2 - x1) / image_width, 2 * min(xc, 1 - xc)))
        height = floor8(min((y2 - y1) / image_height, 2 * min(yc, 1 - yc)))
        line = format_yolo_box((box["class_id"], xc, yc, width, height))
        if validate_yolo_box(parse_yolo_line(line))[1] != "valid":
            raise ValueError(f"Could not encode box {box['xyxy']} as a valid YOLO label")
        lines.append(line)
    return lines


def resize_longest_side(image: np.ndarray, size: int | None) -> np.ndarray:
    if not size:
        return image
    height, width = image.shape[:2]
    scale = size / max(height, width)
    if scale >= 1:
        return image
    return cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)


def find_video_pairs(source: Path, split: str) -> list[tuple[Path, Path]]:
    rgb_dir, mask_dir = source / split / "rgb", source / split / "mask"
    if not rgb_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected {rgb_dir} and {mask_dir}")
    pairs = []
    for video in sorted(path for path in rgb_dir.iterdir() if path.suffix.lower() in VIDEO_SUFFIXES):
        mask = mask_dir / video.name
        if not mask.is_file():
            raise FileNotFoundError(f"Mask video missing for {video}: {mask}")
        pairs.append((video, mask))
    return pairs


def extract_clip(video_path: Path, mask_path: Path, split: str, image_dir: Path, label_dir: Path,
                 frame_step: int, frame_size: int | None, min_area: int) -> dict:
    """Write sampled frames and labels for one clip; returns per-clip statistics."""
    video, mask = cv2.VideoCapture(str(video_path)), cv2.VideoCapture(str(mask_path))
    if not video.isOpened() or not mask.isOpened():
        raise ValueError(f"Cannot open video pair: {video_path}, {mask_path}")
    frames = objects = background = 0
    index = 0
    try:
        while True:
            ok_video, frame = video.read()
            ok_mask, mask_frame = mask.read()
            if not ok_video or not ok_mask:
                break
            if index % frame_step == 0:
                if mask_frame.shape[:2] != frame.shape[:2]:
                    mask_frame = cv2.resize(mask_frame, (frame.shape[1], frame.shape[0]),
                                            interpolation=cv2.INTER_NEAREST)
                boxes = mask_to_boxes(mask_frame, min_area=min_area)
                # Normalized labels are computed on the source size, so resizing is label-safe.
                lines = boxes_to_yolo_lines(boxes, frame.shape[1], frame.shape[0])
                stem = f"{split}_{video_path.stem}_f{index:04d}"
                if not cv2.imwrite(str(image_dir / f"{stem}.jpg"), resize_longest_side(frame, frame_size),
                                   [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f"Could not write frame {stem}")
                (label_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                frames += 1
                objects += len(lines)
                background += not lines
            index += 1
    finally:
        video.release()
        mask.release()
    return {"video": video_path.name, "source_frames": index, "sampled_frames": frames,
            "objects": objects, "background_frames": background}


def build_video_frame_dataset(source: str | Path, output: str | Path = DEFAULT_OUTPUT, frame_step: int = 12,
                              frame_size: int | None = 640, min_area: int = 400,
                              max_videos: int | None = None) -> dict:
    root = project_root()
    source, output = Path(source), Path(output)
    if not source.is_absolute():
        source = root / source
    if not output.is_absolute():
        output = root / output
    if frame_step < 1:
        raise ValueError("frame_step must be >= 1")
    if not source.is_dir():
        raise FileNotFoundError(f"Video dataset not found: {source}")

    # Build into a temporary sibling so stale frames from an older build cannot survive.
    staging = output.with_name(output.name + "_building")
    if staging.exists():
        shutil.rmtree(staging)
    stats: dict = {"splits": {}}
    for split in SPLITS:
        image_dir, label_dir = staging / "images" / split, staging / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        pairs = find_video_pairs(source, split)[:max_videos]
        clips = [extract_clip(video, mask, split, image_dir, label_dir, frame_step, frame_size, min_area)
                 for video, mask in pairs]
        counts = Counter()
        for clip in clips:
            counts.update({key: clip[key] for key in ("sampled_frames", "objects", "background_frames")})
        stats["splits"][split] = {"videos": len(clips), "images": counts["sampled_frames"],
                                  "objects": counts["objects"], "background_images": counts["background_frames"],
                                  "clips": clips}
        print(f"{split}: {len(clips)} videos -> {counts['sampled_frames']} frames, {counts['objects']} potholes")

    (staging / "dataset.yaml").write_text(dataset_yaml_text(output), encoding="utf-8")
    manifest = {"source": str(source), "frame_step": frame_step, "frame_size": frame_size,
                "min_area": min_area, "max_videos": max_videos, "classes": CLASS_NAMES,
                "label_origin": "connected components of segmentation mask videos (class 0 = pothole)",
                "split_policy": "source split preserved per clip; no clip spans two splits", **stats}
    (staging / "video_frames_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if output.exists():
        shutil.rmtree(output)
    staging.rename(output)

    report, valid = validate_processed_dataset(output)
    if not valid:
        raise RuntimeError(f"Generated dataset failed validation: {report['errors'][:5]}")
    print(f"Dataset written to {output} (validation PASSED)")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract YOLO pothole frames from RGB/mask video pairs")
    parser.add_argument("--source", default="data/raw/pothole_video",
                        help="Folder containing train|val|test/rgb and train|val|test/mask videos")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--frame-step", type=int, default=12, help="Keep every N-th frame of each clip")
    parser.add_argument("--frame-size", type=int, default=640, help="Longest side of saved frames (0 keeps source)")
    parser.add_argument("--min-area", type=int, default=400, help="Minimum mask region area in source pixels")
    parser.add_argument("--max-videos", type=int, default=None, help="Limit clips per split (debugging)")
    args = parser.parse_args(argv)
    build_video_frame_dataset(args.source, args.output, args.frame_step, args.frame_size or None,
                              args.min_area, args.max_videos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
