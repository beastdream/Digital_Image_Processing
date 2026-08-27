"""Versioned, clean materialization of uint8 preprocessing experiment datasets."""
from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path

import cv2
import yaml

from src.data.dataset_utils import IMAGE_SUFFIXES, SPLITS, dataset_yaml_text, project_root
from src.preprocessing.pipeline import ImagePreprocessingPipeline
from src.utils.reproducibility import set_global_seed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_split_fingerprint(source_dir: Path) -> str:
    """Fingerprint split membership, labels, and source image bytes."""
    digest = hashlib.sha256()
    for path in [source_dir / "dataset.yaml", *sorted(path for folder in (source_dir / "images", source_dir / "labels") for path in folder.rglob("*") if path.is_file())]:
        digest.update(path.relative_to(source_dir).as_posix().encode())
        digest.update(_sha256(path).encode())
    return digest.hexdigest()


def _fingerprint(config: dict, source_fingerprint: str, seed: int) -> str:
    payload = {"preprocessing": config, "source_split_fingerprint": source_fingerprint, "seed": seed}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:10]


def _read_boxes(label_path: Path) -> list[list[float]]:
    return [[int(parts[0]), *map(float, parts[1:])] for raw in label_path.read_text(encoding="utf-8").splitlines() if (parts := raw.split())]


def materialize_preprocessed_dataset(config_path: str | Path, experiment_name: str | None = None, seed: int = 42, root_dir: str | Path | None = None) -> Path:
    root = Path(root_dir) if root_dir else project_root(); config_path = Path(config_path)
    set_global_seed(seed)
    if not config_path.is_absolute(): config_path = root / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    name = experiment_name or config.get("experiment_name", config_path.stem)
    source_dir = root / "data/processed/road_damage_detection"
    source_yaml = source_dir / "dataset.yaml"
    if not source_yaml.exists():
        raise FileNotFoundError(f"Build the base processed dataset first: {source_yaml}")
    source_fingerprint = _source_split_fingerprint(source_dir)
    version = _fingerprint(config, source_fingerprint, seed)
    output_parent = source_dir / "preprocessed"
    output_dir = output_parent / f"{name}_{version}"
    temporary_dir = output_parent / f".{name}_{version}.{uuid.uuid4().hex}.tmp"
    output_parent.mkdir(parents=True, exist_ok=True)
    temporary_dir.mkdir()
    try:
        pipeline = ImagePreprocessingPipeline(config)
        # Random online augmentation is a training concern, never a frozen file.
        pipeline.config["augmentation"] = {"enabled": False}
        for split in SPLITS:
            source_images, source_labels = source_dir / "images" / split, source_dir / "labels" / split
            destination_images, destination_labels = temporary_dir / "images" / split, temporary_dir / "labels" / split
            destination_images.mkdir(parents=True); destination_labels.mkdir(parents=True)
            for image_path in sorted(path for path in source_images.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
                image = cv2.imread(str(image_path))
                if image is None:
                    raise RuntimeError(f"Unreadable base processed image: {image_path}")
                label_path = source_labels / f"{image_path.stem}.txt"
                boxes = _read_boxes(label_path)
                _, updated_boxes, metadata = pipeline.process(image, boxes, split=split, return_intermediates=True)
                boxes_to_save = boxes if updated_boxes is None else updated_boxes
                output_image = metadata["intermediates"]["final_uint8"]
                if not cv2.imwrite(str(destination_images / image_path.name), output_image):
                    raise RuntimeError(f"Could not write {image_path.name}")
                (destination_labels / f"{image_path.stem}.txt").write_text("\n".join(f"{int(box[0])} {box[1]:.8f} {box[2]:.8f} {box[3]:.8f} {box[4]:.8f}" for box in boxes_to_save), encoding="utf-8")
        (temporary_dir / "dataset.yaml").write_text(dataset_yaml_text(temporary_dir), encoding="utf-8")
        (temporary_dir / "build_manifest.json").write_text(json.dumps({"experiment_name": name, "version": version, "config_path": str(config_path), "config": config, "seed": seed, "source_split_fingerprint": source_fingerprint}, indent=2), encoding="utf-8")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.move(str(temporary_dir), str(output_dir))
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise
    # The path embedded in YAML must point to its final, not temporary location.
    (output_dir / "dataset.yaml").write_text(dataset_yaml_text(output_dir), encoding="utf-8")
    return output_dir / "dataset.yaml"
