"""Bundle code and the extracted video-frame dataset into one zip for Google Colab.

Only what the video preprocessing suite needs is packed: source code, configs,
tests, and ``data/processed/pothole_video_frames`` (images, ``.txt`` labels,
dataset.yaml, manifest). Materialized B-F datasets, Ultralytics caches, and
local training outputs are left out; Colab rebuilds or recreates them.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from src.data.dataset_utils import IMAGE_SUFFIXES, project_root

CODE_ITEMS = ("src", "configs", "tests", "colab", "predict.py", "pytest.ini", "README.md")
DATASET = Path("data/processed/pothole_video_frames")


def _code_files(root: Path) -> list[Path]:
    files = []
    for item in CODE_ITEMS:
        path = root / item
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(p for p in path.rglob("*") if p.is_file()
                         and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pt"})
    return files


def _dataset_files(root: Path) -> list[Path]:
    dataset = root / DATASET
    if not (dataset / "dataset.yaml").is_file():
        raise FileNotFoundError(f"Build the frame dataset first: {dataset / 'dataset.yaml'}")
    files = [dataset / "dataset.yaml", dataset / "video_frames_manifest.json"]
    files += sorted(p for p in (dataset / "images").rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    files += sorted((dataset / "labels").rglob("*.txt"))
    return [path for path in files if path.is_file()]


def package_for_colab(output: str | Path = "colab_package.zip") -> Path:
    root = project_root()
    output = Path(output)
    if not output.is_absolute():
        output = root / output
    files = _code_files(root) + _dataset_files(root)
    with zipfile.ZipFile(output, "w") as archive:
        for path in files:
            # JPEGs are already compressed; storing them keeps packing fast.
            compression = zipfile.ZIP_STORED if path.suffix.lower() in IMAGE_SUFFIXES else zipfile.ZIP_DEFLATED
            archive.write(path, path.relative_to(root).as_posix(), compress_type=compression)
    images = sum(1 for path in files if path.suffix.lower() in IMAGE_SUFFIXES)
    print(f"Packed {len(files)} files ({images} images) into {output} "
          f"({output.stat().st_size / 1024 ** 2:.0f} MB)")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create colab_package.zip for the video experiment suite")
    parser.add_argument("--output", default="colab_package.zip")
    package_for_colab(parser.parse_args(argv).output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
