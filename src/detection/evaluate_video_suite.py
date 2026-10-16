"""Evaluate every suite model on full test videos, with and without temporal tracking.

For each experiment the test clips are run frame by frame against their mask
videos (IoU 0.5). ``raw`` scores every per-frame detection as predicted;
``tracked`` keeps only detections whose track lasted at least
``min_track_frames`` processed frames (offline: track lengths are known after
the clip ends). With a pothole-only model, class voting cannot change
anything, so the temporal benefit measured here is removal of short-lived
false positives. The pooled counts are written to ``video_tracking_results.csv`` beside the image-level
suite results.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.data.dataset_utils import project_root
from src.detection.experiment_suite import DEFAULT_SUITE_CONFIG, load_experiment_suite, suite_experiments
from src.detection.video_predict import run_video_prediction

COLUMNS = ["experiment", "videos", "frames", "raw_tp", "raw_fp", "raw_fn", "raw_precision", "raw_recall", "raw_f1",
           "tracked_tp", "tracked_fp", "tracked_fn", "tracked_precision", "tracked_recall", "tracked_f1",
           "confirmed_tracks", "unconfirmed_tracks", "mean_inference_ms", "processing_fps"]


def _scores(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)


def summarize_experiment(name: str, summary: dict, class_name: str = "pothole") -> dict:
    """Pool per-video frame-level counts for one class into a single result row."""
    videos = [video for video in summary["videos"] if "frame_level_comparison" in video]
    if not videos:
        raise ValueError(f"{name}: no video had a ground-truth mask; nothing to score")
    row = {"experiment": name, "videos": len(videos), "frames": sum(video["frames_processed"] for video in videos)}
    for mode, key in (("raw", "raw_per_frame"), ("tracked", "track_filtered")):
        counts = {item: sum(video["frame_level_comparison"][key][class_name][item] for video in videos)
                  for item in ("tp", "fp", "fn")}
        row.update({f"{mode}_{item}": value for item, value in counts.items()})
        row[f"{mode}_precision"], row[f"{mode}_recall"], row[f"{mode}_f1"] = _scores(**counts)
    row["confirmed_tracks"] = sum(len(video["confirmed_tracks"]) for video in videos)
    row["unconfirmed_tracks"] = sum(video["unconfirmed_track_count"] for video in videos)
    row["mean_inference_ms"] = round(sum(video["mean_inference_ms"] for video in videos) / len(videos), 2)
    row["processing_fps"] = round(sum(video["processing_fps"] for video in videos) / len(videos), 2)
    return row


def expected_video_count(videos: str, max_videos: int | None) -> int:
    from src.detection.video_predict import collect_video_paths
    return len(collect_video_paths(videos, max_videos))


def write_rows(rows: list[dict], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return output


def evaluate_video_suite(videos: str, suite_path: str = DEFAULT_SUITE_CONFIG, conf: float = 0.25,
                         max_videos: int | None = None, experiments: list[str] | None = None,
                         rerun: bool = False) -> Path:
    root = project_root()
    suite = load_experiment_suite(suite_path, root=root)
    project_dir = root / suite["project"]["output"]
    results_dir = root / suite["project"].get("results", "results/experiments")
    rows = []
    for name, _ in suite_experiments(suite):
        if experiments and name not in experiments:
            continue
        weights = project_dir / name / "weights" / "best.pt"
        if not weights.is_file():
            raise FileNotFoundError(f"{name}: train the suite first, missing {weights}")
        print(f"=== {name} ===")
        saved = root / "results/predictions/video_eval" / name / "video_summary.json"
        summary = json.loads(saved.read_text(encoding="utf-8")) if saved.is_file() and not rerun else None
        expected = expected_video_count(videos, max_videos)
        if summary and summary["processed_videos"] == expected and summary["confidence_threshold"] == conf:
            print(f"  reusing {saved}")
        else:
            summary = run_video_prediction(
                str(weights), videos, f"results/predictions/video_eval/{name}", conf, clean_output=True,
                experiment_name=name, max_videos=max_videos, show_ground_truth=True, save_video=False)
        rows.append(summarize_experiment(name, summary))
        row = rows[-1]
        print(f"  raw F1={row['raw_f1']}  tracked F1={row['tracked_f1']}  tracks={row['confirmed_tracks']}")
    output = write_rows(rows, results_dir / "video_tracking_results.csv")
    print(f"Saved {output}")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Frame-level video evaluation of every suite model")
    parser.add_argument("--videos", required=True, help="Folder of test RGB videos with a sibling mask/ folder")
    parser.add_argument("--suite", default=DEFAULT_SUITE_CONFIG)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--experiments", nargs="*", default=None, help="Subset of experiment names")
    parser.add_argument("--rerun", action="store_true",
                        help="Ignore saved per-model summaries (default reuses complete ones)")
    args = parser.parse_args(argv)
    evaluate_video_suite(args.videos, args.suite, args.conf, args.max_videos, args.experiments, args.rerun)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
