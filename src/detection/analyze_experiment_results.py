"""Gate conclusions on fair metrics, latency, per-class trade-offs, and visual review."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path

from src.data.dataset_utils import project_root
from src.detection.experiment_reporting import RESULT_COLUMNS
from src.detection.experiment_suite import load_experiment_suite, suite_experiments
from src.utils.result_paths import experiment_results_dir

EXPECTED_EXPERIMENTS = tuple(name for name, _ in suite_experiments(load_experiment_suite()))
FAIR_COLUMNS = ("epochs", "batch", "imgsz", "fraction", "seed")
VISUAL_COLUMNS = ("experiment", "status", "reviewed_images", "bbox_alignment", "texture_preservation",
                  "representative_failures", "reviewer")


def create_visual_inspection_template(path: Path, expected_experiments=None) -> Path:
    """Create or safely extend the manual-review table without discarding user data."""
    expected_experiments = tuple(expected_experiments or EXPECTED_EXPERIMENTS)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    fieldnames = list(VISUAL_COLUMNS)
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing_fields = list(reader.fieldnames or [])
            rows = list(reader)
        fieldnames = existing_fields + [field for field in VISUAL_COLUMNS if field not in existing_fields]

    existing_experiments = {row.get("experiment") for row in rows}
    for experiment in expected_experiments:
        if experiment not in existing_experiments:
            rows.append({"experiment": experiment, "status": "PENDING", "reviewed_images": 0,
                         "bbox_alignment": "", "texture_preservation": "",
                         "representative_failures": "", "reviewer": ""})

    # Preserve a complete existing file byte-for-byte when no schema/row update is needed.
    if path.exists() and fieldnames == existing_fields and all(
        experiment in existing_experiments for experiment in expected_experiments
    ):
        return path

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", newline="", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return path


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _finite_number(row: dict, key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{row.get('experiment', '<unknown>')}: invalid {key}") from error
    if not math.isfinite(value):
        raise ValueError(f"{row.get('experiment', '<unknown>')}: non-finite {key}")
    return value


def validate_fair_results(rows: list[dict], expected_experiments=None) -> list[str]:
    expected_experiments = tuple(expected_experiments or EXPECTED_EXPERIMENTS)
    reasons = []
    if {row.get("experiment") for row in rows} != set(expected_experiments) or len(rows) != len(expected_experiments):
        reasons.append("Final comparison requires every configured experiment exactly once.")
        return reasons
    reference = {key: rows[0].get(key) for key in FAIR_COLUMNS}
    for row in rows[1:]:
        differences = [key for key in FAIR_COLUMNS if row.get(key) != reference[key]]
        if differences:
            reasons.append(f"Unfair training settings for {row['experiment']}: {', '.join(differences)}")
    try:
        if any(_finite_number(row, "fraction") != 1.0 for row in rows):
            reasons.append("Final conclusions require fraction=1.0 for all experiments.")
        for row in rows:
            for key in ("precision", "recall", "mAP50", "mAP50_95", "pothole_mAP50",
                        "crack_mAP50", "manhole_mAP50", "preprocessing_ms", "inference_ms", "total_ms"):
                _finite_number(row, key)
        if sum(row.get("preprocessing") == "raw" for row in rows) != 1:
            reasons.append("Final comparison requires exactly one raw preprocessing reference.")
    except ValueError as error:
        reasons.append(str(error))
    return reasons


def validate_visual_inspection(rows: list[dict], expected_experiments=None) -> list[str]:
    expected_experiments = tuple(expected_experiments or EXPECTED_EXPERIMENTS)
    reasons = []
    by_name = {row.get("experiment"): row for row in rows}
    if set(by_name) != set(expected_experiments) or len(rows) != len(expected_experiments):
        return ["Visual inspection must contain exactly one row for every final experiment."]
    for experiment in expected_experiments:
        row = by_name[experiment]
        if row.get("status", "").upper() != "COMPLETE":
            reasons.append(f"Visual inspection is not COMPLETE for {experiment}.")
            continue
        try:
            if int(row.get("reviewed_images", 0)) < 10:
                reasons.append(f"Visual inspection for {experiment} must review at least 10 images.")
        except ValueError:
            reasons.append(f"Visual inspection for {experiment} has invalid reviewed_images.")
        for field in ("bbox_alignment", "texture_preservation", "representative_failures", "reviewer"):
            if not row.get(field, "").strip(): reasons.append(f"Visual inspection for {experiment} is missing {field}.")
    return reasons


def _winner(rows: list[dict], metric: str, minimize: bool = False) -> dict:
    selected = min(rows, key=lambda row: _finite_number(row, metric)) if minimize else max(rows, key=lambda row: _finite_number(row, metric))
    return {"experiment": selected["experiment"], "value": _finite_number(selected, metric)}


def _pareto(rows: list[dict]) -> list[str]:
    frontier = []
    for candidate in rows:
        score, latency = _finite_number(candidate, "mAP50"), _finite_number(candidate, "total_ms")
        dominated = any((_finite_number(other, "mAP50") >= score and _finite_number(other, "total_ms") <= latency)
                        and (_finite_number(other, "mAP50") > score or _finite_number(other, "total_ms") < latency)
                        for other in rows if other is not candidate)
        if not dominated: frontier.append(candidate["experiment"])
    return frontier


def analyze_results(results_path: Path, visual_path: Path, expected_experiments=None) -> dict:
    expected_experiments = tuple(expected_experiments or EXPECTED_EXPERIMENTS)
    reasons = []
    if not results_path.exists():
        reasons.append(f"Missing final fair experiment table: {results_path}")
        result_rows = []
    else:
        result_rows = _read_csv(results_path)
        missing_columns = [column for column in RESULT_COLUMNS if not result_rows or column not in result_rows[0]]
        if missing_columns: reasons.append(f"experiment_results.csv is missing columns: {', '.join(missing_columns)}")
        else: reasons.extend(validate_fair_results(result_rows, expected_experiments))
    if not visual_path.exists():
        reasons.append(f"Missing visual inspection: {visual_path}")
        visual_rows = []
    else:
        visual_rows = _read_csv(visual_path); reasons.extend(validate_visual_inspection(visual_rows, expected_experiments))
    if reasons:
        return {"conclusion_status": "BLOCKED", "winner": None,
                "message": "No claim that raw or any preprocessing method is best is permitted yet.",
                "blocking_reasons": reasons,
                "required_evidence": ["fair overall metrics", "per-class metrics", "end-to-end latency", "completed visual inspection"]}
    raw = next(row for row in result_rows if row["preprocessing"] == "raw")
    tradeoffs = []
    for row in result_rows:
        tradeoffs.append({"experiment": row["experiment"], "preprocessing": row["preprocessing"],
            "overall": {key: _finite_number(row, key) for key in ("precision", "recall", "mAP50", "mAP50_95")},
            "per_class_mAP50": {name: _finite_number(row, f"{name}_mAP50") for name in ("pothole", "crack", "manhole")},
            "mAP50_delta_vs_raw": {name: round(_finite_number(row, f"{name}_mAP50") - _finite_number(raw, f"{name}_mAP50"), 6)
                                    for name in ("pothole", "crack", "manhole")},
            "total_ms": _finite_number(row, "total_ms")})
    return {"conclusion_status": "READY", "winner": None,
            "message": "No universal best method is declared; choose using target-class accuracy, overall metrics, latency, and visual findings.",
            "metric_leaders": {metric: _winner(result_rows, metric) for metric in
                               ("precision", "recall", "mAP50", "mAP50_95", "pothole_mAP50", "crack_mAP50", "manhole_mAP50")},
            "latency_leader": _winner(result_rows, "total_ms", minimize=True),
            "accuracy_latency_pareto_frontier": _pareto(result_rows), "per_experiment_tradeoffs": tradeoffs,
            "visual_inspection": visual_rows}


def write_analysis(report: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path, markdown_path = output_dir / "experiment_analysis.json", output_dir / "experiment_analysis.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Experiment analysis", "", f"Status: **{report['conclusion_status']}**", "", report["message"], ""]
    if report["conclusion_status"] == "BLOCKED":
        lines.extend(["## Blocking reasons", ""] + [f"- {reason}" for reason in report["blocking_reasons"]])
    else:
        lines.extend(["## Metric leaders (not a universal winner)", ""] +
                     [f"- {metric}: {item['experiment']} ({item['value']:.4f})" for metric, item in report["metric_leaders"].items()])
        lines.extend(["", f"Accuracy/latency Pareto frontier: {', '.join(report['accuracy_latency_pareto_frontier'])}"])
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser()
    default_output = experiment_results_dir(project_root())
    parser.add_argument("--results", default=str(default_output / "experiment_results.csv"))
    parser.add_argument("--visual", default=str(default_output / "visual_inspection.csv"))
    parser.add_argument("--output", default=str(default_output))
    parser.add_argument("--create-visual-template", action="store_true")
    args = parser.parse_args(); root = project_root()
    expected_experiments = [name for name, _ in suite_experiments(load_experiment_suite(root=root))]
    results, visual, output = (Path(args.results), Path(args.visual), Path(args.output))
    results = results if results.is_absolute() else root / results
    visual = visual if visual.is_absolute() else root / visual
    output = output if output.is_absolute() else root / output
    if args.create_visual_template: create_visual_inspection_template(visual, expected_experiments)
    paths = write_analysis(analyze_results(results, visual, expected_experiments), output)
    print("Saved:", *(str(path) for path in paths), sep="\n")


if __name__ == "__main__":
    main()
