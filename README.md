# Road damage detection (static images)

The project detects exactly three classes: `0=pothole`, `1=crack`, and
`2=manhole`. Class IDs are fixed across every stage.

## Pipeline

```text
raw images + YOLO labels
  -> dataset / annotation validation
  -> duplicate and annotation-conflict detection
  -> group-aware train/val/test split + leakage audit
  -> optional uint8 preprocessing / online training augmentation
  -> YOLO training
  -> evaluation on the held-out test split
  -> static-image prediction
```

The primary reference experiment is **A_raw**: valid raw RGB images, with only
Ultralytics' standard letterbox/normalization and `imgsz=640`. Gaussian,
Median, CLAHE, and brightness are excluded from this baseline. All preprocessing
comparison experiments use `imgsz=640` so their metrics remain comparable.

## Commands

Run from the repository root:

```powershell
python -m src.data.clean_dataset
python -m src.data.validate_processed_dataset
python -m src.data.visual_sanity_check --per-split 10
python -m src.detection.audit_dataset
python -m src.detection.train --config configs/yolo_training.yaml
python -m src.detection.run_preprocessing_experiments
python -m src.detection.analyze_class_balance --metrics results/evaluations/A_raw/metrics.json
python -m src.detection.analyze_experiment_results --create-visual-template
python -m src.detection.evaluate --model experiments/yolo/baseline/weights/best.pt
python predict.py --experiment experiments/yolo/A_raw --image path/to/image.jpg
python predict.py --model experiments/yolo/A_raw/weights/best.pt --source data/test_images/
```

`clean_dataset` writes the dataset and its reports under
`data/processed/road_damage_detection/`. It records unreadable images and
missing/orphan labels without letting one bad file abort processing, drops
invalid boxes, clips valid out-of-bound boxes, removes exact duplicates, and
excludes duplicate groups with conflicting annotations. The split is based on
capture groups, never individual frames.

Preprocessing scripts write uint8 images because Ultralytics normalizes image
pixels internally. Random augmentation is applied online only during training;
validation and test data are never augmented.

The final comparison consists of `A_raw`, `B_gaussian`, `C_median`, `D_clahe`,
`E_brightness`, and `F_combined`. Every run uses the same training settings and
`fraction: 1.0`. A partial fraction is allowed only with `debug_only: true`; it
is labelled `DEBUG ONLY` and cannot enter the final comparison. The comparison
CSV reads `fraction`, image size, and the complete runtime configuration from
Ultralytics' generated `args.yaml`, rather than hard-coding dataset claims.
After all six runs finish, it writes `results/experiments/experiment_results.csv`
directly from those runtime args and evaluator outputs. Latency is measured on
raw test images as disk read → model-contract preprocessing → YOLO inference.
The table reports `preprocessing_ms`, `inference_ms`, and end-to-end `total_ms`;
each run also stores `latency_breakdown.json` with `read_image_ms` for auditing.

No preprocessing winner is declared automatically. Experiment analysis remains
`BLOCKED` until all six final rows have identical training settings with
`fraction=1.0`, all overall/per-class metrics and latency values are present,
and `visual_inspection.csv` contains a completed review of at least ten images
per experiment. Once ready, the report lists separate overall and per-class
leaders, deltas versus `A_raw`, visual findings, and the accuracy/latency Pareto
frontier; it intentionally does not label one method universally best.
Creating the visual-inspection template is non-destructive: completed/manual
rows and additional user columns are preserved, while newly configured
experiments are appended as `PENDING` rows.

The final suite is config-driven by `configs/preprocessing_experiments.yaml`:
class mapping, seed, dataset, training parameters, experiment names, and each
preprocessing YAML live there rather than in runner logic. Immediately before
creating a YOLO model, training validates dataset YAML, all split directories,
image/label pairing, readable images, class IDs, bbox geometry, exact duplicate
and filename leakage, and capture-group leakage. It prints image/object
statistics and writes `training_integrity_report.json`; any failure aborts
before `YOLO(...)` or `model.train()`.

Preprocessing experiments are materialized below
`data/processed/road_damage_detection/preprocessed/<experiment>_<hash>/`. The
hash includes preprocessing config, source split manifest, and seed. Every
materialization builds in a temporary directory and replaces that version's
output, so stale images and labels cannot survive a rebuild.

Every trained experiment stores an `experiment_config.yaml` beside the run and
inside its `weights/` directory. It contains the fixed class map, exact offline
preprocessing configuration, training data, and actual training arguments.
Prediction always loads this contract and applies the same preprocessing to raw
input before YOLO inference; it never infers a method from the model filename.
Evaluation likewise rejects a raw or differently preprocessed dataset when the
model was trained on materialized preprocessing.

Evaluation writes overall and per-class Precision, Recall, mAP50, and mAP50-95
to `metrics.json`, `class_metrics.csv`, and `metrics_by_scope.csv`. It also
writes count and normalized 4×4 confusion matrices for `pothole`, `crack`,
`manhole`, and `background` as CSV and PNG. `confusion_analysis.json` and
`confusion_analysis.md` explicitly report pothole→manhole, manhole→pothole,
crack→background, and pothole→background errors. Matrix rows are predictions;
columns are actual classes.

Class-balance analysis counts objects per class and combines those frequencies
with per-class precision, recall, mAP50, and mAP50-95 when an evaluation
`metrics.json` is supplied. It reports bbox width, height, and area in pixels at
the configured input size and at 320, with COCO small/medium/large area buckets.
The report does not automatically enable weighted sampling, oversampling, or
class-specific augmentation; those changes require supporting per-class metric
evidence. A high small-object share at 320 produces an explicit warning.

`predict.py` accepts only static `.jpg`, `.jpeg`, and `.png` inputs, either as a
single `--image`/`--source` path or a folder passed to `--source`. It validates
every selected image, draws `pothole`, `crack`, or `manhole` with confidence,
and saves the annotated images under `results/predictions/latest/` by default.
Each default quick prediction run safely replaces only that `latest` directory;
an explicit `--output` directory is never recursively deleted. Direct
video input is rejected because video handling is outside this image-only project.

Every prediction run also writes `prediction_summary.json` beside the annotated
images with the resolved model path, experiment name (when `--experiment` is
used), confidence threshold, fixed class map, processed-image count, and each
predicted class/confidence/`xyxy` box. To deliberately replace a custom output,
pass both `--output results/predictions/<name>` and `--clean-output`; guarded
cleanup rejects locations outside `results/predictions/`.

## Output layout

Training models and generated reports have deliberately separate roots:

```text
experiments/yolo/<experiment>/     model and training artifacts (including weights/)
results/
  predictions/latest/             replaceable quick image predictions
  evaluations/latest/             replaceable quick evaluation
  evaluations/A_raw/ ... F_combined/  final experiment evaluations
  experiments/                     comparison CSV/JSON/Markdown reports
  preprocessing/visualizations/   preprocessing sample grids
  analysis/class_balance/          imbalance and bbox reports
  legacy/                          manually retained older results
```

`python -m src.detection.evaluate --model ...` writes to
`results/evaluations/latest/` unless `--output` is supplied. Training always
remains under `experiments/yolo/`; result cleanup never targets that tree or the
`results/` root itself. The final preprocessing runner resets only the current
experiment's `results/evaluations/<experiment>/` folder before evaluating it,
so evaluation artifacts from the other experiments remain untouched.

Training invokes the independent processed-dataset validator and is blocked if
`reports/post_processing_validation.json` is not `PASSED`. Visual sanity
montages are written to `reports/visual_samples/` and show input, augmentation,
resize, and final preprocessing with the corresponding boxes.

The global seed is configured in `configs/dataset_processing.yaml` (default
`42`). Dataset preparation, offline materialization, and training seed Python,
NumPy, PyTorch, and CUDA through one shared helper; YOLO receives the same seed
from its training configuration. All source paths resolve from the project
root, so the repository can be relocated.
