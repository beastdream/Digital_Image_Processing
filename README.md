# Road damage detection (images and video)

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


## Setup after cloning

Git holds code, configs, and small result files only. Model weights, datasets,
and generated images/videos are not in the repository.

```powershell
pip install -r requirements.txt
python -m pytest -q
```

To run predictions, copy the trained models from the team's shared Google
Drive folder `Digital_Image_Processing/outputs/yolo_video/<experiment>/weights/`
into `experiments/yolo_video/<experiment>/weights/` (each folder needs
`best.pt`; `experiment_config.yaml` is already in git). Then:

```powershell
python predict.py --experiment experiments/yolo_video/A_raw --video "path/to/video.mp4" --display
```

To rebuild the dataset or retrain, download the pothole video dataset
(`pothole_video/<split>/rgb|mask/*.mp4`) and follow
[Pothole video dataset and preprocessing suite](#pothole-video-dataset-and-preprocessing-suite).

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
```

<<<<<<< HEAD
=======
### Local web interface

Start the RoadScan interface from the repository root with the project
environment active:

```powershell
.\.venv\Scripts\Activate.ps1
python web\server.py
```

Open `http://127.0.0.1:8000` in a browser. The server listens on the local
machine only; keep its terminal open while using the interface and press
`Ctrl+C` to stop it. Uploaded prediction files are limited to 1 GiB. Each run
writes to a new directory under `results\predictions\roadscan\`, preserving
previous output. Evaluation runs use a new directory under
`results\evaluations\roadscan\`.

The interface reads experiment and evaluation reports from the existing
pipeline outputs. A missing report is shown as unavailable rather than
replaced with sample numbers. Video models trained from the current video
dataset learn pothole only; crack and manhole counts are not evidence of
video-model performance.

>>>>>>> da7e1cd4 (Cap nhat code)
### Safe baseline training workflow

Step 1 — run a short end-to-end smoke test. These overrides apply only at
runtime and do not modify `configs/yolo_training.yaml`:

```powershell
python -m src.detection.train --config configs/yolo_training.yaml --name A_raw_smoke --smoke-test
dir experiments\yolo\A_raw_smoke\weights
```

The directory must contain both `best.pt` and `last.pt` before proceeding.

Step 2 — if the existing `A_raw` is incomplete (it has no
`weights/best.pt`), archive it safely and restart the exact `A_raw` name:

```powershell
python -m src.detection.train --config configs/yolo_training.yaml --name A_raw --overwrite-incomplete
```

The command refuses to overwrite an `A_raw` that already contains
`weights/best.pt`. An incomplete run is moved below
`experiments/yolo/legacy_incomplete/`; other legacy experiment directories are
never moved or deleted.

Step 3 — verify the final checkpoints:

```powershell
dir experiments\yolo\A_raw\weights
```

Both `best.pt` and `last.pt` are required for a completed training run.

Step 4 — predict 20 test images:

```powershell
python predict.py --experiment experiments/yolo/A_raw --source "data/processed/road_damage_detection/images/test" --max-images 20
```

Prediction remains fail-fast when the requested experiment has no model and
writes successful output to `results/predictions/latest/`.

### Prediction

All commands below are single-line commands that work in both Windows CMD and
PowerShell. Predict one image:

```powershell
python predict.py --experiment experiments/yolo/A_raw --image "data/processed/road_damage_detection/images/test/20250223_104730.jpg"
```

Predict up to 20 images from a directory:

```powershell
python predict.py --experiment experiments/yolo/A_raw --source "data/processed/road_damage_detection/images/test" --max-images 20
```

Both commands write to `results/predictions/latest/`. Running prediction again
clears only `results/predictions/latest/` and replaces it with the new run.
Final evaluation and experiment results are not deleted.

For advanced usage, a custom output is preserved rather than cleaned:

```powershell
python predict.py --experiment experiments/yolo/A_raw --source samples_check/images --output results/predictions/demo
```

To deliberately replace that custom output, add `--clean-output`:

```powershell
python predict.py --experiment experiments/yolo/A_raw --source samples_check/images --output results/predictions/demo --clean-output
```

For safety, `--clean-output` may reset only a child directory below
`results/predictions/`. It refuses the prediction root itself and every path
outside that root.

### Evaluation

Run a manual evaluation using the model's explicit preprocessing contract:

```powershell
python -m src.detection.evaluate --model experiments/yolo/A_raw/weights/best.pt --experiment-config experiments/yolo/A_raw/experiment_config.yaml
```

The default output is `results/evaluations/latest/`, which is replaced by the
next manual default evaluation. To select an explicit experiment directory:

```powershell
python -m src.detection.evaluate --model experiments/yolo/A_raw/weights/best.pt --experiment-config experiments/yolo/A_raw/experiment_config.yaml --output results/evaluations/A_raw
```

`clean_dataset` writes the dataset and its reports under
`data/processed/road_damage_detection/`. It records unreadable images and
missing/orphan labels without letting one bad file abort processing, drops
invalid boxes, clips valid out-of-bound boxes, removes exact duplicates, and
excludes duplicate groups with conflicting annotations. The split is based on
capture groups, never individual frames.

Raw and processed datasets are intentionally excluded from Git because they can
be large. Existing local data is never removed by this policy. After placing
the source dataset under `data/raw/`, recreate `data/processed/` with:

```powershell
python -m src.data.clean_dataset
```

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

For images, `predict.py` accepts `.jpg`, `.jpeg`, and `.png` inputs, either as a
single `--image`/`--source` path or a folder passed to `--source`. It validates
every selected image, draws `pothole`, `crack`, or `manhole` with confidence,
and saves the annotated images under `results/predictions/latest/` by default.
Each default quick prediction run safely replaces only that `latest` directory;
an explicit `--output` directory is never recursively deleted. `--image` rejects
video files; videos go through `--video` or `--source` (see below).

Every prediction run also writes `prediction_summary.json` beside the annotated
images with the resolved model path, experiment name (when `--experiment` is
used), confidence threshold, fixed class map, processed-image count, and each
predicted class/confidence/`xyxy` box. To deliberately replace a custom output,
pass both `--output results/predictions/<name>` and `--clean-output`; guarded
cleanup rejects locations outside `results/predictions/`.

## Video detection

`predict.py` also accepts a video file, a folder of videos, or a webcam index.
`--source` routes to video mode automatically when it points at a video (or a
folder holding videos and no images); `--video` forces it.

```powershell
python predict.py --experiment experiments/yolo_video/A_raw --video "D:/Pothole Videos/pothole_video/test/rgb/0081.mp4"
python predict.py --experiment experiments/yolo_video/A_raw --video "D:/Pothole Videos/pothole_video/test/rgb" --max-videos 5 --show-ground-truth
python predict.py --experiment experiments/yolo_video/A_raw --video 0 --display
```

Each frame is first downscaled to `--frame-size` (default 640, the size of the
extracted training frames, so Gaussian/Median/CLAHE act at the trained scale),
then goes through the model's stored preprocessing contract exactly like image
prediction. What video adds over single images:

- **Tracking** (`src/detection/video_tracking.py`): a deterministic IoU tracker
  gives every physical object one `#ID` across frames.
- **Temporal class voting**: a track's class is the confidence-weighted vote over
  all of its frames, so a single frame flickering between `pothole` and
  `manhole` does not change the decision.
- **Unique object counts**: tracks seen in at least `--min-track-frames`
  (default 3) processed frames are counted once per class; shorter tracks are
  reported as unconfirmed noise.

Video options: `--frame-step N` runs the model on every N-th frame (boxes are
held in between), `--max-frames`, `--max-videos`, `--display` (live window, `q`
quits; required with `--max-frames` for a webcam), and `--save-frame-detections`.

Output goes to `results/predictions/video_latest/` (replaced each default run):
`pred_<video>.mp4` plus `video_summary.json` with fps, processing speed,
per-class unique counts, and every confirmed track (voted class, vote shares,
first/last time in seconds). With `--show-ground-truth`, a video under
`.../rgb/<clip>.mp4` automatically uses `.../mask/<clip>.mp4` as ground truth:
mask regions become green `GT` boxes and the summary reports frame-level
TP/FP/FN, precision, and recall at IoU 0.5, both for raw per-frame classes and
for track-voted classes.

### Pothole video dataset and preprocessing suite

The pothole video dataset (`<split>/rgb/*.mp4` with matching `<split>/mask/*.mp4`
segmentation masks) is converted into a YOLO detection dataset, then the same
six preprocessing experiments (A_raw ... F_combined, identical training
settings) run on it:

```powershell
python -m src.data.build_video_frame_dataset --source "D:/Pothole Videos/pothole_video" --frame-step 24
python -m src.detection.run_preprocessing_experiments --suite configs/video_preprocessing_experiments.yaml
python -m src.detection.analyze_experiment_results --suite configs/video_preprocessing_experiments.yaml --results results/video_experiments/experiment_results.csv --visual results/video_experiments/visual_inspection.csv --output results/video_experiments
```

Models go to `experiments/yolo_video/<experiment>/`, evaluations to
`results/evaluations/video/<experiment>/`, and the comparison table and
analysis to `results/video_experiments/`. Latency is measured on the raw test
frames. The suite resumes after an interruption: a run that already has
`best.pt`, `last.pt`, and `experiment_config.yaml` is evaluated again but not
retrained, and an incomplete run is archived and restarted.
#### Visual inspection and video-level evaluation

```powershell
python -m src.detection.visual_inspection_grid --suite configs/video_preprocessing_experiments.yaml
python -m src.detection.evaluate_video_suite --suite configs/video_preprocessing_experiments.yaml --videos "D:/Pothole Videos/pothole_video/test/rgb"
```

The first command writes ten review sheets (seeded sample of test frames) to
`results/video_experiments/visual_inspection/`: ground truth, then every
experiment's prediction drawn on its own preprocessed input, with TP/FP/FN per
panel (`panels.csv`). After reviewing them, complete
`results/video_experiments/visual_inspection.csv` (status `COMPLETE`,
`reviewed_images` 10, notes, reviewer) and rerun `analyze_experiment_results`
with `--suite configs/video_preprocessing_experiments.yaml` and the
`results/video_experiments/` paths shown above.

The second command scores every model frame by frame on all test clips
against their masks and writes `video_tracking_results.csv`. `raw_*` counts
every per-frame detection; `tracked_*` keeps only detections whose track
lasted at least three frames, measured offline after each clip. For this
pothole-only data, class voting cannot change a label, so tracking helps only
by removing short-lived false positives; voting matters for a 3-class model.

#### Running the suite on Google Colab (GPU)

CPU training takes about two hours per experiment; a Colab T4 runs the whole
suite in under an hour with the same code, seed, and training settings
(only `device` changes).

```powershell
python -m src.utils.package_for_colab
```

Upload the resulting `colab_package.zip` (code + frame dataset, ~250 MB) to
`MyDrive/Digital_Image_Processing/`, open `colab/run_video_experiments.ipynb`
in Colab, select a T4 GPU runtime, and run all cells. Models and results are
written to `MyDrive/Digital_Image_Processing/outputs/`, so after a disconnect
"Run all" resumes the suite. The last cell creates `colab_results.zip`; to use
it locally:

```powershell
Rename-Item experiments\yolo_video yolo_video_cpu_old
Expand-Archive colab_results.zip -DestinationPath colab_out
Move-Item colab_out\yolo_video experiments\yolo_video
Copy-Item colab_out\results\* results\ -Recurse -Force
```

Latency columns from a Colab run are GPU timings.

`--frame-step 24` keeps two frames per two-second clip (744 / 248 / 246
train / val / test frames); neighbouring frames are near-identical, so this
halves CPU training time at little cost. Every N-th frame is saved (resized to 640 on the longest side) and every
connected mask region becomes one `pothole` box. The source split is kept per
clip, so frames of one video never appear in two splits. The dataset contains
only potholes; the fixed 3-class map is kept so the model works with every
other command, but it cannot learn `crack` or `manhole` from this data.

## Visualizing ground truth vs predictions

Dataset annotations are loaded independently from YOLO predictions. For an
image under `images/<split>/`, the matching label is found automatically under
`labels/<split>/`. Ground-truth boxes are green and prefixed `GT:`; predictions
are red and prefixed `Pred:` in comparison mode, with a legend identifying both.

Ground truth only (does not load or run a model):

```powershell
python predict.py --image "data/processed/road_damage_detection/images/test/20250216_164325.jpg" --ground-truth-only
```

Prediction only (the unchanged default):

```powershell
python predict.py --experiment experiments/yolo/A_raw --image "data/processed/road_damage_detection/images/test/20250216_164325.jpg"
```

Ground truth and actual model predictions on the same image:

```powershell
python predict.py --experiment experiments/yolo/A_raw_smoke --image "data/processed/road_damage_detection/images/test/20250216_164325.jpg" --show-ground-truth
```

Separate report-friendly panels:

```powershell
python predict.py --experiment experiments/yolo/A_raw_smoke --image "data/processed/road_damage_detection/images/test/20250216_164325.jpg" --show-ground-truth --side-by-side
```

Folder comparison and ground-truth-only modes use the same automatic label
lookup and honor `--max-images`:

```powershell
python predict.py --experiment experiments/yolo/A_raw --source "data/processed/road_damage_detection/images/test" --max-images 10 --show-ground-truth
python predict.py --source "data/processed/road_damage_detection/images/test" --max-images 10 --ground-truth-only
```

Ground-truth-only runs replace only `results/ground_truth/latest/` and write
`gt_<original>.jpg`. Comparison runs remain under
`results/predictions/latest/` and write `compare_<original>.jpg`. A missing
label is fatal in ground-truth-only mode; comparison mode warns and still runs
the model. If the model returns no boxes, the comparison image explicitly says
`Predictions: none` while retaining all available GT boxes.

In comparison mode, `prediction_summary.json` stores `ground_truth` and
`predictions` as separate lists for each image. It also reports per-class
`tp`/`fp`/`fn` using one-to-one, same-class matching at IoU >= 0.5. This quick
comparison is not a replacement for dataset-level mAP evaluation.

## Results directory

The generated-result layout has one purpose for each directory:

```text
results/
  ground_truth/latest/
  predictions/latest/
  evaluations/latest/
  evaluations/<experiment>/
  experiments/
  preprocessing/visualizations/
  analysis/class_balance/
  legacy/
```

Runtime directories are created automatically by prediction, evaluation, and
visualization commands. Run `python -m src.utils.organize_results --init` only
when you want to initialize the complete empty structure in advance; runtime
`latest/` directories do not require `.gitkeep` files.

- `results/predictions/latest/` is the quick prediction test output. Every new
  default prediction run replaces it completely, so stale images cannot remain.
- `results/evaluations/latest/` is the most recent manual evaluation and is
  likewise replaced by the next default manual evaluation.
- `results/evaluations/<experiment>/` stores evaluation artifacts for each fair
  experiment, such as `A_raw` through `F_combined`. A runner replaces only the
  current experiment's directory.
- `results/experiments/` contains comparison tables and scientific-conclusion
  artifacts: `experiment_results.csv`, `experiment_analysis.json`,
  `experiment_analysis.md`, and the manually reviewed `visual_inspection.csv`.
- `results/preprocessing/visualizations/` contains generated grids showing
  Original, Resize, Denoise, CLAHE, Brightness, and Final stages. A new
  visualization run replaces the previous grids.
- `results/analysis/class_balance/` contains class-balance and bbox-size
  analysis in JSON, Markdown, and CSV formats.
- `results/legacy/` preserves outputs created before the directory refactor;
  legacy artifacts are not treated as current or final evidence.

### `experiments/yolo/` versus `results/`

These roots are intentionally different and must not be used interchangeably.
`experiments/yolo/<name>/` contains the trained model and training artifacts,
including `weights/best.pt`, `weights/last.pt`, `args.yaml`,
`experiment_config.yaml`, and reproducibility metadata. For example,
`experiments/yolo/A_raw/weights/best.pt` is a model.

`results/` contains only prediction, evaluation, visualization, comparison, and
analysis outputs. For example, `results/evaluations/A_raw/metrics.json` is the
evaluation result for that model; it is not a model or training artifact.
Cleanup under `results/` never targets `experiments/yolo/` or its weights.

`python -m src.detection.evaluate --model ...` writes to
`results/evaluations/latest/` unless `--output` is supplied.

Class-balance and bbox-size analysis writes its JSON, Markdown, and CSV artifacts
to `results/analysis/class_balance/`. Dataset integrity and validation reports
remain with the processed dataset under
`data/processed/road_damage_detection/reports/`; they are not migrated into
`results/`.

Preview an older generated-results layout first:

```powershell
python -m src.utils.organize_results --migrate --dry-run
```

After reviewing the planned `MOVE` and `CREATE` operations, run the migration:

```powershell
python -m src.utils.organize_results --migrate
```

Initialize missing canonical directories without deleting or replacing any
existing content:

```powershell
python -m src.utils.organize_results --init
```

The migration is idempotent and never treats unknown `results/yolo/` runs as
final experiments. It preserves them under `results/legacy/yolo/`, moves direct
legacy prediction images to `results/legacy/predictions_previous/`, and handles
old preprocessing visualizations without overwriting existing files. It does
not access model weights under `experiments/yolo/`, datasets, configs, source
code, or tests. Always use dry-run first; legacy results are archived rather
than promoted to final experiment evidence.

Training invokes the independent processed-dataset validator and is blocked if
`reports/post_processing_validation.json` is not `PASSED`. Visual sanity
montages are written to `reports/visual_samples/` and show input, augmentation,
resize, and final preprocessing with the corresponding boxes.

The global seed is configured in `configs/dataset_processing.yaml` (default
`42`). Dataset preparation, offline materialization, and training seed Python,
NumPy, PyTorch, and CUDA through one shared helper; YOLO receives the same seed
from its training configuration. All source paths resolve from the project
root, so the repository can be relocated.
