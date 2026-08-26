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

## Commands

Run from the repository root:

```powershell
python -m src.data.clean_dataset
python -m src.detection.audit_dataset
python -m src.detection.train --config configs/yolo_training.yaml
python -m src.detection.evaluate --model experiments/yolo/baseline/weights/best.pt
python -m src.detection.predict --weights experiments/yolo/baseline/weights/best.pt --source data/processed/road_damage_detection/images/test
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
