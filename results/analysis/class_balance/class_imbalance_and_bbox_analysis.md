# Class imbalance and bbox-size analysis

Imbalance: **HIGH** (2.21× largest/smallest).

No weighted sampling, oversampling, or class-specific augmentation is enabled by this analysis.

## Class evidence

- pothole: 1251 objects (26.9%); metrics={'precision': 0.0, 'recall': 0.0093, 'mAP50': 0.0, 'mAP50_95': 0.0}
- crack: 2460 objects (52.8%); metrics={'precision': 0.0019, 'recall': 0.2481, 'mAP50': 0.0007, 'mAP50_95': 0.0002}
- manhole: 945 objects (20.3%); metrics={'precision': 0.0, 'recall': 0.0, 'mAP50': 0.0, 'mAP50_95': 0.0}

## Warnings

- [HIGH] 75.5% of objects are small at imgsz=320; avoid 320 for final training without a targeted recall study.
- [HIGH] 91.8% of pothole objects are small at imgsz=320.
- [HIGH] 58.8% of crack objects are small at imgsz=320.
- [HIGH] 97.5% of manhole objects are small at imgsz=320.
- [HIGH] Largest/smallest class object-count ratio is 2.21x.
