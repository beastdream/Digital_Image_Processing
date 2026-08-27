# Báo cáo sau refactor pipeline Road Damage Detection

## A. Các lỗi đã tìm thấy và cách sửa

| File / function | Hành vi cũ | Vì sao sai | Cách sửa |
|---|---|---|---|
| `src/detection/run_preprocessing_experiments.py` / experiment runner | Tạo config trong Python với `epochs=4`, `fraction=0.15`, trong khi baseline dùng cấu hình khác; tên experiment cũng được hard-code. | So sánh không công bằng và thay experiment buộc sửa code. | Đưa dataset, class map, training args và danh sách experiment vào `configs/preprocessing_experiments.yaml`; validate runtime args và bắt buộc final `fraction=1.0`. |
| `configs/yolo_exp_*.yaml` | Các YAML legacy dùng tên/path `exp_*`, fraction nhỏ và có thể trỏ tới dataset không còn tồn tại. | Tên, folder và preprocessing contract không thống nhất. | Loại bỏ YAML legacy; generated config lấy từ suite YAML và materialized dataset có hash. |
| `src/preprocessing/augmentation.py` / `DataAugmenter` | Thiếu gamma; blur/noise được bật; range hình học có thể quá mạnh. | Crack/pothole phụ thuộc texture và mọi transform hình học phải giữ bbox/class đúng. | Thêm gamma nhẹ, tắt blur/noise mặc định, giới hạn rotation/translation/scale, clip/drop bbox đúng và giữ class ID. |
| `src/preprocessing/augmentation.py` / translation filtering | Bbox biến mất có nguy cơ bị thay lại bằng bbox input hoặc bbox ngoài khung không được xử lý rõ. | Tạo label giả cho object không còn trong ảnh. | Bbox partial được clip; bbox fully outside trả `[]`; không fallback về label cũ. |
| `src/data/dataset_utils.py` và `src/data/clean_dataset.py` / annotation parsing | Annotation biên, width/height bằng 0 và class sai chưa có một policy dùng chung, chặt chẽ. | Có thể đưa bbox vô hiệu hoặc remap class âm thầm vào training. | Parser/validator chung: chỉ sửa overflow trong epsilon, loại lỗi thật, không remap class và lưu lý do/action vào report. |
| `src/data/clean_dataset.py` / duplicate handling | Exact duplicate có annotation khác có thể bị chọn theo filename như duplicate bình thường. | Không thể biết annotation nào đúng một cách tự động. | Loại toàn bộ conflict group khỏi processed dataset và ghi `manual_review_required`; duplicate cùng annotation chỉ giữ một ảnh. |
| `src/data/clean_dataset.py` / splitting | Nguy cơ chia frame gần nhau hoặc near-duplicate sang các split khác nhau. | Gây leakage và làm metric test lạc quan giả. | Group theo capture session, merge group bằng pHash trước split, rồi assign nguyên group; audit filename/hash/group sau split. |
| `src/data/materialize_preprocessed_dataset.py` | Dataset preprocessing có thể chứa file stale hoặc mất liên kết với config/source split. | Experiment không tái lập và có thể lệch split. | Build trong temporary directory, fingerprint config/source/seed, thay nguyên version và lưu `build_manifest.json`. |
| `src/detection/predict.py` | Prediction có thể đưa raw image vào model train trên CLAHE/Gaussian; CLI chỉ dựa weights path và có thể nhận nguồn không phải ảnh. | Train/inference preprocessing mismatch; bbox output không đáng tin. | Load `experiment_config.yaml`, chạy đúng preprocessing trước YOLO, validate ảnh, hỗ trợ single/folder image và từ chối video rõ ràng. |
| `src/detection/evaluate.py` | Chủ yếu nhìn overall mAP; không có confusion artifacts bắt buộc. | Che khuất lỗi class thiểu số và lỗi pothole/manhole/background. | Xuất Precision/Recall/mAP50/mAP50-95 overall + từng class, confusion matrix 4×4, normalized matrix và bốn confusion quan trọng. |
| `src/detection/experiment_reporting.py` | Latency cũ chỉ đo `model.predict`, không tính đọc ảnh/preprocessing và có thể benchmark raw cho model processed. | Không phản ánh end-to-end deployment cost. | Đo `read → contract preprocessing → inference`, lưu breakdown và tự sinh `experiment_results.csv` từ args/metrics thật. |
| `src/detection/analyze_experiment_results.py` | Có nguy cơ tuyên bố raw/CLAHE tốt nhất trước khi đủ bằng chứng. | Kết luận không dựa trên experiment công bằng, per-class trade-off và visual review. | Gate kết luận bằng đủ 6 run, fair args, full fraction, metrics, latency và visual inspection; không khai báo universal winner. |
| `src/detection/train.py` / `train_yolo` | Validation trước train chưa bao phủ duplicate/capture-group leakage và chưa bảo đảm chạy trước khi tạo model. | Dataset hỏng vẫn có thể bắt đầu training. | `run_dataset_integrity_check()` chạy trước `YOLO(...)`: validate YAML/splits/images/labels/classes/bbox/hash/group, in thống kê và fail-fast. |

## B. Các file chính đã sửa hoặc thêm

### Config

- `configs/preprocessing_experiments.yaml`
- `configs/yolo_training.yaml`
- `configs/preprocessing.yaml`
- `configs/experiments/{baseline,gaussian_denoise,median_denoise,clahe,brightness,full_preprocessing,augmentation}.yaml`
- Xóa các `configs/yolo_exp_*.yaml` legacy không còn an toàn.

### Data

- `src/data/clean_dataset.py`
- `src/data/dataset_utils.py`
- `src/data/materialize_preprocessed_dataset.py`
- `src/data/validate_processed_dataset.py`
- `src/data/training_integrity.py`
- `src/data/visual_sanity_check.py`

### Preprocessing

- `src/preprocessing/augmentation.py`
- `src/preprocessing/pipeline.py`
- `src/preprocessing/run_experiments.py`
- `src/preprocessing/visualize.py`

### Detection và reporting

- `src/detection/train.py`
- `src/detection/evaluate.py`
- `src/detection/predict.py`
- `src/detection/run_experiments.py`
- `src/detection/run_preprocessing_experiments.py`
- `src/detection/model_contract.py`
- `src/detection/experiment_suite.py`
- `src/detection/experiment_reporting.py`
- `src/detection/analyze_experiment_results.py`
- `src/detection/analyze_class_balance.py`
- `predict.py`

### Tests

- `tests/test_dataset_utils.py`
- `tests/test_dataset_validation.py`
- `tests/test_duplicate_conflicts.py`
- `tests/test_preprocessing.py`
- `tests/test_required_edge_cases.py`
- `tests/test_training_integrity.py`
- `tests/test_materialize_preprocessed_dataset.py`
- `tests/test_model_contract.py`
- `tests/test_evaluation_metrics.py`
- `tests/test_fair_preprocessing_experiments.py`
- `tests/test_experiment_analysis.py`
- `tests/test_class_balance_analysis.py`
- `tests/test_reproducibility.py`

## C. Pipeline mới

```text
raw images + YOLO annotations
  → validate readable images, label presence and schema
  → validate/classify bbox issues (valid, epsilon-clipped, excluded)
  → exact duplicate + annotation-conflict detection
  → capture-sequence grouping + pHash near-duplicate group merging
  → group-aware train/val/test split
  → post-split filename/hash/group leakage audit
  → optional versioned preprocessing materialization
  → mandatory pre-training integrity gate + statistics
  → fair config-driven YOLO training
  → overall/per-class evaluation + confusion analysis
  → end-to-end latency benchmark + automatic result table
  → fair comparison gate + manual visual inspection
  → image-only prediction using model preprocessing contract
```

Không có bước nào được phép tự đổi class ID. Random augmentation chỉ chạy ở train; val/test không augmentation. Prediction load preprocessing metadata cạnh model, không đoán từ filename.

## D. Dataset statistics hiện tại

Nguồn: `data/processed/road_damage_detection/reports/training_integrity_report.json`.

| Split | Images | Pothole | Crack | Manhole | Total objects |
|---|---:|---:|---:|---:|---:|
| Train | 1,394 | 888 | 1,636 | 741 | 3,265 |
| Validation | 348 | 156 | 493 | 89 | 738 |
| Test | 257 | 207 | 331 | 115 | 653 |
| **Total** | **1,999** | **1,251** | **2,460** | **945** | **4,656** |

Training class distribution là pothole 27.20%, crack 50.11%, manhole 22.70%; largest/smallest ratio 2.21× nên được báo `HIGH`, nhưng chưa tự động oversample/weighted sampling.

## E. Các lỗi chưa thể auto-fix

1. **Annotation conflicts của exact duplicate:** 5 conflict groups, tổng 10 ảnh. Cùng pixel nhưng label khác nhau; tất cả đã bị loại khỏi processed dataset và giữ trạng thái `manual_review_required` trong `duplicate_annotation_conflicts.json`. Cần người kiểm tra ảnh nguồn và chọn/sửa ground truth.
2. **Invalid source annotations:** 64 annotation được flag; 18 lỗi epsilon nhỏ được clip có log, 46 annotation lỗi thật bị loại. Không thể khôi phục bbox đúng từ dữ liệu hiện có; muốn lấy lại object phải relabel thủ công từ ảnh nguồn.
3. **Final experiment conclusion:** chưa có `experiment_results.csv` từ đủ sáu full-fraction run và visual inspection của sáu experiment còn `PENDING`. `experiment_analysis.json` đang `BLOCKED`; không có kết luận Raw/CLAHE/phương pháp nào tốt nhất.
4. **Legacy evaluation metrics:** các artifact model cũ được tạo với training args không công bằng và không có model contract đầy đủ. Chúng không được dùng làm kết luận final; cần chạy lại suite config-driven.

## Coverage test bắt buộc

- Annotation: valid bbox, zero width, zero height, out-of-range, invalid class.
- Augmentation: inside, partial clip, fully outside, empty boxes, class preservation.
- Duplicate/split: same image + same label, same image + conflict label, sequence grouping, exact duplicate leakage gate, pHash near-duplicate merge.
- Preprocessing: valid output shape, immutable input, letterbox bbox alignment.

Các test này là regression gates; không thay thế visual inspection của dữ liệu thật.
