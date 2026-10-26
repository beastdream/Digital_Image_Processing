# RoadScan — Đặc tả thiết kế giao diện (handoff)

Giao diện web cho hệ thống phát hiện hư hỏng mặt đường (YOLO, 3 lớp cố định: `0=pothole`, `1=crack`, `2=manhole`). Tài liệu này dùng để dựng giao diện trong VS Code. Bản mockup gốc là canvas "Road Damage Detection UI" gồm 4 màn hình.

Quy ước dữ liệu: mọi ô chỉ số chưa có số liệu hiển thị `—`. Không được hard-code số mẫu.

---

## 1. Gợi ý kiến trúc

| Lớp | Đề xuất | Ghi chú |
|---|---|---|
| Frontend | React + Vite (hoặc bất kỳ framework nào) | 4 route: `/`, `/video`, `/experiments`, `/evaluation` |
| Backend | FastAPI (Python) bọc các lệnh có sẵn | Gọi `predict.py` và `src.detection.*` bằng subprocess hoặc import trực tiếp, vì repo đã là Python |
| Dữ liệu | Đọc file JSON/CSV do pipeline sinh ra | Xem mục 5 |

Không sửa logic pipeline. Giao diện chỉ gọi lệnh, đọc kết quả và hiển thị.

### API gợi ý

| Method | Đường dẫn | Việc làm | Lệnh tương ứng |
|---|---|---|---|
| GET | `/api/experiments` | Liệt kê thí nghiệm có `weights/best.pt` | quét `experiments/yolo/*` và `experiments/yolo_video/*` |
| POST | `/api/predict/image` | Dự đoán ảnh/thư mục | `python predict.py --experiment ... --image/--source ... [--show-ground-truth] [--side-by-side]` |
| POST | `/api/predict/video` | Dự đoán video | `python predict.py --experiment ... --video ... [--frame-step N] [--min-track-frames N]` |
| POST | `/api/evaluate` | Đánh giá tập test | `python -m src.detection.evaluate --model ... --experiment-config ... --output ...` |
| GET | `/api/results/experiments?suite=image\|video` | Bảng so sánh | đọc `experiment_results.csv` + `experiment_analysis.json` |
| GET | `/api/results/evaluation/{experiment}` | Chỉ số + ma trận nhầm lẫn | đọc `metrics.json`, `class_metrics.csv`, `confusion_*.csv` |
| GET | `/api/results/class-balance` | Phân tích cân bằng lớp | đọc `results/analysis/class_balance/*` |

Lệnh chạy lâu (đánh giá, video) nên trả `job_id` rồi để frontend poll trạng thái.

Ràng buộc từ README cần giữ ở backend:
- Prediction và evaluation luôn đọc `experiment_config.yaml` của mô hình. Không suy ra phương pháp từ tên file.
- `--clean-output` chỉ được xoá thư mục con của `results/predictions/`.
- Prediction báo lỗi ngay nếu thí nghiệm chưa có `best.pt` (fail-fast).

---

## 2. Design tokens

```css
:root {
  /* Nền và chữ */
  --bg:          #F4F2EE;  /* nền trang */
  --surface:     #FFFFFF;  /* thẻ */
  --surface-alt: #FAF9F6;  /* vùng thả file, ô ma trận */
  --ink:         #1A1C1E;  /* chữ chính */
  --muted:       #5B6065;  /* chữ phụ (đạt 4.5:1 trên trắng) */
  --border:      #DAD6CE;
  --border-soft: #E7E3DB;
  --track:       #EFECE6;  /* nền thanh tiến độ */

  /* Sidebar */
  --side-bg:     #17191B;
  --side-ink:    #C9CCCF;
  --side-hover:  #26292C;
  --side-line:   #33373B;

  /* Nhấn (tweak được) */
  --accent:      #F2B705;  /* vàng cảnh báo đường bộ */
  --accent-ink:  #1A1C1E;

  /* Lớp (trên nền sáng) */
  --c-pothole:   #E4572E;
  --c-crack:     #2F6FDE;
  --c-manhole:   #0E8F84;
  /* Lớp (trên nền tối, sidebar) */
  --c-pothole-d: #FF7A4D;
  --c-crack-d:   #6FA3FF;
  --c-manhole-d: #4FD1C5;

  /* So khớp ground truth */
  --gt:          #0F7A37;  /* xanh: GT */
  --pred:        #D62F2F;  /* đỏ: dự đoán */

  /* Trạng thái */
  --ok:          #1E8E4E;
  --danger:      #B42318;
  --danger-bg:   #FDE7E4;
  --warn-bg:     #FFF4D1;
  --warn-line:   #E5C766;
  --warn-ink:    #4A3800;
}
```

- Chữ: `Be Vietnam Pro` (hỗ trợ tiếng Việt) cho giao diện, `JetBrains Mono` cho đường dẫn, lệnh, số liệu, tên thí nghiệm.
- Cỡ chữ: tiêu đề trang 28/700, tiêu đề thẻ 16/600, thân 14, chú thích 12–13.
- Bo góc: thẻ 14px, nút/ô nhập 8–10px, chip 999px.
- Khoảng cách: lề trang `32px 40px`, khoảng cách giữa thẻ 20–24px.
- Vùng bấm tối thiểu 44px. Dùng `<button>`, `<a>`, `<label for>` thật.
- Không chỉ dựa vào màu: lớp luôn kèm tên chữ, GT/Pred luôn kèm tiền tố `GT:` / `Pred:`.

---

## 3. Khung chung (shell)

Bố cục: sidebar trái + nội dung. Dưới ~800px sidebar xếp lên trên (flex-wrap), nội dung `flex: 999 1 560px; min-width: 0`.

**Sidebar** (rộng 220–260px, nền `--side-bg`)
- Logo ô vuông `--accent` + "RoadScan" / "Road damage detection".
- 4 mục điều hướng, mục đang chọn nền `--accent`, chữ `--accent-ink`, đậm:
  1. Dự đoán ảnh (`/`)
  2. Dự đoán video (`/video`)
  3. So sánh thí nghiệm (`/experiments`)
  4. Đánh giá mô hình (`/evaluation`)
- Chân sidebar: chú giải 3 lớp kèm class ID.

Component dùng chung: `Sidebar`, `PageHeader`, `Card`, `ExperimentSelect`, `ToggleRow`, `CommandPreview`, `ClassDot`, `StatTile`, `DataTable`, `Banner` (info/warn/danger).

---

## 4. Từng màn hình

### 4.1 Dự đoán ảnh (`/`)

Hai cột: cấu hình (rộng tối đa 400px) và kết quả.

**Cấu hình**
- `ExperimentSelect`: A_raw, B_gaussian, C_median, D_clahe, E_brightness, F_combined. Chỉ cho chọn thí nghiệm có `best.pt`.
- Vùng thả file/thư mục (`.jpg .jpeg .png`).
- Thanh trượt `conf` (mặc định 0.25), ô số `--max-images` (mặc định 20).
- Hai nút bật/tắt: "So sánh với ground truth" (`--show-ground-truth`, mặc định bật), "Hiển thị song song" (`--side-by-side`).
- Nút chính "Chạy dự đoán".
- `CommandPreview`: hiển thị lệnh sẽ chạy, cập nhật theo form.

**Kết quả**
- Ảnh có box. Chú giải: GT xanh (`GT:`), Pred đỏ (`Pred:` + confidence).
- Dải thumbnail các ảnh đã xử lý.
- Bảng so khớp theo lớp: TP / FP / FN (khớp 1–1, cùng lớp, IoU ≥ 0.5). Ghi chú: kiểm tra nhanh, không thay thế mAP.
- Thẻ `prediction_summary.json`: model, experiment, confidence, số ảnh, thư mục xuất `results/predictions/latest/`. Nút tải kết quả.

**Trạng thái đặc biệt**
- Không có box: ảnh ghi `Predictions: none`, vẫn hiện GT.
- Thiếu label ở chế độ so sánh: cảnh báo nhưng vẫn chạy mô hình. Ở chế độ chỉ-GT thì báo lỗi.
- Chưa có model: lỗi rõ ràng, hướng dẫn copy `best.pt` từ Google Drive vào `experiments/yolo_video/<exp>/weights/`.

### 4.2 Dự đoán video (`/video`)

**Banner cảnh báo (luôn hiện):** bộ video chỉ chứa pothole, mô hình không học được crack/manhole từ dữ liệu này.

**Cấu hình**
- `ExperimentSelect` (thư mục `experiments/yolo_video/*`).
- Nguồn: chuyển đổi "Video / thư mục" hoặc "Webcam" (chỉ số, mặc định 0). Với webcam phải có giới hạn khung khi dùng `--max-frames`.
- Lưới 2 cột: `--frame-size` (640), `--frame-step` (1), `--min-track-frames` (3), `--max-videos` (5).
- Checkbox: `--show-ground-truth`, `--save-frame-detections`.
- Nút "Chạy phát hiện video".

**Kết quả**
- Trình phát `pred_<video>.mp4`, box kèm `#ID` và lớp đã bỏ phiếu. Track dưới ngưỡng vẽ nét đứt, ghi "< 3 khung".
- Dòng thời gian: mỗi thanh là vòng đời một track.
- 3 ô đếm vật thể duy nhất theo lớp.
- Bảng so sánh ground truth (IoU 0.5): TP/FP/FN/Precision/Recall, hai cột "Theo khung" và "Sau bỏ phiếu".
- Bảng track đã xác nhận: ID, lớp bỏ phiếu, tỉ lệ phiếu (thanh), giây bắt đầu/kết thúc.

### 4.3 So sánh thí nghiệm (`/experiments`)

- Chuyển đổi bộ dữ liệu: "Bộ ảnh" / "Bộ video".
- **Thẻ trạng thái BLOCKED** (viền trái đỏ) với 3 điều kiện mở khoá:
  1. 6 dòng cùng cấu hình huấn luyện, `fraction = 1.0`.
  2. Đủ chỉ số tổng và theo lớp (P, R, mAP50, mAP50-95) cùng độ trễ.
  3. `visual_inspection.csv` hoàn tất ≥ 10 ảnh mỗi thí nghiệm.
  Khi đủ điều kiện, thẻ chuyển sang trạng thái sẵn sàng. Backend là nguồn quyết định, frontend không tự tính.
- **Kiểm tra trực quan:** 6 thanh tiến độ `x / 10`, nút mở lưới kiểm tra (10 trang từ `visual_inspection_grid`).
- **Bảng `experiment_results.csv`:** cột Thí nghiệm, Tiền xử lý, fraction, imgsz, Precision, Recall, mAP50, mAP50-95, preprocessing_ms, inference_ms, total_ms. Run có `debug_only` gắn nhãn `DEBUG ONLY` và không vào so sánh cuối.
- **Pareto** (mAP50 theo total_ms) và bảng **dẫn đầu theo chỉ số** (tổng thể và từng lớp, kèm Δ so với A_raw). Chỉ hiện khi hết BLOCKED. Tuyệt đối không gắn nhãn "tốt nhất".

### 4.4 Đánh giá mô hình (`/evaluation`)

- Thanh cấu hình: chọn `--model`, `--experiment-config`, `--output` (mặc định `results/evaluations/<exp>`), nút "Chạy đánh giá".
- 4 `StatTile`: Precision, Recall, mAP50, mAP50-95.
- **Ma trận nhầm lẫn 4×4** (pothole, crack, manhole, background). Hàng là dự đoán, cột là thực tế. Nút chuyển "Số lượng" / "Chuẩn hoá". Viền đỏ ở 4 ô lỗi được theo dõi: pothole→manhole, manhole→pothole, crack→background, pothole→background.
- Bảng chỉ số theo lớp (P, R, mAP50, mAP50-95).
- Danh sách 4 lỗi nhầm lẫn kèm số lượng (đọc từ `confusion_analysis.json`).
- **Cân bằng lớp:** thanh số vật thể theo lớp, 3 ô Small/Medium/Large (COCO, tại 320px), banner cảnh báo khi tỉ lệ vật thể nhỏ cao. Ghi chú: không tự bật oversampling hay trọng số lớp.
- Từ chối rõ ràng khi mô hình huấn luyện trên dữ liệu đã tiền xử lý mà bộ đánh giá là ảnh thô hoặc tiền xử lý khác.

---

## 5. Ánh xạ dữ liệu: ô giao diện đọc từ file nào

| Thành phần | File nguồn |
|---|---|
| Tóm tắt dự đoán ảnh, TP/FP/FN | `results/predictions/latest/prediction_summary.json` |
| Ảnh kết quả | `results/predictions/latest/pred_*.jpg`, `compare_*.jpg` |
| Ảnh chỉ GT | `results/ground_truth/latest/gt_*.jpg` |
| Tóm tắt video, track, đếm | `results/predictions/video_latest/video_summary.json`, `pred_<video>.mp4` |
| Chỉ số tổng | `results/evaluations/<exp>/metrics.json` |
| Chỉ số theo lớp | `class_metrics.csv`, `metrics_by_scope.csv` |
| Ma trận nhầm lẫn | `confusion_matrix*.csv` / `.png` (đếm và chuẩn hoá) |
| Phân tích lỗi nhầm lẫn | `confusion_analysis.json`, `confusion_analysis.md` |
| Độ trễ | `latency_breakdown.json` (mỗi run) |
| So sánh ảnh | `results/experiments/experiment_results.csv`, `experiment_analysis.json`, `visual_inspection.csv` |
| So sánh video | `results/video_experiments/experiment_results.csv`, `visual_inspection.csv`, `video_tracking_results.csv` |
| Cân bằng lớp | `results/analysis/class_balance/` |
| Cấu hình huấn luyện thực | `experiments/yolo*/<exp>/args.yaml`, `experiment_config.yaml` |

Tên cột và khoá JSON chính xác cần đọc trực tiếp từ file thật trong repo, vì README không liệt kê đầy đủ.

---

## 6. Hành vi tương tác

- Mọi nút bật/tắt dùng `aria-pressed`. Thẻ chọn nguồn (Video/Webcam) và chuyển chế độ ma trận cũng vậy.
- Khi chạy lệnh: nút chuyển sang trạng thái đang chạy, hiển thị log tối giản, không chặn điều hướng.
- Thư mục `results/predictions/latest/` bị thay thế mỗi lần chạy mặc định. Giao diện nên nói rõ "kết quả lần trước sẽ bị thay".
- Thao tác xoá hoặc ghi đè thư mục xuất tuỳ chỉnh phải có hộp xác nhận (tương ứng `--clean-output`).

## 7. Trạng thái cần thiết kế thêm (chưa có trong mockup)

- Trạng thái trống: chưa chọn ảnh, chưa có model, chưa chạy lần nào.
- Trạng thái đang chạy và lỗi cho từng màn.
- Màn kiểm tra trực quan (lưới 10 trang từ `visual_inspection_grid`, form hoàn tất `COMPLETE`, `reviewed_images`, ghi chú, người đánh giá).
- Hiển thị Colab: cột độ trễ từ Colab là thời gian GPU, nên gắn nhãn nguồn thiết bị.
