# 🎥 Nhận Diện Hành Vi Bạo Lực Từ Camera Giám Sát

> **Deep Learning cho bài toán phát hiện bạo lực trên CCTV**  
> Kiến trúc: `MobileNetV2` + `BiLSTM` + `Temporal Attention Mechanism`

---

## 📋 Tổng quan

Nghiên cứu này xây dựng hệ thống phát hiện hành vi bạo lực tự động từ video camera giám sát (CCTV), sử dụng kết hợp:

- **MobileNetV2** — trích xuất đặc trưng không gian từng khung hình (Transfer Learning từ ImageNet)
- **BiLSTM** — mô hình hóa quan hệ thời gian giữa các khung hình
- **Temporal Attention** — tập trung vào các khung hình quan trọng nhất, bỏ qua nền tĩnh

Kết quả đạt được **~90% accuracy** trên tập test với 3 dataset thực tế.

---

## 🏗️ Kiến trúc mô hình

```
Video Input (T frames)
        ↓
MobileNetV2 (TimeDistributed) — trích xuất spatial features
        ↓
BiLSTM (128 units) — học temporal dynamics
        ↓
Temporal Attention — tập trung khung hình quan trọng
        ↓
Dense(128) → Dense(64) → Softmax(2)
        ↓
  Normal / Violence
```

---

## 📊 Dataset

| Dataset | Nguồn | Số video |
|---|---|---|
| **RWF-2000** | Kaggle: `rwf2000` | 2,000 |
| **Real Life Violence Situations** | Kaggle: `real-life-violence-situations-dataset` | 2,000 |
| **SCVD** | Kaggle: `smartcity-cctv-violence-detection-dataset-scvd` | ~2,000 |

**Phân chia dữ liệu:** 80% Train / 10% Val / 10% Test  
**Chống Scene Leakage:** Dùng `GroupShuffleSplit` theo scene gốc

---

## 📈 Kết quả

| Metric | Giá trị |
|---|---|
| Accuracy | ~90.77% |
| ROC-AUC | ~0.95 |
| Violence Recall | ~96% |
| Avg Precision | ~0.96 |

---

## 🚀 Hướng dẫn chạy trên Kaggle

### Bước 1: Upload notebook
Vào [kaggle.com](https://kaggle.com) → **New Notebook** → upload file `v8_chay.ipynb`

### Bước 2: Add 3 dataset
Click **+ Add Data** và tìm lần lượt:
- `rwf2000`
- `real-life-violence-situations-dataset`
- `smartcity-cctv-violence-detection-dataset-scvd`

### Bước 3: Bật GPU
**Settings** → **Accelerator** → **GPU T4 x2**

### Bước 4: Chạy
Click **Run All** và chờ khoảng 4–6 tiếng.

---

## 📁 Cấu trúc project

```
📦 violence-detection-cctv/
├── 📓 v8_chay.ipynb          # Notebook chính (chạy trên Kaggle)
├── 📄 README.md              # File này
└── 📂 output_results/        # Sinh ra sau khi train (không commit)
    ├── models/
    │   ├── model_phase1_best.keras
    │   ├── model_phase2_best.keras
    │   └── attention_model.keras
    ├── charts/
    │   ├── 00_sanity_check.png
    │   ├── 01_learning_curves.png
    │   ├── 02_evaluation_charts.png
    │   └── 03_xai_*.png
    └── reports/
        └── classification_report.txt
```

---

## 🔧 Yêu cầu thư viện

```
tensorflow >= 2.12
opencv-python
scikit-learn
matplotlib
seaborn
numpy
```

> Tất cả đã có sẵn trên môi trường Kaggle, không cần cài thêm.

---

## 🧠 Chi tiết kỹ thuật

### Training Pipeline
- **Phase 1** (Frozen Base): Train toàn bộ head với MobileNetV2 đóng băng — `lr=1e-4`, 15 epochs
- **Phase 2** (Fine-tuning): Unfreeze 30 layer cuối MobileNetV2 — `lr=1e-5`, 10 epochs
- **Loss Function**: `CategoricalFocalCrossentropy` (alpha=0.25, gamma=2.0) — xử lý class imbalance

### Data Augmentation
- Random horizontal flip
- Random crop (85%) + resize
- Brightness jitter
- Temporal dropout (zeroing ngẫu nhiên 1–2 frames)

### Chống overfitting
- Dropout (0.3 + 0.2)
- EarlyStopping (patience=7)
- ReduceLROnPlateau
- GroupShuffleSplit (chống scene leakage)

---

## 📊 Explainable AI (XAI)

Hệ thống tích hợp **Temporal Attention Visualization** giúp giải thích quyết định của model — hiển thị những khung hình nào được model chú ý nhiều nhất khi phân loại.

![XAI Example](output_results/charts/03_xai_005.png)

---

## 👤 Tác giả

- **Họ tên:** Nguyễn Trần (ln1703nguyнtrnh)
- **Kaggle:** [kaggle.com/ln1703nguyнtrnh](https://kaggle.com)

---

## 📄 License

MIT License — free to use for research and educational purposes.
