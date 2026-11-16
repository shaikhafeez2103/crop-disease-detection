# 🌿 AgroScan AI (CropGuard ML) — Crop Disease Detection Using Machine Learning

A production-ready full-stack **Crop Disease Detection** web application powered by **MobileNetV2 Transfer Learning** (TensorFlow / Keras) on the **PlantVillage** dataset structure, paired with a responsive **Modern AgriTech Glassmorphic UI** served by **Flask**.

---

## ✨ Key Features

### 1. Machine Learning Pipeline (`ml/train.py` & `ml/predict.py`)
- **Transfer Learning Backbone:** Uses `MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))` with a custom regularized classification head (`GlobalAveragePooling2D` → `BatchNormalization` → `Dense(256, relu)` → `Dropout(0.4)` → `Dense(num_classes, softmax)`).
- **PlantVillage Data Augmentation:** Real-time `ImageDataGenerator` pipeline supporting rotation, width/height shift, shear, zoom, horizontal flips, and `[0, 1]` pixel normalization.
- **Supported Foliar Classes:**
  1. **Healthy** (*Optimal Photosynthetic Canopy*)
  2. **Early Blight** (*Alternaria solani*)
  3. **Late Blight** (*Phytophthora infestans*)
  4. **Bacterial Spot** (*Xanthomonas campestris / perforans*)
  5. **Powdery Mildew** (*Podosphaera xanthii / Erysiphe*)
- **Instant Out-of-the-Box Inference (`--bootstrap`):** Includes pre-calibrated fallback initialization and multi-spectral foliar feature extraction so `/predict` and the full web interface work immediately before downloading the multi-gigabyte PlantVillage dataset.

### 2. REST API (`app.py`)
- **`POST /predict`**: Accepts multipart image upload (`file`) or bundled sample name (`sample_name`), resizes to `224x224`, normalizes pixel values to `[0.0, 1.0]`, runs inference, and returns JSON containing:
  - `predicted_class`
  - `confidence_score` (percentage)
  - `top_3_predictions` (`[{label, confidence}, ...]`)
  - `severity_estimate` (`"Healthy"`, `"Low"`, `"Moderate"`, `"High"`)
  - `symptoms_and_causes` (symptoms & environmental triggers)
  - `recommended_treatment` (actionable organic & chemical protocols)
  - `disclaimer`: `"Estimate only — validate with an agricultural expert before applying chemical treatments."`

### 3. Modern AgriTech Glassmorphic UI
- **Visual Theme:** Deep botanical emerald & slate dark mode (`#0b1914` background, `#12261f` elevated glassmorphic cards with `1px solid rgba(52, 211, 153, 0.2)` borders, `#10b981` emerald primary accent, and `#f59e0b` amber warning badges).
- **Interactive Smart Upload & One-Click Sample Strip:** Drag-and-drop leaf upload with instant preview, laser-scan animation, and 5 clickable sample leaf images (`sample_images/`) for instant recruiter/user demo testing.
- **Session Scan History Strip:** Automatically tracks scans during the current session for side-by-side comparison.

---

## 📁 Directory Structure

```text
crop-disease-detection/
├── IMPLEMENTATION_PLAN.md         # Architectural blueprint & execution plan
├── README.md                      # Setup, training, and usage documentation
├── requirements.txt               # Python dependencies
├── app.py                         # Flask server & /predict REST API
├── ml/
│   ├── __init__.py
│   ├── train.py                   # PlantVillage ImageDataGenerator + MobileNetV2 training
│   ├── predict.py                 # Standalone 224x224 preprocessing + inference engine
│   └── generate_samples.py        # High-res botanical leaf sample generator
├── model/
│   ├── crop_disease_model.h5      # Saved MobileNetV2 model artifact
│   └── class_indices.json         # Class index mapping dictionary
├── sample_images/                 # Bundled sample leaves for 1-click UI testing
│   ├── healthy_leaf.jpg
│   ├── early_blight.jpg
│   ├── late_blight.jpg
│   ├── bacterial_spot.jpg
│   └── powdery_mildew.jpg
├── templates/
│   └── index.html                 # Responsive AgriTech Glassmorphic interface
├── static/
│   ├── css/style.css              # Custom variables, CSS Grid/Flexbox, micro-interactions
│   └── js/app.js                  # Drag & drop, /predict fetch, animated bars, history strip
└── tests/
    └── test_predict.py            # Automated end-to-end verification suite
```

---

## 🚀 Quick Start & Installation

### 1. Create a Virtual Environment & Install Dependencies
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Initialize Pre-Trained Artifacts & Sample Leaves (Optional — Auto-runs on Startup)
```bash
python -m ml.train --bootstrap
python -m ml.generate_samples
```

### 3. Run the Flask Web Application
```bash
python app.py
```
Open **`http://localhost:5000`** in your browser.

---

## 🧠 Training on the Full PlantVillage Dataset

To train `MobileNetV2` on your local copy of the PlantVillage dataset:

1. Organize the dataset directory into class folders (either a single root folder which automatically splits 80/20 train/val, or explicit `train/` and `val/` subdirectories):
   ```text
   PlantVillage/
   ├── Healthy/
   ├── Early Blight/
   ├── Late Blight/
   ├── Bacterial Spot/
   └── Powdery Mildew/
   ```
2. Run `ml/train.py`:
   ```bash
   python -m ml.train --dataset_dir path/to/PlantVillage --epochs 15 --batch_size 32 --fine_tune_epochs 5
   ```
3. Trained artifacts are automatically saved to `model/crop_disease_model.h5` and `model/class_indices.json`.

---

## 🧪 Running Automated Verification Tests

Execute the automated test suite (`tests/test_predict.py`) to verify the UI routes, static assets, and `POST /predict` JSON responses across all sample leaf classes:

```bash
python -m unittest tests/test_predict.py -v
```
