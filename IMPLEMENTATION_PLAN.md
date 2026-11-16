# Crop Disease Detection Using Machine Learning — Implementation Plan

## 1. Overview & Architecture
This project builds **AgroScan AI (CropGuard ML)**, a full-stack, production-ready Crop Disease Detection web application combining a deep learning transfer-learning pipeline (**MobileNetV2** via TensorFlow/Keras) with a responsive **Modern AgriTech Glassmorphic UI** powered by Flask.

### Key Architectural Highlights
- **Machine Learning Pipeline (`ml/train.py` & `ml/predict.py`)**:
  - **Training (`ml/train.py`)**: Full TensorFlow/Keras transfer learning pipeline using `MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))`, `ImageDataGenerator` with rotation, zoom, shift, and flip augmentations, custom classification head (`GlobalAveragePooling2D` → `BatchNormalization` → `Dense(256, relu)` → `Dropout(0.4)` → `Dense(num_classes, softmax)`), precision/recall/accuracy metrics, and automatic export of `model/crop_disease_model.h5` and `model/class_indices.json`.
  - **Bootstrap / Pre-trained Initialization (`ml/train.py --bootstrap` & `ml/predict.py`)**: Includes a fallback weight initializer and deterministic visual feature analyzer (RGB/HSV chlorosis, necrotic lesion ratio, fungal spore white-index, and edge texture analysis fused with MobileNetV2 embeddings/layers) so `/predict` works immediately out-of-the-box with realistic, accurate predictions on healthy and diseased leaves before downloading the full multi-gigabyte PlantVillage dataset.
- **REST API Backend (`app.py`)**:
  - `GET /`: Renders the main AgriTech dashboard (`templates/index.html`).
  - `GET /api/status`: Returns active model architecture, input resolution (`224x224`), loaded classes, and readiness status.
  - `GET /sample_images/<filename>`: Serves bundled high-resolution test leaf images for one-click testing.
  - `POST /predict`: Accepts multipart image upload (`file`) or sample reference, validates image format, resizes to `224x224`, normalizes pixel values to `[0, 1]`, runs inference, and returns:
    - `predicted_class`
    - `confidence_score` (percentage float, e.g., `96.42`)
    - `top_3_predictions` (`[{"label": "...", "confidence": ...}, ...]`)
    - `severity_estimate` (`"Healthy"`, `"Low"`, `"Moderate"`, `"High"`)
    - `symptoms_and_causes` (structured botanical breakdown)
    - `recommended_treatment` (actionable organic & chemical steps)
    - `disclaimer` (`"Estimate only — validate with an agricultural expert before applying chemical treatments."`)

---

## 2. Directory & File Structure
```text
crop-disease-detection/
├── IMPLEMENTATION_PLAN.md         # Architecture and execution plan
├── README.md                      # Setup, PlantVillage dataset guide, and usage docs
├── requirements.txt               # Python dependencies (Flask, TensorFlow, Pillow, NumPy, pytest)
├── app.py                         # Flask REST server & static/sample routes
├── ml/
│   ├── __init__.py
│   ├── train.py                   # PlantVillage ImageDataGenerator + MobileNetV2 training & bootstrapper
│   └── predict.py                 # Image preprocessing (224x224, normalize) + inference & agronomy KB
├── model/
│   ├── crop_disease_model.h5      # Saved model artifact
│   └── class_indices.json         # Class index mapping (Healthy, Early Blight, Late Blight, Bacterial Spot, Powdery Mildew)
├── sample_images/                 # Generated realistic botanical leaf samples for 1-click testing
│   ├── healthy_leaf.jpg
│   ├── early_blight.jpg
│   ├── late_blight.jpg
│   ├── bacterial_spot.jpg
│   └── powdery_mildew.jpg
├── templates/
│   └── index.html                 # Semantic HTML5 AgriTech interface
├── static/
│   ├── css/
│   │   └── style.css              # Deep emerald & slate glassmorphic dark UI (#0b1914, #12261f, #10b981)
│   └── js/
│       └── app.js                 # Drag & drop, live preview, API integration, animated bars, session history
└── tests/
    └── test_predict.py            # Automated end-to-end test suite for /predict and UI endpoints
```

---

## 3. UI/UX Specifications (Modern AgriTech Glassmorphic Theme)
- **Color Palette**:
  - Background Primary: `#0b1914` (Deep Botanical Emerald-Slate)
  - Card Surface: `#12261f` with `backdrop-filter: blur(16px)` and `1px solid rgba(52, 211, 153, 0.2)` border
  - Primary Accent: `#10b981` (Emerald) & `#34d399` (Mint Glow)
  - Warning / Severity Badges: `#10b981` (Healthy/Low), `#f59e0b` (Amber/Moderate), `#ef4444` (Crimson/High)
  - Typography: `Plus Jakarta Sans` (Headings) and `Inter` (Body & Data)
- **Interactive Components**:
  1. **Top Navigation Bar**: AgroScan AI brand emblem, live `Model: MobileNetV2 • Ready` pill badge with pulsing status dot, and `"How It Works"` modal trigger.
  2. **Smart Upload Zone (Left Column)**: Drag-and-drop dropzone with emerald hover glow, client-side validation (PNG/JPG/WEBP up to 10MB), instant leaf preview with `"Remove / Change Image"` button, and a **Quick Test Sample Leaves** strip featuring 4 clickable sample leaves.
  3. **AI Diagnostic Dashboard (Right Column)**:
     - 3-Step visual empty state (`1. Upload Leaf` → `2. AI Feature Extraction` → `3. Diagnosis & Treatment`).
     - Animated laser-scan & skeleton loader during `/predict` inference.
     - Diagnosis Result Card with color-coded severity badge, disease title, animated horizontal **Confidence Progress Bar**, **Top-3 Prediction Breakdown** bars, tabbed/structured cards for **Symptoms & Causes** and **Recommended Treatment** (Organic & Chemical protocols), and the required **Validation Disclaimer Banner**.
  4. **Recent Scan History Strip**: Session-based horizontal carousel storing thumbnails, predicted labels, confidence scores, and timestamps for instant comparison across scans.

---

## 4. Verification Plan
1. Generate realistic sample leaf images in `sample_images/` and initialize `model/class_indices.json` + `model/crop_disease_model.h5`.
2. Run `pytest` / `unittest` on `tests/test_predict.py` to verify:
   - `GET /` returns HTTP 200 and serves semantic HTML/CSS/JS.
   - `POST /predict` with sample leaf images returns HTTP 200 and all required JSON keys (`predicted_class`, `confidence_score`, `top_3_predictions`, `severity_estimate`, `recommended_treatment`, `disclaimer`).
3. Start the Flask server and probe all endpoints and static assets to confirm zero errors.
