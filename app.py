"""
app.py — AgroScan AI (CropGuard ML) Flask Web Server & REST API

Serves:
- `GET /`: Modern AgriTech Glassmorphic Web Interface (`templates/index.html`)
- `GET /api/status`: Model readiness, MobileNetV2 metadata, and supported classes
- `GET /sample_images/<filename>`: Bundled sample leaf images for instant 1-click testing
- `POST /predict`: Accepts uploaded leaf image (`file`) or sample reference (`sample_name`),
  resizes to 224x224, normalizes pixel values, runs MobileNetV2 inference, and returns JSON.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from ml.generate_samples import SAMPLE_DIR, generate_sample_leaves
from ml.predict import get_predictor, predict_leaf_disease

BASE_DIR = Path(__file__).resolve().parent
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB upload limit

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def ensure_startup_assets() -> None:
    """Ensures sample leaf images and MobileNetV2 model artifacts exist at startup."""
    expected_samples = [
        "healthy_leaf.jpg",
        "early_blight.jpg",
        "late_blight.jpg",
        "bacterial_spot.jpg",
        "powdery_mildew.jpg",
    ]
    if not all((SAMPLE_DIR / name).exists() for name in expected_samples):
        generate_sample_leaves()

    # Initialize predictor singleton (auto-bootstraps model/class_indices.json & .h5 if needed)
    get_predictor()


ensure_startup_assets()


@app.route("/", methods=["GET"])
def index():
    """Renders the main AgroScan AI Crop Disease Detection interface."""
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def api_status():
    """Returns live MobileNetV2 model metadata and supported PlantVillage classes."""
    predictor = get_predictor()
    return jsonify(
        {
            "status": "ready",
            "model": "MobileNetV2",
            "architecture": predictor.model_mode,
            "input_shape": [224, 224, 3],
            "classes": list(predictor.class_indices.keys()),
            "num_classes": len(predictor.class_indices),
        }
    )


@app.route("/sample_images/<path:filename>", methods=["GET"])
def serve_sample_image(filename: str):
    """Serves bundled sample leaf images for instant one-click testing in the UI."""
    safe_name = secure_filename(filename)
    target_path = SAMPLE_DIR / safe_name
    if not target_path.exists():
        return jsonify({"error": f"Sample image '{safe_name}' not found."}), 404
    return send_from_directory(str(SAMPLE_DIR), safe_name)


@app.route("/predict", methods=["POST"])
def predict():
    """
    REST endpoint that accepts an uploaded leaf image (`file` in multipart/form-data)
    or a bundled sample leaf name (`sample_name` in form/JSON), preprocesses it
    (resize to 224x224, normalize pixel values to [0,1]), runs MobileNetV2 inference,
    and returns the diagnostic JSON payload.
    """
    start_time = time.perf_counter()

    # 1. Check if a bundled sample leaf was requested by name
    sample_name = request.form.get("sample_name")
    if not sample_name and request.is_json:
        json_body: Dict[str, Any] = request.get_json(silent=True) or {}
        sample_name = json_body.get("sample_name")

    if sample_name:
        safe_sample = secure_filename(sample_name)
        sample_path = SAMPLE_DIR / safe_sample
        if not sample_path.exists():
            return (
                jsonify(
                    {
                        "error": f"Sample image '{safe_sample}' does not exist.",
                        "status": "error",
                    }
                ),
                400,
            )
        try:
            result = predict_leaf_disease(sample_path)
            result["inference_time_ms"] = round((time.perf_counter() - start_time) * 1000.0, 1)
            result["source_filename"] = safe_sample
            return jsonify(result), 200
        except Exception as exc:
            return jsonify({"error": f"Inference failed: {str(exc)}", "status": "error"}), 500

    # 2. Standard multipart file upload (`file` or `image` field)
    upload_file = request.files.get("file") or request.files.get("image")
    if upload_file is None or upload_file.filename == "":
        return (
            jsonify(
                {
                    "error": "No leaf image provided. Please upload a valid PNG, JPG, or WEBP leaf image.",
                    "status": "error",
                }
            ),
            400,
        )

    if not allowed_file(upload_file.filename):
        return (
            jsonify(
                {
                    "error": "Unsupported file type. Allowed formats: PNG, JPG, JPEG, WEBP, BMP.",
                    "status": "error",
                }
            ),
            400,
        )

    try:
        image_bytes = upload_file.read()
        if not image_bytes:
            return jsonify({"error": "Uploaded image file is empty.", "status": "error"}), 400

        result = predict_leaf_disease(image_bytes)
        result["inference_time_ms"] = round((time.perf_counter() - start_time) * 1000.0, 1)
        result["source_filename"] = secure_filename(upload_file.filename)
        return jsonify(result), 200
    except Exception as exc:
        return (
            jsonify(
                {
                    "error": f"Unable to process leaf image: {str(exc)}",
                    "status": "error",
                }
            ),
            400,
        )


@app.errorhandler(413)
def request_entity_too_large(_error):
    return (
        jsonify(
            {
                "error": "File too large. Maximum allowed leaf image size is 10 MB.",
                "status": "error",
            }
        ),
        413,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
