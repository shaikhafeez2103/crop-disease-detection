"""
tests/test_predict.py — Automated Verification Suite for AgroScan AI

Verifies:
1. `GET /` serves the AgriTech Glassmorphic UI (HTTP 200) and static CSS/JS assets load cleanly.
2. `GET /api/status` returns MobileNetV2 readiness metadata and PlantVillage classes.
3. `POST /predict` with multipart leaf image upload returns HTTP 200 and all required JSON keys:
   - `predicted_class`
   - `confidence_score` (percentage)
   - `top_3_predictions` (list of 3 `{label, confidence}` objects)
   - `severity_estimate` ("Healthy", "Low", "Moderate", "High")
   - `recommended_treatment` (organic & chemical steps)
   - `disclaimer` ("Estimate only — validate with an agricultural expert before applying chemical treatments.")
4. `POST /predict` accurately classifies all 5 bundled sample leaf images (`healthy_leaf.jpg`,
   `early_blight.jpg`, `late_blight.jpg`, `bacterial_spot.jpg`, `powdery_mildew.jpg`).
"""

import io
import json
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import app
from ml.generate_samples import SAMPLE_DIR, generate_sample_leaves


class TestCropDiseaseDetectionApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()
        cls.sample_map = generate_sample_leaves()

    def test_01_index_and_static_assets(self):
        """Verify GET / and static CSS/JS assets return HTTP 200."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn("AgroScan", html)
        self.assertIn("Model: MobileNetV2 • Ready", html)
        self.assertIn("Quick Test Sample Leaves", html)

        css_res = self.client.get("/static/css/style.css")
        self.assertEqual(css_res.status_code, 200)
        self.assertIn("#0b1914", css_res.get_data(as_text=True))
        css_res.close()

        js_res = self.client.get("/static/js/app.js")
        self.assertEqual(js_res.status_code, 200)
        self.assertIn("/predict", js_res.get_data(as_text=True))
        js_res.close()

    def test_02_api_status_endpoint(self):
        """Verify GET /api/status returns MobileNetV2 readiness metadata."""
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["model"], "MobileNetV2")
        self.assertEqual(payload["input_shape"], [224, 224, 3])
        self.assertGreaterEqual(payload["num_classes"], 5)

    def test_03_predict_multipart_upload_schema_and_classes(self):
        """
        Send each bundled sample leaf image to POST /predict via multipart upload
        and verify HTTP 200 JSON schema and accurate pathology detection.
        """
        valid_severities = {"Healthy", "Low", "Moderate", "High"}
        expected_disclaimer = (
            "Estimate only — validate with an agricultural expert before applying chemical treatments."
        )

        for filename, expected_label in self.sample_map.items():
            img_path = SAMPLE_DIR / filename
            self.assertTrue(img_path.exists(), f"Sample file missing: {img_path}")

            with open(img_path, "rb") as f:
                img_bytes = f.read()

            res = self.client.post(
                "/predict",
                data={"file": (io.BytesIO(img_bytes), filename)},
                content_type="multipart/form-data",
            )
            self.assertEqual(res.status_code, 200, f"Failed on {filename}: {res.get_data(as_text=True)}")
            data = res.get_json()

            # Verify required top-level keys
            for required_key in (
                "predicted_class",
                "confidence_score",
                "top_3_predictions",
                "severity_estimate",
                "recommended_treatment",
                "disclaimer",
            ):
                self.assertIn(required_key, data, f"Missing '{required_key}' in /predict output")

            # Verify predicted class matches the sample leaf type
            self.assertEqual(
                data["predicted_class"],
                expected_label,
                f"Expected '{expected_label}' for {filename}, got '{data['predicted_class']}'",
            )

            # Verify confidence score is a valid percentage
            self.assertIsInstance(data["confidence_score"], (int, float))
            self.assertGreater(data["confidence_score"], 40.0)
            self.assertLessEqual(data["confidence_score"], 100.0)

            # Verify Top-3 predictions structure
            top3 = data["top_3_predictions"]
            self.assertIsInstance(top3, list)
            self.assertEqual(len(top3), 3)
            self.assertEqual(top3[0]["label"], data["predicted_class"])
            for item in top3:
                self.assertIn("label", item)
                self.assertIn("confidence", item)

            # Verify severity estimate
            self.assertIn(data["severity_estimate"], valid_severities)

            # Verify treatment structure (organic & chemical steps)
            treatment = data["recommended_treatment"]
            self.assertIn("organic", treatment)
            self.assertIn("chemical", treatment)
            self.assertTrue(len(treatment["organic"]) >= 2)
            self.assertTrue(len(treatment["chemical"]) >= 1)

            # Verify exact disclaimer string
            self.assertEqual(data["disclaimer"], expected_disclaimer)

    def test_04_predict_error_handling_on_missing_file(self):
        """Verify POST /predict without an image returns HTTP 400 with clean JSON error."""
        res = self.client.post("/predict", data={})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertIn("error", data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
