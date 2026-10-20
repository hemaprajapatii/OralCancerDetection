"""
API integration tests for Flask REST endpoints /api/health and /api/predict.
"""

import io
import unittest
from PIL import Image

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import create_app
import inference
from tests.test_inference import MockPredictionModel


class TestFlaskAPI(unittest.TestCase):
    """Test suite for Flask endpoints /api/health and /api/predict."""

    def setUp(self):
        # Create app without auto-loading the large model file for unit tests
        self.app = create_app(load_model_on_init=False)
        self.client = self.app.test_client()

        # Inject a predictable mock model
        self.mock_model = MockPredictionModel(fixed_score=0.20)
        inference.set_custom_model_for_testing(self.mock_model)

        # Generate a valid synthetic JPEG
        self.valid_jpeg_stream = io.BytesIO()
        Image.new("RGB", (100, 100), color="red").save(self.valid_jpeg_stream, format="JPEG")
        self.valid_jpeg_bytes = self.valid_jpeg_stream.getvalue()

    def test_health_endpoint_healthy(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["decision_threshold"], 0.35)
        self.assertIn("JPEG", data["allowed_formats"])

    def test_health_endpoint_degraded_when_model_unloaded(self):
        inference._loaded_model = None
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "degraded")
        self.assertFalse(data["model_loaded"])

    def test_predict_service_unavailable_when_model_not_loaded(self):
        inference._loaded_model = None
        data = {"image": (io.BytesIO(self.valid_jpeg_bytes), "sample.jpg")}
        response = self.client.post("/api/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 503)
        res_json = response.get_json()
        self.assertFalse(res_json["success"])

    def test_predict_missing_file_field(self):
        # Post multipart without 'image' key
        response = self.client.post("/api/predict", data={"other_field": "val"}, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("No image file provided", data["error"])

    def test_predict_empty_filename(self):
        data = {"image": (io.BytesIO(self.valid_jpeg_bytes), "")}
        response = self.client.post("/api/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("empty filename", data["error"])

    def test_predict_empty_file_content(self):
        data = {"image": (io.BytesIO(b""), "empty.jpg")}
        response = self.client.post("/api/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("empty", data["error"])

    def test_predict_unsupported_file_extension(self):
        data = {"image": (io.BytesIO(self.valid_jpeg_bytes), "document.pdf")}
        response = self.client.post("/api/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("Unsupported file extension", data["error"])

    def test_predict_corrupt_image_content(self):
        corrupt_bytes = b"not an actual image file"
        data = {"image": (io.BytesIO(corrupt_bytes), "corrupt.jpg")}
        response = self.client.post("/api/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("Could not decode image", data["error"])

    def test_predict_oversized_payload(self):
        # Simulate payload exceeding 10MB limit
        oversized_bytes = b"0" * (11 * 1024 * 1024)
        data = {"image": (io.BytesIO(oversized_bytes), "huge.jpg")}
        response = self.client.post("/api/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 413)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("too large", data["error"])

    def test_predict_valid_image_success(self):
        # Mock score 0.20 <= 0.35 -> CANCER
        inference.set_custom_model_for_testing(MockPredictionModel(0.20))
        data = {"image": (io.BytesIO(self.valid_jpeg_bytes), "oral_photo.jpg")}
        response = self.client.post("/api/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["prediction"], "CANCER")
        self.assertEqual(data["class_index"], 0)
        self.assertEqual(data["raw_sigmoid_score"], 0.20)
        self.assertAlmostEqual(data["cancer_score"], 0.80, places=4)
        self.assertAlmostEqual(data["non_cancer_score"], 0.20, places=4)
        self.assertEqual(data["threshold"], 0.35)
        self.assertIn("research model", data["note"])

    def test_predict_valid_image_non_cancer(self):
        # Mock score 0.85 > 0.35 -> NON CANCER
        inference.set_custom_model_for_testing(MockPredictionModel(0.85))
        data = {"image": (io.BytesIO(self.valid_jpeg_bytes), "oral_photo.png")}
        response = self.client.post("/api/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["prediction"], "NON CANCER")
        self.assertEqual(data["class_index"], 1)
        self.assertAlmostEqual(data["cancer_score"], 0.15, places=4)
        self.assertAlmostEqual(data["non_cancer_score"], 0.85, places=4)

    def test_cors_headers_allowed_origin(self):
        headers = {"Origin": "http://localhost:5173"}
        response = self.client.get("/api/health", headers=headers)
        self.assertEqual(response.headers.get("Access-Control-Allow-Origin"), "http://localhost:5173")

    def test_cors_headers_disallowed_origin(self):
        headers = {"Origin": "http://malicious-site.com"}
        response = self.client.get("/api/health", headers=headers)
        self.assertNotEqual(response.headers.get("Access-Control-Allow-Origin"), "http://malicious-site.com")


if __name__ == "__main__":
    unittest.main()
