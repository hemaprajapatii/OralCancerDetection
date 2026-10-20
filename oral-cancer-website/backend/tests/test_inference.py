"""
Unit tests for inference module: preprocessing, mock threshold boundaries,
score complement consistency, and one-time model caching.
"""

import io
import unittest
import numpy as np
from PIL import Image

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference import (
    preprocess_image,
    predict_image,
    load_inference_model,
    is_model_loaded,
    set_custom_model_for_testing,
    InvalidImageError,
    ModelNotLoadedError,
    DECISION_THRESHOLD
)


class MockPredictionModel:
    """Mock model that returns a predetermined sigmoid output for threshold testing."""
    def __init__(self, fixed_score: float):
        self.fixed_score = fixed_score
        self.input_shape = (None, 224, 224, 3)
        self.output_shape = (None, 1)

    def predict(self, x, verbose=0):
        return np.array([[self.fixed_score]], dtype=np.float32)


class TestInferencePreprocessing(unittest.TestCase):
    """Tests for in-memory image preprocessing pipeline."""

    def _create_test_image_bytes(self, mode="RGB", size=(300, 300), fmt="JPEG"):
        stream = io.BytesIO()
        img = Image.new(mode, size, color="blue" if mode != "L" else 128)
        img.save(stream, format=fmt)
        return stream.getvalue()

    def test_valid_rgb_image(self):
        img_bytes = self._create_test_image_bytes("RGB", (320, 240), "JPEG")
        arr = preprocess_image(img_bytes)
        self.assertEqual(arr.shape, (1, 224, 224, 3))
        self.assertEqual(arr.dtype, np.float32)
        self.assertTrue(0.0 <= arr.min() <= arr.max() <= 1.0)

    def test_grayscale_image_conversion(self):
        img_bytes = self._create_test_image_bytes("L", (200, 200), "PNG")
        arr = preprocess_image(img_bytes)
        self.assertEqual(arr.shape, (1, 224, 224, 3))
        self.assertEqual(arr.dtype, np.float32)

    def test_rgba_image_conversion(self):
        img_bytes = self._create_test_image_bytes("RGBA", (150, 150), "PNG")
        arr = preprocess_image(img_bytes)
        self.assertEqual(arr.shape, (1, 224, 224, 3))
        self.assertEqual(arr.dtype, np.float32)

    def test_webp_image(self):
        img_bytes = self._create_test_image_bytes("RGB", (250, 250), "WEBP")
        arr = preprocess_image(img_bytes)
        self.assertEqual(arr.shape, (1, 224, 224, 3))

    def test_empty_bytes_raises_error(self):
        with self.assertRaises(InvalidImageError):
            preprocess_image(b"")

    def test_corrupt_bytes_raises_error(self):
        with self.assertRaises(InvalidImageError):
            preprocess_image(b"this is corrupt non-image data 1234567890")

    def test_unsupported_format_raises_error(self):
        # Create a BMP image (not in allowed list JPEG, PNG, WEBP)
        stream = io.BytesIO()
        img = Image.new("RGB", (100, 100), color="red")
        img.save(stream, format="BMP")
        with self.assertRaises(InvalidImageError):
            preprocess_image(stream.getvalue())


class TestThresholdAndScoring(unittest.TestCase):
    """Tests for threshold boundary decisions and score complement logic using mock models."""

    def setUp(self):
        self.valid_img_bytes = io.BytesIO()
        Image.new("RGB", (100, 100), color="green").save(self.valid_img_bytes, format="JPEG")
        self.valid_img_bytes = self.valid_img_bytes.getvalue()

    def test_threshold_boundary_034_predicts_cancer(self):
        set_custom_model_for_testing(MockPredictionModel(0.34))
        result = predict_image(self.valid_img_bytes)
        self.assertEqual(result["prediction"], "CANCER")
        self.assertEqual(result["class_index"], 0)
        self.assertEqual(result["raw_sigmoid_score"], 0.34)
        self.assertAlmostEqual(result["cancer_score"], 0.66, places=4)
        self.assertAlmostEqual(result["non_cancer_score"], 0.34, places=4)

    def test_threshold_boundary_035_predicts_cancer(self):
        # s <= 0.35 must predict CANCER (class 0)
        set_custom_model_for_testing(MockPredictionModel(0.35))
        result = predict_image(self.valid_img_bytes)
        self.assertEqual(result["prediction"], "CANCER")
        self.assertEqual(result["class_index"], 0)
        self.assertEqual(result["raw_sigmoid_score"], 0.35)
        self.assertAlmostEqual(result["cancer_score"], 0.65, places=4)
        self.assertAlmostEqual(result["non_cancer_score"], 0.35, places=4)

    def test_threshold_boundary_036_predicts_non_cancer(self):
        # s > 0.35 must predict NON CANCER (class 1)
        set_custom_model_for_testing(MockPredictionModel(0.36))
        result = predict_image(self.valid_img_bytes)
        self.assertEqual(result["prediction"], "NON CANCER")
        self.assertEqual(result["class_index"], 1)
        self.assertEqual(result["raw_sigmoid_score"], 0.36)
        self.assertAlmostEqual(result["cancer_score"], 0.64, places=4)
        self.assertAlmostEqual(result["non_cancer_score"], 0.36, places=4)

    def test_extreme_00_predicts_cancer(self):
        set_custom_model_for_testing(MockPredictionModel(0.0))
        result = predict_image(self.valid_img_bytes)
        self.assertEqual(result["prediction"], "CANCER")
        self.assertAlmostEqual(result["cancer_score"], 1.0, places=4)
        self.assertAlmostEqual(result["non_cancer_score"], 0.0, places=4)

    def test_extreme_10_predicts_non_cancer(self):
        set_custom_model_for_testing(MockPredictionModel(1.0))
        result = predict_image(self.valid_img_bytes)
        self.assertEqual(result["prediction"], "NON CANCER")
        self.assertAlmostEqual(result["cancer_score"], 0.0, places=4)
        self.assertAlmostEqual(result["non_cancer_score"], 1.0, places=4)

    def test_score_complement_consistency(self):
        # In all cases, cancer_score + non_cancer_score must equal 1.0
        for test_s in [0.0, 0.15, 0.34, 0.35, 0.35001, 0.50, 0.85, 1.0]:
            set_custom_model_for_testing(MockPredictionModel(test_s))
            result = predict_image(self.valid_img_bytes)
            sum_scores = result["cancer_score"] + result["non_cancer_score"]
            self.assertAlmostEqual(sum_scores, 1.0, places=4)

    def test_model_not_loaded_raises_error(self):
        # Unset model
        import inference
        inference._loaded_model = None
        with self.assertRaises(ModelNotLoadedError):
            predict_image(self.valid_img_bytes)


class TestOneTimeModelLoading(unittest.TestCase):
    """Verifies that the model is cached and not reloaded on every call."""

    def test_caching_returns_same_instance(self):
        set_custom_model_for_testing(MockPredictionModel(0.5))
        m1 = load_inference_model()
        m2 = load_inference_model()
        self.assertIs(m1, m2)


if __name__ == "__main__":
    unittest.main()
