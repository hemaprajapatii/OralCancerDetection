"""
Real-model integration test: loads the actual saved H5 model and tests inference
against a real oral image from the dataset without modifying the dataset or model.
"""

import os
import unittest
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference import (
    load_inference_model,
    predict_image,
    is_model_loaded,
    get_default_model_path,
    DECISION_THRESHOLD
)


class TestRealModelInference(unittest.TestCase):
    """End-to-end inference verification using the real MobileNetV2 H5 model."""

    @classmethod
    def setUpClass(cls):
        # Load the real H5 model once
        model_path = get_default_model_path()
        if not os.path.isfile(model_path):
            raise unittest.SkipTest(f"Model file not found at: {model_path}")
        load_inference_model(model_path, force_reload=True)

    def test_real_model_is_loaded(self):
        self.assertTrue(is_model_loaded())

    def test_real_inference_on_sample_dataset_image(self):
        # Locate sample test image in source folder (read-only)
        test_cancer_dir = r"D:\Oral Cancer-20261019T114553Z-1-001\Oral Cancer\datasets\dataset2\OralCancer_new_Split\test\CANCER"
        if not os.path.isdir(test_cancer_dir):
            raise unittest.SkipTest("Source test dataset directory not available.")

        images = [f for f in os.listdir(test_cancer_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not images:
            raise unittest.SkipTest("No sample images found in test cancer directory.")

        sample_img_path = os.path.join(test_cancer_dir, images[0])
        with open(sample_img_path, "rb") as f:
            img_bytes = f.read()

        result = predict_image(img_bytes)

        # Validate result schema and values
        self.assertIn("prediction", result)
        self.assertIn(result["prediction"], ["CANCER", "NON CANCER"])
        self.assertIn(result["class_index"], [0, 1])
        self.assertTrue(0.0 <= result["raw_sigmoid_score"] <= 1.0)
        self.assertTrue(0.0 <= result["cancer_score"] <= 1.0)
        self.assertTrue(0.0 <= result["non_cancer_score"] <= 1.0)
        self.assertEqual(result["threshold"], DECISION_THRESHOLD)

        # Confirm mathematical complement relationship
        self.assertAlmostEqual(result["cancer_score"] + result["non_cancer_score"], 1.0, places=4)

        print(f"\n[Real Model Verification] File: {images[0]} | Sigmoid: {result['raw_sigmoid_score']} | Cancer Score: {result['cancer_score']} | Prediction: {result['prediction']}")


if __name__ == "__main__":
    unittest.main()
