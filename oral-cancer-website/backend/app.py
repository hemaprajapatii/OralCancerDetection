"""
Flask REST API for Standalone Oral Cancer Detection Research Website.
Exposes /api/health and /api/predict endpoints with strict validation and CORS.
"""

import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge

from inference import (
    load_inference_model,
    predict_image,
    is_model_loaded,
    ModelNotLoadedError,
    InvalidImageError,
    DECISION_THRESHOLD
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Constants
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 Megabytes limit
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

ALLOWED_FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]


def create_app(load_model_on_init: bool = True) -> Flask:
    """Application factory for Flask inference API."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    CORS(
        app,
        resources={r"/api/*": {"origins": ALLOWED_FRONTEND_ORIGINS}},
        methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"]
    )

    # Load model during application startup if requested
    if load_model_on_init:
        try:
            load_inference_model()
            logger.info("Inference model ready on application startup.")
        except Exception as exc:
            logger.error("Failed to load model on application startup: %s", exc)


    @app.errorhandler(RequestEntityTooLarge)
    def handle_oversized_request(error):
        return jsonify({
            "success": False,
            "error": "Uploaded file is too large. Maximum allowed size is 10 MB."
        }), 413

    @app.errorhandler(500)
    def handle_internal_error(error):
        logger.error("Unhandled internal server error: %s", error)
        return jsonify({
            "success": False,
            "error": "An internal server error occurred while processing the request."
        }), 500

    @app.route("/api/health", methods=["GET"])
    def health_check():
        """Health check endpoint confirming API status and model readiness."""
        loaded = is_model_loaded()
        return jsonify({
            "status": "healthy" if loaded else "degraded",
            "model_loaded": loaded,
            "model_architecture": "MobileNetV2 (Without ROI)",
            "decision_threshold": DECISION_THRESHOLD,
            "allowed_formats": ["JPEG", "PNG", "WEBP"],
            "max_file_size_mb": 10
        }), 200

    @app.route("/api/predict", methods=["POST"])
    def predict():
        """
        Receives an uploaded oral image, performs preprocessing and inference,
        and returns predicted class and confidence scores.
        """
        # Ensure model is ready
        if not is_model_loaded():
            return jsonify({
                "success": False,
                "error": "Inference model is not currently loaded. Please check server health."
            }), 503

        # Validate multipart form-data presence
        if "image" not in request.files:
            return jsonify({
                "success": False,
                "error": "No image file provided. Please include an 'image' file field in multipart request."
            }), 400

        file = request.files["image"]
        filename = (file.filename or "").strip()

        if not filename:
            return jsonify({
                "success": False,
                "error": "Selected file has an empty filename."
            }), 400

        # Validate file extension
        _, ext = os.path.splitext(filename.lower())
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({
                "success": False,
                "error": f"Unsupported file extension '{ext}'. Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
            }), 400

        # Read image content into memory
        try:
            image_bytes = file.read()
        except Exception as exc:
            logger.warning("Failed to read uploaded stream: %s", exc)
            return jsonify({
                "success": False,
                "error": "Could not read uploaded file content."
            }), 400

        if not image_bytes or len(image_bytes) == 0:
            return jsonify({
                "success": False,
                "error": "Uploaded image file is empty."
            }), 400

        # Perform inference in memory
        try:
            result = predict_image(image_bytes)
        except InvalidImageError as err:
            return jsonify({
                "success": False,
                "error": str(err)
            }), 400
        except ModelNotLoadedError as err:
            return jsonify({
                "success": False,
                "error": "Inference service is currently unavailable."
            }), 503
        except Exception as exc:
            logger.error("Unexpected error during inference: %s", exc)
            return jsonify({
                "success": False,
                "error": "Inference failed due to an unexpected processing error."
            }), 500

        # Return standardized response with medical research disclaimer note
        return jsonify({
            "success": True,
            "prediction": result["prediction"],
            "class_index": result["class_index"],
            "raw_sigmoid_score": result["raw_sigmoid_score"],
            "cancer_score": result["cancer_score"],
            "non_cancer_score": result["non_cancer_score"],
            "threshold": result["threshold"],
            "note": "This prediction is derived from an experimental research model and must not be used as a clinical diagnosis."
        }), 200

    return app


# Default app instance for WSGI/Gunicorn/Flask CLI
app = create_app()

if __name__ == "__main__":
    # Run locally on port 5000
    app.run(host="127.0.0.1", port=5000, debug=False)
