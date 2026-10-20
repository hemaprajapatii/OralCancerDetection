"""
Inference module for Oral Cancer Detection using fine-tuned MobileNetV2 (Without ROI).
Loads the model once at startup and performs in-memory image preprocessing and inference.
"""

import io
import os
import logging
from typing import Dict, Any, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Constants matching training configuration
IMG_SIZE = 224
DECISION_THRESHOLD = 0.35
ALLOWED_PIL_FORMATS = {"JPEG", "PNG", "WEBP"}

# Class index mapping from training generator:
# train_gen.class_indices = {'CANCER': 0, 'NON CANCER': 1}
CLASS_NAMES = {
    0: "CANCER",
    1: "NON CANCER"
}

# Module-level model cache (loaded once at startup)
_loaded_model = None
_model_path_used = None


class InferenceError(Exception):
    """Base exception for inference errors."""
    pass


class ModelNotLoadedError(InferenceError):
    """Raised when inference is attempted before the model is loaded."""
    pass


class InvalidImageError(InferenceError):
    """Raised when an uploaded file cannot be parsed as a valid image."""
    pass


def get_default_model_path() -> str:
    """Returns the default absolute path to the saved H5 model."""
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(backend_dir, "models", "WR_oral_cancer_mobilenetv2_finetuned.h5")


def load_inference_model(model_path: Optional[str] = None, force_reload: bool = False):
    """
    Loads the Keras model once for inference with compile=False.
    Validates input and output tensor shapes.
    """
    global _loaded_model, _model_path_used

    if _loaded_model is not None and not force_reload:
        return _loaded_model

    path = model_path or get_default_model_path()
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found at expected location: {path}")

    # Lazy import tensorflow only when loading model to keep imports clean
    import tensorflow as tf

    logger.info("Loading inference model from: %s", path)
    model = tf.keras.models.load_model(path, compile=False)

    # Validate model architecture contract
    input_shape = model.input_shape
    output_shape = model.output_shape

    # Expected: (None, 224, 224, 3) and (None, 1)
    if len(input_shape) != 4 or input_shape[1:4] != (IMG_SIZE, IMG_SIZE, 3):
        raise ValueError(f"Unexpected model input shape: {input_shape}. Expected: (None, {IMG_SIZE}, {IMG_SIZE}, 3)")

    if len(output_shape) != 2 or output_shape[1] != 1:
        raise ValueError(f"Unexpected model output shape: {output_shape}. Expected: (None, 1)")

    _loaded_model = model
    _model_path_used = path
    logger.info("Model successfully loaded and validated. Input: %s, Output: %s", input_shape, output_shape)
    return _loaded_model


def set_custom_model_for_testing(model):
    """Allows test fixtures to inject a mock or test model without loading H5 file."""
    global _loaded_model, _model_path_used
    _loaded_model = model
    _model_path_used = "mock_model"


def is_model_loaded() -> bool:
    """Returns True if the inference model is currently loaded in memory."""
    return _loaded_model is not None


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Validates, converts, and normalizes an image from raw bytes:
    1. Reads image into memory via PIL (no disk writes).
    2. Verifies format (JPEG, PNG, WEBP).
    3. Converts to RGB (handles RGBA, grayscale, palette).
    4. Resizes to 224x224 using Bilinear interpolation (without ROI cropping).
    5. Normalizes pixel values by 1/255.0 to [0.0, 1.0] (float32).
    6. Expands batch dimension to (1, 224, 224, 3).
    """
    if not image_bytes:
        raise InvalidImageError("Image payload is empty.")

    try:
        stream = io.BytesIO(image_bytes)
        img = Image.open(stream)
        
        # Verify valid image by loading image raster data
        img.load()
        raw_format = (img.format or "").upper()
        if raw_format not in ALLOWED_PIL_FORMATS and raw_format != "MPO":
            # Allow MPO if it decodes as JPEG (common on some mobile cameras)
            if raw_format != "JPEG":
                raise InvalidImageError(f"Unsupported image format: '{raw_format}'. Allowed: JPEG, PNG, WEBP.")

        # Convert to RGB (guarantees 3 color channels)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize to standard input dimensions without ROI cropping
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)

        # Convert to numpy array and normalize to [0.0, 1.0] float32
        img_array = np.array(img, dtype=np.float32) / 255.0

        # Expand batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
        img_batch = np.expand_dims(img_array, axis=0)
        return img_batch

    except InvalidImageError:
        raise
    except Exception as exc:
        logger.warning("Failed to decode image bytes: %s", exc)
        raise InvalidImageError("Could not decode image. The file may be corrupt or not a valid image.") from exc


def predict_image(image_bytes: bytes) -> Dict[str, Any]:
    """
    Performs inference on the uploaded image bytes:
    1. Preprocesses image to (1, 224, 224, 3).
    2. Runs model prediction without gradient tracking.
    3. Interprets raw sigmoid score:
       - train_gen.class_indices: CANCER = 0, NON CANCER = 1
       - sigmoid output s is P(NON CANCER)
       - If s > 0.35: predict NON CANCER (class 1)
       - If s <= 0.35: predict CANCER (class 0)
       - non_cancer_score = s
       - cancer_score = 1.0 - s
    """
    if _loaded_model is None:
        raise ModelNotLoadedError("Inference model is not loaded. Call load_inference_model() first.")

    input_tensor = preprocess_image(image_bytes)

    # Perform inference (verbose=0 suppresses stdout noise)
    raw_output = _loaded_model.predict(input_tensor, verbose=0)
    raw_sigmoid = float(raw_output[0][0])

    # Bound check to avoid numerical issues
    raw_sigmoid = max(0.0, min(1.0, raw_sigmoid))

    # Apply verified threshold and class mapping
    if raw_sigmoid > DECISION_THRESHOLD:
        predicted_class = CLASS_NAMES[1]  # "NON CANCER"
        class_index = 1
    else:
        predicted_class = CLASS_NAMES[0]  # "CANCER"
        class_index = 0

    non_cancer_score = raw_sigmoid
    cancer_score = 1.0 - raw_sigmoid

    return {
        "prediction": predicted_class,
        "class_index": class_index,
        "raw_sigmoid_score": round(raw_sigmoid, 5),
        "cancer_score": round(cancer_score, 5),
        "non_cancer_score": round(non_cancer_score, 5),
        "threshold": DECISION_THRESHOLD
    }
