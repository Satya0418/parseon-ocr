"""
=============================================================
predict.py  —  Inference: Extract Text from a New Image
=============================================================

HOW INFERENCE WORKS:
---------------------
Inference = using the TRAINED model to make predictions on NEW images.

The pipeline is:
  1. Load a new image from disk
  2. Preprocess it (same steps as during training — CRITICAL!)
  3. Run it through the inference model (image → probability matrix)
  4. Decode the probability matrix into text (CTC beam/greedy)
  5. Return the extracted text

WHY IS STEP 2 CRITICAL?
The model learned from images in a specific format (64×512, grayscale,
normalized 0–1, text pixels = bright). If you feed it an image in a
different format, the model sees something it was never trained on →
garbage output. ALWAYS apply the exact same preprocessing as training.

USAGE FROM COMMAND LINE:
  python backend/inference/predict.py \
      --image_path sample_images/test.png \
      --model_path saved_models/crnn_iam_v1_inference.keras

USAGE FROM PYTHON:
  from backend.inference.predict import OCRPredictor
  predictor = OCRPredictor("saved_models/crnn_iam_v1_inference.keras")
  text = predictor.predict("my_image.png")
  print(text)
=============================================================
"""

import os
import sys
import argparse
import numpy as np
import cv2
import tensorflow as tf
import h5py

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from backend.preprocessing.image_processor import preprocess_image, preprocess_from_array
from backend.utils.char_map                import build_char_maps, ALPHABET, BLANK_INDEX
from backend.inference.decoder             import decode_batch, greedy_decode


def _rebuild_inference_model_from_legacy_weights(model_path: str) -> tf.keras.Model:
    """Rebuild an inference .keras model using legacy Keras 2.x weight files."""
    from keras.src.legacy.saving import legacy_h5_format
    from backend.models.crnn_model import build_crnn_model, build_inference_model

    save_dir = os.path.dirname(model_path)
    candidates = [
        os.path.join(save_dir, "crnn_camera_local_best.weights.h5"),
        os.path.join(save_dir, "crnn_camera_local_final.weights.h5"),
    ]

    legacy_weights = None
    for path in candidates:
        if os.path.isfile(path):
            legacy_weights = path
            break

    if legacy_weights is None:
        raise FileNotFoundError(
            "Could not rebuild inference model: no legacy weights found in saved_models. "
            "Expected one of: crnn_camera_local_best.weights.h5, crnn_camera_local_final.weights.h5"
        )

    print(f"[Predictor] Rebuilding inference model from legacy weights: {legacy_weights}")
    training_model = build_crnn_model(lstm_units=256, dropout_rate=0.25)

    with h5py.File(legacy_weights, "r") as h5f:
        legacy_h5_format.load_weights_from_hdf5_group(h5f, training_model)

    inference_model = build_inference_model(training_model)
    inference_model.save(model_path)
    print(f"[Predictor] Rebuilt and saved valid inference model: {model_path}")
    return inference_model


class OCRPredictor:
    """
    High-level OCR predictor class.

    WHY A CLASS?
    Encapsulating state (loaded model, char maps) in a class means you
    load the model ONCE and reuse it for many predictions without the
    overhead of reloading each time. Ideal for a web server or batch processing.

    Usage
    -----
    predictor = OCRPredictor("saved_models/crnn_iam_v1_inference.keras")
    text = predictor.predict("path/to/my/image.png")
    """

    def __init__(self, model_path: str, decode_method: str = "greedy"):
        """
        Load the trained model and character maps.

        Parameters
        ----------
        model_path    : str  — path to the saved inference model (.keras file)
        decode_method : str  — 'greedy' (fast) or 'beam' (more accurate)
        """
        print(f"[Predictor] Loading model from: {model_path}")
        try:
            self.model = tf.keras.models.load_model(model_path)
        except (ValueError, OSError) as e:
            msg = str(e)
            if "accessible `.keras` zip file" not in msg:
                raise
            print("[Predictor] Detected invalid .keras archive. Attempting automatic recovery...")
            self.model = _rebuild_inference_model_from_legacy_weights(model_path)

        self.decode_method = decode_method

        # Build character maps using the same convention as training.
        self.char_to_int, self.int_to_char = build_char_maps(ALPHABET)

        print(f"[Predictor] Model loaded (using training-consistent char mapping)")
        print(f"[Predictor] Decode method: {decode_method}")
        print(f"[Predictor] Ready to extract text from images.")

    def predict(self, image_path: str) -> str:
        """
        Extract text from an image file.

        Parameters
        ----------
        image_path : str  — path to the image file (.png, .jpg, .tif)

        Returns
        -------
        str  — the extracted text string

        Raises
        ------
        FileNotFoundError if image_path does not exist.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Step 1: Preprocess image (same pipeline as training!)
        img = preprocess_image(image_path)   # shape: (H, W, 1), float32, 0-1

        # Step 2: Add batch dimension  (H, W, 1) → (1, H, W, 1)
        img_batch = np.expand_dims(img, axis=0)

        # Step 3: Run the model
        # predictions shape: (1, time_steps, num_classes)
        predictions = self.model.predict(img_batch, verbose=0)

        # Step 4: Decode predictions to text
        texts = decode_batch(
            predictions,
            self.int_to_char,
            method=self.decode_method,
            blank_index=BLANK_INDEX,
        )

        return texts[0]  # return the single decoded string

    def predict_from_array(self, img_array: np.ndarray) -> str:
        """
        Extract text from an already-loaded image array.

        WHY: When integrating with a web backend that receives image bytes
        rather than file paths, we preprocess the array directly.

        Parameters
        ----------
        img_array : np.ndarray  — raw image array (BGR, RGB, or grayscale)

        Returns
        -------
        str  — the extracted text string
        """
        # Preprocess the array through the same pipeline
        img = preprocess_from_array(img_array)       # (H, W, 1), float32
        img_batch = np.expand_dims(img, axis=0)      # (1, H, W, 1)

        predictions = self.model.predict(img_batch, verbose=0)
        texts = decode_batch(
            predictions,
            self.int_to_char,
            method=self.decode_method,
            blank_index=BLANK_INDEX,
        )
        return texts[0]

    def predict_batch(self, image_paths: list) -> list:
        """
        Extract text from a batch of image files at once.

        WHY: Processing images in batches is significantly faster than
        one at a time because the GPU handles multiple samples in parallel.

        Parameters
        ----------
        image_paths : list of str  — list of image file paths

        Returns
        -------
        list of str  — one extracted text string per image
        """
        imgs = []
        valid_paths = []

        for path in image_paths:
            if not os.path.exists(path):
                print(f"[Predictor] WARNING: Missing file — {path}")
                continue
            img = preprocess_image(path)
            imgs.append(img)
            valid_paths.append(path)

        if not imgs:
            return []

        # Stack into a batch: (N, H, W, 1)
        img_batch   = np.stack(imgs, axis=0)
        predictions = self.model.predict(img_batch, verbose=0)
        texts       = decode_batch(
            predictions,
            self.int_to_char,
            method=self.decode_method,
            blank_index=BLANK_INDEX,
        )

        return list(zip(valid_paths, texts))

    def visualize_prediction(self, image_path: str, save_path: str = None):
        """
        Show the image alongside its OCR prediction.

        WHY: A quick visual debug tool — if the prediction looks wrong,
        you can immediately see whether the image itself is the problem
        (bad preprocessing) or a model accuracy issue.

        Parameters
        ----------
        image_path : str        — path to the image
        save_path  : str | None — if provided, save the visualization to this path;
                                  otherwise, open in a window
        """
        import matplotlib
        matplotlib.use("Agg" if save_path else "TkAgg")
        import matplotlib.pyplot as plt

        # Predict
        predicted_text = self.predict(image_path)

        # Load original image for display
        original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        fig, axes = plt.subplots(1, 2, figsize=(14, 3))

        axes[0].imshow(original, cmap="gray")
        axes[0].set_title("Input Image", fontsize=11)
        axes[0].axis("off")

        # Preprocessed view
        preprocessed = preprocess_image(image_path)
        axes[1].imshow(preprocessed[:, :, 0], cmap="gray")
        axes[1].set_title(f"Preprocessed  |  Predicted: \"{predicted_text}\"", fontsize=11)
        axes[1].axis("off")

        plt.suptitle("OCR Inference Result", fontsize=13, fontweight="bold")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[Predictor] Saved visualization → {save_path}")
        else:
            plt.show()

        plt.close()
        return predicted_text


# ── Command-line interface ─────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Run OCR inference on an image")
    p.add_argument("--image_path",  required=True,  help="Path to input image file")
    p.add_argument("--model_path",  required=True,  help="Path to inference model (.keras)")
    p.add_argument("--method",      default="greedy", choices=["greedy", "beam"],
                   help="Decoding method: greedy (fast) or beam (accurate)")
    p.add_argument("--save_viz",    default=None,
                   help="Optional: save visualization image to this path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    predictor = OCRPredictor(args.model_path, decode_method=args.method)

    if args.save_viz:
        text = predictor.visualize_prediction(args.image_path, save_path=args.save_viz)
    else:
        text = predictor.predict(args.image_path)

    print("\n" + "=" * 60)
    print(f"  EXTRACTED TEXT:")
    print(f"  \"{text}\"")
    print("=" * 60)