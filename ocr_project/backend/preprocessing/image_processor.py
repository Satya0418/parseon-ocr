"""
=============================================================
image_processor.py  —  Image Preprocessing Pipeline
=============================================================

WHY DO WE NEED THIS?
--------------------
Raw images from the IAM dataset cannot be fed directly into the model.
Problems with raw images:
  1. Different sizes  → model needs a FIXED input shape
  2. Color/BGR       → OCR doesn't need color, grayscale is enough
  3. Pixel values 0–255 → neural networks train better with values 0.0–1.0
  4. Dark background / noise → needs cleaning (binarization)

This module standardizes every image to:
  Shape  : (HEIGHT, WIDTH, 1)  — single grayscale channel
  Values : float32 in [0.0, 1.0]
  Width  : padded to fixed width (not squeezed, to preserve aspect ratio)

The IAM images are HANDWRITTEN lines of text, so we process them
as full-line images (not cropped individual words by default).
=============================================================
"""

import cv2
import numpy as np
from PIL import Image


# ── Default dimensions ─────────────────────────────────────────────
# Tuned for IAM WORD-level images (single words, not full lines).
# HEIGHT = 64:  tall enough to capture ascenders/descenders
# WIDTH  = 256: wide enough for the longest handwritten words.
#               (Full-line width was 512; halved because words are shorter)
# CNN reduces width by 4× (two 2×2 MaxPools + one 2×1 MaxPool)
# → 256 // 4 = 64 time steps fed into the LSTM (plenty for words ≤ 25 chars)
IMG_HEIGHT = 64
IMG_WIDTH  = 256


def read_image(image_path: str) -> np.ndarray:
    """
    Read an image from disk and return it as a NumPy array.

    WHY: All subsequent processing works on NumPy arrays (not file paths).
    OpenCV reads images in BGR format by default, so we convert to grayscale
    immediately since color information is irrelevant for OCR.

    Parameters
    ----------
    image_path : str
        Absolute or relative path to the image file (.png, .jpg, .tif, etc.)

    Returns
    -------
    np.ndarray  shape (H, W), dtype uint8, values 0–255
        Grayscale image. Returns None if file not found.
    """
    # Normalize path separators for cross-platform compatibility
    image_path = image_path.replace('\\', '/')
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return None

    return img


def binarize_image(img: np.ndarray, method: str = "otsu") -> np.ndarray:
    """
    Convert a grayscale image to a binary (black & white) image.

    WHY: Handwritten images have varying ink density, shadows, and paper
    texture. Binarization removes background noise and makes ink strokes
    crisp and clear, which helps the model focus on the actual text shapes.

    METHODS:
      - 'otsu'     : Automatically finds the best threshold (recommended)
      - 'adaptive' : Uses local neighborhood thresholds — better for
                     images with uneven lighting
      - 'simple'   : Fixed threshold at 127 — fast but less robust

    Parameters
    ----------
    img    : np.ndarray  — grayscale image (values 0–255)
    method : str         — binarization method: 'otsu', 'adaptive', 'simple'

    Returns
    -------
    np.ndarray  — binary image (values: 0 or 255)
    """
    if method == "otsu":
        # Otsu's algorithm automatically finds the ideal threshold
        # by minimizing intra-class variance between ink and paper pixels
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif method == "adaptive":
        # Adaptive thresholding computes a threshold for each small
        # neighborhood block, handling uneven illumination well
        binary = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,   # Neighborhood size (must be odd)
            C=2             # Constant subtracted from the mean
        )

    elif method == "simple":
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    else:
        raise ValueError(f"Unknown binarization method: '{method}'")

    return binary


def resize_and_pad(img: np.ndarray,
                   target_height: int = IMG_HEIGHT,
                   target_width:  int = IMG_WIDTH) -> np.ndarray:
    """
    Resize an image to a fixed height, then pad the width to target_width.

    WHY: The CRNN model requires a fixed input shape. Instead of squeezing
    the image (which distorts character proportions), we:
      1. Resize to fixed HEIGHT while keeping aspect ratio
      2. Pad the right side with white (255) to reach target_width
      3. Crop if the image is wider than target_width

    This preserves the relative widths of characters — critical because
    narrow letters (i, l) and wide letters (m, w) must look different.

    Parameters
    ----------
    img           : np.ndarray  — input grayscale image
    target_height : int         — output image height in pixels
    target_width  : int         — output image width in pixels

    Returns
    -------
    np.ndarray  — resized + padded image, shape (target_height, target_width)
    """
    h, w = img.shape[:2]

    # Step 1: Scale height to target_height, maintain aspect ratio
    scale  = target_height / h
    new_w  = int(w * scale)
    resized = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA)

    # Step 2: If wider than target, crop; if narrower, pad with white on the right
    if new_w >= target_width:
        final = resized[:, :target_width]     # crop excess width
    else:
        # Create a white canvas and paste the image on the left
        pad_width = target_width - new_w
        padding   = np.ones((target_height, pad_width), dtype=np.uint8) * 255
        final     = np.concatenate([resized, padding], axis=1)

    return final


def normalize(img: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values from [0, 255] to [0.0, 1.0].

    WHY: Neural networks are sensitive to the scale of input values.
    Large values (0–255) cause larger gradients during backpropagation,
    making training unstable. Normalizing to 0–1 keeps gradients small
    and consistent.

    Additionally, we INVERT the image so that:
      - Text pixels  = HIGH values (close to 1.0)  — model attends to them
      - Background   = LOW values  (close to 0.0)  — model ignores them

    Parameters
    ----------
    img : np.ndarray  — uint8 image (values 0–255)

    Returns
    -------
    np.ndarray  — float32 array, values in [0.0, 1.0], text=bright
    """
    # Invert: 0 (black ink) → 255 → after division → 1.0 (bright)
    #        255 (white bg) →   0 → after division → 0.0 (dark)
    img_float = (255.0 - img.astype(np.float32)) / 255.0
    return img_float


def add_channel_dim(img: np.ndarray) -> np.ndarray:
    """
    Add a channel dimension: (H, W) → (H, W, 1).

    WHY: TensorFlow/Keras Conv2D layers expect input shape (H, W, C)
    where C is the number of channels. For grayscale, C=1.
    Without this, the model will raise a shape mismatch error.

    Parameters
    ----------
    img : np.ndarray  — shape (H, W)

    Returns
    -------
    np.ndarray  — shape (H, W, 1)
    """
    return np.expand_dims(img, axis=-1)


def preprocess_image(image_path: str,
                     target_height: int = IMG_HEIGHT,
                     target_width:  int = IMG_WIDTH,
                     binarize:      bool = True,
                     binarize_method: str = "otsu") -> np.ndarray:
    """
    Full preprocessing pipeline for a single image.

    WHY: This is the single entry point that chains all preprocessing steps
    in the correct order. Every image — whether training or inference —
    must go through exactly these same steps, in this exact order.

    PIPELINE:
        Read → (Binarize) → Resize+Pad → Normalize → Add Channel

    Parameters
    ----------
    image_path       : str   — path to image file
    target_height    : int   — output height (default 64)
    target_width     : int   — output width  (default 512)
    binarize         : bool  — apply binarization? (True for most cases)
    binarize_method  : str   — 'otsu', 'adaptive', or 'simple'

    Returns
    -------
    np.ndarray  — shape (target_height, target_width, 1), dtype float32
                  Values 0.0–1.0, text=1.0, background=0.0
    """
    # Step 1: Read image as grayscale
    img = read_image(image_path)
    if img is None:
        # Return a blank image as fallback for missing files
        return np.zeros((target_height, target_width, 1), dtype=np.float32)

    # Step 2: Binarize (remove noise, sharpen ink strokes)
    if binarize:
        img = binarize_image(img, method=binarize_method)

    # Step 3: Resize to fixed height + pad width
    img = resize_and_pad(img, target_height, target_width)

    # Step 4: Normalize to [0.0, 1.0] with inversion
    img = normalize(img)

    # Step 5: Add channel dimension for Keras compatibility
    img = add_channel_dim(img)

    return img.astype(np.float32)


def preprocess_from_array(img_array: np.ndarray,
                           target_height: int = IMG_HEIGHT,
                           target_width:  int = IMG_WIDTH,
                           binarize:      bool = True,
                           binarize_method: str = "otsu") -> np.ndarray:
    """
    Same as preprocess_image() but takes an array instead of a file path.

    WHY: During inference (and some augmentation pipelines), we already have
    the image loaded in memory. Re-reading from disk would be wasteful.

    Parameters
    ----------
    img_array : np.ndarray  — already-loaded grayscale image (H, W)
    (all other params same as preprocess_image)

    Returns
    -------
    np.ndarray  — shape (target_height, target_width, 1), dtype float32
    """
    img = img_array.copy()

    # Convert to grayscale if accidentally passed a color image
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if binarize:
        img = binarize_image(img, method=binarize_method)

    img = resize_and_pad(img, target_height, target_width)
    img = normalize(img)
    img = add_channel_dim(img)

    return img.astype(np.float32)


# ── Quick self-test ───────────────────────────────────────────────
if __name__ == "__main__":
    import os

    print("Image Processor Self-Test")
    print("-" * 40)

    # Create a dummy test image (white with some black text simulation)
    dummy = np.ones((100, 400), dtype=np.uint8) * 240
    dummy[20:80, 30:370] = 50   # simulate dark text region

    result = preprocess_from_array(dummy)
    print(f"Input shape  : {dummy.shape}")
    print(f"Output shape : {result.shape}   (expected {IMG_HEIGHT}, {IMG_WIDTH}, 1)")
    print(f"Value range  : [{result.min():.3f}, {result.max():.3f}]")
    print(f"dtype        : {result.dtype}")
    print("Self-test PASSED" if result.shape == (IMG_HEIGHT, IMG_WIDTH, 1) else "FAILED")
