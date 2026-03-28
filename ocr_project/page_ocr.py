"""
=============================================================
page_ocr.py  —  Full-Page OCR Pipeline
=============================================================

PURPOSE:
--------
Extract ALL text from a full-page document image containing
multiple lines or paragraphs of text.

The trained CRNN model only works on single-line/word images.
This pipeline bridges the gap by:
  1. Detecting all text regions on the page using computer vision
  2. Cropping each text region
  3. Preprocessing each crop to match training format
  4. Running the CRNN model on each crop
  5. Combining results into the full page text

NO EXTERNAL OCR ENGINES ARE USED.
Everything is built with OpenCV + NumPy + TensorFlow/Keras.

USAGE:
------
  python page_ocr.py --image path/to/full_page.png

  Or from Python:
    from page_ocr import extract_text_from_page
    text = extract_text_from_page("path/to/full_page.png")
    print(text)
=============================================================
"""

import os
import sys
import argparse
import difflib
import numpy as np
import cv2
import tensorflow as tf
import h5py

# ── Project imports ────────────────────────────────────────────────
# Add project root to path so backend modules can be imported
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from backend.preprocessing.image_processor import (
    IMG_HEIGHT, IMG_WIDTH,
    preprocess_from_array, binarize_image, resize_and_pad, normalize, add_channel_dim
)
from backend.utils.char_map import (
    ALPHABET, BLANK_INDEX, build_char_maps
)
from backend.inference.decoder import decode_batch


# ── Default paths and constants ────────────────────────────────────
DEFAULT_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "saved_models", "crnn_iam_v1_inference.keras"
)

# Check if model file exists
if not os.path.isfile(DEFAULT_MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {DEFAULT_MODEL_PATH}\nPlease ensure the file exists and the path is correct.")

# ── Minimum bounding box dimensions to filter noise ───────────────
# Contours smaller than these are treated as noise, not text.
# These values work well for typical document scans (200-300 DPI).
MIN_REGION_WIDTH  = 20   # pixels — filters tiny specks
MIN_REGION_HEIGHT = 18   # pixels — filters thin horizontal noise strips

# ── Line grouping tolerance ───────────────────────────────────────
# Two bounding boxes whose vertical centers are within this fraction
# of the average box height are considered to be on the same line.
LINE_GROUP_TOLERANCE = 0.6

# Filters for noisy word crops during full-page extraction.
MIN_WORD_WIDTH_PX = 20
MIN_WORD_HEIGHT_PX = 16
MIN_DECODE_CONFIDENCE = 0.65

_LEXICON_CACHE = None


def _load_lexicon() -> set:
    """Load a word lexicon for lightweight post-correction of noisy tokens."""
    global _LEXICON_CACHE
    if _LEXICON_CACHE is not None:
        return _LEXICON_CACHE

    candidates = [
        os.path.join(PROJECT_ROOT, "data", "iam_words", "words_new.txt"),
        os.path.join(PROJECT_ROOT, "data", "iam_words", "iam_words", "words.txt"),
        os.path.join(PROJECT_ROOT, "data", "iam", "words_new.txt"),
    ]

    words = set()
    for path in candidates:
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                token = line.strip().lower()
                if token and token.isalpha() and 2 <= len(token) <= 20:
                    words.add(token)

    _LEXICON_CACHE = words
    if words:
        print(f"[PageOCR] Lexicon loaded: {len(words)} words")
    return words


def _correct_token(token: str, confidence: float, lexicon: set) -> str:
    """Correct likely noisy OCR tokens using nearest lexicon match."""
    if not token:
        return token

    cleaned = "".join(ch for ch in token if ch.isalpha())
    if len(cleaned) < 3:
        return token

    lower = cleaned.lower()
    if lower in lexicon:
        return token

    # Apply correction only when confidence is modest/low.
    if confidence >= 0.82 or not lexicon:
        return token

    # Length-constrained candidate pool keeps matching fast and relevant.
    n = len(lower)
    candidates = [w for w in lexicon if abs(len(w) - n) <= 2 and w[:1] == lower[:1]]
    if not candidates:
        candidates = [w for w in lexicon if abs(len(w) - n) <= 2]

    match = difflib.get_close_matches(lower, candidates, n=1, cutoff=0.72)
    if not match:
        return token

    corrected = match[0]
    if token and token[0].isupper():
        corrected = corrected.capitalize()
    return corrected


# =================================================================
#  STEP 1: Load Model
# =================================================================

def _rebuild_inference_model_from_legacy_weights(model_path: str) -> tf.keras.Model:
    """Rebuild a valid inference .keras model from legacy Keras 2.x weight files."""
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

    print(f"[PageOCR] Rebuilding inference model from legacy weights: {legacy_weights}")
    training_model = build_crnn_model(lstm_units=256, dropout_rate=0.25)

    with h5py.File(legacy_weights, "r") as h5f:
        legacy_h5_format.load_weights_from_hdf5_group(h5f, training_model)

    inference_model = build_inference_model(training_model)
    inference_model.save(model_path)
    print(f"[PageOCR] Rebuilt and saved valid inference model: {model_path}")
    return inference_model

def load_model(model_path: str = DEFAULT_MODEL_PATH) -> tf.keras.Model:
    """
    Load the trained CRNN inference model from disk.

    The inference model takes a single image tensor as input and outputs
    a softmax probability matrix of shape (1, time_steps, num_classes).

    Parameters
    ----------
    model_path : str
        Path to the saved .keras model file.

    Returns
    -------
    tf.keras.Model
        The loaded inference model, ready for prediction.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Please ensure the trained model exists at this path."
        )

    print(f"[PageOCR] Loading model from: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
    except (ValueError, OSError) as e:
        msg = str(e)
        if "accessible `.keras` zip file" not in msg:
            raise
        print(f"[PageOCR] Detected invalid .keras archive. Attempting automatic recovery...")
        model = _rebuild_inference_model_from_legacy_weights(model_path)

    print(f"[PageOCR] Model loaded successfully.")
    print(f"[PageOCR] Input shape : {model.input_shape}")
    print(f"[PageOCR] Output shape: {model.output_shape}")
    return model


# =================================================================
#  STEP 2: Detect Text Regions
# =================================================================

def _subtract_background(gray: np.ndarray) -> np.ndarray:
    """
    Estimate and subtract the background to isolate text strokes.

    WHY: Phone camera photos have uneven lighting — one side of the page
    might be brighter than the other. Global thresholding fails because
    no single threshold separates text from background everywhere.

    SOLUTION: Estimate the background using a large median blur (which
    smooths out text but preserves the background illumination pattern),
    then subtract it. The result contains only the local deviations from
    the background — i.e., the text strokes.

    Parameters
    ----------
    gray : np.ndarray — grayscale image

    Returns
    -------
    np.ndarray — background-subtracted image (text = bright pixels)
    """
    # Median blur with large kernel estimates the background
    # (text details are lost in the blur, only overall illumination remains)
    bg = cv2.medianBlur(gray, 51)

    # Absolute difference: pixels where text deviates from background
    diff = cv2.absdiff(gray, bg)

    return diff


def _remove_ruled_lines(binary: np.ndarray) -> np.ndarray:
    """
    Remove horizontal and vertical ruled lines from a binary image.

    WHY: Notebook paper has horizontal ruled lines that get detected as
    text. We remove them using morphological opening with a long
    horizontal kernel — this keeps only structures that are at least
    100px wide and only 1px tall (i.e., ruled lines), then subtracts
    them from the image. Same for vertical lines.

    Parameters
    ----------
    binary : np.ndarray — binary image with text + ruled lines

    Returns
    -------
    np.ndarray — binary image with ruled lines removed
    """
    # Detect horizontal lines: structures wider than 100px, 1px tall
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    result = cv2.subtract(binary, h_lines)

    # Detect vertical lines: structures taller than 100px, 1px wide
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
    v_lines = cv2.morphologyEx(result, cv2.MORPH_OPEN, v_kernel)
    result = cv2.subtract(result, v_lines)

    return result


def detect_text_regions(image: np.ndarray,
                        debug: bool = False) -> list:
    """
    Detect all text regions in a full-page image using computer vision.

    Handles both clean scanned documents AND phone camera photos
    (including ruled notebook paper) using a robust multi-step pipeline:

      1. Grayscale conversion
      2. Background subtraction (handles uneven lighting)
      3. Otsu thresholding on the difference image
      4. Removal of horizontal/vertical ruled lines
      5. Morphological noise cleanup
      6. Horizontal dilation to merge characters into word/line segments
      7. Contour detection and bounding box extraction
      8. Height-based filtering and splitting of overly tall boxes
      9. Grouping boxes into text lines by vertical center proximity
      10. Merging groups into final line bounding boxes

    Parameters
    ----------
    image : np.ndarray
        The full-page image (BGR color or grayscale).
    debug : bool
        If True, saves intermediate processing images for debugging.

    Returns
    -------
    list of tuple
        Each tuple is (x, y, w, h) — the bounding box of a text region.
    """
    # --- Step 2a: Convert to grayscale ---
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    img_h, img_w = gray.shape[:2]

    # --- Step 2b: Background subtraction ---
    # Removes uneven lighting from camera photos
    diff = _subtract_background(gray)

    # --- Step 2c: Otsu thresholding on the difference image ---
    # Otsu automatically finds the optimal threshold to separate
    # text strokes from the (now uniform) background
    _, binary = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- Step 2d: Remove ruled lines (notebook paper) ---
    cleaned = _remove_ruled_lines(binary)

    # --- Step 2e: Remove small noise specks ---
    noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, noise_kernel)

    # --- Step 2f: Horizontal dilation to merge characters ---
    # Wide horizontal kernel (50px) bridges gaps between letters
    # Thin vertical (3px) connects stroke parts without merging lines
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 3))
    dilated = cv2.dilate(cleaned, dilate_kernel, iterations=2)

    # Close remaining gaps within word segments
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, close_kernel)

    # --- Step 2g: Find contours ---
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # --- Step 2h: Extract and filter bounding boxes ---
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Skip tiny noise
        if w < MIN_REGION_WIDTH or h < MIN_REGION_HEIGHT:
            continue
        # Skip full-page blobs (borders/artifacts)
        if w > img_w * 0.95 and h > img_h * 0.5:
            continue
        boxes.append((x, y, w, h))

    if not boxes:
        if debug:
            _save_debug_images(image, gray, diff, binary, cleaned, dilated, [], img_h, img_w)
        print("[PageOCR] Detected 0 text regions.")
        return []

    # --- Step 2i: Split overly tall boxes ---
    # Boxes that span multiple text lines need to be split.
    # Estimate typical line height from the 75th percentile of box heights.
    heights = sorted([h for _, _, _, h in boxes])
    ref_h = heights[int(len(heights) * 0.75)] if len(heights) > 4 else max(heights)
    ref_h = max(ref_h, 20)  # minimum reference height

    good_boxes = []
    for (x, y, w, h) in boxes:
        if h <= ref_h * 2.5:
            good_boxes.append((x, y, w, h))
        else:
            # Split this tall box into chunks of ~ref_h
            n_splits = max(1, round(h / ref_h))
            split_h = h / n_splits
            for i in range(n_splits):
                sy = int(y + i * split_h)
                sh = int(split_h)
                good_boxes.append((x, sy, w, sh))

    # --- Step 2j: Group boxes into text lines ---
    # Boxes whose vertical centers are within tolerance of each other
    # belong to the same text line.
    tolerance = max(ref_h * 0.6, 15)

    boxes_with_center = [(x, y, w, h, y + h / 2) for (x, y, w, h) in good_boxes]
    boxes_with_center.sort(key=lambda b: b[4])

    line_groups = []
    current_group = [boxes_with_center[0]]

    for box in boxes_with_center[1:]:
        current_avg_y = np.mean([b[4] for b in current_group])
        if abs(box[4] - current_avg_y) <= tolerance:
            current_group.append(box)
        else:
            line_groups.append(current_group)
            current_group = [box]
    line_groups.append(current_group)

    # --- Step 2k: Merge each line group into a single bounding box ---
    merged = []
    min_line_height = max(MIN_REGION_HEIGHT, int(ref_h * 0.60))
    for group in line_groups:
        x_min = min(b[0] for b in group)
        y_min = min(b[1] for b in group)
        x_max = max(b[0] + b[2] for b in group)
        y_max = max(b[1] + b[3] for b in group)
        w = x_max - x_min
        h = y_max - y_min
        # Drop likely ruling/header artifacts: very wide but too thin.
        if w > img_w * 0.75 and h < max(min_line_height, int(ref_h * 0.85)):
            continue

        if w >= MIN_REGION_WIDTH and h >= min_line_height:
            merged.append((x_min, y_min, w, h))

    merged.sort(key=lambda b: b[1])

    # --- Debug: save intermediate images ---
    if debug:
        _save_debug_images(image, gray, diff, binary, cleaned, dilated, merged, img_h, img_w)

    print(f"[PageOCR] Detected {len(merged)} text regions.")
    return merged


def segment_line_into_words(line_crop: np.ndarray,
                            line_x: int = 0,
                            line_y: int = 0) -> list:
    """
    Split a text line crop into chunks suitable for the CRNN model.

    Uses a hybrid strategy:
      1. Find vertical projection gaps (ink-free columns) in the line.
      2. Try word-level segmentation using large gaps.
      3. If word crops are too small on average (camera photos with low
         contrast produce over-fragmented tiny crops), merge small runs
         into word-like chunks.
    """
    h, w = line_crop.shape[:2]
    min_token_width = max(12, int(h * 0.35))

    # ── Binarize for projection analysis ───────────────────────────
    # Use stronger cleanup only for low-contrast camera crops.
    p1 = np.percentile(line_crop, 1)
    p99 = np.percentile(line_crop, 99)
    pixel_span = p99 - p1
    pixel_std = np.std(line_crop)
    low_contrast = (pixel_span <= 120 or pixel_std <= 50)

    if low_contrast:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(line_crop)
        _, binary = cv2.threshold(enhanced, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Remove ruled lines/noise so projection gaps reflect spaces,
        # not paper texture.
        horiz_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(20, w // 8), 1)
        )
        horiz_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)
        binary = cv2.subtract(binary, horiz_lines)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    else:
        _, binary = cv2.threshold(line_crop, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ── Vertical projection: count ink pixels per column ───────────
    projection = np.sum(binary > 0, axis=0).astype(np.float32)
    projection = cv2.blur(projection.reshape(1, -1), (1, 9)).reshape(-1)
    ink_threshold = max(2, int(h * (0.12 if low_contrast else 0.06)))
    is_ink = projection > ink_threshold

    # ── Find all gap locations (columns with no/little ink) ────────
    gaps = []
    in_gap = False
    gap_start = 0
    for col in range(w):
        if not is_ink[col] and not in_gap:
            gap_start = col
            in_gap = True
        elif is_ink[col] and in_gap:
            gaps.append((gap_start, col, col - gap_start))
            in_gap = False
    if in_gap:
        gaps.append((gap_start, w, w - gap_start))

    # ── Find ink runs (contiguous ink columns) ─────────────────────
    runs = []
    in_run = False
    run_start = 0
    for col in range(w):
        if is_ink[col] and not in_run:
            run_start = col
            in_run = True
        elif not is_ink[col] and in_run:
            runs.append((run_start, col))
            in_run = False
    if in_run:
        runs.append((run_start, w))

    if not runs:
        return [(line_x, line_y, w, h)]

    # ── Merge ink runs separated by narrow gaps (always merge these) ──
    min_char_gap = max(3, int(h * 0.08))
    merged_runs = [runs[0]]
    for start, end in runs[1:]:
        prev_start, prev_end = merged_runs[-1]
        gap = start - prev_end
        if gap < min_char_gap:
            merged_runs[-1] = (prev_start, end)
        else:
            merged_runs.append((start, end))

    # ── Determine ink extent ───────────────────────────────────────
    ink_start = merged_runs[0][0]
    ink_end = merged_runs[-1][1]
    ink_width = ink_end - ink_start

    # ── If the line fits within model width, return it as-is ───────
    if ink_width <= IMG_WIDTH:
        return [(line_x + ink_start, line_y, ink_width, h)]

    # ── Try word-level segmentation first ──────────────────────────
    # Previous threshold over-merged words on full notebook pages.
    # Use a smaller gap threshold so inter-word spaces are preserved.
    min_word_gap = max(3, int(h * 0.10))
    word_runs = [merged_runs[0]]
    for start, end in merged_runs[1:]:
        prev_start, prev_end = word_runs[-1]
        gap = start - prev_end
        if gap < min_word_gap:
            word_runs[-1] = (prev_start, end)
        else:
            word_runs.append((start, end))

    # Split overly wide "word" runs at strong internal gaps.
    # This prevents multi-word blobs from being sent as one crop.
    split_gap = max(4, int(h * 0.12))
    refined_runs = []
    for ws, we in word_runs:
        width = we - ws
        if width <= max(350, int(IMG_WIDTH * 1.35)):
            refined_runs.append((ws, we))
            continue

        internal_gaps = [
            (gs, ge, gw)
            for (gs, ge, gw) in gaps
            if ws < gs < we and gw >= split_gap
        ]
        if not internal_gaps:
            refined_runs.append((ws, we))
            continue

        cut_points = [
            (gs + ge) // 2
            for (gs, ge, _) in internal_gaps
        ]
        cut_points = [cp for cp in cut_points if ws + 8 < cp < we - 8]
        if not cut_points:
            refined_runs.append((ws, we))
            continue

        prev = ws
        for cp in cut_points:
            if cp - prev >= 8:
                refined_runs.append((prev, cp))
            prev = cp
        if we - prev >= 8:
            refined_runs.append((prev, we))

    word_runs = refined_runs

    word_widths = [end - start for start, end in word_runs]
    avg_word_width = np.mean(word_widths) if word_widths else 0

    if avg_word_width >= 40:
        word_regions = []
        for start, end in word_runs:
            word_w = end - start
            if word_w >= min_token_width:
                word_regions.append((line_x + start, line_y, word_w, h))
        if word_regions:
            return word_regions

    # If a very wide line still failed to split, force coarse chunks using
    # projection minima so multi-word blobs are not sent as a single sample.
    if len(word_runs) <= 1 and ink_width > int(IMG_WIDTH * 1.6):
        forced = []
        target = int(IMG_WIDTH * 0.9)
        start = ink_start
        while start < ink_end:
            end = min(ink_end, start + target)
            if end >= ink_end:
                forced.append((start, ink_end))
                break

            # Search for a low-ink cut near the chunk end.
            lo = max(start + 20, end - 40)
            hi = min(ink_end - 20, end + 40)
            if hi > lo:
                cut = lo + int(np.argmin(projection[lo:hi]))
            else:
                cut = end

            if cut - start < 20:
                cut = min(ink_end, start + target)

            forced.append((start, cut))
            start = cut

        forced_regions = []
        for ws, we in forced:
            ww = we - ws
            if ww >= min_token_width:
                forced_regions.append((line_x + ws, line_y, ww, h))
        if forced_regions:
            return forced_regions

    # ── Fallback: merge tiny fragments into word-like regions ──────
    min_word_width = max(24, int(h * 0.35))
    max_word_width = IMG_WIDTH

    merged_words = []
    cur_start, cur_end = word_runs[0]
    for start, end in word_runs[1:]:
        proposed_w = end - cur_start

        if (cur_end - cur_start) < min_word_width and proposed_w <= max_word_width:
            cur_end = end
            continue

        if (cur_end - cur_start) >= min_word_width:
            merged_words.append((cur_start, cur_end))
            cur_start, cur_end = start, end
            continue

        merged_words.append((cur_start, cur_end))
        cur_start, cur_end = start, end

    merged_words.append((cur_start, cur_end))

    merged_regions = []
    for ws, we in merged_words:
        ww = we - ws
        if ww >= min_token_width:
            merged_regions.append((line_x + ws, line_y, ww, h))

    return merged_regions if merged_regions else [(line_x, line_y, w, h)]


def _save_debug_images(image, gray, diff, binary, cleaned, dilated, boxes, img_h, img_w):
    """Save all intermediate processing images for debugging."""
    debug_dir = os.path.join(PROJECT_ROOT, "debug_output")
    os.makedirs(debug_dir, exist_ok=True)

    cv2.imwrite(os.path.join(debug_dir, "1_gray.png"), gray)
    cv2.imwrite(os.path.join(debug_dir, "2_bg_diff.png"), diff)
    cv2.imwrite(os.path.join(debug_dir, "3_binary.png"), binary)
    cv2.imwrite(os.path.join(debug_dir, "4_cleaned.png"), cleaned)
    cv2.imwrite(os.path.join(debug_dir, "5_dilated.png"), dilated)

    # Draw detected boxes on the original image
    debug_img = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(debug_img, str(i + 1), (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(os.path.join(debug_dir, "6_detected_boxes.png"), debug_img)

    print(f"[PageOCR] Debug images saved to: {debug_dir}")


# =================================================================
#  STEP 3: Sort Text Boxes in Reading Order
# =================================================================

def sort_text_boxes(boxes: list,
                    line_tolerance_ratio: float = LINE_GROUP_TOLERANCE) -> list:
    """
    Sort bounding boxes in natural reading order: top-to-bottom, left-to-right.

    Simple top-to-bottom sorting by y-coordinate fails when text lines are
    slightly tilted or when two boxes on the same line have slightly different
    y-values. Instead, we:

      1. Group boxes that share approximately the same vertical position
         into "lines" (using their vertical center).
      2. Sort these groups from top to bottom (by average y-center).
      3. Within each line group, sort boxes from left to right (by x).

    Parameters
    ----------
    boxes : list of (x, y, w, h)
        Unsorted bounding boxes from contour detection.
    line_tolerance_ratio : float
        Two boxes are on the same line if the difference between their
        vertical centers is less than this fraction of the average box height.

    Returns
    -------
    list of (x, y, w, h)
        Bounding boxes sorted in reading order.
    """
    if not boxes:
        return []

    # Calculate the average box height to set a reasonable grouping threshold
    avg_height = np.mean([h for (_, _, _, h) in boxes])
    tolerance = avg_height * line_tolerance_ratio

    # Sort boxes by their vertical center (y + h/2) as a starting point
    boxes_with_center = [(x, y, w, h, y + h / 2) for (x, y, w, h) in boxes]
    boxes_with_center.sort(key=lambda b: b[4])  # sort by y_center

    # Group boxes into lines: boxes whose y-centers are within tolerance
    lines = []
    current_line = [boxes_with_center[0]]

    for box in boxes_with_center[1:]:
        # Compare this box's y-center with the current line's average y-center
        current_avg_y = np.mean([b[4] for b in current_line])

        if abs(box[4] - current_avg_y) <= tolerance:
            # Same line — add to current group
            current_line.append(box)
        else:
            # New line — save current line and start a new one
            lines.append(current_line)
            current_line = [box]

    lines.append(current_line)  # don't forget the last line

    # Sort lines from top to bottom (by their average y-center)
    lines.sort(key=lambda line: np.mean([b[4] for b in line]))

    # Within each line, sort boxes from left to right (by x-coordinate)
    sorted_boxes = []
    for line in lines:
        line.sort(key=lambda b: b[0])  # sort by x
        for (x, y, w, h, _) in line:
            sorted_boxes.append((x, y, w, h))

    return sorted_boxes


# =================================================================
#  STEP 4: Preprocess a Cropped Text Region
# =================================================================

def preprocess_crop(crop: np.ndarray,
                    target_height: int = IMG_HEIGHT,
                    target_width:  int = IMG_WIDTH,
                    low_contrast_hint: bool = False) -> np.ndarray:
    """
    Preprocess a single cropped text region for model input.

    Transforms ANY input image (scan, phone camera, screenshot) to look
    like an IAM training image before applying the training pipeline:

      1. Convert to grayscale (if color)
      2. Domain adaptation: stretch pixel range to match IAM distribution
         - IAM images use the full 0-255 range (std ~80, span ~200+)
         - Camera crops often use a narrow band (std ~15, span ~50)
         - We stretch to fill 0-255 so Otsu can separate ink from paper
      3. Otsu binarization (matching training preprocessing)
      4. Resize to target height while preserving aspect ratio
      5. Pad with white to reach target width
      6. Normalize to [0, 1] with inversion (text=bright, bg=dark)
      7. Add channel dimension for Keras: (H, W) → (H, W, 1)

    Parameters
    ----------
    crop : np.ndarray
        Cropped text region from the page image.
    target_height : int
        Model input height (default: 64, matching training).
    target_width : int
        Model input width (default: 256, matching training).

    Returns
    -------
    np.ndarray
        Preprocessed image of shape (target_height, target_width, 1),
        dtype float32, values in [0.0, 1.0].
    """
    # Step 4a: Convert to grayscale if the crop is color
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    # Keep page-mode preprocessing aligned with single-word inference.
    # For low-contrast pages, lightly enhance contrast before running the
    # same preprocess_from_array pipeline used elsewhere.
    if low_contrast_hint:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
        gray = clahe.apply(gray)

    out = preprocess_from_array(
        gray,
        target_height=target_height,
        target_width=target_width,
        binarize=True,
        binarize_method="otsu",
    )
    return out.astype(np.float32)


# =================================================================
#  STEP 5: Run Model Prediction on a Single Crop
# =================================================================

def predict_text(model: tf.keras.Model,
                 preprocessed_img: np.ndarray) -> np.ndarray:
    """
    Run the CRNN model on a single preprocessed image and return
    raw softmax predictions.

    Parameters
    ----------
    model : tf.keras.Model
        The loaded CRNN inference model.
    preprocessed_img : np.ndarray
        Preprocessed image of shape (H, W, 1), dtype float32.

    Returns
    -------
    np.ndarray
        Softmax prediction matrix of shape (time_steps, num_classes).
        Each row is a probability distribution over all characters
        at that time step.
    """
    # Add batch dimension: (H, W, 1) → (1, H, W, 1)
    batch = np.expand_dims(preprocessed_img, axis=0)

    # Run model inference — verbose=0 suppresses progress bar
    predictions = model.predict(batch, verbose=0)

    # Remove batch dimension: (1, T, C) → (T, C)
    return predictions[0]


# =================================================================
#  STEP 6: Decode Prediction to Text
# =================================================================

def decode_prediction(prediction: np.ndarray,
                      int_to_char: dict) -> str:
    """
    Convert a raw softmax prediction matrix into a readable text string
    using CTC greedy decoding.

    CTC decoding works in three steps:
      1. argmax — pick the most likely character at each time step
      2. collapse — remove consecutive duplicate characters
      3. remove blanks — delete all CTC blank tokens

    Parameters
    ----------
    prediction : np.ndarray
        Softmax matrix of shape (time_steps, num_classes).
    int_to_char : dict
        Integer-to-character mapping from training.

    Returns
    -------
    str
        The decoded text string.
    """
    # Use the same greedy decode path as single-word predictor for consistency.
    texts = decode_batch(
        np.expand_dims(prediction, axis=0),
        int_to_char,
        method="greedy",
        blank_index=BLANK_INDEX,
    )
    return texts[0] if texts else ""


# =================================================================
#  STEP 7: Full Pipeline — Extract Text from a Page Image
# =================================================================

def extract_text_from_page(image_path: str,
                           model: tf.keras.Model = None,
                           model_path: str = DEFAULT_MODEL_PATH,
                           debug: bool = False) -> str:
    """
    Complete OCR pipeline: extract all text from a full-page image.

    This is the main entry point. It orchestrates the entire pipeline:
      1. Load the page image
      2. Detect text regions using computer vision
      3. Sort regions in reading order (top→bottom, left→right)
      4. Crop, preprocess, and run the model on each region
      5. Decode each prediction into text
      6. Combine all text lines into the final paragraph

    Parameters
    ----------
    image_path : str
        Path to the full-page image file (.png, .jpg, .tif, etc.).
    model : tf.keras.Model or None
        Pre-loaded model. If None, the model is loaded from model_path.
        Pass a pre-loaded model when processing multiple pages to avoid
        reloading the model each time.
    model_path : str
        Path to the .keras model file (used only if model is None).
    debug : bool
        If True, saves intermediate processing images for debugging.

    Returns
    -------
    str
        The complete extracted text from the page, with lines separated
        by newline characters.
    """
    # ── Validate input ─────────────────────────────────────────────
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # ── Load model if not provided ─────────────────────────────────
    if model is None:
        model = load_model(model_path)

    # ── Build character map for decoding ───────────────────────────
    _, int_to_char = build_char_maps(ALPHABET)
    lexicon = _load_lexicon()

    # ── Step 1: Read the full-page image ───────────────────────────
    print(f"\n[PageOCR] Processing image: {image_path}")
    page_image = cv2.imread(image_path)
    if page_image is None:
        raise ValueError(f"Could not read image: {image_path}")

    page_gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
    print(f"[PageOCR] Image size: {page_image.shape[1]}×{page_image.shape[0]} pixels")

    # ── Step 1b: Page-level quality enhancement for low-contrast images ──
    # Camera photos have compressed dynamic range. Enhance the full page
    # BEFORE cropping so each crop benefits from global context.
    p1_page = np.percentile(page_gray, 1)
    p99_page = np.percentile(page_gray, 99)
    page_span = p99_page - p1_page
    page_std = np.std(page_gray)

    page_low_contrast = (page_span <= 120 or page_std <= 50)

    if page_low_contrast:
        # Low-contrast page (camera photo) — enhance globally
        # 1. Background estimation via large median blur
        bg = cv2.medianBlur(page_gray, 51)
        # 2. Divide image by background to normalize illumination
        #    Result: text becomes dark, background becomes uniform ~255
        normalized = cv2.divide(page_gray, bg, scale=255)
        # 3. Stretch contrast of the normalized result
        pn1 = np.percentile(normalized, 1)
        pn99 = np.percentile(normalized, 99)
        if pn99 - pn1 > 5:
            crop_source = np.clip(
                (normalized.astype(np.float32) - pn1) / (pn99 - pn1) * 255,
                0, 255).astype(np.uint8)
        else:
            crop_source = normalized
        print(f"[PageOCR] Low-contrast image detected (span={page_span:.0f}, std={page_std:.0f})")
        print(f"[PageOCR] Applied illumination normalization + contrast stretch")
    else:
        crop_source = page_gray

    # ── Step 2: Detect text regions ────────────────────────────────
    boxes = detect_text_regions(page_image, debug=debug)

    if not boxes:
        print("[PageOCR] No text regions detected.")
        return ""

    # ── Step 3: Sort boxes in reading order ────────────────────────
    sorted_boxes = sort_text_boxes(boxes)
    print(f"[PageOCR] Processing {len(sorted_boxes)} text regions in reading order...")

    # ── Steps 4–6: Process each text region ────────────────────────
    # The model was trained on individual words, not full lines.
    # Each detected line must be segmented into words first.
    extracted_lines = []

    if debug:
        crops_dir = os.path.join(PROJECT_ROOT, "debug_output", "crops")
        os.makedirs(crops_dir, exist_ok=True)

    word_count = 0
    for i, (x, y, w, h) in enumerate(sorted_boxes):
        # Crop the line region with padding
        pad_y = max(4, int(h * 0.35))
        pad_x = max(2, int(w * 0.05))
        y1 = max(0, y - pad_y)
        y2 = min(crop_source.shape[0], y + h + pad_y)
        x1 = max(0, x - pad_x)
        x2 = min(crop_source.shape[1], x + w + pad_x)
        # Use original grayscale for segmentation to avoid artifacts from
        # page-level enhancement that can create false gaps/components.
        line_crop = page_gray[y1:y2, x1:x2]

        if line_crop.size == 0 or line_crop.shape[0] < 5 or line_crop.shape[1] < 5:
            continue

        # Segment the line into individual words
        word_boxes = segment_line_into_words(line_crop, line_x=x1, line_y=y1)

        # Process each word
        line_words = []
        for wx, wy, ww, wh in word_boxes:
            # Skip tiny regions that are usually punctuation/noise fragments.
            if ww < MIN_WORD_WIDTH_PX or wh < MIN_WORD_HEIGHT_PX:
                continue

            # Expand word crop vertically so strokes are not clipped.
            wpad_y = max(2, int(wh * 0.2))
            wy1 = max(0, wy - wpad_y)
            wy2 = min(page_gray.shape[0], wy + wh + wpad_y)
            wx1 = max(0, wx)
            wx2 = min(page_gray.shape[1], wx + ww)

            # Crop word from ORIGINAL grayscale page for recognition.
            # Enhanced/normalized page is useful for detection, but feeding it
            # to recognition can introduce artifacts and random decodes.
            word_crop = page_gray[wy1:wy2, wx1:wx2]
            if word_crop.size == 0 or word_crop.shape[0] < 5 or word_crop.shape[1] < 5:
                continue

            word_count += 1

            # Preprocess to match training format
            preprocessed = preprocess_crop(
                word_crop,
                low_contrast_hint=page_low_contrast,
            )

            # Save debug crops
            if debug:
                cv2.imwrite(os.path.join(crops_dir, f"line{i+1:02d}_word{word_count:03d}_raw.png"), word_crop)
                vis = ((1.0 - preprocessed[:, :, 0]) * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(crops_dir, f"line{i+1:02d}_word{word_count:03d}_prep.png"), vis)

            # Run model and decode
            prediction = predict_text(model, preprocessed)
            confidence = float(np.mean(np.max(prediction, axis=-1)))

            # Skip low-confidence decodes to avoid random garbage tokens.
            if confidence < MIN_DECODE_CONFIDENCE:
                continue

            text = decode_prediction(prediction, int_to_char)
            if text.strip():
                corrected = _correct_token(text.strip(), confidence, lexicon)
                line_words.append(corrected)

        # Join words from the same line with spaces
        if line_words:
            line_text = " ".join(line_words)
            extracted_lines.append(line_text)
            print(f"  Line {i+1:3d} [{x:4d},{y:4d} {w:4d}×{h:3d}] ({len(line_words)} words): {line_text}")

    # ── Step 7: Reconstruct paragraph ──────────────────────────────
    # Join all extracted text lines with newlines to preserve structure
    full_text = "\n".join(extracted_lines)

    print(f"\n[PageOCR] Extraction complete. {len(extracted_lines)} lines found.")
    return full_text


# =================================================================
#  Debug Visualization Helper
# =================================================================

def visualize_detections(image_path: str,
                         save_path: str = None,
                         debug: bool = True):
    """
    Visualize the text detection step: draws all detected bounding boxes
    on the original image and optionally saves the result.

    Useful for debugging detection issues — if text is being missed or
    too much noise is being detected, this makes it immediately visible.

    Parameters
    ----------
    image_path : str
        Path to the full-page image.
    save_path : str or None
        If provided, saves the annotated image to this path.
        If None, displays it in a window.
    debug : bool
        If True, also saves intermediate pipeline images.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    boxes = detect_text_regions(image, debug=debug)
    sorted_boxes = sort_text_boxes(boxes)

    # Draw numbered boxes on a copy of the image
    annotated = image.copy()
    for i, (x, y, w, h) in enumerate(sorted_boxes):
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            annotated, str(i + 1),
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        cv2.imwrite(save_path, annotated)
        print(f"[PageOCR] Annotated image saved to: {save_path}")
    else:
        cv2.imshow("Detected Text Regions", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# =================================================================
#  Command-Line Interface
# =================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Full-Page OCR Pipeline — Extract text from document images "
                    "using CRNN + CTC without any external OCR engines."
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to the full-page image file (.png, .jpg, .tif)"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_PATH,
        help="Path to the trained CRNN inference model (.keras file)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Save intermediate processing images for debugging"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Save an annotated image showing detected text regions"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # If --visualize is set, just show detection boxes and exit
    if args.visualize:
        viz_path = os.path.splitext(args.image)[0] + "_detections.png"
        visualize_detections(args.image, save_path=viz_path, debug=args.debug)
        sys.exit(0)

    # Load model once (so it can be reused if we later add batch page processing)
    model = load_model(args.model)

    # Run the full pipeline
    extracted_text = extract_text_from_page(
        image_path=args.image,
        model=model,
        debug=args.debug
    )

    # Print the final result
    print("\n" + "=" * 60)
    print("  EXTRACTED TEXT")
    print("=" * 60)
    print(extracted_text)
    print("=" * 60)
