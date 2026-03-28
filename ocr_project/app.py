"""
#new updates 

OCR Web Interface - Multi-mode text extraction
  - PDF files: direct text extraction (pdfplumber)
  - Printed text images: Tesseract OCR
  - Handwritten text images: CRNN model (page_ocr pipeline)
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import re
import csv
#comments
import json
#hi

import random
from datetime import datetime
from pathlib import Path

import gradio as gr
from PIL import Image
import numpy as np
import cv2

# ── Tesseract setup ───────────────────────────────────────────────
import pytesseract

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# ── PDF extraction ────────────────────────────────────────────────
import pdfplumber

# ── Handwriting model (lazy load) ─────────────────────────────────
_handwriting_model = None
_handwriting_predictor = None
_word_lexicon = None

def _get_handwriting_model():
    global _handwriting_model
    if _handwriting_model is None:
        from page_ocr import load_model
        print("Loading CRNN handwriting model...")
        _handwriting_model = load_model()
        print("CRNN model ready!")
    return _handwriting_model


def _get_handwriting_predictor():
    global _handwriting_predictor
    if _handwriting_predictor is None:
        from backend.inference.predict import OCRPredictor
        model_path = os.path.join(os.path.dirname(__file__), "saved_models", "crnn_iam_v1_inference.keras")
        # Beam decoding is slower but typically more stable than greedy for noisy samples.
        _handwriting_predictor = OCRPredictor(model_path, decode_method="beam")
    return _handwriting_predictor


def _looks_like_document_image(image: Image.Image) -> bool:
    """Heuristic: page OCR works best for document-like images, not single-word crops."""
    w, h = image.size
    if w < 320 or h < 140:
        return False
    if (w / max(h, 1)) > 12:
        return False
    return True


def _load_word_lexicon() -> set:
    """Load a lightweight word list to score OCR output quality."""
    global _word_lexicon
    if _word_lexicon is not None:
        return _word_lexicon

    base = os.path.dirname(__file__)
    candidates = [
        os.path.join(base, "data", "iam_words", "words_new.txt"),
        os.path.join(base, "data", "iam_words", "iam_words", "words.txt"),
        os.path.join(base, "data", "iam", "words_new.txt"),
    ]

    words = set()
    for path in candidates:
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                token = line.strip().lower()
                if token.isalpha() and 2 <= len(token) <= 20:
                    words.add(token)

    _word_lexicon = words
    return words


def _text_quality_score(text: str) -> float:
    """Heuristic quality score: higher means more plausible English OCR text."""
    if not text or not text.strip():
        return -1.0

    cleaned = text.replace("\n", " ").strip()
    n_chars = len(cleaned)
    if n_chars == 0:
        return -1.0

    letters = sum(ch.isalpha() for ch in cleaned)
    spaces = sum(ch.isspace() for ch in cleaned)
    punctuation = sum(ch in ".,;:!?'-\"()[]" for ch in cleaned)
    weird = sum((not ch.isalnum()) and (not ch.isspace()) and (ch not in ".,;:!?'-\"()[]") for ch in cleaned)

    alpha_ratio = letters / n_chars
    space_ratio = spaces / n_chars
    weird_ratio = weird / n_chars
    punc_ratio = punctuation / n_chars

    tokens = re.findall(r"[A-Za-z]{2,}", cleaned)
    lex = _load_word_lexicon()
    lex_hits = sum(1 for t in tokens if t.lower() in lex) if lex else 0
    lex_ratio = (lex_hits / len(tokens)) if tokens else 0.0

    # Penalize repeated tiny token patterns like "io io io".
    rep_penalty = 0.0
    if len(tokens) >= 6:
        short = [t.lower() for t in tokens if len(t) <= 3]
        if short:
            most = max(short.count(t) for t in set(short))
            rep_penalty = min(0.25, most / max(1, len(tokens)) * 0.5)

    score = (
        1.2 * alpha_ratio
        + 0.7 * lex_ratio
        + 0.2 * min(space_ratio, 0.25)
        + 0.1 * min(punc_ratio, 0.12)
        - 1.1 * weird_ratio
        - rep_penalty
    )
    return score


def extract_from_pdf(pdf_path: str) -> str:
    """Extract text directly from a PDF file using pdfplumber."""
    text_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text_pages.append(page_text.strip())
    return "\n\n".join(text_pages)


def extract_printed_text(image) -> str:
    """Extract printed/typed text from an image using Tesseract OCR."""
    if isinstance(image, np.ndarray):
        rgb = image
    else:
        rgb = np.array(image)

    if rgb.ndim == 2:
        gray = rgb.copy()
    else:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Build OCR variants: raw, enhanced grayscale, and adaptive-thresholded.
    h, w = gray.shape[:2]
    scale = 2.0 if max(h, w) < 1800 else 1.4
    up = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    den = cv2.bilateralFilter(up, d=7, sigmaColor=50, sigmaSpace=50)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enh = clahe.apply(den)
    thr = cv2.adaptiveThreshold(
        enh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11
    )

    variants = [
        ("raw", up),
        ("enhanced", enh),
        ("adaptive", thr),
    ]

    configs = [
        ("psm6", "--oem 3 --psm 6 -c preserve_interword_spaces=1"),
        ("psm4", "--oem 3 --psm 4 -c preserve_interword_spaces=1"),
        ("psm11", "--oem 3 --psm 11"),
    ]

    best_text = ""
    best_score = -1.0

    for _, img_var in variants:
        for _, cfg in configs:
            text = pytesseract.image_to_string(img_var, lang="eng", config=cfg).strip()
            score = _text_quality_score(text)
            if score > best_score:
                best_score = score
                best_text = text

    return best_text


def extract_handwriting_full_page(image) -> str:
    """Extract full-page handwriting using page OCR + Tesseract fallback."""
    from page_ocr import extract_text_from_page

    temp_path = "temp_upload.png"
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image.save(temp_path)

    try:
        backend_used = "CRNN"
        crnn_score = -1.0
        tess_score = -1.0

        if not _looks_like_document_image(image):
            return "Please upload a document-like page image for Full-Page mode."

        model = _get_handwriting_model()
        crnn_text = extract_text_from_page(temp_path, model=model, debug=True)

        # Hybrid fallback: compare CRNN page OCR vs Tesseract and choose
        # whichever output looks more linguistically plausible.
        tess_text = extract_printed_text(image)
        crnn_score = _text_quality_score(crnn_text)
        tess_score = _text_quality_score(tess_text)

        if tess_score > crnn_score + 0.05:
            text = tess_text
            backend_used = "Tesseract fallback"
            print(f"[App] Using Tesseract fallback (score {tess_score:.3f} > CRNN {crnn_score:.3f})")
        else:
            text = crnn_text
            backend_used = "CRNN page OCR"

        # Final fallback if page path is empty.
        if not text.strip():
            predictor = _get_handwriting_predictor()
            text = predictor.predict(temp_path)
            backend_used = "CRNN single-word"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    final_text = text if text else "(No text detected)"
    info = (
        f"[Backend: {backend_used}]\n"
        f"[Quality scores -> CRNN: {crnn_score:.3f}, Tesseract: {tess_score:.3f}]\n\n"
    )
    return info + final_text


def extract_handwriting_single_word(image) -> str:
    """Extract single-word handwriting using only the CRNN predictor path."""
    temp_path = "temp_upload.png"
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image.save(temp_path)

    try:
        predictor = _get_handwriting_predictor()
        text = predictor.predict(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    final_text = text if text else "(No text detected)"
    return "[Backend: CRNN single-word]\n\n" + final_text


def _resolve_file_path(file_obj) -> str:
    """Return a file path from Gradio file objects or plain path strings."""
    if file_obj is None:
        return ""
    if isinstance(file_obj, str):
        return file_obj
    return getattr(file_obj, "name", "")


def _build_export_file(records: list, export_format: str) -> str:
    """Create an export file and return its path."""
    export_dir = os.path.join(os.path.dirname(__file__), "debug_output", "exports")
    os.makedirs(export_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"ocr_batch_{timestamp}"

    if export_format == "CSV":
        out_path = os.path.join(export_dir, f"{base_name}.csv")
        fieldnames = ["index", "file_name", "status", "mode", "text", "error"]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for item in records:
                writer.writerow({k: item.get(k, "") for k in fieldnames})
        return out_path

    if export_format == "JSON":
        out_path = os.path.join(export_dir, f"{base_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        return out_path

    if export_format == "TXT":
        out_path = os.path.join(export_dir, f"{base_name}.txt")
        chunks = []
        for item in records:
            chunks.append(
                (
                    f"[{item.get('index')}] {item.get('file_name')}\n"
                    f"Status: {item.get('status')}\n"
                    f"Mode: {item.get('mode')}\n"
                    f"Error: {item.get('error') or '-'}\n\n"
                    f"{item.get('text') or '(No text detected)'}"
                )
            )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n\n" + ("\n" + ("-" * 80) + "\n\n").join(chunks))
        return out_path

    return ""


def _filter_export_records(records: list, filter_mode: str, random_count: float) -> list:
    """Filter export records by selection mode before writing files."""
    if not records:
        return []

    if filter_mode != "Random Sample":
        return records

    try:
        count = int(random_count)
    except (TypeError, ValueError):
        count = len(records)

    if count <= 0 or count >= len(records):
        return records

    return random.sample(records, count)


def process_files(files, mode, export_format, export_filter_mode, random_sample_count):
    """
    Batch processing function for multiple uploaded files.

    The selected mode is applied to each compatible file:
      - PDF mode only processes .pdf files.
      - Image modes only process image files.
    """
    if not files:
        return "Please upload one or more files.", None

    results = []
    export_records = []

    for idx, file_obj in enumerate(files, start=1):
        file_path = _resolve_file_path(file_obj)
        if not file_path:
            msg = f"[{idx}] Unknown file: could not resolve path"
            results.append(msg)
            export_records.append({
                "index": idx,
                "file_name": "Unknown",
                "status": "error",
                "mode": mode,
                "text": "",
                "error": "Could not resolve uploaded file path.",
            })
            continue

        name = Path(file_path).name
        suffix = Path(file_path).suffix.lower()

        try:
            if mode == "PDF (extract text directly)":
                if suffix != ".pdf":
                    msg = f"[{idx}] {name}\nSkipped: not a PDF file."
                    results.append(msg)
                    export_records.append({
                        "index": idx,
                        "file_name": name,
                        "status": "skipped",
                        "mode": mode,
                        "text": "",
                        "error": "Not a PDF file.",
                    })
                    continue

                text = extract_from_pdf(file_path)
                if not text:
                    text = (
                        "PDF has no selectable text (it may be scanned).\n"
                        "Try Printed Text mode for image-based OCR."
                    )

            else:
                if suffix == ".pdf":
                    msg = f"[{idx}] {name}\nSkipped: PDF file in image OCR mode."
                    results.append(msg)
                    export_records.append({
                        "index": idx,
                        "file_name": name,
                        "status": "skipped",
                        "mode": mode,
                        "text": "",
                        "error": "PDF file used in image OCR mode.",
                    })
                    continue

                image = Image.open(file_path).convert("RGB")

                if mode == "Printed Text (Tesseract)":
                    text = extract_printed_text(image)
                elif mode == "Handwriting Single Word (CRNN)":
                    text = extract_handwriting_single_word(image)
                elif mode == "Handwriting Full Page (CRNN + fallback)":
                    text = extract_handwriting_full_page(image)
                else:
                    text = "Unknown mode selected."

            text = text if text and text.strip() else "(No text detected)"
            results.append(f"[{idx}] {name}\n{text}")
            export_records.append({
                "index": idx,
                "file_name": name,
                "status": "processed",
                "mode": mode,
                "text": text,
                "error": "",
            })

        except Exception as e:
            results.append(f"[{idx}] {name}\nError: {e}")
            export_records.append({
                "index": idx,
                "file_name": name,
                "status": "error",
                "mode": mode,
                "text": "",
                "error": str(e),
            })

    if not results:
        return "No compatible files were processed.", None

    export_path = None
    if export_format in {"CSV", "TXT", "JSON"} and export_records:
        selected_records = _filter_export_records(
            export_records,
            export_filter_mode,
            random_sample_count,
        )
        export_path = _build_export_file(selected_records, export_format)

    text_output = "\n\n" + ("\n" + ("-" * 80) + "\n\n").join(results)
    return text_output, export_path


# ── Gradio Interface ──────────────────────────────────────────────
with gr.Blocks(title="OCR System") as demo:
    gr.Markdown(
        "# OCR Text Extraction System\n"
        "Upload one or more files and select the appropriate mode:\n"
        "- **PDF** — extracts text directly from PDF files (100% accurate for digital PDFs)\n"
        "- **Printed Text** — uses Tesseract OCR for typed/printed documents\n"
        "- **Handwriting Single Word** — optimized for one handwritten word\n"
        "- **Handwriting Full Page** — optimized for page extraction with fallback\n\n"
        "You can upload multiple files together for batch extraction."
    )

    with gr.Row():
        with gr.Column():
            mode = gr.Radio(
                choices=[
                    "Printed Text (Tesseract)",
                    "Handwriting Single Word (CRNN)",
                    "Handwriting Full Page (CRNN + fallback)",
                    "PDF (extract text directly)",
                ],
                value="Printed Text (Tesseract)",
                label="Select Mode",
            )
            files_input = gr.File(
                label="Upload Files (Images/PDF)",
                file_count="multiple",
                file_types=["image", ".pdf"],
            )
            export_format = gr.Dropdown(
                choices=["None", "CSV", "TXT", "JSON"],
                value="None",
                label="Export Results As",
            )
            export_filter_mode = gr.Dropdown(
                choices=["All", "Random Sample"],
                value="All",
                label="Export Filter",
            )
            random_sample_count = gr.Number(
                value=5,
                precision=0,
                minimum=1,
                label="Random Sample Count (used only for Random Sample)",
            )
            submit_btn = gr.Button("Extract Text from All Files", variant="primary")

        with gr.Column():
            output = gr.Textbox(label="Extracted Text", lines=20)
            export_file = gr.File(label="Download Exported Results")

    submit_btn.click(
        fn=process_files,
        inputs=[files_input, mode, export_format, export_filter_mode, random_sample_count],
        outputs=[output, export_file],
    )


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Starting OCR Web Interface...")
    print("  Modes: PDF | Printed Text (Tesseract) | Handwriting Single Word | Handwriting Full Page")
    print("=" * 60 + "\n")

    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Soft(),
    )