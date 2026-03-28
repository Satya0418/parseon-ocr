from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

# Keep the alphabet in sync with your training scripts.
ALPHABET = (
    " !\"#&'()*+,-./0123456789:;?"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
)

IMG_HEIGHT = 64
IMG_WIDTH = 256
MODEL_PATH = Path(__file__).resolve().parent.parent / "ocr_project" / "crnn_iam_v1_inference.keras"

app = FastAPI(title="OCR Model Checker", version="1.0.0")
app.mount("/static", StaticFiles(directory=Path(__file__).resolve().parent / "static"), name="static")


class OCRService:
    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: Path) -> tf.keras.Model:
        # Support both single-file .keras and directory-style bundles.
        if model_path.is_dir():
            config_path = model_path / "config.json"
            weights_path = model_path / "model.weights.h5"
            if not config_path.exists() or not weights_path.exists():
                raise FileNotFoundError(
                    f"Expected {config_path.name} and {weights_path.name} in {model_path}"
                )

            config_text = config_path.read_text(encoding="utf-8")
            model = tf.keras.models.model_from_json(config_text)
            model.load_weights(weights_path)
            return model

        return tf.keras.models.load_model(model_path, compile=False)

    def preprocess(self, raw_bytes: bytes) -> np.ndarray:
        try:
            image = Image.open(BytesIO(raw_bytes)).convert("L")
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc

        # Normalize orientation and preserve aspect ratio with white padding.
        image = ImageOps.exif_transpose(image)
        image = ImageOps.autocontrast(image)

        src_w, src_h = image.size
        if src_w == 0 or src_h == 0:
            raise HTTPException(status_code=400, detail="Image has invalid size")

        scale = min(IMG_WIDTH / src_w, IMG_HEIGHT / src_h)
        resized_w = max(1, int(src_w * scale))
        resized_h = max(1, int(src_h * scale))

        resized = image.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (IMG_WIDTH, IMG_HEIGHT), color=255)
        offset_x = (IMG_WIDTH - resized_w) // 2
        offset_y = (IMG_HEIGHT - resized_h) // 2
        canvas.paste(resized, (offset_x, offset_y))

        arr = np.asarray(canvas, dtype=np.float32)

        # If text appears white on black, invert so training-style contrast is closer.
        if arr.mean() < 90:
            arr = 255.0 - arr

        arr = arr / 255.0
        arr = np.expand_dims(arr, axis=(0, -1))
        return arr

    def decode_prediction(self, probs: np.ndarray) -> tuple[str, float]:
        input_len = np.full((probs.shape[0],), probs.shape[1], dtype=np.int32)
        decoded, _ = tf.keras.backend.ctc_decode(probs, input_length=input_len, greedy=True)
        token_ids = decoded[0].numpy()[0]

        chars: list[str] = []
        conf_parts: list[float] = []

        timestep_argmax = probs[0].argmax(axis=-1)
        timestep_max = probs[0].max(axis=-1)

        for idx in token_ids:
            if idx < 0 or idx >= len(ALPHABET):
                continue
            chars.append(ALPHABET[int(idx)])

        for t, cls_id in enumerate(timestep_argmax):
            if int(cls_id) < len(ALPHABET):
                conf_parts.append(float(timestep_max[t]))

        text = "".join(chars)
        confidence = float(np.mean(conf_parts)) if conf_parts else 0.0
        return text, confidence


ocr = OCRService(MODEL_PATH)


def run_ocr_from_image_bytes(raw: bytes) -> dict[str, object]:
    batch = ocr.preprocess(raw)
    probs = ocr.model.predict(batch, verbose=0)
    text, confidence = ocr.decode_prediction(probs)
    return {
        "prediction": text,
        "confidence": round(confidence, 4),
        "timesteps": int(probs.shape[1]),
        "num_classes": int(probs.shape[2]),
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(__file__).resolve().parent / "static" / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": str(MODEL_PATH)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    return run_ocr_from_image_bytes(raw)


@app.post("/predict-handwritten")
async def predict_handwritten(file: UploadFile = File(...)) -> dict[str, object]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a handwritten image file")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    result = run_ocr_from_image_bytes(raw)
    result["source"] = "handwritten"
    return result


@app.post("/predict-pdf")
async def predict_pdf(file: UploadFile = File(...)) -> dict[str, object]:
    if file.content_type not in {"application/pdf", "application/x-pdf"}:
        raise HTTPException(status_code=400, detail="Please upload a PDF file")

    raw_pdf = await file.read()
    if not raw_pdf:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty")

    try:
        pdf = pdfium.PdfDocument(raw_pdf)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"Invalid PDF: {exc}") from exc

    page_results: list[dict[str, object]] = []
    extracted_lines: list[str] = []

    for page_idx in range(len(pdf)):
        page = pdf.get_page(page_idx)
        try:
            bitmap = page.render(scale=2.0)
            pil_img = bitmap.to_pil().convert("L")
            buf = BytesIO()
            pil_img.save(buf, format="PNG")
            prediction = run_ocr_from_image_bytes(buf.getvalue())
            prediction["page"] = page_idx + 1
            page_results.append(prediction)
            extracted_lines.append(str(prediction["prediction"]))
        finally:
            page.close()

    pdf.close()

    return {
        "source": "pdf",
        "pages": len(page_results),
        "extracted_text": "\n".join(extracted_lines),
        "page_results": page_results,
    }
