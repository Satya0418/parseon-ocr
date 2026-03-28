"""
Kaggle training script for full-page handwriting OCR via LINE-LEVEL CRNN.

Why line-level first?
- Full pages are best handled as: page -> line segmentation -> line recognizer.
- This is much more stable than forcing a word model on full pages.

Designed for Kaggle with IAM Handwritten Forms dataset attached as Input.
Expected IAM structure somewhere under /kaggle/input:
- lines.txt
- lines/...

Usage on Kaggle terminal (or notebook cell):
  python kaggle_train_iam_forms_lines.py --epochs 60 --batch_size 16

Outputs in /kaggle/working/fullpage_saved_models:
- crnn_iam_lines_best.weights.h5
- crnn_iam_lines_final.weights.h5
- crnn_iam_lines_inference.keras
- training_lines.csv
- char_map_lines.json
"""

import os
import json
import random
import argparse
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# -----------------------------
# Config
# -----------------------------
IMG_HEIGHT = 64
IMG_WIDTH = 1024
MAX_LABEL_LEN = 128
PAD_VALUE = -1
DEFAULT_SEED = 42

ALPHABET = (
    " !\"#&'()*+,-./0123456789:;?"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
)
BLANK_INDEX = len(ALPHABET)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[GPU] Found {len(gpus)} GPU(s). Memory growth enabled.")
    else:
        print("[GPU] No GPU found. Training will run on CPU.")


# -----------------------------
# IAM discovery + parsing
# -----------------------------
def find_iam_lines_root(base: str = "/kaggle/input"):
    """Find dataset root that contains both lines.txt and lines/ directory."""
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Input directory not found: {base}")

    for root, dirs, files in os.walk(base):
        if "lines.txt" in files and "lines" in dirs:
            return root

    # fallback: allow nested lines/ and lines.txt in nearby folders
    candidates = []
    for root, dirs, files in os.walk(base):
        if "lines.txt" in files:
            candidates.append(root)

    for root in candidates:
        for maybe in [root, os.path.dirname(root), os.path.join(root, "iam"), os.path.join(root, "iam_lines")]:
            if os.path.isfile(os.path.join(maybe, "lines.txt")) and os.path.isdir(os.path.join(maybe, "lines")):
                return maybe

    raise FileNotFoundError(
        "Could not find IAM lines dataset root with lines.txt + lines/. "
        "Attach IAM Handwritten Forms dataset in Kaggle Input."
    )


def parse_iam_lines(lines_txt_path: str, lines_dir: str) -> pd.DataFrame:
    """Parse IAM lines.txt and resolve existing image paths."""
    rows = []
    skipped_status = 0
    skipped_missing = 0

    with open(lines_txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 9:
                continue

            line_id = parts[0]            # e.g., a01-000u-00
            status = parts[1]             # ok / err
            if status != "ok":
                skipped_status += 1
                continue

            text = " ".join(parts[8:]).replace("|", " ").strip()
            if not text:
                continue

            id_parts = line_id.split("-")
            if len(id_parts) < 3:
                continue

            form1 = id_parts[0]                   # a01
            form2 = f"{id_parts[0]}-{id_parts[1]}"  # a01-000u
            img_path = os.path.join(lines_dir, form1, form2, f"{line_id}.png")

            if not os.path.isfile(img_path):
                skipped_missing += 1
                continue

            rows.append({
                "image_path": img_path,
                "label": text,
                "form_id": form2,
                "line_id": line_id,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No valid line samples found from lines.txt")

    print(f"[Data] Parsed lines: {len(df):,}")
    print(f"[Data] Skipped non-ok: {skipped_status:,}")
    print(f"[Data] Skipped missing: {skipped_missing:,}")
    return df


def split_by_form(df: pd.DataFrame, seed: int = DEFAULT_SEED):
    forms = sorted(df["form_id"].unique())
    rng = random.Random(seed)
    rng.shuffle(forms)

    n = len(forms)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train_forms = set(forms[:n_train])
    val_forms = set(forms[n_train:n_train + n_val])
    test_forms = set(forms[n_train + n_val:])

    train_df = df[df["form_id"].isin(train_forms)].reset_index(drop=True)
    val_df = df[df["form_id"].isin(val_forms)].reset_index(drop=True)
    test_df = df[df["form_id"].isin(test_forms)].reset_index(drop=True)

    print(f"[Split] Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")
    return train_df, val_df, test_df


# -----------------------------
# Preprocessing + labels
# -----------------------------
def encode_label(text: str, char_to_int: dict):
    out = []
    for ch in text:
        if ch in char_to_int:
            out.append(char_to_int[ch])
    return out


def preprocess_line_image(path: str, target_h: int = IMG_HEIGHT, target_w: int = IMG_WIDTH):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Robust contrast normalize
    p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
    if p99 - p1 > 5:
        img = np.clip((img.astype(np.float32) - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)

    # IAM-style binarization
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h, w = img.shape[:2]
    if h < 2 or w < 2:
        return None

    scale = target_h / h
    new_w = max(1, int(w * scale))
    resized = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

    if new_w >= target_w:
        out = resized[:, :target_w]
    else:
        pad = np.full((target_h, target_w - new_w), 255, dtype=np.uint8)
        out = np.concatenate([resized, pad], axis=1)

    # Normalize and invert: text bright, bg dark
    out = (255.0 - out.astype(np.float32)) / 255.0
    out = np.expand_dims(out, axis=-1)
    return out.astype(np.float32)


class IAMLineSequence(tf.keras.utils.Sequence):
    def __init__(self, df: pd.DataFrame, char_to_int: dict, batch_size: int, augment: bool, shuffle: bool):
        self.df = df
        self.char_to_int = char_to_int
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.indices = np.arange(len(df))
        self.on_epoch_end()

    def __len__(self):
        return max(1, len(self.indices) // self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, i):
        s = i * self.batch_size
        e = s + self.batch_size
        batch_ids = self.indices[s:e]

        images = []
        labels = []

        for idx in batch_ids:
            row = self.df.iloc[idx]
            img = preprocess_line_image(row["image_path"])
            if img is None:
                continue

            if self.augment and np.random.rand() < 0.35:
                # lightweight camera-like augmentation
                noise = np.random.normal(0, 0.02, img.shape).astype(np.float32)
                img = np.clip(img + noise, 0.0, 1.0)

            enc = encode_label(row["label"], self.char_to_int)
            if not enc:
                continue

            images.append(img)
            labels.append(enc)

        if not images:
            # extremely unlikely fallback to avoid Keras crash
            dummy_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
            images = [dummy_img]
            labels = [[0]]

        bsz = len(images)
        x = np.stack(images, axis=0)

        y = np.full((bsz, MAX_LABEL_LEN), PAD_VALUE, dtype=np.int32)
        y_len = np.zeros((bsz,), dtype=np.int32)
        for j, enc in enumerate(labels):
            enc = enc[:MAX_LABEL_LEN]
            y[j, :len(enc)] = enc
            y_len[j] = len(enc)

        input_len = np.full((bsz,), IMG_WIDTH // 4, dtype=np.int32)

        inputs = {
            "input_images": x,
            "label_encoded": y,
            "image_widths": input_len,
            "label_lengths": y_len,
        }
        return inputs, np.zeros((bsz,), dtype=np.float32)


# -----------------------------
# Model
# -----------------------------
class CTCLayer(layers.Layer):
    def call(self, inputs):
        y_true, y_pred, input_len, label_len = inputs
        if len(tf.shape(input_len)) == 1:
            input_len = tf.expand_dims(input_len, axis=-1)
        if len(tf.shape(label_len)) == 1:
            label_len = tf.expand_dims(label_len, axis=-1)

        loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_len, label_len)
        self.add_loss(tf.reduce_mean(loss))
        return y_pred


def build_crnn_lines(num_classes: int):
    in_img = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="input_images", dtype="float32")
    in_lbl = layers.Input(shape=(None,), name="label_encoded", dtype="int32")
    in_w = layers.Input(shape=(), name="image_widths", dtype="int32")
    in_l = layers.Input(shape=(), name="label_lengths", dtype="int32")

    x = in_img
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 1))(x)

    # (B, H, W, C) -> (B, W, H*C)
    shp = x.shape
    t_steps = int(shp[2])
    feat_dim = int(shp[1]) * int(shp[3])
    x = layers.Permute((2, 1, 3))(x)
    x = layers.Reshape((t_steps, feat_dim))(x)

    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Dropout(0.25)(x)

    y_pred = layers.Dense(num_classes, activation="softmax", name="output_softmax")(x)
    out = CTCLayer(name="ctc_loss")([in_lbl, y_pred, in_w, in_l])

    train_model = keras.Model(inputs=[in_img, in_lbl, in_w, in_l], outputs=out, name="crnn_lines_train")
    infer_model = keras.Model(inputs=in_img, outputs=y_pred, name="crnn_lines_infer")
    return train_model, infer_model


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 16
    lr: float = 1e-3
    seed: int = DEFAULT_SEED


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/fullpage_saved_models")
    args = parser.parse_args()

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.learning_rate, seed=args.seed)

    set_seed(cfg.seed)
    configure_gpu()

    root = find_iam_lines_root("/kaggle/input")
    lines_txt = os.path.join(root, "lines.txt")
    lines_dir = os.path.join(root, "lines")

    print(f"[Data] Using root: {root}")
    df = parse_iam_lines(lines_txt, lines_dir)
    train_df, val_df, test_df = split_by_form(df, cfg.seed)

    char_to_int = {c: i for i, c in enumerate(ALPHABET)}
    int_to_char = {i: c for i, c in enumerate(ALPHABET)}
    int_to_char[BLANK_INDEX] = "[BLANK]"

    train_gen = IAMLineSequence(train_df, char_to_int, cfg.batch_size, augment=True, shuffle=True)
    val_gen = IAMLineSequence(val_df, char_to_int, cfg.batch_size, augment=False, shuffle=False)

    num_classes = len(ALPHABET) + 1
    train_model, infer_model = build_crnn_lines(num_classes=num_classes)

    train_model.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg.lr))

    os.makedirs(args.output_dir, exist_ok=True)
    best_w = os.path.join(args.output_dir, "crnn_iam_lines_best.weights.h5")
    final_w = os.path.join(args.output_dir, "crnn_iam_lines_final.weights.h5")
    infer_p = os.path.join(args.output_dir, "crnn_iam_lines_inference.keras")
    hist_p = os.path.join(args.output_dir, "training_lines.csv")
    cmap_p = os.path.join(args.output_dir, "char_map_lines.json")

    callbacks = [
        keras.callbacks.ModelCheckpoint(best_w, monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        keras.callbacks.CSVLogger(hist_p),
    ]

    print("=" * 80)
    print("Training line-level CRNN for full-page OCR")
    print(f"Train samples: {len(train_df):,} | Val samples: {len(val_df):,}")
    print(f"Image size: {IMG_HEIGHT}x{IMG_WIDTH} | Classes: {num_classes}")
    print(f"Epochs: {cfg.epochs} | Batch size: {cfg.batch_size} | LR: {cfg.lr}")
    print("=" * 80)

    train_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=cfg.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    train_model.save_weights(final_w)
    infer_model.save(infer_p)

    with open(cmap_p, "w", encoding="utf-8") as f:
        json.dump(
            {
                "alphabet": ALPHABET,
                "blank_index": BLANK_INDEX,
                "num_classes": num_classes,
                "image_height": IMG_HEIGHT,
                "image_width": IMG_WIDTH,
            },
            f,
            indent=2,
        )

    print("\nSaved outputs:")
    print(f"- {best_w}")
    print(f"- {final_w}")
    print(f"- {infer_p}")
    print(f"- {hist_p}")
    print(f"- {cmap_p}")

    # Save test split too for later evaluation notebook
    test_csv = os.path.join(args.output_dir, "test_split_lines.csv")
    test_df[["image_path", "label", "form_id", "line_id"]].to_csv(test_csv, index=False)
    print(f"- {test_csv}")


if __name__ == "__main__":
    main()
