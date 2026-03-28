"""
=============================================================
download_iam.py  —  IAM Word Database Setup
=============================================================

Dataset : nibinv23/iam-handwriting-word-database  (Kaggle, no terms wall)
  - 115,000+ handwritten WORD images (.png)
  - Labels in iam_words/words.txt (format: word-id ok graylevel x y w h tag text)

Difference from line-level IAM:
  Line-level → each image is a full sentence (~50–80 chars)
  Word-level → each image is a single word (1–25 chars)  ← THIS DATASET
  Word-level is easier to start with and trains faster.

This script will:
  STEP 0 → Download the zip (if not already present)
  STEP 1 → Extract the zip (~1.1 GB)
  STEP 2 → Parse words.txt → labels.csv
  STEP 3 → Split into train / val / test by form (80/10/10)
  STEP 4 → Verify a few samples

ONE-TIME SETUP:
  Place kaggle.json at  C:\\Users\\<YourName>\\.kaggle\\kaggle.json
  Get it from: kaggle.com → Profile → Settings → API → Create New Token

HOW TO RUN:
  python backend/dataset/download_iam.py

WHAT IT PRODUCES:
  data/iam_words/iam_words/words/   ← 115k word images
  data/iam_words/labels.csv         ← all samples
  data/iam_words/splits/train.csv
  data/iam_words/splits/val.csv
  data/iam_words/splits/test.csv
============================================================="""

import os
import sys
import subprocess
import shutil
import argparse
import zipfile
import random
import pandas as pd
from pathlib import Path


# ── Paths ──────────────────────────────────────────────────────────
IAM_ROOT     = os.path.join("data", "iam_words")
ZIP_PATH     = os.path.join(IAM_ROOT, "iam-handwriting-word-database.zip")
WORDS_DIR    = os.path.join(IAM_ROOT, "iam_words", "words")
WORDS_TXT    = os.path.join(IAM_ROOT, "iam_words", "words.txt")
OUTPUT_CSV   = os.path.join(IAM_ROOT, "labels.csv")
SPLITS_DIR   = os.path.join(IAM_ROOT, "splits")

# Kaggle dataset — NO terms acceptance required (unlike vikramtiwari/iam-dataset)
KAGGLE_DATASET = "nibinv23/iam-handwriting-word-database"



# ═══════════════════════════════════════════════════════════════════
# STEP 0 — Download if zip is missing
# ═══════════════════════════════════════════════════════════════════

def download_if_missing():
    """
    Download the IAM word database zip via Kaggle Python API if not present.

    The dataset 'nibinv23/iam-handwriting-word-database' does NOT require
    terms acceptance — it downloads freely once your kaggle.json is placed at:
      Windows : C:\\Users\\<YourName>\\.kaggle\\kaggle.json
      Linux   : ~/.kaggle/kaggle.json

    Get your token at kaggle.com → Profile → Settings → API → Create New Token
    """
    if os.path.exists(ZIP_PATH) and os.path.getsize(ZIP_PATH) > 100_000_000:
        print(f"[Step 0] Zip already present ({os.path.getsize(ZIP_PATH) // 1_000_000} MB)  ✓")
        return

    print("[Step 0] Zip not found. Downloading from Kaggle (~1.1 GB)...")
    try:
        from kaggle import KaggleApi
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "-q"])
        from kaggle import KaggleApi

    api = KaggleApi()
    api.authenticate()
    os.makedirs(IAM_ROOT, exist_ok=True)
    api.dataset_download_files(KAGGLE_DATASET, path=IAM_ROOT, quiet=False, unzip=False)
    print("[Step 0] Download complete  ✓")


# ═══════════════════════════════════════════════════════════════════
# STEP 1 — Extract the zip
# ═══════════════════════════════════════════════════════════════════

def extract_zip():
    """
    Unzip iam-handwriting-word-database.zip into data/iam_words/.

    The zip contains:
      iam_words/words.txt          ← label file
      iam_words/words/a01/...      ← PNG images

    After extraction:
      data/iam_words/iam_words/words/a01/a01-000u/a01-000u-00-00.png

    Skips if images already extracted.
    """
    if os.path.isdir(WORDS_DIR):
        count = sum(1 for _ in Path(WORDS_DIR).rglob("*.png"))
        if count > 50_000:
            print(f"[Step 1] Already extracted ({count:,} images)  ✓")
            return

    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError(
            f"Zip not found at {ZIP_PATH}. Run the script without --skip_download."
        )

    zip_size_mb = os.path.getsize(ZIP_PATH) / (1024 * 1024)
    print(f"[Step 1] Extracting {ZIP_PATH} ({zip_size_mb:.0f} MB)...")
    print("         This takes 2–5 minutes...")

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(IAM_ROOT)

    count = sum(1 for _ in Path(WORDS_DIR).rglob("*.png"))
    print(f"[Step 1] Extracted {count:,} word images  ✓")


# ═══════════════════════════════════════════════════════════════════
# STEP 2 — Parse words.txt → labels.csv
# ═══════════════════════════════════════════════════════════════════

def parse_words_txt() -> pd.DataFrame:
    """
    Parse the IAM words.txt label file into a clean DataFrame.

    FILE FORMAT — each non-comment line:
      a01-000u-00-00  ok  154  408  768  27  51  AT  A
      |               |                           |   |
      word-id      result                       tag  TRANSCRIPTION

    Rules:
      - Skip lines where result == 'er'  (bad segmentation → noisy training data)
      - Map word-id → image path:
          a01-000u-00-00 → data/iam_words/iam_words/words/a01/a01-000u/a01-000u-00-00.png
      - Only include images that actually exist on disk

    WHY filter 'er' segmentations?
      The bounding box was drawn incorrectly — the image may contain partial
      letters from neighbouring words.  Training on such noisy pairs
      confuses the model and slows convergence.
    """
    if not os.path.exists(WORDS_TXT):
        raise FileNotFoundError(
            f"Label file not found at {WORDS_TXT}.  Run extraction first."
        )

    print(f"[Step 2] Parsing {WORDS_TXT}...")
    records        = []
    skipped_er     = 0
    skipped_missing = 0

    with open(WORDS_TXT, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split(" ")
            if len(parts) < 9:
                continue

            word_id = parts[0]   # e.g. a01-000u-00-00
            result  = parts[1]   # 'ok' or 'er'
            label   = parts[8].strip()   # transcription

            if result == "er" or not label:
                skipped_er += 1
                continue

            # Build path: a01-000u-00-00 → words/a01/a01-000u/a01-000u-00-00.png
            id_parts = word_id.split("-")
            folder1  = id_parts[0]              # a01
            folder2  = "-".join(id_parts[:2])   # a01-000u
            img_abs  = os.path.join(WORDS_DIR, folder1, folder2, f"{word_id}.png")

            if not os.path.exists(img_abs):
                skipped_missing += 1
                continue

            records.append({
                "image_path": img_abs,
                "label":      label,
                "form_id":    folder2,   # used for train/val/test split
            })

    df = pd.DataFrame(records)
    print(f"[Step 2] Found {len(df):,} valid word samples")
    print(f"         Skipped {skipped_er:,} 'er' (bad segmentation)")
    print(f"         Skipped {skipped_missing:,} missing image files")
    return df


# ═══════════════════════════════════════════════════════════════════
# STEP 3 — Train / Val / Test split by form
# ═══════════════════════════════════════════════════════════════════

def split_dataset(df: pd.DataFrame,
                  train_ratio: float = 0.80,
                  val_ratio:   float = 0.10,
                  seed:        int   = 42):
    """
    Split samples into train / val / test by FORM ID (not by individual word).

    WHY split by form?
      Each form (e.g. a01-000u) is written by one person in one sitting.
      A random word-level split would put words from the same handwriting
      style into both train and test — the model would 'cheat' by recognising
      the writer's style rather than the actual characters.
      Form-level splitting ensures test writers are completely unseen.

    Splits: 80% train | 10% val | 10% test
    """
    random.seed(seed)
    forms = sorted(df["form_id"].unique())
    random.shuffle(forms)

    n_train = int(len(forms) * train_ratio)
    n_val   = int(len(forms) * val_ratio)
    train_forms = set(forms[:n_train])
    val_forms   = set(forms[n_train : n_train + n_val])
    test_forms  = set(forms[n_train + n_val :])

    train_df = df[df["form_id"].isin(train_forms)].reset_index(drop=True)
    val_df   = df[df["form_id"].isin(val_forms)].reset_index(drop=True)
    test_df  = df[df["form_id"].isin(test_forms)].reset_index(drop=True)

    print(f"[Step 3] Train : {len(train_df):>7,} samples  ({len(train_forms)} forms)")
    print(f"         Val   : {len(val_df):>7,} samples  ({len(val_forms)} forms)")
    print(f"         Test  : {len(test_df):>7,} samples  ({len(test_forms)} forms)")
    return train_df, val_df, test_df


# ═══════════════════════════════════════════════════════════════════
# STEP 4 — Save CSVs
# ═══════════════════════════════════════════════════════════════════

def save_csvs(df, train_df, val_df, test_df):
    """Write labels.csv and the three split CSVs."""
    os.makedirs(SPLITS_DIR, exist_ok=True)
    cols = ["image_path", "label"]
    df[cols].to_csv(OUTPUT_CSV,                              index=False)
    train_df[cols].to_csv(os.path.join(SPLITS_DIR, "train.csv"), index=False)
    val_df[cols].to_csv(  os.path.join(SPLITS_DIR, "val.csv"),   index=False)
    test_df[cols].to_csv( os.path.join(SPLITS_DIR, "test.csv"),  index=False)
    print(f"[Step 4] Saved → {OUTPUT_CSV}")
    print(f"         Saved → {SPLITS_DIR}/train.csv | val.csv | test.csv")


# ═══════════════════════════════════════════════════════════════════
# STEP 5 — Sanity check
# ═══════════════════════════════════════════════════════════════════

def verify_dataset():
    """Load 5 random images to confirm they're readable and print label stats."""
    import cv2

    train_csv = os.path.join(SPLITS_DIR, "train.csv")
    df        = pd.read_csv(train_csv)
    all_df    = pd.read_csv(OUTPUT_CSV)

    print(f"[Step 5] Verifying dataset...")
    sample = df.sample(min(5, len(df)), random_state=42)
    ok = 0
    for _, row in sample.iterrows():
        img = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  [WARN] Cannot load: {row['image_path']}")
        else:
            h, w = img.shape
            print(f"  ✓  {w}×{h}  '{row['label']}'")
            ok += 1

    lengths = all_df["label"].str.len()
    chars   = set("".join(all_df["label"].tolist()))
    print(f"\n[Stats] Total     : {len(all_df):,} samples")
    print(f"[Stats] Max label : {lengths.max()} chars  (min={lengths.min()}, mean={lengths.mean():.1f})")
    print(f"[Stats] Unique chars ({len(chars)}): {''.join(sorted(chars))}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Set up IAM word database for OCR training")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip download step (zip already present)")
    parser.add_argument("--skip_extract",  action="store_true",
                        help="Skip extraction step (images already extracted)")
    parser.add_argument("--skip_parse",    action="store_true",
                        help="Skip parsing step (labels.csv already built)")
    args = parser.parse_args()

    print("=" * 60)
    print("  IAM Word Database — Setup")
    print("=" * 60)

    if not args.skip_download:
        download_if_missing()

    if not args.skip_extract:
        extract_zip()

    if not args.skip_parse:
        df = parse_words_txt()
        if len(df) == 0:
            print("[ERROR] No samples found after parsing.")
            sys.exit(1)
        train_df, val_df, test_df = split_dataset(df)
        save_csvs(df, train_df, val_df, test_df)

    verify_dataset()

    print("\n" + "=" * 60)
    print("  Done! Next steps:")
    print("  1. Upload data/iam_words/ to Google Drive")
    print("  2. Open notebooks/OCR_CRNN_Training_Colab.ipynb in Colab")
    print("  3. Train on a free T4 GPU")
    print("=" * 60)


if __name__ == "__main__":
    main()
