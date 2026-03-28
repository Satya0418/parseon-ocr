"""
=============================================================
dataloader.py  —  TensorFlow Data Pipeline for IAM Dataset
=============================================================

WHY DO WE NEED THIS?
--------------------
Loading ALL 100,000+ IAM images into RAM at once would require
~10 GB of memory — your machine would crash.

Instead, we build a streaming data pipeline that:
  1. Reads image paths from the CSV (tiny — fits in RAM easily)
  2. Loads and preprocesses images ONE BATCH at a time while training
  3. Prefetches the next batch while the GPU processes the current one
  4. Applies augmentation ON THE FLY during training (saves disk space)

This is called a DATA GENERATOR or DATA PIPELINE.
TensorFlow's tf.data API is the standard way to build this.

WHAT IS A BATCH?
  Instead of training on 1 image at a time (too slow) or all images
  at once (too much RAM), we train on BATCHES of e.g. 16 or 32 images.
  The model averages the gradient across the whole batch → more stable.

CTC LABEL PADDING:
  CTC Loss requires labels to be the SAME length within a batch.
  Since different text lines have different lengths ("Hi" vs "Hello world"),
  we PAD shorter labels with -1 (ignored by TF during loss computation).
=============================================================
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf

# Import our own modules (relative imports work when run as package)
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.preprocessing.image_processor import preprocess_image, IMG_HEIGHT, IMG_WIDTH
from backend.preprocessing.augmentation   import augment_image
from backend.utils.char_map               import build_char_maps, encode_label, ALPHABET


# ── Constants ─────────────────────────────────────────────────────
BATCH_SIZE   = 16      # Number of samples per training step
MAX_LABEL_LEN = 32     # Max characters per WORD (longest IAM word ≤ 25; 32 gives headroom)
PAD_VALUE    = -1      # Value used to pad shorter labels to MAX_LABEL_LEN


def load_split_csv(csv_path: str) -> pd.DataFrame:
    """
    Load a train/val/test split CSV file.

    WHY: Separating data loading from data processing keeps code modular.
    This function is the single entry point for reading split files.

    Parameters
    ----------
    csv_path : str  — path to train.csv, val.csv, or test.csv

    Returns
    -------
    pd.DataFrame  — columns: ['image_path', 'label', 'form_id', 'line_id']
    """
    df = pd.read_csv(csv_path)
    # Drop rows with missing labels or paths
    df = df.dropna(subset=["image_path", "label"])
    df = df[df["label"].str.strip().str.len() > 0]
    df = df.reset_index(drop=True)
    print(f"[Dataloader] Loaded {len(df)} samples from {csv_path}")
    return df


def encode_batch_labels(labels: list, char_to_int: dict) -> tuple:
    """
    Encode a batch of text labels into padded integer arrays.

    WHY: TensorFlow operations need uniform-shape tensors.
    Since text lines have different lengths, we pad all labels in a batch
    to the same length using PAD_VALUE = -1.

    CTC Loss knows to IGNORE positions with value -1, so padding
    does not affect the training signal.

    Parameters
    ----------
    labels       : list of str   — batch of text label strings
    char_to_int  : dict          — character → integer map

    Returns
    -------
    encoded      : np.ndarray  — shape (batch_size, MAX_LABEL_LEN), int32
    label_lengths: np.ndarray  — shape (batch_size,), the real length of each label (before padding)

    Example
    -------
    labels = ["Hi", "Hello"]
    encoded → [[34, 35, -1, -1, -1], [34, 5, 12, 12, 15]]   (with MAX=5)
    label_lengths → [2, 5]
    """
    batch_size = len(labels)
    encoded    = np.full((batch_size, MAX_LABEL_LEN), PAD_VALUE, dtype=np.int32)
    lengths    = np.zeros(batch_size, dtype=np.int32)

    for i, text in enumerate(labels):
        enc = encode_label(text, char_to_int)[:MAX_LABEL_LEN]  # truncate if too long
        encoded[i, :len(enc)] = enc
        lengths[i] = len(enc)

    return encoded, lengths


class OCRDataGenerator(tf.keras.utils.Sequence):
    """
    Keras data generator for OCR training.

    WHY USE A Sequence GENERATOR?
    tf.keras.utils.Sequence is the standard Keras interface for custom
    data generators. It handles:
      - Shuffling at the end of each epoch
      - Correct batch indexing
      - Safe multi-process loading

    HOW IT WORKS:
    The generator knows the full list of image paths and labels.
    For each batch index, it:
      1. Selects batch_size samples from the list
      2. Loads and preprocesses each image
      3. Applies augmentation if in training mode
      4. Encodes the labels
      5. Returns a ready-to-train (inputs_dict, outputs_dict) pair

    The inputs_dict and outputs_dict format is required by our custom
    CTC loss layer (explained in ctc_loss.py).
    """

    def __init__(self,
                 df:           pd.DataFrame,
                 char_to_int:  dict,
                 batch_size:   int  = BATCH_SIZE,
                 augment:      bool = False,
                 shuffle:      bool = True):
        """
        Parameters
        ----------
        df          : pd.DataFrame  — split dataframe (image_path + label columns)
        char_to_int : dict          — character maps from char_map.py
        batch_size  : int           — samples per batch
        augment     : bool          — apply augmentation? (True for train only)
        shuffle     : bool          — shuffle data each epoch?
        """
        self.df          = df.reset_index(drop=True)
        self.char_to_int = char_to_int
        self.batch_size  = batch_size
        self.augment     = augment
        self.shuffle     = shuffle
        self.indices     = np.arange(len(self.df))

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        """
        Return the number of batches per epoch.

        WHY: Keras calls this to know when one epoch ends.
        We use floor division so the last incomplete batch is dropped.
        (Dropping the last batch avoids shape errors with variable-size batches.)
        """
        return len(self.df) // self.batch_size

    def __getitem__(self, batch_idx: int) -> tuple:
        """
        Load and return one batch at index `batch_idx`.

        WHY: Keras calls __getitem__(0), __getitem__(1), ... for each batch.
        This is where image loading, preprocessing, and augmentation happen.

        Parameters
        ----------
        batch_idx : int  — which batch to return

        Returns
        -------
        tuple (inputs_dict, dummy_outputs)
          inputs_dict = {
              'input_images'  : float32 array (batch, H, W, 1)
              'label_encoded' : int32 array   (batch, MAX_LABEL_LEN)
              'image_widths'  : int32 array   (batch,) = W/4 for each image
              'label_lengths' : int32 array   (batch,) real label length
          }
          dummy_outputs = np.zeros(batch_size)  — CTC loss is computed inside model
        """
        # Select the indices for this batch
        start = batch_idx * self.batch_size
        end   = start + self.batch_size
        batch_indices = self.indices[start:end]

        images      = []
        raw_labels  = []

        for idx in batch_indices:
            row = self.df.iloc[idx]

            # Remap Windows local paths to relative paths (needed when running on Colab)
            img_path = row["image_path"].replace('\\', '/')
            marker = 'data/iam_words'
            _idx = img_path.find(marker)
            if _idx != -1:
                img_path = img_path[_idx:]

            # Load and preprocess the image
            img = preprocess_image(img_path)

            # Apply augmentation during training only
            if self.augment:
                img = augment_image(img, augment_prob=0.4)

            images.append(img)
            raw_labels.append(row["label"])

        # Stack images into a batch array: (batch_size, H, W, 1)
        images_batch = np.stack(images, axis=0).astype(np.float32)

        # Encode and pad the text labels
        encoded_labels, label_lengths = encode_batch_labels(raw_labels, self.char_to_int)

        # Image widths after CNN feature extraction.
        # The CNN reduces width by a factor of 4 (due to 2 MaxPool layers with stride 2).
        # This tells CTC how many time steps the model outputs.
        image_widths = np.full(
            (self.batch_size,),
            IMG_WIDTH // 4,     # e.g. 256 // 4 = 64 time steps
            dtype=np.int32
        )

        inputs = {
            "input_images":  images_batch,
            "label_encoded": encoded_labels,
            "image_widths":  image_widths,
            "label_lengths": label_lengths,
        }

        # The actual loss is computed inside the model's CTC layer.
        # Keras still needs a "y_true" — we pass zeros as a dummy.
        dummy_outputs = np.zeros(self.batch_size, dtype=np.float32)

        return inputs, dummy_outputs

    def on_epoch_end(self):
        """
        Called automatically by Keras at the end of each epoch.

        WHY: We reshuffle the data at the end of every epoch so the model
        sees samples in a different order each time. This prevents the model
        from learning the ORDER of the data rather than the content.
        """
        if self.shuffle:
            np.random.shuffle(self.indices)


def build_generators(splits_dir: str = "data/iam_words/splits",
                     batch_size: int = BATCH_SIZE) -> tuple:
    """
    Build train, validation, and test data generators in one call.

    WHY: This is a convenience function that wires together all the
    dataloader components with the correct settings for each split:
      - Train  : augment=True,  shuffle=True
      - Val    : augment=False, shuffle=False
      - Test   : augment=False, shuffle=False

    Parameters
    ----------
    splits_dir : str  — directory containing train.csv, val.csv, test.csv
    batch_size : int  — samples per batch

    Returns
    -------
    tuple  (train_gen, val_gen, test_gen, char_to_int, int_to_char)
    """
    char_to_int, int_to_char = build_char_maps(ALPHABET)

    train_df = load_split_csv(os.path.join(splits_dir, "train.csv"))
    val_df   = load_split_csv(os.path.join(splits_dir, "val.csv"))
    test_df  = load_split_csv(os.path.join(splits_dir, "test.csv"))

    train_gen = OCRDataGenerator(train_df, char_to_int, batch_size, augment=True,  shuffle=True)
    val_gen   = OCRDataGenerator(val_df,   char_to_int, batch_size, augment=False, shuffle=False)
    test_gen  = OCRDataGenerator(test_df,  char_to_int, batch_size, augment=False, shuffle=False)

    print(f"[Dataloader] Generators ready:")
    print(f"  Train batches : {len(train_gen)}")
    print(f"  Val   batches : {len(val_gen)}")
    print(f"  Test  batches : {len(test_gen)}")

    return train_gen, val_gen, test_gen, char_to_int, int_to_char
