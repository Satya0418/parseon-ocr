"""
=============================================================
crnn_model.py  —  CRNN Architecture for OCR
=============================================================

WHAT IS CRNN?
-------------
CRNN = Convolutional Recurrent Neural Network

It is the standard architecture for handwriting and scene text recognition.
It combines two types of neural networks:

  CNN  (Convolutional Neural Network)    → sees the image, extracts visual features
  RNN  (Recurrent Neural Network)        → reads features left-to-right, like reading

Think of it like this:
  - CNN is your EYES — it sees shapes, curves, edges (what letters look like)
  - RNN is your BRAIN — it reads the sequence of shapes from left to right

DETAILED ARCHITECTURE:
======================

INPUT IMAGE  (64 × 512 × 1)   — height=64, width=512, grayscale
     ↓
┌─────────────────────────────────────────────────────┐
│  CNN BLOCK (Feature Extractor)                      │
│                                                     │
│  Conv2D(32)  + BatchNorm + ReLU                    │
│  MaxPool2D(2×2)   → size: 32 × 256 × 32            │
│                                                     │
│  Conv2D(64)  + BatchNorm + ReLU                    │
│  MaxPool2D(2×2)   → size: 16 × 128 × 64            │
│                                                     │
│  Conv2D(128) + BatchNorm + ReLU                    │
│  Conv2D(128) + BatchNorm + ReLU  (no pool)         │
│                 → size: 16 × 128 × 128              │
│                                                     │
│  Conv2D(256) + BatchNorm + ReLU                    │
│  MaxPool(2×1) (only height pooled)  → 8 × 128 × 256│
│                                                     │
│  Conv2D(512) + BatchNorm + ReLU                    │
│                 → size: 8 × 128 × 512               │
└─────────────────────────────────────────────────────┘
     ↓
  RESHAPE: Collapse height×channels into feature vector per column
  (8 × 128 × 512) → (128 time_steps × 4096 features)
     ↓
┌─────────────────────────────────────────────────────┐
│  RNN BLOCK (Sequence Reader)                        │
│                                                     │
│  Bidirectional LSTM(256)   → reads L→R and R→L     │
│  Dropout(0.25)                                      │
│  Bidirectional LSTM(256)                            │
│  Dropout(0.25)                                      │
└─────────────────────────────────────────────────────┘
     ↓
  Dense(num_classes)  →  softmax  →  (128, num_classes)
     ↓
  CTC Loss Layer
     ↓
OUTPUT: text string

WHY BIDIRECTIONAL LSTM?
  A regular LSTM reads left→right only. In handwriting, understanding
  letter 't' at position 50 might require seeing what comes AFTER it
  (the crossbar connects to adjacent letters). Bidirectional LSTM reads
  both directions simultaneously, giving richer context.

WHY TWO LSTM LAYERS?
  The first LSTM learns short-range dependencies (individual characters).
  The second LSTM learns longer-range dependencies (words, letter pairs).
=============================================================
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.utils.char_map  import get_num_classes
from backend.models.ctc_loss import CTCLayer
from backend.preprocessing.image_processor import IMG_HEIGHT, IMG_WIDTH


def build_crnn_model(img_height:   int = IMG_HEIGHT,
                     img_width:    int = IMG_WIDTH,
                     num_classes:  int = None,
                     lstm_units:   int = 256,
                     dropout_rate: float = 0.25) -> keras.Model:
    """
    Build the CRNN model architecture with integrated CTC Loss.

    WHY FUNCTIONAL API (not Sequential)?
    We have MULTIPLE INPUTS:
      - The image tensor
      - The encoded labels (needed for CTC loss during training)
      - The image width (needed for CTC loss)
      - The label length (needed for CTC loss)

    Keras Functional API lets us define multi-input models cleanly.

    Parameters
    ----------
    img_height   : int    — image height (default 64)
    img_width    : int    — image width  (default 512)
    num_classes  : int    — total output classes = len(alphabet) + 1 (blank)
                           If None, uses get_num_classes() from char_map.py
    lstm_units   : int    — size of each LSTM layer (default 256)
    dropout_rate : float  — dropout probability (0.0 = no dropout)

    Returns
    -------
    keras.Model  — compiled training model
                   Inputs : ['input_images', 'label_encoded', 'image_widths', 'label_lengths']
                   Output : softmax predictions (batch, time_steps, num_classes)
    """
    if num_classes is None:
        num_classes = get_num_classes()

    # ── Define Inputs ──────────────────────────────────────────────
    # All four inputs are defined here. This is the multi-input design.
    input_images  = layers.Input(shape=(img_height, img_width, 1),
                                 name="input_images",   dtype="float32")
    label_encoded = layers.Input(shape=(None,),
                                 name="label_encoded",  dtype="int32")
    image_widths  = layers.Input(shape=(),
                                 name="image_widths",   dtype="int32")
    label_lengths = layers.Input(shape=(),
                                 name="label_lengths",  dtype="int32")

    # ── CNN Block ──────────────────────────────────────────────────
    # Each Conv2D + BatchNorm + ReLU trio is called a "ConvBlock".
    # BatchNormalization normalizes activations → speeds up training.
    # ReLU = max(0, x) — introduces non-linearity so the model can
    # learn complex patterns (not just linear relationships).

    x = input_images  # start with the raw image

    # --- ConvBlock 1 ---
    # 32 filters, 3×3 kernel, 'same' padding keeps spatial dimensions
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    # MaxPool halves height AND width: (64, 512) → (32, 256)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # --- ConvBlock 2 ---
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    # (32, 256) → (16, 128)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # --- ConvBlock 3 ---
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)

    # --- ConvBlock 4 (no pooling — preserve spatial resolution) ---
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.Activation("relu", name="relu4")(x)

    # --- ConvBlock 5 (pool only height, not width) ---
    x = layers.Conv2D(256, (3, 3), padding="same", name="conv5")(x)
    x = layers.BatchNormalization(name="bn5")(x)
    x = layers.Activation("relu", name="relu5")(x)
    # Pool only height: (16, 128) → (8, 128)
    x = layers.MaxPooling2D((2, 1), name="pool3")(x)

    # --- ConvBlock 6 ---
    x = layers.Conv2D(512, (3, 3), padding="same", name="conv6")(x)
    x = layers.BatchNormalization(name="bn6")(x)
    x = layers.Activation("relu", name="relu6")(x)

    # At this point: x.shape = (batch, 8, 128, 512)
    # 8 = compressed height, 128 = time steps (columns), 512 = features

    # ── Reshape: (batch, H, W, C) → (batch, W, H*C) ──────────────
    # We need to give the RNN a sequence of feature vectors.
    # Each "time step" corresponds to one COLUMN of the feature map.
    # We collapse the height (8) and channels (512) into a single
    # feature vector per column: 8 * 512 = 4096 features per time step.
    #
    # WHY: LSTM processes sequences shape (batch, time_steps, features).
    # The CNN output (batch, H, W, C) needs to be rearranged to this format.

    # Get shape dynamically (handles variable batch sizes)
    cnn_out_shape = x.shape       # (None, 8, 128, 512)
    seq_len = cnn_out_shape[2]    # 128 time steps
    feat_dim = cnn_out_shape[1] * cnn_out_shape[3]  # 8 * 512 = 4096

    # Permute: (batch, H, W, C) → (batch, W, H, C)
    x = layers.Permute((2, 1, 3), name="permute")(x)

    # Reshape: (batch, W, H, C) → (batch, W, H*C)
    x = layers.Reshape((seq_len, feat_dim), name="reshape_to_seq")(x)

    # ── RNN Block ──────────────────────────────────────────────────
    # Bidirectional LSTM reads the sequence both left→right and right→left.
    # return_sequences=True passes ALL time step outputs to the next layer.
    # (Without it, only the final hidden state is passed — not what we want.)

    # --- BiLSTM Layer 1 ---
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True),
        name="bilstm1"
    )(x)
    x = layers.Dropout(dropout_rate, name="dropout1")(x)

    # --- BiLSTM Layer 2 ---
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True),
        name="bilstm2"
    )(x)
    x = layers.Dropout(dropout_rate, name="dropout2")(x)

    # ── Output Layer ───────────────────────────────────────────────
    # For each time step, predict a probability distribution over all classes.
    # Dense(num_classes) + softmax → (batch, 128, num_classes)
    # Softmax converts raw scores (logits) to probabilities that sum to 1.
    output = layers.Dense(num_classes, activation="softmax", name="output_softmax")(x)

    # ── CTC Loss Layer ─────────────────────────────────────────────
    # This layer computes CTC Loss using output predictions + labels.
    # During inference, we bypass this layer and use `output` directly.
    ctc_output = CTCLayer(name="ctc_loss")(
        [label_encoded, output, image_widths, label_lengths]
    )

    # ── Build Training Model ───────────────────────────────────────
    # This model has 4 inputs and 1 output.
    # Loss is computed inside CTCLayer via add_loss().
    training_model = keras.Model(
        inputs  = [input_images, label_encoded, image_widths, label_lengths],
        outputs = ctc_output,
        name    = "CRNN_OCR_Training"
    )

    return training_model


def build_inference_model(training_model: keras.Model) -> keras.Model:
    """
    Extract an inference-only model from the trained CRNN.

    WHY: The training model has 4 inputs (image + labels + lengths).
    During inference, we only have an IMAGE. No labels.
    So we build a simpler model that:
      - Takes ONLY the image as input
      - Outputs the softmax probability matrix
      - We then decode this matrix to text using the CTC decoder

    HOW: We reuse the EXACT same layers from the training model.
    No retraining needed — weights are shared.

    Parameters
    ----------
    training_model : keras.Model  — the trained model with 4 inputs

    Returns
    -------
    keras.Model  — inference model
                   Input  : image tensor (1, H, W, 1)
                   Output : softmax probabilities (1, time_steps, num_classes)
    """
    # Get the image input tensor from the training model's inputs list
    # training_model.inputs = [input_images, label_encoded, image_widths, label_lengths]
    image_input = training_model.inputs[0]  # First input is 'input_images'

    # Get the softmax output (output of the dense layer, before CTC)
    softmax_output = training_model.get_layer("output_softmax").output

    inference_model = keras.Model(
        inputs  = image_input,
        outputs = softmax_output,
        name    = "CRNN_OCR_Inference"
    )

    return inference_model


def model_summary(model: keras.Model):
    """
    Print a detailed model summary with parameter counts.

    WHY: Before training, always check:
      1. Total parameters — too many = overfitting risk, too few = underfitting
      2. Layer shapes — ensure data flows correctly through the network
      3. Names — verify CTC layer is attached correctly

    A typical CRNN for OCR has ~5–10 million trainable parameters.
    """
    model.summary(line_length=100)
    total = model.count_params()
    print(f"\n  Total parameters    : {total:,}")
    print(f"  Trainable params    : {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")


# ── Quick self-test ───────────────────────────────────────────────
if __name__ == "__main__":
    print("Building CRNN Model...")
    model = build_crnn_model()
    model_summary(model)

    print("\nBuilding Inference Model...")
    inf_model = build_inference_model(model)
    inf_model.summary(line_length=80)

    print("\nTest forward pass...")
    import numpy as np
    dummy_imgs    = np.zeros((2, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
    dummy_labels  = np.zeros((2, 128),                      dtype=np.int32)
    dummy_widths  = np.full((2, 1), IMG_WIDTH // 4,         dtype=np.int32)
    dummy_lengths = np.array([[5], [8]],                    dtype=np.int32)

    out = model.predict({
        "input_images":  dummy_imgs,
        "label_encoded": dummy_labels,
        "image_widths":  dummy_widths,
        "label_lengths": dummy_lengths,
    }, verbose=0)

    print(f"  Output shape: {out.shape}  (expected: (2, 128, {get_num_classes()}))")
    print("  Self-test PASSED")
