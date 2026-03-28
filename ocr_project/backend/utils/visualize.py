"""
=============================================================
visualize.py  —  Training Curves & Prediction Visualization
=============================================================

WHY DO WE NEED THIS?
--------------------
Numbers alone during training don't tell the full story.
Visualizations help you:
  1. Spot overfitting (train loss goes down but val loss goes up)
  2. See if learning rate is too high (loss spikes)
  3. Visually compare what the model predicted vs ground truth
  4. Debug the preprocessing pipeline (see what the model actually sees)

All plots are saved as PNG files so they work in Colab and locally.
=============================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # Use non-interactive backend (works in Colab too)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2


def plot_training_history(history: dict, save_path: str = "saved_models/training_history.png"):
    """
    Plot training and validation loss curves side by side.

    WHY: The loss curve shows HOW the model learned over time.
    - Both curves going down  → model is learning correctly
    - Train down, Val up      → OVERFITTING (memorizing, not learning)
    - Both staying flat       → model is stuck (learning rate too low?)

    Parameters
    ----------
    history : dict
        Dictionary with keys 'loss' and 'val_loss' (lists of float values).
        This is the .history attribute of TensorFlow's model.fit() return value.

        Example:
            history = {
                'loss':     [3.2, 2.8, 2.4, 2.1],
                'val_loss': [3.4, 3.1, 2.9, 2.7]
            }

    save_path : str
        Where to save the plot PNG file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = range(1, len(history["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    # ── Loss Curve ──────────────────────────────────────────────
    axes[0].plot(epochs, history["loss"],     "b-o", label="Train Loss",  markersize=4)
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Val Loss",    markersize=4)
    axes[0].set_title("CTC Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── CER Curve (if available) ─────────────────────────────────
    if "cer" in history:
        axes[1].plot(epochs, history["cer"],     "g-o", label="Train CER", markersize=4)
        axes[1].plot(epochs, history["val_cer"], "m-o", label="Val CER",   markersize=4)
        axes[1].set_title("Character Error Rate over Epochs")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("CER (lower = better)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "CER not tracked yet", ha="center", va="center",
                     fontsize=12, color="gray")
        axes[1].set_title("CER (not available)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualize] Saved training history → {save_path}")


def show_predictions(images: list, predictions: list, ground_truths: list,
                     save_path: str = "saved_models/predictions_sample.png",
                     n: int = 8):
    """
    Display a grid of images with their predicted and true text labels.

    WHY: Seeing actual images + predictions during evaluation tells you
    WHAT kinds of characters the model struggles with (e.g., '1' vs 'l',
    'O' vs '0', connected cursive letters).

    Parameters
    ----------
    images       : list of np.ndarray  — preprocessed image arrays (H, W) grayscale
    predictions  : list of str         — model-predicted text for each image
    ground_truths: list of str         — actual true text for each image
    save_path    : str                 — where to save the output PNG
    n            : int                 — how many samples to display (max)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    n = min(n, len(images))
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 2.5))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes.flatten()

    for i in range(len(axes)):
        ax = axes[i]
        if i < n:
            img = images[i]
            # Squeeze channel dim if present
            if img.ndim == 3:
                img = img[:, :, 0]

            ax.imshow(img, cmap="gray", aspect="auto")

            pred = predictions[i]
            truth = ground_truths[i]
            color = "green" if pred == truth else "red"

            ax.set_title(f"GT:   {truth}\nPred: {pred}",
                         fontsize=8, color=color, loc="left")
            ax.axis("off")
        else:
            ax.axis("off")   # hide empty subplot slots

    plt.suptitle("Model Predictions (Green=Correct, Red=Wrong)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualize] Saved predictions grid → {save_path}")


def show_preprocessed_sample(original: np.ndarray, processed: np.ndarray,
                              label: str = "",
                              save_path: str = "saved_models/preprocess_sample.png"):
    """
    Show an original image side-by-side with its preprocessed version.

    WHY: A critical debugging tool. When model performance is bad, the first
    thing to check is: 'Is the input data actually correct?'
    This function lets you visually verify the preprocessing pipeline.

    Parameters
    ----------
    original  : np.ndarray  — raw image before preprocessing
    processed : np.ndarray  — image after preprocessing (model input)
    label     : str         — ground truth text for this image
    save_path : str         — where to save the comparison PNG
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3))

    axes[0].imshow(original if original.ndim == 3 else original, cmap="gray")
    axes[0].set_title(f"Original Image\nLabel: '{label}'", fontsize=10)
    axes[0].axis("off")

    disp = processed[:, :, 0] if processed.ndim == 3 else processed
    axes[1].imshow(disp, cmap="gray")
    axes[1].set_title(f"After Preprocessing\nShape: {processed.shape}", fontsize=10)
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualize] Saved preprocessing comparison → {save_path}")


def plot_sample_batch(images: list, labels: list,
                      save_path: str = "saved_models/sample_batch.png"):
    """
    Visualize a batch of training samples with their labels.

    WHY: Before training starts, always verify your dataloader is producing
    correct batches. If labels don't match images, the model learns nothing.

    Parameters
    ----------
    images : list of np.ndarray  — batch of preprocessed images
    labels : list of str         — decoded text labels for each image
    save_path : str
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    n = min(12, len(images))
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 2.5))
    axes = axes.flatten()

    for i in range(len(axes)):
        ax = axes[i]
        if i < n:
            img = images[i]
            if img.ndim == 3:
                img = img[:, :, 0]
            ax.imshow(img, cmap="gray", aspect="auto")
            ax.set_title(labels[i], fontsize=8)
            ax.axis("off")
        else:
            ax.axis("off")

    plt.suptitle("Sample Training Batch", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualize] Saved sample batch → {save_path}")
