"""
=============================================================
train.py  —  Main Training Script
=============================================================

HOW TO RUN THIS ON GOOGLE COLAB:
----------------------------------
1. Upload your entire ocr_project/ folder to Google Drive.
2. In Colab, mount Drive:
       from google.colab import drive
       drive.mount('/content/drive')
3. Set sys.path:
       import sys
       sys.path.insert(0, '/content/drive/MyDrive/ocr_project')
4. Run this script:
       !python backend/training/train.py

THE TRAINING LOOP — HOW DOES IT WORK?
======================================
One "epoch" = the model sees ALL training samples once.
In each epoch:

  FOR EACH BATCH:
    1. Forward Pass:  Image → CNN → RNN → softmax predictions
    2. Loss:          CTC Loss compares predictions to true labels
    3. Backward Pass: Gradients flow backward through all layers
    4. Update:        Adam optimizer updates every weight in the network

  After all batches:
    5. Validation: Run on val set (no gradient update)
    6. Callbacks:  Save checkpoint, maybe reduce LR, check early stop

The model gets progressively better at predicting text from images.
Early epochs: random outputs. By epoch 30+: readable text.
=============================================================
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf

# Make sure Python can find our backend modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from backend.dataset.dataloader   import build_generators
from backend.models.crnn_model    import build_crnn_model, build_inference_model, model_summary
from backend.training.callbacks   import get_callbacks
from backend.utils.char_map       import build_char_maps, ALPHABET, get_num_classes
from backend.utils.visualize      import plot_training_history


# ── Default Hyperparameters ────────────────────────────────────────
# These can be overridden via command-line arguments.
# Start with these defaults — only tune after you see initial results.
DEFAULTS = {
    "splits_dir":      "data/iam_words/splits",   # path to train/val/test CSVs
    "checkpoint_dir":  "saved_models",       # where to save model weights
    "epochs":          100,                  # max training epochs (early stop kicks in)
    "batch_size":      16,                   # samples per gradient update
    "learning_rate":   1e-3,                 # initial Adam learning rate
    "lstm_units":      256,                  # LSTM hidden size
    "dropout":         0.25,                 # dropout rate
    "experiment_name": "crnn_iam_v1",        # name prefix for saved files
}


def parse_args():
    """
    Parse command-line arguments so you can override defaults easily.

    WHY: Hardcoding hyperparameters makes it painful to experiment.
    With argparse, you can run:
        python train.py --epochs 50 --batch_size 32 --learning_rate 0.0005
    without touching the code.
    """
    parser = argparse.ArgumentParser(description="Train CRNN OCR model on IAM dataset")

    parser.add_argument("--splits_dir",      default=DEFAULTS["splits_dir"])
    parser.add_argument("--checkpoint_dir",  default=DEFAULTS["checkpoint_dir"])
    parser.add_argument("--epochs",          type=int,   default=DEFAULTS["epochs"])
    parser.add_argument("--batch_size",      type=int,   default=DEFAULTS["batch_size"])
    parser.add_argument("--learning_rate",   type=float, default=DEFAULTS["learning_rate"])
    parser.add_argument("--lstm_units",      type=int,   default=DEFAULTS["lstm_units"])
    parser.add_argument("--dropout",         type=float, default=DEFAULTS["dropout"])
    parser.add_argument("--experiment_name", default=DEFAULTS["experiment_name"])
    parser.add_argument("--resume_from",     default=None,
                        help="Path to .weights.h5 file to resume training from")

    return parser.parse_args()


def configure_gpu():
    """
    Configure GPU memory growth to prevent TensorFlow from grabbing ALL VRAM.

    WHY: By default, TensorFlow pre-allocates ALL available GPU memory.
    On shared systems (Colab) this causes Out-Of-Memory errors.
    Memory growth mode allocates GPU RAM incrementally as needed.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[Train] GPU found: {[g.name for g in gpus]}")
        print(f"[Train] Memory growth enabled.")
    else:
        print("[Train] No GPU found — training on CPU.")
        print("[Train] TIP: Use Google Colab for free GPU access.")


def print_training_config(args):
    """Print a clear summary of all training settings before starting."""
    print()
    print("=" * 60)
    print(f"{'TRAINING CONFIGURATION':^60}")
    print("=" * 60)
    print(f"  {'Experiment':<22} {args.experiment_name}")
    print(f"  {'Splits Dir':<22} {args.splits_dir}")
    print(f"  {'Checkpoint Dir':<22} {args.checkpoint_dir}")
    print(f"  {'Epochs (max)':<22} {args.epochs}")
    print(f"  {'Batch Size':<22} {args.batch_size}")
    print(f"  {'Learning Rate':<22} {args.learning_rate}")
    print(f"  {'LSTM Units':<22} {args.lstm_units}")
    print(f"  {'Dropout Rate':<22} {args.dropout}")
    print(f"  {'Resume From':<22} {args.resume_from or 'N/A (fresh start)'}")
    print(f"  {'Num Classes':<22} {get_num_classes()}")
    print("=" * 60)
    print()


def train(args=None):
    """
    Main training function.

    Call this directly or pass a Namespace of args.
    If args=None, defaults from DEFAULTS dict are used.

    FLOW:
      1. Configure GPU
      2. Build data generators
      3. Build model
      4. (Optional) Load checkpoint to resume training
      5. Compile model with Adam optimizer
      6. Train with model.fit()
      7. Save final model
      8. Plot training history
    """
    if args is None:
        # Use defaults when called programmatically (not from CLI)
        class Args:
            pass
        args = Args()
        for k, v in DEFAULTS.items():
            setattr(args, k, v)
        args.resume_from = None

    print_training_config(args)

    # Step 1: GPU setup
    configure_gpu()

    # Step 2: Build data generators
    print("\n[Train] Building data generators...")
    train_gen, val_gen, test_gen, char_to_int, int_to_char = build_generators(
        splits_dir = args.splits_dir,
        batch_size = args.batch_size,
    )

    # Step 3: Build CRNN model
    print("\n[Train] Building CRNN model...")
    model = build_crnn_model(
        lstm_units   = args.lstm_units,
        dropout_rate = args.dropout,
    )
    model_summary(model)

    # Step 4: Resume from checkpoint (optional)
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"\n[Train] Loading weights from: {args.resume_from}")
        model.load_weights(args.resume_from)
        print("[Train] Weights loaded — resuming training.")

    # Step 5: Compile
    # WHY Adam? Adam (Adaptive Moment Estimation) is the best default optimizer.
    # It adapts the learning rate per-parameter based on gradient history.
    # Most practitioners use Adam with lr=1e-3 as a starting point.
    #
    # NOTE: We pass loss=None because the loss is computed INSIDE the CTCLayer
    # via add_loss(). Keras automatically picks it up.
    print("\n[Train] Compiling model...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(optimizer=optimizer)

    # Step 6: Get callbacks
    callbacks = get_callbacks(
        checkpoint_dir  = args.checkpoint_dir,
        experiment_name = args.experiment_name,
    )

    # Step 7: TRAIN
    print("\n[Train] Starting training...")
    print("  Watch for:")
    print("  - val_loss going DOWN   → model is learning correctly")
    print("  - val_loss going UP     → overfitting (reduce epochs or add dropout)")
    print("  - loss stops changing   → learning rate too low (callbacks will fix it)")
    print()

    history = model.fit(
        train_gen,
        validation_data = val_gen,
        epochs          = args.epochs,
        callbacks       = callbacks,
        verbose         = 1,     # 1 = progress bar per epoch
    )

    # Step 8: Save the final model
    final_weights_path = os.path.join(args.checkpoint_dir, f"{args.experiment_name}_final.weights.h5")
    model.save_weights(final_weights_path)
    print(f"\n[Train] Final weights saved → {final_weights_path}")

    # Save inference model separately
    inference_model = build_inference_model(model)
    inference_path  = os.path.join(args.checkpoint_dir, f"{args.experiment_name}_inference.keras")
    inference_model.save(inference_path)
    print(f"[Train] Inference model saved → {inference_path}")

    # Step 9: Plot training history
    plot_training_history(
        history.history,
        save_path=os.path.join(args.checkpoint_dir, f"{args.experiment_name}_history.png")
    )

    print("\n[Train] Training complete!")
    print(f"[Train] To run inference:")
    print(f"  python backend/inference/predict.py --image_path sample_images/test.png")
    print(f"  --model_path {inference_path}")

    return model, history


# ── Entry Point ───────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    train(args)
