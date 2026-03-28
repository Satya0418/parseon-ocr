"""
=============================================================
callbacks.py  —  Keras Training Callbacks
=============================================================

WHY DO WE NEED CALLBACKS?
--------------------------
Callbacks are functions that Keras automatically calls at specific
moments during training. They let you:

  1. SAVE THE BEST MODEL — so you don't lose your best weights
     if training keeps going and validation loss gets worse.

  2. EARLY STOPPING — automatically stop training when the model
     stops improving. Saves hours of useless GPU time.

  3. REDUCE LEARNING RATE — when loss plateaus, reduce the step size
     so the optimizer can find a better minimum.

  4. LOG METRICS — save a CSV of every epoch's metrics for analysis.

Think of callbacks as a "smart supervisor" watching training and
making decisions automatically.

LEARNING RATE EXPLAINED:
The learning rate controls HOW BIG each optimizer step is.
  - Too high → model overshoots, loss oscillates, training unstable
  - Too low  → model learns very slowly, takes forever
  - Just right → smooth convergence

A common strategy: start with 0.001, reduce by factor 0.5 when
validation loss doesn't improve for 5 epochs.
=============================================================
"""

import os
import tensorflow as tf
from tensorflow import keras


def get_callbacks(checkpoint_dir: str = "saved_models",
                  experiment_name: str = "crnn_ocr",
                  patience_stop:   int = 15,
                  patience_lr:     int = 5,
                  lr_factor:       float = 0.5,
                  min_lr:          float = 1e-7) -> list:
    """
    Build and return a list of Keras callbacks for training.

    WHY: Returning a list of callbacks lets us pass them directly to
    model.fit(callbacks=...) with one clean call. Each callback is
    configured here with sensible defaults.

    Parameters
    ----------
    checkpoint_dir  : str   — directory to save model checkpoint files
    experiment_name : str   — prefix for all saved files (for organization)
    patience_stop   : int   — how many epochs to wait before early stopping
                              (15 means: stop if no improvement for 15 epochs)
    patience_lr     : int   — epochs with no improvement before reducing LR
    lr_factor       : float — factor to multiply LR by (0.5 = halve it)
    min_lr          : float — minimum LR (never go below this)

    Returns
    -------
    list  — list of keras Callback objects to pass to model.fit()
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    callbacks = []

    # ── 1. ModelCheckpoint ─────────────────────────────────────────
    # Saves model weights whenever validation loss improves.
    # WHY: Training can take hours. If you get an even better result
    # 10 epochs later but then overfit, the checkpoint protects you.
    # 'save_best_only=True' keeps only the best-so-far weights.
    # 'save_weights_only=True' is faster — saves just the weights (.h5),
    # not the full model architecture.
    best_model_path = os.path.join(checkpoint_dir, f"{experiment_name}_best.weights.h5")
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath        = best_model_path,
        monitor         = "val_loss",     # watch validation loss
        save_best_only  = True,           # only save when it improves
        save_weights_only = True,         # skip architecture, just weights
        mode            = "min",          # loss should DECREASE (min = better)
        verbose         = 1,
    )
    callbacks.append(checkpoint_cb)

    # ── 2. EarlyStopping ──────────────────────────────────────────
    # Stops training when validation loss hasn't improved in `patience` epochs.
    # 'restore_best_weights=True' reloads the best checkpoint automatically.
    # WHY: Without early stopping, training runs for all specified epochs even
    # if the model has already peaked. This wastes GPU time and can overfit.
    early_stop_cb = keras.callbacks.EarlyStopping(
        monitor              = "val_loss",
        patience             = patience_stop,
        restore_best_weights = True,   # reload best weights at the end
        mode                 = "min",
        verbose              = 1,
    )
    callbacks.append(early_stop_cb)

    # ── 3. ReduceLROnPlateau ───────────────────────────────────────
    # Reduces the learning rate when training hits a plateau.
    # WHY: When the model is close to the optimum, a large learning rate
    # "overshoots" past it. Reducing LR lets the optimizer take smaller,
    # more precise steps to find the true minimum.
    # Example: LR 0.001 → 0.0005 → 0.00025 → ... (until min_lr)
    reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(
        monitor   = "val_loss",
        factor    = lr_factor,    # multiply LR by this factor
        patience  = patience_lr,  # wait this many epochs before reducing
        min_lr    = min_lr,       # never go below this
        mode      = "min",
        verbose   = 1,
    )
    callbacks.append(reduce_lr_cb)

    # ── 4. CSVLogger ──────────────────────────────────────────────
    # Saves training metrics to a CSV file after every epoch.
    # WHY: Allows you to plot training history even after the session ends.
    # Especially useful in Colab where output gets cleared between sessions.
    csv_log_path = os.path.join(checkpoint_dir, f"{experiment_name}_log.csv")
    csv_logger_cb = keras.callbacks.CSVLogger(
        filename = csv_log_path,
        append   = True,   # append so resumed training continues the log
    )
    callbacks.append(csv_logger_cb)

    # ── 5. TensorBoard (optional — comment out if not using) ────────
    # WHY: TensorBoard provides a beautiful interactive dashboard for
    # real-time training visualization directly in your browser.
    # Start it with: tensorboard --logdir saved_models/logs
    tb_log_dir = os.path.join(checkpoint_dir, "logs", experiment_name)
    tensorboard_cb = keras.callbacks.TensorBoard(
        log_dir          = tb_log_dir,
        histogram_freq   = 1,    # compute weight histograms every epoch
        update_freq      = "epoch",
    )
    callbacks.append(tensorboard_cb)

    print(f"[Callbacks] Configured {len(callbacks)} callbacks:")
    print(f"  ✓ ModelCheckpoint  → {best_model_path}")
    print(f"  ✓ EarlyStopping    (patience={patience_stop})")
    print(f"  ✓ ReduceLROnPlateau (patience={patience_lr}, factor={lr_factor})")
    print(f"  ✓ CSVLogger        → {csv_log_path}")
    print(f"  ✓ TensorBoard      → {tb_log_dir}")

    return callbacks
