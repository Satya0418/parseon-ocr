"""
Train CRNN OCR with Camera Augmentation - LOCAL VERSION
Trains from scratch on your laptop (no pre-trained weights needed)
Expected time: 10-20 hours on CPU
"""
import os
import sys
os.chdir("d:/text/ocr_project")
sys.path.insert(0, "d:/text/ocr_project")

import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from pathlib import Path

# Limit CPU threads to avoid overheating
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

print("=" * 80)
print("CAMERA-AUGMENTED OCR TRAINING (LOCAL)")
print("=" * 80)
print(f"TensorFlow: {tf.__version__}")
print(f"Training from SCRATCH (no pre-trained weights)")
print("=" * 80)

# Import modules
from backend.models.crnn_model import build_crnn_model
from backend.dataset.dataloader import OCRDataGenerator, load_split_csv, encode_batch_labels
from backend.preprocessing.image_processor import preprocess_image, IMG_WIDTH
from backend.preprocessing.augmentation import augment_image_camera, augment_image
from backend.utils.char_map import build_char_maps
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

# Safe image loader with retry
def _safe_load_image(df, idx, indices, augment_fn=None):
    """Load image with retry on failure."""
    row = df.iloc[idx]
    img_path = row["image_path"].replace("\\", "/")
    marker = "data/iam_words"
    _idx = img_path.find(marker)
    if _idx != -1:
        img_path = img_path[_idx:]
    
    img = preprocess_image(img_path)
    
    if img is None or np.max(img) == 0:
        for _ in range(5):
            alt_idx = np.random.choice(indices)
            alt_row = df.iloc[alt_idx]
            alt_path = alt_row["image_path"].replace("\\", "/")
            _aidx = alt_path.find(marker)
            if _aidx != -1:
                alt_path = alt_path[_aidx:]
            img = preprocess_image(alt_path)
            if img is not None and np.max(img) > 0:
                row = alt_row
                break
    
    if augment_fn and img is not None and np.max(img) > 0:
        img = augment_fn(img)
    
    return img, row["label"]


class CameraAugDataGenerator(OCRDataGenerator):
    """Camera-augmented data generator (50/50 mix)."""
    
    def __getitem__(self, batch_idx):
        start = batch_idx * self.batch_size
        end = start + self.batch_size
        batch_indices = self.indices[start:end]
        
        images = []
        raw_labels = []
        
        for idx in batch_indices:
            def _augment(img):
                if np.random.random() < 0.5:
                    return augment_image_camera(img, augment_prob=0.5)
                else:
                    return augment_image(img, augment_prob=0.4)
            
            aug_fn = _augment if self.augment else None
            img, label = _safe_load_image(self.df, idx, self.indices, augment_fn=aug_fn)
            images.append(img)
            raw_labels.append(label)
        
        images_batch = np.stack(images, axis=0).astype(np.float32)
        encoded_labels, label_lengths = encode_batch_labels(raw_labels, self.char_to_int)
        image_widths = np.full((len(images),), IMG_WIDTH // 4, dtype=np.int32)
        
        inputs = {
            "input_images": images_batch,
            "label_encoded": encoded_labels,
            "image_widths": image_widths,
            "label_lengths": label_lengths,
        }
        return inputs, np.zeros(len(images), dtype=np.float32)


class RobustOCRDataGenerator(OCRDataGenerator):
    """Robust validation generator."""
    
    def __getitem__(self, batch_idx):
        start = batch_idx * self.batch_size
        end = start + self.batch_size
        batch_indices = self.indices[start:end]
        
        images = []
        raw_labels = []
        
        for idx in batch_indices:
            aug_fn = (lambda img: augment_image(img, augment_prob=0.4)) if self.augment else None
            img, label = _safe_load_image(self.df, idx, self.indices, augment_fn=aug_fn)
            images.append(img)
            raw_labels.append(label)
        
        images_batch = np.stack(images, axis=0).astype(np.float32)
        encoded_labels, label_lengths = encode_batch_labels(raw_labels, self.char_to_int)
        image_widths = np.full((len(images),), IMG_WIDTH // 4, dtype=np.int32)
        
        inputs = {
            "input_images": images_batch,
            "label_encoded": encoded_labels,
            "image_widths": image_widths,
            "label_lengths": label_lengths,
        }
        return inputs, np.zeros(len(images), dtype=np.float32)


# Filter unreadable images
def filter_missing_images(df, label=""):
    """Remove unreadable images."""
    valid = []
    missing = 0
    for idx, row in df.iterrows():
        img_path = row["image_path"].replace("\\", "/")
        marker = "data/iam_words"
        _idx = img_path.find(marker)
        if _idx != -1:
            img_path = img_path[_idx:]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None and img.size > 0:
            valid.append(idx)
        else:
            missing += 1
    filtered = df.loc[valid].reset_index(drop=True)
    print(f"  {label}: {len(df)} -> {len(filtered)} ({missing} unreadable removed)")
    return filtered


print("\n[1/5] Loading data splits...")
char_to_int, int_to_char = build_char_maps()
BATCH_SIZE = 32

splits_dir = "data/iam_words/splits"
train_df = load_split_csv(os.path.join(splits_dir, "train.csv"))
val_df = load_split_csv(os.path.join(splits_dir, "val.csv"))

print("\n[2/5] Filtering unreadable images...")
train_df = filter_missing_images(train_df, "Train")
val_df = filter_missing_images(val_df, "Val")

print("\n[3/5] Creating data generators...")
train_gen = CameraAugDataGenerator(
    df=train_df, char_to_int=char_to_int, batch_size=BATCH_SIZE, augment=True, shuffle=True
)
val_gen = RobustOCRDataGenerator(
    df=val_df, char_to_int=char_to_int, batch_size=BATCH_SIZE, augment=False, shuffle=False
)
print(f"  Train batches: {len(train_gen)}, Val batches: {len(val_gen)}")

print("\n[4/5] Building model...")
model = build_crnn_model(lstm_units=256, dropout_rate=0.25)
# TRAINING FROM SCRATCH - no weights loaded
model.compile(optimizer=Adam(learning_rate=1e-3))  # Higher LR for training from scratch
print("  Model compiled with lr=1e-3 (training from scratch)")

print("\n[5/5] Setting up callbacks...")
EPOCHS = 35
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

callbacks = [
    ModelCheckpoint(
        filepath=os.path.join(SAVE_DIR, "crnn_camera_local_best.weights.h5"),
        save_weights_only=True,
        save_best_only=True,
        monitor="val_loss",
        verbose=1
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-7,
        verbose=1
    ),
    CSVLogger(os.path.join(SAVE_DIR, "training_camera_local.csv"))
]

print("\n" + "=" * 80)
print(f"STARTING TRAINING")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Train samples: {len(train_df)}")
print(f"  Val samples: {len(val_df)}")
print(f"  Learning rate: 1e-3 (training from scratch)")
print(f"  Camera augmentation: 50/50 mix")
print(f"  Expected time: 10-20 hours on CPU")
print("=" * 80)
print()

# Train
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)

# Save inference model
print("\nSaving inference model...")
from backend.models.crnn_model import build_inference_model

best_weights = os.path.join(SAVE_DIR, "crnn_camera_local_best.weights.h5")
if os.path.exists(best_weights):
    model.load_weights(best_weights)
    print(f"Loaded best weights from {best_weights}")

inference_model = build_inference_model(model)
inference_path = os.path.join(SAVE_DIR, "crnn_iam_v1_inference.keras")
inference_model.save(inference_path)
print(f"Inference model saved: {inference_path}")

# Save final weights
final_weights = os.path.join(SAVE_DIR, "crnn_camera_local_final.weights.h5")
model.save_weights(final_weights)
print(f"Final weights saved: {final_weights}")

print("\n" + "=" * 80)
print("ALL DONE! Model ready to use.")
print("=" * 80)
