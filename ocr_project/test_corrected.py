"""
Test with corrected index mapping (shift -1)
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf
from backend.preprocessing.image_processor import preprocess_image
from backend.utils.char_map import ALPHABET

# Load model
model = tf.keras.models.load_model("saved_models/crnn_iam_v1_inference.keras")

# Build CORRECTED int_to_char mapping (shift all indices by -1)
# Model was trained with indices off-by-one
int_to_char_corrected = {}
for idx, char in enumerate(ALPHABET):
    int_to_char_corrected[idx + 1] = char  # Shift by +1 to compensate for training shift

print(f"Testing with corrected mapping (indices shifted +1)")
print()

# Simple decode with correction
def decode_corrected(predictions, blank_index=79):
    texts = []
    for sample_pred in predictions:
        best_path = np.argmax(sample_pred, axis=-1)
        
        # Collapse
        collapsed = [best_path[0]]
        for i in range(1, len(best_path)):
            if best_path[i] != best_path[i - 1]:
                collapsed.append(best_path[i])
        
        # Remove blanks
        result = [idx for idx in collapsed if idx != blank_index]
        
        # Convert to text with CORRECTED mapping
        text = "".join(int_to_char_corrected.get(idx, "?") for idx in result)
        texts.append(text)
    
    return texts

# Test
test_df = pd.read_csv("data/iam_words/splits/test.csv")
samples = test_df.sample(n=10, random_state=42)

print("Testing 10 samples with corrected mapping:")
print("="*70)

correct = 0
for _, row in samples.iterrows():
    img_path = row['image_path']
    gt = row['label']
    
    img = preprocess_image(img_path)
    img_batch = np.expand_dims(img, axis=0)
    preds = model.predict(img_batch, verbose=0)
    
    pred_text = decode_corrected(preds, blank_index=79)[0]
    
    match = "✓" if pred_text == gt else "✗"
    if pred_text == gt:
        correct += 1
    
    print(f"{match} GT:   '{gt}'")
    print(f"  Pred: '{pred_text}'")
    print()

print("="*70)
print(f"Accuracy: {correct}/10 = {correct*10}%")
print("="*70)
