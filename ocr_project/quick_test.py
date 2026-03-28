"""
Simple direct test of the model - minimal imports
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import numpy as np
import pandas as pd
import tensorflow as tf

# Load model
print("Loading model...")
model = tf.keras.models.load_model("saved_models/crnn_iam_v1_inference.keras")
print("✓ Model loaded")

model = tf.keras.models.load_model("saved_models/crnn_iam_v1_inference.keras")
print("✓ Model loaded")

# Load character mappings using the SAME function as training
from backend.utils.char_map import build_char_maps, ALPHABET
char_to_int, int_to_char = build_char_maps(ALPHABET)
print(f"Alphabet: {len(ALPHABET)} chars, blank at {len(ALPHABET)}")

# Simple greedy decode with CORRECT blank_index
def decode_greedy(predictions, blank_index=79):
    texts = []
    for sample_pred in predictions:
        best_path = np.argmax(sample_pred, axis=-1)
        
        # Collapse consecutive duplicates
        collapsed = [best_path[0]]
        for i in range(1, len(best_path)):
            if best_path[i] != best_path[i - 1]:
                collapsed.append(best_path[i])
        
        # Remove blanks
        result = [idx for idx in collapsed if idx != blank_index]
        
        # Convert to text
        text = "".join(int_to_char.get(idx, "?") for idx in result)
        texts.append(text)
    
    return texts

# Load test data and preprocess
from backend.preprocessing.image_processor import preprocess_image

test_df = pd.read_csv("data/iam_words/splits/test.csv")
samples = test_df.sample(n=5, random_state=42)

print("\nTesting 5 samples:")
print("="*70)

for _, row in samples.iterrows():
    img_path = row['image_path']
    gt = row['label']
    
    # Preprocess and predict
    img = preprocess_image(img_path)
    img_batch = np.expand_dims(img, axis=0)
    preds = model.predict(img_batch, verbose=0)
    
    # Decode
    pred_text = decode_greedy(preds, blank_index=79)[0]
    
    match = "✓" if pred_text == gt else "✗"
    print(f"{match} GT:   '{gt}'")
    print(f"  Pred: '{pred_text}'")
    print()

print("="*70)
print("✅ Test complete! Check if predictions look correct.")
