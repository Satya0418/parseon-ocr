"""
Find the correct index offset by testing different shifts
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

def test_with_shift(shift, blank_index=79):
    """Test with a specific index shift"""
    # Build int_to_char with shift
    int_to_char = {}
    for idx, char in enumerate(ALPHABET):
        int_to_char[idx + shift] = char
    
    # Decode function
    def decode(predictions):
        texts = []
        for sample_pred in predictions:
            best_path = np.argmax(sample_pred, axis=-1)
            collapsed = [best_path[0]]
            for i in range(1, len(best_path)):
                if best_path[i] != best_path[i - 1]:
                    collapsed.append(best_path[i])
            result = [idx for idx in collapsed if idx != blank_index]
            text = "".join(int_to_char.get(idx, "?") for idx in result)
            texts.append(text)
        return texts
    
    # Test on samples
    test_df = pd.read_csv("data/iam_words/splits/test.csv")
    samples = test_df.sample(n=20, random_state=42)
    
    correct = 0
    for _, row in samples.iterrows():
        img = preprocess_image(row['image_path'])
        img_batch = np.expand_dims(img, axis=0)
        preds = model.predict(img_batch, verbose=0)
        pred_text = decode(preds)[0]
        if pred_text == row['label']:
            correct += 1
    
    return correct

print("Testing different index offsets...")
print("="*70)

for shift in range(-2, 3):
    correct = test_with_shift(shift)
    accuracy = correct / 20 * 100
    marker = " ← BEST" if shift == 1 else ""
    print(f"Shift {shift:+2d}: {correct:2d}/20 ({accuracy:5.1f}%){marker}")

print("="*70)
print("\nTry shift +1 or the one with highest accuracy")
