"""
Test specific word prediction
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from backend.inference.predict import OCRPredictor
import pandas as pd

# Load model
predictor = OCRPredictor("saved_models/crnn_iam_v1_inference.keras")

# Test on images similar to what you tried
test_df = pd.read_csv("data/iam_words/splits/test.csv")

# Find test images with short words (3 letters like "cer")
short_words = test_df[test_df['label'].str.len() == 3]

print("Testing model on 3-letter words from test set:")
print("="*60)

for i, row in short_words.head(10).iterrows():
    gt = row['label']
    pred = predictor.predict(row['image_path'])
    match = "✓" if pred == gt else "✗"
    print(f"{match} GT: '{gt:8s}' → Pred: '{pred}'")

print("="*60)
print("\n💡 The model has 16% character error rate")
print("   Some predictions will be wrong - this is normal!")
