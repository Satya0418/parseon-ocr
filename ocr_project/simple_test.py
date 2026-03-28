"""
Simple test - predict text from 5 random images
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from backend.inference.predict import OCRPredictor
import pandas as pd

# Load model
print("Loading model...")
predictor = OCRPredictor("saved_models/crnn_iam_v1_inference.keras")
print("✓ Model ready\n")

# Test on 5 random images
test_df = pd.read_csv("data/iam_words/splits/test.csv")
samples = test_df.sample(5, random_state=999)

print("="*70)
print("TESTING MODEL ON 5 RANDOM IMAGES")
print("="*70)

for i, (_, row) in enumerate(samples.iterrows(), 1):
    img_path = row['image_path']
    gt = row['label']
    
    pred = predictor.predict(img_path)
    
    match = "✓" if pred == gt else "✗"
    
    print(f"\nImage {i}:")
    print(f"  {match} Ground Truth: '{gt}'")
    print(f"     Prediction:   '{pred}'")

print("\n" + "="*70)
print("✅ Model test complete!")
print("="*70)
