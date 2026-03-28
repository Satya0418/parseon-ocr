"""
Test the model on a single image with visualization
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import matplotlib.pyplot as plt
from backend.inference.predict import OCRPredictor
import pandas as pd

# Load the model
print("Loading model...")
predictor = OCRPredictor("saved_models/crnn_iam_v1_inference.keras")
print("✓ Model loaded\n")

# Get a random test image
test_df = pd.read_csv("data/iam_words/splits/test.csv")
sample = test_df.sample(1, random_state=123).iloc[0]

img_path = sample['image_path']
ground_truth = sample['label']

# Predict
print(f"Testing image: {img_path}")
print(f"Ground truth: '{ground_truth}'")
print("\nRunning OCR...")

prediction = predictor.predict(img_path)

print(f"Prediction:   '{prediction}'")
print()

# Show result
if prediction == ground_truth:
    print("✓ PERFECT MATCH!")
else:
    print("✗ Mismatch")
    # Character-by-character comparison
    max_len = max(len(ground_truth), len(prediction))
    print("\nCharacter comparison:")
    print(f"  GT:   {' '.join(ground_truth)}")
    print(f"  Pred: {' '.join(prediction)}")

# Display the image
print("\nDisplaying image...")
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(10, 3))
plt.imshow(img, cmap='gray')
plt.title(f"GT: '{ground_truth}' | Prediction: '{prediction}'", fontsize=12)
plt.axis('off')
plt.tight_layout()
plt.show()

print("\n✓ Test complete!")
