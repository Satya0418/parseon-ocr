"""
Quick test script to validate the trained OCR model
"""
import pandas as pd
from backend.inference.predict import OCRPredictor
from backend.utils.metrics import character_error_rate

# Load predictor with the trained model
print("Loading model...")
predictor = OCRPredictor("saved_models/crnn_iam_v1_inference.keras")

# Load test CSV to get some test images
test_df = pd.read_csv("data/iam_words/splits/test.csv")

# Test on 10 random samples
print("\nTesting on 10 random samples from test set:")
print("="*70)

samples = test_df.sample(n=10, random_state=42)
correct = 0
all_preds = []
all_gts = []

for idx, row in samples.iterrows():
    img_path = row['image_path']
    ground_truth = row['label']
    
    # Run prediction
    prediction = predictor.predict(img_path)
    
    all_preds.append(prediction)
    all_gts.append(ground_truth)
    
    match = "✓" if prediction == ground_truth else "✗"
    if prediction == ground_truth:
        correct += 1
    
    print(f"{match} GT:   '{ground_truth}'")
    print(f"  Pred: '{prediction}'")
    print()

# Calculate overall CER
cer = character_error_rate(all_preds, all_gts)

print("="*70)
print(f"Accuracy: {correct}/10 = {correct*10}%")
print(f"CER: {cer:.4f} ({cer*100:.2f}%)")
print("="*70)
print("\n✅ Model test complete!")
print("If CER is around 5-15%, the decoder fix worked perfectly!")
print("If CER is still >100%, something went wrong.")
