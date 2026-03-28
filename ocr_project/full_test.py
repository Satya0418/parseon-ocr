"""
Full test - evaluate on entire test set (4,384 samples)
"""
import pandas as pd
from backend.inference.predict import OCRPredictor
from backend.utils.metrics import character_error_rate, word_error_rate

print("Loading model...")
predictor = OCRPredictor("saved_models/crnn_iam_v1_inference.keras")

# Load test set
test_df = pd.read_csv("data/iam_words/splits/test.csv")
print(f"Test set: {len(test_df)} samples")
print()

all_preds = []
all_gts = []

print("Running predictions...")
for idx, row in test_df.iterrows():
    if idx % 500 == 0:
        print(f"  Progress: {idx}/{len(test_df)} ({idx/len(test_df)*100:.1f}%)")
    
    try:
        pred = predictor.predict(row['image_path'])
        all_preds.append(pred)
        all_gts.append(row['label'])
    except Exception as e:
        print(f"  Error on {row['image_path']}: {e}")
        all_preds.append("")
        all_gts.append(row['label'])

# Calculate metrics
cer = character_error_rate(all_preds, all_gts)
wer = word_error_rate(all_preds, all_gts)

print()
print("="*70)
print("FULL TEST SET EVALUATION")
print("="*70)
print(f"Samples:  {len(all_preds)}")
print(f"CER:      {cer:.4f} ({cer*100:.2f}%)")
print(f"WER:      {wer:.4f} ({wer*100:.2f}%)")
print("="*70)

# Show some examples
print("\nSample predictions:")
import random
random.seed(42)
indices = random.sample(range(len(all_preds)), min(10, len(all_preds)))
for i in indices:
    gt = all_gts[i]
    pred = all_preds[i]
    match = "✓" if pred == gt else "✗"
    print(f"{match} GT:   '{gt}'")
    print(f"  Pred: '{pred}'")
    print()
