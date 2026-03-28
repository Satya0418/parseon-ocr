"""Test encode_label to see if it produces index 79"""

import pandas as pd
import sys
import os
sys.path.append('ocr_project')

from backend.utils.char_map import build_char_maps, encode_label, ALPHABET

char_to_int, int_to_char = build_char_maps()

print(f"ALPHABET length: {len(ALPHABET)}")
print(f"char_to_int keys: {len(char_to_int)}")
print(f"Max index in char_to_int: {max(char_to_int.values())}")
print(f"Blank index: {len(ALPHABET)}")

# Load sample labels
df = pd.read_csv('ocr_project/data/iam_words/splits/train.csv')

# Check first 100 labels
problem_labels = []
for idx, row in df.head(1000).iterrows():
    label = row['label']
    encoded = encode_label(label, char_to_int)
    if any(idx >= 79 for idx in encoded):
        problem_labels.append((idx, label, encoded))
        print(f"Row {idx}: '{label}' → {encoded}")
        
if not problem_labels:
    print("\n✓ No labels produce index >= 79 in first 1000 samples")
else:
    print(f"\n✗ Found {len(problem_labels)} problem labels")
