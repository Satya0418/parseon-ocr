"""Check if dataset labels contain characters outside ALPHABET"""

import pandas as pd

ALPHABET = (
    " !\"#&'()*+,-./0123456789:;?"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
)

alphabet_set = set(ALPHABET)

# Load labels
labels_df = pd.read_csv('ocr_project/data/iam_words/labels.csv')

invalid_labels = []

for idx, row in labels_df.iterrows():
    label = row['label']
    for char in label:
        if char not in alphabet_set:
            invalid_labels.append((idx, label, char))
            if len(invalid_labels) <= 10:  # Show first 10
                print(f"Row {idx}: label='{label}' contains invalid char '{char}' (ord={ord(char)})")

print(f"\nTotal labels with invalid characters: {len(invalid_labels)}")

if len(invalid_labels) > 10:
    print("Showing first 10 only...")
