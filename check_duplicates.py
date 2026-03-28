"""Check for duplicate characters in ALPHABET"""

ALPHABET = (
    " !\"#&'()*+,-./0123456789:;?"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
)

from collections import Counter

char_counts = Counter(ALPHABET)
duplicates = {char: count for char, count in char_counts.items() if count > 1}

if duplicates:
    print("✗ DUPLICATE CHARACTERS FOUND:")
    for char, count in duplicates.items():
        print(f"  '{char}' appears {count} times")
        indices = [i for i, c in enumerate(ALPHABET) if c == char]
        print(f"  At indices: {indices}")
else:
    print("✓ No duplicates found")

# Check if comma appears multiple times (common mistake)
if ',' in ALPHABET:
    indices = [i for i, c in enumerate(ALPHABET) if c == ',']
    print(f"\nComma ',' at indices: {indices} (count: {len(indices)})")

# Check if minus/hyphen appears multiple times
if '-' in ALPHABET:
    indices = [i for i, c in enumerate(ALPHABET) if c == '-']
    print(f"Minus '-' at indices: {indices} (count: {len(indices)})")
