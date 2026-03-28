"""Quick script to check ALPHABET length and character mappings"""

ALPHABET = (
    " !\"#&'()*+,-./0123456789:;?"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
)

print(f"ALPHABET length: {len(ALPHABET)}")
print(f"BLANK_INDEX: {len(ALPHABET)}")
print(f"num_classes: {len(ALPHABET) + 1}")
print(f"\nFirst 10 chars: {[f'{i}:{repr(c)}' for i, c in enumerate(ALPHABET[:10])]}")
print(f"Last 10 chars:  {[f'{i}:{repr(c)}' for i, c in enumerate(ALPHABET[-10:], len(ALPHABET)-10)]}")

# Check if there are 79 or 80 characters
if len(ALPHABET) == 79:
    print("\n✓ Alphabet has 79 chars (indices 0-78), blank at 79, num_classes=80")
elif len(ALPHABET) == 80:
    print("\n✗ Alphabet has 80 chars (indices 0-79), blank should be at 80, num_classes should be 81!")
    print("  This is the problem! Model has 80 classes but alphabet uses 0-79.")
else:
    print(f"\n? Unexpected alphabet length: {len(ALPHABET)}")

# Show character at index 79 if it exists
if len(ALPHABET) > 79:
    print(f"\nCharacter at index 79: {repr(ALPHABET[79])}")
