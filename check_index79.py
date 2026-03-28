"""Check which character maps to index 79"""

ALPHABET = (
    " !\"#&'()*+,-./0123456789:;?"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
)

char_to_int = {char: idx for idx, char in enumerate(ALPHABET)}

print("Checking key characters:")
print(f"  'z' → {char_to_int.get('z', 'NOT FOUND')}")
print(f"  'Z' → {char_to_int.get('Z', 'NOT FOUND')}")
print(f"  Last char in alphabet (index {len(ALPHABET)-1}): '{ALPHABET[-1]}' → {char_to_int.get(ALPHABET[-1])}")

print(f"\nLabel from error: [32,65,62,79,54,55,58,73,61]")
print("Decoding:")
int_to_char = {idx: char for idx, char in enumerate(ALPHABET)}
for idx in [32, 65, 62, 79, 54, 55, 58, 73, 61]:
    if idx < len(ALPHABET):
        print(f"  {idx} → '{int_to_char.get(idx, '?')}'")
    else:
        print(f"  {idx} → BLANK TOKEN (out of alphabet range 0-{len(ALPHABET)-1})")

# Check if there's a char whose ASCII or position could be 79
print(f"\nCharacter at position 32 in alphabet: '{ALPHABET[32]}' (should be '?')")
print(f"Character at position 65 in alphabet: '{ALPHABET[65]}' (should be 'k')")
print(f"Character at position 78 (last valid): '{ALPHABET[78]}' (should be 'z')")
