"""
=============================================================
char_map.py  —  Character ↔ Number Mapping
=============================================================

WHY DO WE NEED THIS?
--------------------
Neural networks work with NUMBERS, not letters.
So we need a dictionary that converts:
  - Characters  →  Integers  (for feeding into the model)
  - Integers    →  Characters (for converting model output back to text)

EXAMPLE:
  char_to_int['a'] = 1
  char_to_int['b'] = 2
  int_to_char[1]   = 'a'

The index 0 is RESERVED for the CTC "blank" token.
CTC Loss needs a special blank symbol to handle repeated
characters and spaces between characters. Think of it as a
separator the model uses internally.

Input:  None (builds map from a hard-coded alphabet)
Output: Two dictionaries (char→int, int→char) + helper functions
=============================================================
"""

# The full set of characters the model can recognize.
# This covers all lowercase, uppercase, digits, and common symbols
# found in the IAM Handwriting dataset.
ALPHABET = (
    " !\"#&'()*+,-./0123456789:;?"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
)

# TensorFlow CTC expects blank at the LAST index (num_classes - 1)
# Characters are mapped to indices 0 through len(alphabet)-1
BLANK_INDEX = len(ALPHABET)  # Will be the last index


def build_char_maps(alphabet: str = ALPHABET):
    """
    Build character-to-integer and integer-to-character dictionaries.

    WHY: The model outputs a probability distribution over all characters
    at each time step. We need a fixed mapping so character index 5 always
    means the same letter, both during training and inference.

    Parameters
    ----------
    alphabet : str
        A string of all characters the model should learn to recognize.
        Default is the full IAM alphabet (letters, digits, punctuation).

    Returns
    -------
    char_to_int : dict  {str → int}
        Maps each character to a unique integer (starting from 0).
        Example: {'a': 0, 'b': 1, 'c': 2, ...}

    int_to_char : dict  {int → str}
        The reverse mapping — integer back to character.
        Example: {0: 'a', 1: 'b', 2: 'c', ...}
    """
    # Map characters to indices 0 through len(alphabet)-1
    char_to_int = {char: idx for idx, char in enumerate(alphabet)}
    int_to_char = {idx: char for idx, char in enumerate(alphabet)}

    # Blank token is at index len(alphabet) (last position)
    int_to_char[len(alphabet)] = "[BLANK]"

    return char_to_int, int_to_char


def build_corrected_char_maps_for_kaggle_model(alphabet: str = ALPHABET):
    """
    Build corrected character mappings for the Kaggle-trained model.
    
    ISSUE: The Kaggle model was trained with indices offset by +1.
    This function compensates for that offset during inference.
    
    Returns
    -------
    char_to_int : dict  — same as normal (not needed for inference)
    int_to_char : dict  — CORRECTED mapping with +1 offset
    """
    char_to_int = {char: idx for idx, char in enumerate(alphabet)}
    
    # Build int_to_char with +1 offset to match Kaggle training
    int_to_char = {idx + 1: char for idx, char in enumerate(alphabet)}
    int_to_char[len(alphabet)] = "[BLANK]"  # Blank at index 79
    
    return char_to_int, int_to_char


def encode_label(text: str, char_to_int: dict) -> list:
    """
    Convert a text string into a list of integers.

    WHY: Labels in the IAM dataset are strings like "hello world".
    The model needs them as integer sequences like [8, 5, 12, 12, 15, ...].

    Parameters
    ----------
    text : str
        The ground-truth text label. Example: "Hello"

    char_to_int : dict
        The character-to-integer mapping built by build_char_maps().

    Returns
    -------
    list of int
        Integer-encoded label. Unknown characters are skipped with a warning.

    Example
    -------
    encode_label("Hi", char_to_int)  →  [34, 35]  (indices of 'H' and 'i')
    """
    encoded = []
    for char in text:
        if char in char_to_int:
            encoded.append(char_to_int[char])
        else:
            # Skip characters not in our alphabet (rare unicode, etc.)
            print(f"[WARNING] Character '{char}' not in alphabet — skipping.")
    return encoded


def decode_label(indices: list, int_to_char: dict) -> str:
    """
    Convert a list of integers back into a text string.

    WHY: After training, the model outputs integer indices.
    We need to convert them back to human-readable text.

    Parameters
    ----------
    indices : list of int
        Model output after CTC decoding. Example: [34, 35]

    int_to_char : dict
        The integer-to-character mapping built by build_char_maps().

    Returns
    -------
    str
        The decoded text string. Example: "Hi"
    """
    return "".join(
        int_to_char.get(idx, "?")   # '?' for any unknown index
        for idx in indices
        if idx != BLANK_INDEX        # skip CTC blank tokens
    )


def get_num_classes(alphabet: str = ALPHABET) -> int:
    """
    Return the total number of output classes for the model.

    WHY: The final Dense layer of the CRNN must have exactly this many
    output neurons — one per character + one for the CTC blank.

    Returns
    -------
    int
        len(alphabet) + 1  (the +1 accounts for the CTC blank token)

    Example
    -------
    If alphabet has 80 characters → model outputs 81 classes
    """
    return len(alphabet) + 1  # +1 for the blank token at index 0


# ── Quick self-test ───────────────────────────────────────────────
if __name__ == "__main__":
    char_to_int, int_to_char = build_char_maps()

    test_text = "Hello, World!"
    encoded = encode_label(test_text, char_to_int)
    decoded = decode_label(encoded, int_to_char)

    print(f"Alphabet size  : {len(ALPHABET)} characters")
    print(f"Model classes  : {get_num_classes()} (includes blank)")
    print(f"Original text  : {test_text}")
    print(f"Encoded        : {encoded}")
    print(f"Decoded back   : {decoded}")
    print(f"Round-trip OK  : {test_text == decoded}")
