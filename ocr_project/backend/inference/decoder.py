"""
=============================================================
decoder.py  —  CTC Decoding Strategies
=============================================================

WHY DO WE NEED A SEPARATE DECODER?
------------------------------------
The model's final output is a matrix of shape:
    (1, 128, 81)   →  128 time steps × 81 character probabilities

This raw matrix is NOT text. A decoder reads this matrix and
converts it into a human-readable string.

TWO DECODING STRATEGIES:
=========================

1. GREEDY DECODING (fast, good for most cases):
   At each time step, just pick the character with the highest probability.
   Then collapse duplicates and remove blanks.

   Example output probabilities (simplified to 4 classes: a, b, c, blank):
   t=1: [0.8, 0.1, 0.05, 0.05] → 'a'  (highest = a)
   t=2: [0.7, 0.2, 0.05, 0.05] → 'a'  (highest = a, same as before)
   t=3: [0.1, 0.1, 0.05, 0.75] → '_'  (blank)
   t=4: [0.1, 0.8, 0.05, 0.05] → 'b'
   
   After collapse: [a, a, _, b] → [a, _, b] → 'ab'

2. BEAM SEARCH DECODING (slower, more accurate):
   Instead of picking top-1, keep the top-K candidate sequences
   at each step and score them at the end.
   
   This finds the GLOBALLY best sequence, not just the locally
   greedy one. Better when similar characters are easily confused
   (e.g., 'l' vs '1', 'O' vs '0').
=============================================================
"""

import numpy as np
import tensorflow as tf

from backend.utils.char_map import BLANK_INDEX


def greedy_decode(predictions: np.ndarray, blank_index: int = BLANK_INDEX) -> list:
    """
    Greedy CTC Decoder — fast, straightforward, works well for most OCR.

    WHY: When model confidence is high (well-trained model), greedy
    decoding gives the same result as beam search while being much faster.
    Use this for production inference on a trained model.

    ALGORITHM:
    ----------
    Step 1: argmax — at each time step, pick the class with highest prob
    Step 2: collapse — remove consecutive duplicate classes
    Step 3: remove blanks — delete all blank tokens

    Parameters
    ----------
    predictions : np.ndarray  shape (time_steps, num_classes)
                  Softmax output for ONE sample (not a batch).
                  Values are probabilities summing to 1 per time step.

    blank_index : int  — the index reserved for the CTC blank token (default 0)

    Returns
    -------
    list of int  — decoded integer sequence (without blanks or duplicates)

    Example
    -------
    If predictions output argmax = [3, 3, 0, 5, 5, 0, 3]
    Step 1 (argmax)  : already done above
    Step 2 (collapse): [3, 0, 5, 0, 3]
    Step 3 (rm blank): [3, 5, 3]  → "aba" (if 3='a', 5='b')
    """
    # Step 1: Pick the most probable class at each time step
    best_path = np.argmax(predictions, axis=-1)  # shape: (time_steps,)

    # Step 2: Collapse consecutive duplicates
    # Compare each element to the previous; keep only where they differ
    collapsed = [best_path[0]]
    for i in range(1, len(best_path)):
        if best_path[i] != best_path[i - 1]:
            collapsed.append(best_path[i])

    # Step 3: Remove blank tokens
    result = [idx for idx in collapsed if idx != blank_index]

    return result


def beam_search_decode(predictions: np.ndarray,
                       beam_width:  int = 10,
                       blank_index: int = BLANK_INDEX) -> list:
    """
    Beam Search CTC Decoder — more accurate, slower than greedy.

    WHY: Greedy decoding picks the locally best character at each step,
    but that local best might not lead to the globally best sequence.
    Beam search explores the top-K paths simultaneously and returns
    the globally most probable sequence.

    WHEN TO USE:
    - When character error rate needs to be minimized at any cost
    - When model confidence is low (ambiguous predictions)
    - Post-training evaluation / benchmarking

    HOW IT WORKS:
    -------------
    We maintain a dictionary of K "beams" = partial sequences.
    At each time step, we:
      1. Extend each beam with EVERY possible next character
      2. Score each extended beam = old_score × P(new_char | t)
      3. Keep only the top-K beams (prune the rest)
    After all time steps, return the highest-scoring beam.

    Parameters
    ----------
    predictions : np.ndarray  shape (time_steps, num_classes)
                  Softmax probabilities for ONE sample.
    beam_width  : int   — how many candidate sequences to track (K)
                          Higher = more accurate but slower.
                          10–20 is a good balance.
    blank_index : int   — CTC blank token index

    Returns
    -------
    list of int  — best decoded integer sequence

    NOTE: This is a simplified beam search without CTC prefix merging.
    A full CTC beam search (with prefix score merging) is more complex.
    For production use, consider: pip install ctc_decoder
    """
    time_steps, num_classes = predictions.shape

    # Initialize beams as: { sequence_tuple: log_probability }
    # Start with one empty beam with probability 1.0 → log(1) = 0
    beams = {(): 0.0}

    for t in range(time_steps):
        new_beams = {}

        for beam, score in beams.items():
            for c in range(num_classes):
                # Log probability: log(score × P(c)) = score + log(P(c))
                # Using log probabilities prevents numerical underflow
                # (multiplying many small numbers → 0, but adding logs → fine)
                new_score = score + np.log(predictions[t, c] + 1e-10)
                new_beam  = beam + (c,)
                new_beams[new_beam] = max(
                    new_beams.get(new_beam, -np.inf),
                    new_score
                )

        # Keep only the top-K beams
        beams = dict(
            sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:beam_width]
        )

    # Get the best beam sequence
    best_beam = max(beams, key=beams.get)

    # Apply CTC collapse rules: remove consecutive duplicates then blanks
    collapsed = greedy_decode(
        # Rebuild one-hot-like array so we can reuse greedy_decode's logic
        np.eye(num_classes)[[c for c in best_beam]],
        blank_index=blank_index
    )

    return collapsed


def decode_batch(predictions: np.ndarray,
                 int_to_char: dict,
                 method:      str = "greedy",
                 beam_width:  int = 10,
                 blank_index: int = BLANK_INDEX) -> list:
    """
    Decode a full batch of model predictions into text strings.

    WHY: After model.predict() returns predictions for a whole batch,
    we need to decode each sample individually and collect the results.

    Parameters
    ----------
    predictions : np.ndarray  shape (batch_size, time_steps, num_classes)
                  Batch of softmax probability matrices from the model.
    int_to_char : dict         — integer → character mapping from char_map.py
    method      : str          — 'greedy' or 'beam'
    beam_width  : int          — only used if method='beam'

    Returns
    -------
    list of str  — one decoded text string per sample in the batch
    """
    texts = []

    for sample_pred in predictions:
        # sample_pred shape: (time_steps, num_classes)
        if method == "greedy":
            indices = greedy_decode(sample_pred, blank_index=blank_index)
        elif method == "beam":
            indices = beam_search_decode(sample_pred, beam_width=beam_width, blank_index=blank_index)
        else:
            raise ValueError(f"Unknown decode method '{method}'. Use 'greedy' or 'beam'.")

        # Convert integer indices to characters
        text = "".join(int_to_char.get(idx, "?") for idx in indices)
        texts.append(text)

    return texts


# ── Quick self-test ───────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from backend.utils.char_map import build_char_maps, ALPHABET

    _, int_to_char = build_char_maps(ALPHABET)
    num_classes    = len(ALPHABET) + 1

    # Simulate model output: random probabilities (batch=2, time=20, classes=81)
    np.random.seed(42)
    fake_predictions = np.random.dirichlet(
        np.ones(num_classes), size=(2, 20)
    ).astype(np.float32)

    print("Decoder Self-Test")
    print("-" * 40)
    texts_g = decode_batch(fake_predictions, int_to_char, method="greedy")
    texts_b = decode_batch(fake_predictions, int_to_char, method="beam", beam_width=5)

    for i, (g, b) in enumerate(zip(texts_g, texts_b)):
        print(f"  Sample {i+1}:")
        print(f"    Greedy : '{g}'")
        print(f"    Beam   : '{b}'")
