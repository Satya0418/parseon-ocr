"""
=============================================================
metrics.py  —  Evaluation Metrics (CER and WER)
=============================================================

WHY DO WE NEED THIS?
--------------------
Accuracy alone tells you "right or wrong" for full strings.
But for OCR, we need finer measurement:

  CER (Character Error Rate):
    Measures what % of individual characters the model got wrong.
    If the answer is "hello" and model says "helo", CER = 1/5 = 0.20

  WER (Word Error Rate):
    Same idea but at the word level.
    If answer is "hello world" and model says "hello wrold", WER = 1/2 = 0.50

Both are based on "edit distance" — how many edits (insert, delete, replace)
are needed to turn the prediction into the ground truth.

Lower CER/WER = better model.
CER of 0.0 = perfect character-level accuracy.
=============================================================
"""

import numpy as np


def edit_distance(seq1: str, seq2: str) -> int:
    """
    Compute the Levenshtein (edit) distance between two strings.

    WHY: Edit distance counts the minimum number of single-character
    operations (insert, delete, replace) needed to transform seq1 into seq2.
    It is the foundation for both CER and WER.

    Parameters
    ----------
    seq1 : str  — the reference (ground truth) string
    seq2 : str  — the hypothesis (model prediction) string

    Returns
    -------
    int  — the edit distance (0 = identical strings)

    HOW IT WORKS (Dynamic Programming):
    We build a 2D table where cell [i][j] = min edits to convert
    seq1[:i] → seq2[:j]. We fill it row by row using:
      - If characters match: carry the diagonal value
      - If they don't: 1 + min(replace, insert, delete)

    Example:
      edit_distance("cat", "cut")  →  1  (replace 'a' with 'u')
      edit_distance("hello", "helo")  →  1  (delete one 'l')
    """
    len1, len2 = len(seq1), len(seq2)

    # Create a 2D matrix filled with zeros
    # dp[i][j] = edit distance between seq1[:i] and seq2[:j]
    dp = np.zeros((len1 + 1, len2 + 1), dtype=int)

    # Base cases: converting to/from empty string costs len insertions/deletions
    dp[:, 0] = np.arange(len1 + 1)
    dp[0, :] = np.arange(len2 + 1)

    # Fill the table
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if seq1[i - 1] == seq2[j - 1]:
                # Characters match — no operation needed
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Characters differ — pick cheapest operation
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # delete from seq1
                    dp[i][j - 1],      # insert into seq1
                    dp[i - 1][j - 1],  # replace
                )

    return dp[len1][len2]


def character_error_rate(predictions: list, ground_truths: list) -> float:
    """
    Compute the Character Error Rate (CER) across a batch.

    WHY: CER tells us what fraction of characters the model predicted wrong.
    It is the standard metric for OCR evaluation.

    Formula:
        CER = total_edit_distance / total_number_of_reference_characters

    Parameters
    ----------
    predictions  : list of str  — model-predicted texts
    ground_truths: list of str  — the true labels

    Returns
    -------
    float  — CER value between 0.0 (perfect) and 1.0+ (many errors)

    Example
    -------
    predictions   = ["helo", "wrold"]
    ground_truths = ["hello", "world"]
    CER = (1 + 1) / (5 + 5) = 0.20   (20% character error)
    """
    total_distance = 0
    total_chars = 0

    for pred, truth in zip(predictions, ground_truths):
        total_distance += edit_distance(pred, truth)
        total_chars += len(truth)

    if total_chars == 0:
        return 0.0  # edge case: empty references

    return total_distance / total_chars


def word_error_rate(predictions: list, ground_truths: list) -> float:
    """
    Compute the Word Error Rate (WER) across a batch.

    WHY: WER measures errors at the word level, useful for
    evaluating sentences and phrases.

    Formula:
        WER = total_word_edit_distance / total_number_of_reference_words

    Parameters
    ----------
    predictions  : list of str  — model-predicted texts
    ground_truths: list of str  — the true labels

    Returns
    -------
    float  — WER value between 0.0 (perfect) and 1.0+ (many errors)
    """
    total_distance = 0
    total_words = 0

    for pred, truth in zip(predictions, ground_truths):
        # Split into word lists and compute edit distance on word sequences
        pred_words = pred.split()
        truth_words = truth.split()
        total_distance += edit_distance(" ".join(pred_words), " ".join(truth_words))
        total_words += len(truth_words)

    if total_words == 0:
        return 0.0

    return total_distance / total_words


def print_metrics_report(predictions: list, ground_truths: list):
    """
    Print a formatted metrics report comparing predictions to ground truth.

    WHY: Quick diagnostic tool during training evaluation to see both
    per-sample predictions and aggregate metrics in one clean view.

    Parameters
    ----------
    predictions  : list of str
    ground_truths: list of str
    """
    cer = character_error_rate(predictions, ground_truths)
    wer = word_error_rate(predictions, ground_truths)

    print("=" * 60)
    print(f"{'METRICS REPORT':^60}")
    print("=" * 60)
    print(f"{'Samples evaluated':<30} {len(predictions)}")
    print(f"{'Character Error Rate (CER)':<30} {cer:.4f}  ({cer*100:.2f}%)")
    print(f"{'Word Error Rate (WER)':<30} {wer:.4f}  ({wer*100:.2f}%)")
    print("-" * 60)
    print(f"{'#':<5} {'Ground Truth':<30} {'Prediction':<30}")
    print("-" * 60)
    for i, (gt, pred) in enumerate(zip(ground_truths, predictions)):
        marker = "✓" if gt == pred else "✗"
        print(f"{marker} {i+1:<4} {gt:<30} {pred:<30}")
    print("=" * 60)


# ── Quick self-test ───────────────────────────────────────────────
if __name__ == "__main__":
    preds  = ["helo world", "ocr systm", "hello"]
    truths = ["hello world", "ocr system", "hello"]
    print_metrics_report(preds, truths)
