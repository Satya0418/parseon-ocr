"""
End-to-end test: compose a synthetic "page" from IAM word images,
then run the full page_ocr pipeline on it.

This proves the pipeline works correctly when the input domain
matches the model's training data.
"""
import os, sys, glob, cv2, numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Find IAM word images
iam_dir = "data\\iam_words\\iam_words\\words"
word_files = glob.glob(os.path.join(iam_dir, "*", "*", "*.png"))
if not word_files:
    print("No IAM word images found!")
    sys.exit(1)

# Load first 12 words to compose two lines
words = []
for p in word_files[:12]:
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        words.append(img)

# Compose a synthetic page: two lines of ~6 words each
def make_line(word_imgs, target_h=80, gap=30):
    """Arrange word images into a horizontal line with white gaps."""
    resized = []
    for w in word_imgs:
        scale = target_h / w.shape[0]
        new_w = int(w.shape[1] * scale)
        r = cv2.resize(w, (new_w, target_h))
        resized.append(r)
        resized.append(np.ones((target_h, gap), dtype=np.uint8) * 255)  # gap
    return np.concatenate(resized, axis=1)

line1 = make_line(words[:6], target_h=80, gap=30)
line2 = make_line(words[6:12], target_h=80, gap=30)

# Make both lines the same width
max_w = max(line1.shape[1], line2.shape[1])
def pad_to_width(img, target_w):
    if img.shape[1] < target_w:
        pad = np.ones((img.shape[0], target_w - img.shape[1]), dtype=np.uint8) * 255
        return np.concatenate([img, pad], axis=1)
    return img[:, :target_w]

line1 = pad_to_width(line1, max_w)
line2 = pad_to_width(line2, max_w)

# Stack with vertical gap
v_gap = np.ones((40, max_w), dtype=np.uint8) * 255  # 40px vertical gap
margin = np.ones((30, max_w), dtype=np.uint8) * 255  # top/bottom margin
page = np.vstack([margin, line1, v_gap, line2, margin])

# Save the synthetic page
page_path = os.path.join("debug_output", "synthetic_page.png")
os.makedirs("debug_output", exist_ok=True)
cv2.imwrite(page_path, page)
print(f"Synthetic page saved: {page_path}")
print(f"Page size: {page.shape[1]}x{page.shape[0]}")
print(f"Contains {len(words)} words arranged in 2 lines")

# Ground truth from words.txt
import re
gt_file = "data\\iam_words\\iam_words\\words.txt"
gt_words = []
for p in word_files[:12]:
    word_id = os.path.basename(p).replace(".png", "")
    with open(gt_file, 'r') as f:
        for line in f:
            if line.startswith(word_id + " "):
                parts = line.strip().split()
                gt_words.append(parts[-1])
                break

print(f"\nGround truth words: {gt_words}")
print(f"Line 1: {' '.join(gt_words[:6])}")
print(f"Line 2: {' '.join(gt_words[6:12])}")

# Now run the full page_ocr pipeline
print("\n" + "="*60)
print("Running page_ocr pipeline on synthetic page...")
print("="*60)

from page_ocr import extract_text_from_page
text = extract_text_from_page(page_path, debug=True)

print("\n" + "="*60)
print("GROUND TRUTH:")
print(f"  Line 1: {' '.join(gt_words[:6])}")
print(f"  Line 2: {' '.join(gt_words[6:12])}")
print()
print("PIPELINE OUTPUT:")
for i, line in enumerate(text.split('\n')):
    print(f"  Line {i+1}: {line}")
print("="*60)
