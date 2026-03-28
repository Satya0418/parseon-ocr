import zipfile
import os
from pathlib import Path

base = Path("d:/text/ocr_project")
zip_path = "d:/text/ocr_kaggle_VERIFIED.zip"

print("Creating VERIFIED Kaggle zip file...")
print("=" * 60)

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    
    # 1. Backend code
    print("Adding backend code...")
    backend_count = 0
    for py_file in base.glob("backend/**/*.py"):
        arcname = f"ocr_project/{py_file.relative_to(base).as_posix()}"
        zf.write(py_file, arcname)
        backend_count += 1
    print(f"  ✓ {backend_count} Python files")
    
    # 2. CSV splits
    print("Adding CSV splits...")
    csv_count = 0
    for csv_file in base.glob("data/iam_words/splits/*.csv"):
        arcname = f"ocr_project/{csv_file.relative_to(base).as_posix()}"
        zf.write(csv_file, arcname)
        csv_count += 1
    print(f"  ✓ {csv_count} CSV files")
    
    # 3. Weights file
    print("Adding weights file...")
    weights = base / "saved_models/crnn_iam_v1_best.weights.h5"
    if weights.exists():
        arcname = f"ocr_project/saved_models/crnn_iam_v1_best.weights.h5"
        zf.write(weights, arcname)
        size_mb = weights.stat().st_size / 1024 / 1024
        print(f"  ✓ Weights ({size_mb:.1f} MB)")
    else:
        print(f"  ✗ WEIGHTS NOT FOUND!")
    
    # 4. ALL IMAGE FILES
    print("Adding images (this takes 2-3 minutes)...")
    img_count = 0
    for png_file in base.glob("data/iam_words/iam_words/words/**/*.png"):
        arcname = f"ocr_project/{png_file.relative_to(base).as_posix()}"
        zf.write(png_file, arcname)
        img_count += 1
        if img_count % 15000 == 0:
            print(f"  ... {img_count} images added")
    
    print(f"  ✓ {img_count} images")

print("=" * 60)
size_mb = os.path.getsize(zip_path) / 1024 / 1024
print(f"DONE: {zip_path}")
print(f"Size: {size_mb:.1f} MB")
print(f"Images: {img_count} (expected: 115,320)")
if img_count == 115320:
    print("✓ ZIP IS PERFECT!")
else:
    print(f"✗ WARNING: Missing {115320 - img_count} images")
