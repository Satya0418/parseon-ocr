import zipfile
import os
from pathlib import Path

print("Creating Kaggle dataset zip...")
zip_path = "d:/text/ocr_kaggle_dataset_NEW.zip"

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    base = Path("d:/text/ocr_project")
    
    # Add all backend code
    for f in base.glob("backend/**/*.py"):
        zf.write(f, f"ocr_project/{f.relative_to(base)}")
    
    # Add splits
    for f in base.glob("data/iam_words/splits/*.csv"):
        zf.write(f, f"ocr_project/{f.relative_to(base)}")
    
    # Add weights
    weights = base / "saved_models/crnn_iam_v1_best.weights.h5"
    if weights.exists():
        zf.write(weights, f"ocr_project/{weights.relative_to(base)}")
    
    # Add all images
    print("Adding images...")
    count = 0
    for f in base.glob("data/iam_words/iam_words/words/**/*.png"):
        zf.write(f, f"ocr_project/{f.relative_to(base)}")
        count += 1
        if count % 10000 == 0:
            print(f"  {count} images added...")
    
    print(f"Total: {count} images")

size_mb = os.path.getsize(zip_path) / 1024 / 1024
print(f"\nDone! {zip_path} ({size_mb:.1f} MB)")
