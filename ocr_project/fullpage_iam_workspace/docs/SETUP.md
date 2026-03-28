# Full-Page IAM Workspace Setup

## Folder Map
- data/raw: downloaded Kaggle zip
- data/extracted: extracted dataset content
- code: training script
- notebooks: Kaggle training notebook copy
- scripts: one-click setup scripts

## Run Steps (PowerShell)
1. cd d:/text/ocr_project/fullpage_iam_workspace
2. ./scripts/01_download_dataset.ps1
3. ./scripts/02_validate_and_extract.ps1

## Notes
- If step 2 reports ZIP_BAD or invalid zip, run step 1 again.
- The dataset is large. A partial download produces a corrupt zip.
- After extraction, you can point training code to data/extracted.
