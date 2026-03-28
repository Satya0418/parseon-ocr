# Kaggle Full-Page Handwriting Training (IAM Forms)

This trains a **separate line-level model** for full-page extraction, so your single-word model stays untouched.

## 1. Kaggle setup
1. Create a new Kaggle Notebook.
2. Enable GPU (`T4` is fine).
3. Add dataset input: `IAM Handwritten Forms Dataset`.
4. Upload this file from your local project to Kaggle notebook files:
   - `kaggle_train_iam_forms_lines.py`

## 2. Run training
In a Kaggle notebook cell:

```bash
!python kaggle_train_iam_forms_lines.py --epochs 60 --batch_size 16 --learning_rate 1e-3
```

If you want a quick smoke run first:

```bash
!python kaggle_train_iam_forms_lines.py --epochs 3 --batch_size 8 --learning_rate 1e-3
```

## 3. Output files (download only these)
From `/kaggle/working/fullpage_saved_models/`:
- `crnn_iam_lines_best.weights.h5`
- `crnn_iam_lines_final.weights.h5`
- `crnn_iam_lines_inference.keras`
- `training_lines.csv`
- `char_map_lines.json`
- `test_split_lines.csv`

## 4. Next integration step
After training, share these with me:
1. `crnn_iam_lines_inference.keras`
2. `char_map_lines.json`

Then I will wire this model into **Handwriting Full Page** mode while keeping **Handwriting Single Word** unchanged.
