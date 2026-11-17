# Evaluate NeuroVision AI CNN

1. Copy `evaluate_model.py` into your project root (same folder as `app.py`).
2. Confirm your trained model exists at one of:
   - `models/experimental_best_brain_tumor_cnn.keras`
   - `models/best_brain_tumor_cnn.keras`
3. Confirm your labeled test images are in a folder such as `Testing/`, with one subfolder per class.
4. Install any missing packages:
   `python -m pip install tensorflow scikit-learn matplotlib numpy`
5. Run:
   `python evaluate_model.py`

The script creates `evaluation_results.json`, `confusion_matrix.png`, and `classification_report.txt`.
It prints accuracy, macro precision, macro recall, macro F1, and per-class metrics.

Important:
- Class mapping must match the mapping used during model training. Prefer `models/class_names.json`.
- Preprocessing is set to RGB 224x224 and pixel values divided by 255. Change it if training used something else.
- Only use unique held-out test images; don't use training images, duplicates, or augmented copies as test samples.
