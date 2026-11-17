# NeuroVision AI — Connected Streamlit Prototype

This is the updated Streamlit app based on the frontend you supplied. It keeps the NeuroVision AI visual design and pages, and connects the MRI Analyzer page to your trained TensorFlow/Keras CNN.

## 1. Project layout

Keep your existing trained model files and copy these updated frontend files into the project root:

```text
brain-tumor-detection/
├── app.py
├── requirements.txt
├── README.md
└── models/
    ├── experimental_best_brain_tumor_cnn.keras   # preferred
    ├── best_brain_tumor_cnn.keras                 # fallback
    └── class_names.json                           # recommended
```

The app checks the preferred model first, then the fallback. You only need one of the `.keras` files.

## 2. Class names

Create `models/class_names.json` using the exact output order from your model training. Example only:

```json
["glioma", "meningioma", "notumor", "pituitary"]
```

**Important:** do not use that order unless it matches the class-index order used when training. If your existing `class_names.json` already exists, keep it.

## 3. Install and run (Windows / VS Code)

Open a terminal in the project root and run:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If TensorFlow is already installed in the project's environment, you can install only the other missing packages. TensorFlow installation depends on your Python version and operating system; use a Python version supported by the TensorFlow release you install.

## 4. Image preprocessing assumption

The app currently converts images to RGB, resizes to `224 × 224`, and scales pixel values to `[0, 1]`. This matches the project details previously provided. If the model was trained with a different size or preprocessing pipeline, update `IMAGE_SIZE` and `predict_mri()` accordingly.

## 5. Model Insights page

The metrics remain placeholders until you provide real evaluation outputs. Do not report placeholder values as measured performance. The analyzer displays model scores as raw outputs; they are not guaranteed to be calibrated probabilities.

## 6. Important limitation

This is an academic demonstration only, not a medical diagnostic tool. Predictions must not be used for clinical decisions.
