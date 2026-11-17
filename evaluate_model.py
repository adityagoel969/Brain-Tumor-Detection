"""
Evaluate a trained 4-class brain MRI CNN on a held-out test folder.

Expected dataset structure:
Testing/
  glioma/
  meningioma/
  notumor/       (or no_tumor / no tumor, depending on your folder names)
  pituitary/

Run from your project root:
    python evaluate_model.py

Outputs:
    evaluation_results.json
    confusion_matrix.png
    classification_report.txt

IMPORTANT:
- Do not use training images as test images.
- Do not duplicate or augment test images to inflate the test count.
- Ensure model input preprocessing and class order match training.
"""
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

# Adjust these if your project uses different names/locations.
MODEL_CANDIDATES = [
    ROOT / "models" / "experimental_best_brain_tumor_cnn.keras",
    ROOT / "models" / "best_brain_tumor_cnn.keras",
]
TEST_DIR_CANDIDATES = [
    ROOT / "data" / "raw" / "Testing",
    ROOT / "data" / "raw" / "testing",
    ROOT / "Testing",
    ROOT / "testing",
    ROOT / "data" / "Testing",
    ROOT / "data" / "testing",
    ROOT / "dataset" / "Testing",
    ROOT / "dataset" / "testing",
]
CLASS_NAMES_FILE = ROOT / "models" / "class_names.json"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
OUTPUT_JSON = ROOT / "evaluation_results.json"
OUTPUT_CM = ROOT / "confusion_matrix.png"
OUTPUT_REPORT = ROOT / "classification_report.txt"

def first_existing(paths, kind):
    for path in paths:
        if path.exists():
            return path
    checked = "\n".join(f"  - {p}" for p in paths)
    raise FileNotFoundError(f"Could not find {kind}. Checked:\n{checked}")

def canonical_class(name):
    value = name.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    aliases = {
        "glioma": "glioma",
        "meningioma": "meningioma",
        "notumor": "notumor",
        "pituitary": "pituitary",
        "pituitarytumor": "pituitary",
    }
    return aliases.get(value, value)

def load_class_names(test_dir):
    # Prefer the class mapping saved during training.
    if CLASS_NAMES_FILE.exists():
        raw = json.loads(CLASS_NAMES_FILE.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            # Support {"0":"glioma", ...} or {"glioma":0, ...}.
            if all(str(k).isdigit() for k in raw.keys()):
                names = [raw[str(i)] for i in range(len(raw))]
            else:
                names = [k for k, _ in sorted(raw.items(), key=lambda kv: kv[1])]
        elif isinstance(raw, list):
            names = raw
        else:
            raise ValueError("class_names.json must contain a list or a class-index dictionary.")
        return [str(x) for x in names]

    # Fallback uses the dataset folder names; confirm this order matches training.
    folders = sorted(p.name for p in test_dir.iterdir() if p.is_dir())
    return folders

def main():
    model_path = first_existing(MODEL_CANDIDATES, "model file")
    test_dir = first_existing(TEST_DIR_CANDIDATES, "test directory")
    class_names = load_class_names(test_dir)

    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"Test folder: {test_dir}")
    print(f"Class order: {class_names}")

    # Keras assigns class indices alphabetically when class_names isn't specified.
    # Explicitly pass class_names so labels align with the model's training mapping.
    dataset = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        color_mode="rgb",
    )

    # The model was described as trained on RGB images normalized to [0, 1].
    # Apply this only if your training pipeline used this same normalization.
    dataset = dataset.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)

    y_true_parts, y_prob_parts = [], []
    for images, labels in dataset:
        outputs = model.predict(images, verbose=0)
        outputs = np.asarray(outputs)
        # Supports a single-output multiclass softmax model.
        if outputs.ndim != 2 or outputs.shape[1] != len(class_names):
            raise ValueError(
                f"Expected model output shape (batch, {len(class_names)}), got {outputs.shape}. "
                "Check class count and model output layer."
            )
        y_prob_parts.append(outputs)
        y_true_parts.append(labels.numpy())

    y_true = np.concatenate(y_true_parts)
    y_prob = np.concatenate(y_prob_parts)
    y_pred = np.argmax(y_prob, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))),
        average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    report_dict = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, zero_division=0
    )

    results = {
        "model_path": str(model_path.relative_to(ROOT)),
        "test_dir": str(test_dir.relative_to(ROOT)),
        "image_size": list(IMAGE_SIZE),
        "normalization": "pixel values divided by 255 (0-1)",
        "class_names": class_names,
        "test_images": int(len(y_true)),
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
    }
    OUTPUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    OUTPUT_REPORT.write_text(report_text, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Brain MRI CNN — Confusion Matrix")
    fig.colorbar(im, ax=ax)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks, labels=class_names, rotation=35, ha="right")
    ax.set_yticks(ticks, labels=class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    threshold = cm.max() / 2 if cm.size and cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black"
            )
    fig.tight_layout()
    fig.savefig(OUTPUT_CM, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("\n=== EVALUATION RESULTS ===")
    print(f"Test images: {len(y_true)}")
    print(f"Accuracy:    {accuracy:.4%}")
    print(f"Precision:   {precision:.4%} (macro)")
    print(f"Recall:      {recall:.4%} (macro)")
    print(f"F1-score:    {f1:.4%} (macro)")
    print("\nPer-class report:\n")
    print(report_text)
    print(f"\nSaved: {OUTPUT_JSON}")
    print(f"Saved: {OUTPUT_CM}")
    print(f"Saved: {OUTPUT_REPORT}")

if __name__ == "__main__":
    main()
