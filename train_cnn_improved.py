"""
AI-Based Brain Tumor Detection and Classification
Class-weighted training and evaluation for the existing 4-class CNN.
Authors: Aditya Goel, Piyush Raj (CSET214)

This script does NOT use MobileNetV2. It trains the custom CNN defined below
and saves experimental checkpoints separately from the existing baseline.

Run from the project root:
    python train.py --epochs 20 --batch-size 32

Evaluate a saved model without training:
    python train.py --evaluate-only models/experimental_best_brain_tumor_cnn.keras
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers


IMAGE_SIZE = (224, 224)


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate the custom brain tumor CNN")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--evaluate-only",
        metavar="MODEL_PATH",
        help="Evaluate an existing .keras model without training, e.g. models/best_brain_tumor_cnn.keras",
    )
    return parser.parse_args()


def build_cnn(input_shape=(224, 224, 3), num_classes=4, dropout_rate=0.4):
    """Custom CNN architecture; no pretrained/transfer-learning model."""
    inputs = keras.Input(shape=input_shape, name="mri_input")
    x = inputs

    for block, filters in enumerate((32, 64, 128), start=1):
        x = layers.Conv2D(
            filters, 3, padding="same", use_bias=False, name=f"block{block}_conv1"
        )(x)
        x = layers.BatchNormalization(name=f"block{block}_bn1")(x)
        x = layers.Activation("relu", name=f"block{block}_relu1")(x)

        x = layers.Conv2D(
            filters, 3, padding="same", use_bias=False, name=f"block{block}_conv2"
        )(x)
        x = layers.BatchNormalization(name=f"block{block}_bn2")(x)
        x = layers.Activation("relu", name=f"block{block}_relu2")(x)
        x = layers.MaxPooling2D(2, name=f"block{block}_pool")(x)
        x = layers.Dropout(0.10, name=f"block{block}_dropout")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="dense_1")(x)
    x = layers.Dropout(dropout_rate, name="classifier_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name="brain_tumor_cnn")


def load_class_names(models_dir):
    path = models_dir / "class_names.json"
    if not path.exists():
        raise FileNotFoundError(f"Class names file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    # Accept either ["glioma", ...] or {"0": "glioma", ...} formats.
    if isinstance(data, list):
        names = data
    elif isinstance(data, dict):
        try:
            names = [data[str(i)] if str(i) in data else data[i] for i in range(len(data))]
        except (KeyError, TypeError):
            names = list(data.values())
    else:
        raise ValueError("class_names.json must contain a list or dictionary.")

    return [str(name) for name in names]


def resolve_image_path(value, root_dir, data_dir):
    """Resolve old absolute paths and project-relative dataset paths."""
    raw = Path(str(value).strip().strip('"'))

    candidates = []

    if raw.is_absolute():
        candidates.append(raw)

        # Recover paths from an older project location.
        parts = raw.parts
        data_index = next(
            (i for i, part in enumerate(parts) if part.lower() == "data"),
            None
        )

        if data_index is not None:
            relative_part = parts[data_index + 1:]
            candidates.append(data_dir.joinpath(*relative_part))
            candidates.append(root_dir.joinpath("brain-tumor-detection", "data", *relative_part))
    else:
        candidates.extend([
            root_dir / raw,
            data_dir / raw,
            data_dir / "raw" / raw,
            root_dir / "brain-tumor-detection" / raw,
        ])

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())

    raise FileNotFoundError(
        f"Could not find image path from CSV: {value}\n"
        "Checked these locations:\n"
        + "\n".join(str(p) for p in candidates)
    )



def load_split_csv(csv_path, root_dir, data_dir, class_names):
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Required split CSV not found: {csv_path}\n"
            "Check that train_split.csv, val_split.csv, and test_split.csv are in data/."
        )

    frame = pd.read_csv(csv_path)
    if "filepath" not in frame.columns or "label" not in frame.columns:
        raise ValueError(
            f"{csv_path.name} must contain 'filepath' and 'label' columns. "
            f"Found: {list(frame.columns)}"
        )

    frame = frame.copy()
    frame["filepath"] = [
        resolve_image_path(value, root_dir, data_dir) for value in frame["filepath"]
    ]

    # Support numeric labels or class-name labels in the CSV.
    label_values = frame["label"]
    if pd.api.types.is_numeric_dtype(label_values):
        frame["label"] = label_values.astype("int32")
    else:
        name_to_id = {name: i for i, name in enumerate(class_names)}
        converted = label_values.astype(str).map(name_to_id)
        if converted.isna().any():
            unknown = sorted(label_values[converted.isna()].astype(str).unique())
            raise ValueError(f"Unknown class labels in {csv_path.name}: {unknown}")
        frame["label"] = converted.astype("int32")

    valid_ids = set(range(len(class_names)))
    found_ids = set(frame["label"].unique().tolist())
    if not found_ids.issubset(valid_ids):
        raise ValueError(
            f"Labels in {csv_path.name} are {sorted(found_ids)}, but class IDs "
            f"should be in {sorted(valid_ids)} according to class_names.json."
        )
    return frame


def load_image(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def make_dataset(frame, batch_size, training=False, use_augmentation=True, seed=42):
    autotune = tf.data.AUTOTUNE
    paths = frame["filepath"].astype(str).to_numpy()
    labels = frame["label"].astype("int32").to_numpy()
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(frame), seed=seed, reshuffle_each_iteration=True
        )

    dataset = dataset.map(load_image, num_parallel_calls=autotune)

    if training and use_augmentation:
        augmentation = keras.Sequential(
            [
                layers.RandomRotation(0.03, fill_mode="reflect", seed=seed),
                layers.RandomZoom(0.05, fill_mode="reflect", seed=seed + 1),
                layers.RandomContrast(0.10, seed=seed + 2),
            ],
            name="mri_augmentation",
        )
        dataset = dataset.map(
            lambda image, label: (augmentation(image, training=True), label),
            num_parallel_calls=autotune,
        )

    return dataset.batch(batch_size).prefetch(autotune)


def save_evaluation(model, test_ds, test_frame, class_names, reports_dir, figures_dir, output_prefix="evaluation"):
    """Evaluate once, then use the same predictions for all classification metrics."""
    print("\n--- Evaluating CNN on the held-out test set ---")
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

    # Dataset is finite and re-iterable. Concatenate labels and predictions in order.
    y_true = test_frame["label"].to_numpy(dtype=np.int32)
    probabilities = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)

    if len(y_true) != len(y_pred):
        raise RuntimeError(
            f"Prediction count ({len(y_pred)}) does not match test labels ({len(y_true)})."
        )

    report_text = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, zero_division=0
    )
    report_dict = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, output_dict=True, zero_division=0
    )

    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print("\nClassification report:\n")
    print(report_text)

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "classification_report": report_dict,
        "class_names": class_names,
        "test_samples": int(len(y_true)),
    }
    metrics_path = reports_dir / f"{output_prefix}_test_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    report_path = figures_dir / f"{output_prefix}_classification_report.txt"
    with report_path.open("w", encoding="utf-8") as file:
        file.write(report_text)

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(8, 7))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    display.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45, values_format="d")
    ax.set_title(f"CNN Test Confusion Matrix — Accuracy: {test_accuracy * 100:.2f}%")
    plt.tight_layout()
    matrix_path = figures_dir / f"{output_prefix}_test_confusion_matrix.png"
    plt.savefig(matrix_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved classification report: {report_path}")
    print(f"Saved confusion matrix: {matrix_path}")
    return test_accuracy


def main():
    args = parse_args()
    keras.utils.set_random_seed(args.seed)

    root_dir = Path(__file__).resolve().parent
    data_dir = root_dir / "data"
    models_dir = root_dir / "models"
    reports_dir = root_dir / "reports"
    figures_dir = reports_dir / "figures"
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(models_dir)
    print("Class order:", list(enumerate(class_names)))
    print("TensorFlow version:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))

    # Evaluation-only mode does not train or overwrite any model.
    if args.evaluate_only:
        model_path = Path(args.evaluate_only)
        if not model_path.is_absolute():
            model_path = root_dir / model_path
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        test_frame = load_split_csv(
            data_dir / "test_split.csv", root_dir, data_dir, class_names
        )
        test_ds = make_dataset(
            test_frame, args.batch_size, training=False, use_augmentation=False
        )
        model = keras.models.load_model(str(model_path))
        print(f"Loaded model: {model_path}")
        prefix = "experimental" if "experimental" in model_path.name.lower() else "evaluation"
        save_evaluation(model, test_ds, test_frame, class_names, reports_dir, figures_dir, prefix)
        return

    train_frame = load_split_csv(
        data_dir / "train_split.csv", root_dir, data_dir, class_names
    )
    val_frame = load_split_csv(
        data_dir / "val_split.csv", root_dir, data_dir, class_names
    )
    test_frame = load_split_csv(
        data_dir / "test_split.csv", root_dir, data_dir, class_names
    )

    print(
        f"Dataset splits -> Train: {len(train_frame)}, "
        f"Validation: {len(val_frame)}, Test: {len(test_frame)}"
    )
    print("Train label counts:", train_frame["label"].value_counts().sort_index().to_dict())
    print("Validation label counts:", val_frame["label"].value_counts().sort_index().to_dict())
    print("Test label counts:", test_frame["label"].value_counts().sort_index().to_dict())

    train_ds = make_dataset(
        train_frame, args.batch_size, training=True,
        use_augmentation=not args.no_augmentation, seed=args.seed
    )
    val_ds = make_dataset(
        val_frame, args.batch_size, training=False, use_augmentation=False
    )
    test_ds = make_dataset(
        test_frame, args.batch_size, training=False, use_augmentation=False
    )

    # Inverse-frequency weighting helps the model pay attention to every class.
    labels = train_frame["label"].to_numpy(dtype=np.int32)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(class_names)),
        y=labels,
    )
    class_weights = {int(class_id): float(weight) for class_id, weight in enumerate(weights)}
    print("Class weights:", {class_names[i]: round(class_weights[i], 3) for i in range(len(class_names))})

    model = build_cnn(
        input_shape=(224, 224, 3),
        num_classes=len(class_names),
        dropout_rate=args.dropout,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # Use experimental filenames so existing baseline files remain untouched.
    best_path = models_dir / "experiment_noaug_best.keras"
    last_path = models_dir / "experiment_noaug_last.keras"
    history_path = reports_dir / "experiment_noaug_history.csv"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_path), monitor="val_loss", mode="min",
            save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=args.patience,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min", factor=0.5,
            patience=2, min_lr=1e-6, verbose=1
        ),
    ]

    print(f"\nTraining custom CNN for up to {args.epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    model.save(str(last_path))
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(history_path, index=False)
    print(f"Saved final experimental model: {last_path}")
    print(f"Saved training history: {history_path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epoch_numbers = range(1, len(history_df) + 1)
    axes[0].plot(epoch_numbers, history_df["accuracy"], marker="o", label="Training")
    axes[0].plot(epoch_numbers, history_df["val_accuracy"], marker="s", label="Validation")
    axes[0].set_title("CNN Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].plot(epoch_numbers, history_df["loss"], marker="o", label="Training")
    axes[1].plot(epoch_numbers, history_df["val_loss"], marker="s", label="Validation")
    axes[1].set_title("CNN Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    plt.tight_layout()
    curve_path = figures_dir / "experimental_training_curves.png"
    plt.savefig(curve_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curves: {curve_path}")

    # Evaluate the best validation checkpoint on the held-out test set.
    best_model = keras.models.load_model(str(best_path))
    save_evaluation(best_model, test_ds, test_frame, class_names, reports_dir, figures_dir, "noaugS")
    print("\nTraining and evaluation completed.")


if __name__ == "__main__":
    main()
