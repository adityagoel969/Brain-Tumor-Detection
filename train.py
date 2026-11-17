"""
AI-Based Brain Tumor Detection and Classification
CNN-only training and evaluation script.

This script does not use MobileNetV2.
It can train a fresh CNN, continue training a saved CNN checkpoint,
or evaluate an existing checkpoint without retraining.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.utils.class_weight import compute_class_weight


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate the brain tumor CNN")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a compatible saved CNN .keras model to continue training from.",
    )
    parser.add_argument(
        "--evaluate-only",
        type=str,
        default=None,
        help="Path to a saved .keras model to evaluate without training.",
    )
    parser.add_argument(
        "--class-weights",
        choices=("none", "balanced"),
        default="none",
        help="Optionally use balanced class weights during training.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="best_brain_tumor_cnn.keras",
        help="Filename for the best checkpoint saved in models/.",
    )
    return parser.parse_args()


def build_cnn(input_shape=(224, 224, 3), num_classes=4, dropout_rate=0.3):
    """Custom CNN architecture; no transfer-learning backbone is used."""
    inputs = keras.Input(shape=input_shape, name="mri_input")
    x = inputs

    for i, filters in enumerate((32, 64, 128), start=1):
        x = layers.Conv2D(
            filters, 3, padding="same", use_bias=False, name=f"block{i}_conv"
        )(x)
        x = layers.BatchNormalization(name=f"block{i}_bn")(x)
        x = layers.ReLU(name=f"block{i}_relu")(x)
        x = layers.MaxPooling2D(2, name=f"block{i}_pool")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="dense_1")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name="brain_tumor_cnn")


def load_class_names(path):
    with open(path, "r", encoding="utf-8") as f:
        value = json.load(f)

    # Support either ["glioma", ...] or {"0": "glioma", ...}.
    if isinstance(value, list):
        names = value
    elif isinstance(value, dict):
        try:
            names = [value[str(i)] if str(i) in value else value[i]
                     for i in range(len(value))]
        except (KeyError, TypeError):
            # Also support {"glioma": 0, ...}.
            names = [name for name, index in sorted(value.items(), key=lambda item: item[1])]
    else:
        raise ValueError("class_names.json must contain a list or dictionary.")

    names = [str(name) for name in names]
    if len(names) != 4:
        raise ValueError(f"Expected 4 class names, found {len(names)}: {names}")
    return names


def resolve_image_path(raw_path, project_root):
    """Resolve absolute paths, project-relative paths, or stale paths containing data/."""
    path_text = str(raw_path).strip().strip('"')
    candidate = Path(path_text)

    if candidate.is_file():
        return str(candidate)

    if not candidate.is_absolute():
        candidate2 = project_root / candidate
        if candidate2.is_file():
            return str(candidate2)

    # Handles CSVs created on another machine/location, e.g. .../data/raw/Training/...
    normalized_parts = list(candidate.parts)
    lowered = [part.lower() for part in normalized_parts]
    if "data" in lowered:
        data_index = lowered.index("data")
        relative_suffix = Path(*normalized_parts[data_index:])
        candidate3 = project_root / relative_suffix
        if candidate3.is_file():
            return str(candidate3)

    raise FileNotFoundError(
        f"Image path does not exist: {path_text}\n"
        "Check data/train_split.csv, val_split.csv, and test_split.csv."
    )


def get_data_loaders(data_dir, project_root, batch_size=32,
                     use_augmentation=True, seed=42):
    img_size = (224, 224)
    autotune = tf.data.AUTOTUNE

    csv_paths = {
        "train": data_dir / "train_split.csv",
        "val": data_dir / "val_split.csv",
        "test": data_dir / "test_split.csv",
    }
    missing_csvs = [str(p) for p in csv_paths.values() if not p.exists()]
    if missing_csvs:
        raise FileNotFoundError(
            "Missing split CSV file(s): " + ", ".join(missing_csvs)
            + "\nRun the dataset setup/preprocessing notebook first."
        )

    frames = {name: pd.read_csv(path) for name, path in csv_paths.items()}
    for split_name, frame in frames.items():
        required = {"filepath", "label"}
        missing_columns = required - set(frame.columns)
        if missing_columns:
            raise ValueError(
                f"{split_name}_split.csv is missing columns {sorted(missing_columns)}. "
                f"Found columns: {list(frame.columns)}"
            )
        if frame.empty:
            raise ValueError(f"{split_name}_split.csv contains no rows.")

        frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype("int32")
        if not frame["label"].between(0, 3).all():
            raise ValueError(f"{split_name} split contains labels outside 0..3.")

        print(f"Resolving {split_name} image paths...")
        frame["filepath"] = frame["filepath"].map(
            lambda p: resolve_image_path(p, project_root)
        )

    print(
        "Dataset splits -> "
        f"Train: {len(frames['train'])}, "
        f"Validation: {len(frames['val'])}, "
        f"Test: {len(frames['test'])}"
    )
    for name, frame in frames.items():
        print(f"{name.title()} label counts:")
        print(frame["label"].value_counts().sort_index().to_dict())

    def load_image(path, label):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(
            image_bytes, channels=3, expand_animations=False
        )
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, img_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    augmentation = None
    if use_augmentation:
        augmentation = keras.Sequential(
            [
                layers.RandomRotation(
                    0.03, fill_mode="constant", fill_value=0.0
                ),
                layers.RandomZoom(
                    0.05, fill_mode="constant", fill_value=0.0
                ),
            ],
            name="mri_augmentation",
        )

    def make_dataset(frame, training=False):
        paths = frame["filepath"].astype(str).to_numpy()
        labels = frame["label"].to_numpy(dtype=np.int32)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training:
            ds = ds.shuffle(
                buffer_size=len(frame),
                seed=seed,
                reshuffle_each_iteration=True,
            )
        ds = ds.map(load_image, num_parallel_calls=autotune)
        if training and augmentation is not None:
            ds = ds.map(
                lambda image, label: (augmentation(image, training=True), label),
                num_parallel_calls=autotune,
            )
        return ds.batch(batch_size).prefetch(autotune)

    return (
        make_dataset(frames["train"], training=True),
        make_dataset(frames["val"]),
        make_dataset(frames["test"]),
        frames["train"],
    )


def save_evaluation(model, test_ds, class_names, reports_dir, figures_dir):
    print("\n--- Evaluating CNN on the held-out test set ---")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)

    y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
    probabilities = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)

    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print("\nClassification report:\n", report_text)

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "classification_report": report_dict,
        "class_names": class_names,
    }
    with open(reports_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report_path = figures_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=class_names
    ).plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45, values_format="d")
    ax.set_title(f"CNN Test Confusion Matrix (Accuracy: {test_acc * 100:.1f}%)")
    plt.tight_layout()
    cm_path = figures_dir / "test_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close(fig)

    print(f"Saved metrics: {reports_dir / 'test_metrics.json'}")
    print(f"Saved classification report: {report_path}")
    print(f"Saved confusion matrix: {cm_path}")


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
    reports_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(models_dir / "class_names.json")
    print("Class order:", list(enumerate(class_names)))
    print("TensorFlow version:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))

    _, _, test_ds, train_df = get_data_loaders(
        data_dir,
        root_dir,
        batch_size=args.batch_size,
        use_augmentation=False,
        seed=args.seed,
    )

    # Evaluation-only mode does not train or overwrite a checkpoint.
    if args.evaluate_only:
        evaluation_path = Path(args.evaluate_only)
        if not evaluation_path.is_absolute():
            evaluation_path = root_dir / evaluation_path
        if not evaluation_path.is_file():
            raise FileNotFoundError(f"Model not found: {evaluation_path}")
        model = keras.models.load_model(str(evaluation_path))
        if model.output_shape[-1] != len(class_names):
            raise ValueError(
                f"Model has {model.output_shape[-1]} outputs but "
                f"class_names.json contains {len(class_names)} classes."
            )
        save_evaluation(model, test_ds, class_names, reports_dir, figures_dir)
        return

    # Build training datasets separately so augmentation is applied only to training.
    train_ds, val_ds, _, train_df = get_data_loaders(
        data_dir,
        root_dir,
        batch_size=args.batch_size,
        use_augmentation=not args.no_augmentation,
        seed=args.seed,
    )

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = root_dir / resume_path
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume model not found: {resume_path}")
        print(f"Loading existing CNN checkpoint: {resume_path}")
        model = keras.models.load_model(str(resume_path))
        if tuple(model.input_shape[1:]) != (224, 224, 3):
            raise ValueError(
                f"Saved model input shape is {model.input_shape}; expected "
                "(None, 224, 224, 3). Do not resume this checkpoint with this script."
            )
        if model.output_shape[-1] != len(class_names):
            raise ValueError(
                f"Saved model has {model.output_shape[-1]} outputs but "
                f"there are {len(class_names)} classes."
            )
    else:
        print("Building a fresh custom CNN (no MobileNetV2)...")
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

    output_path = models_dir / args.output_name
    last_path = models_dir / "last_brain_tumor_cnn.keras"
    history_path = reports_dir / "training_history.csv"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    class_weight = None
    if args.class_weights == "balanced":
        present_classes = np.unique(train_df["label"].to_numpy())
        weights = compute_class_weight(
            class_weight="balanced",
            classes=present_classes,
            y=train_df["label"].to_numpy(),
        )
        class_weight = {int(c): float(w) for c, w in zip(present_classes, weights)}
        print("Using class weights:", class_weight)

    print(f"\nTraining CNN for up to {args.epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    model.save(str(last_path))
    print(f"Saved final-epoch model: {last_path}")
    print(f"Saved best-validation-loss model: {output_path}")

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(history_path, index=False)
    print(f"Saved training history: {history_path}")

    epochs_range = range(1, len(history_df) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs_range, history_df["accuracy"], label="Training", marker="o")
    axes[0].plot(epochs_range, history_df["val_accuracy"], label="Validation", marker="s")
    axes[0].set_title("CNN Accuracy Across Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].plot(epochs_range, history_df["loss"], label="Training", marker="o")
    axes[1].plot(epochs_range, history_df["val_loss"], label="Validation", marker="s")
    axes[1].set_title("CNN Loss Across Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    curves_path = figures_dir / "training_curves.png"
    plt.savefig(curves_path, dpi=150)
    plt.close(fig)
    print(f"Saved training curves: {curves_path}")

    # Always evaluate the best checkpoint, not the final epoch checkpoint.
    best_model = keras.models.load_model(str(output_path))
    # Recreate the test dataset without augmentation (same held-out split).
    _, _, test_ds, _ = get_data_loaders(
        data_dir, root_dir, batch_size=args.batch_size,
        use_augmentation=False, seed=args.seed
    )
    save_evaluation(best_model, test_ds, class_names, reports_dir, figures_dir)
    print("\nCNN training and evaluation completed.")


if __name__ == "__main__":
    main()
