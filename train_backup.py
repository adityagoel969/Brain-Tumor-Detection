"""
AI-Based Brain Tumor Detection and Classification
Training Script for 4-Class MRI CNN Model
Authors: Aditya Goel, Piyush Raj (CSET214)
"""

import argparse
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def parse_args():
    parser = argparse.ArgumentParser(description="Train Brain Tumor CNN")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience")
    parser.add_argument("--no-augmentation", action="store_true", help="Disable data augmentation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def build_cnn(input_shape=(224, 224, 3), num_classes=4, dropout_rate=0.3):
    inputs = keras.Input(shape=input_shape, name="mri_input")
    x = inputs
    for i, filters in enumerate((32, 64, 128), start=1):
        x = layers.Conv2D(filters, 3, padding="same", name=f"block{i}_conv")(x)
        x = layers.BatchNormalization(name=f"block{i}_bn")(x)
        x = layers.ReLU(name=f"block{i}_relu")(x)
        x = layers.MaxPooling2D(2, name=f"block{i}_pool")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="dense_1")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name="brain_tumor_cnn")


def get_data_loaders(data_dir, batch_size=32, use_augmentation=True, seed=42):
    img_size = (224, 224)
    autotune = tf.data.AUTOTUNE

    train_csv = data_dir / "train_split.csv"
    val_csv = data_dir / "val_split.csv"
    test_csv = data_dir / "test_split.csv"

    assert train_csv.exists() and val_csv.exists() and test_csv.exists(), (
        f"Split files not found in {data_dir}. Run notebook 01/03 first."
    )

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    print(f"Dataset splits -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    def load_image(path, label):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, img_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    augmentation_layer = keras.Sequential([
        layers.RandomRotation(0.03, fill_mode="constant", fill_value=0.0),
        layers.RandomZoom(0.05, fill_mode="constant", fill_value=0.0),
    ], name="mri_augmentation") if use_augmentation else None

    def make_dataset(frame, training=False):
        paths = frame["filepath"].astype(str).to_numpy()
        labels = frame["label"].astype("int32").to_numpy()
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training:
            ds = ds.shuffle(len(frame), seed=seed, reshuffle_each_iteration=True)
        ds = ds.map(load_image, num_parallel_calls=autotune)
        if training and augmentation_layer is not None:
            ds = ds.map(lambda x, y: (augmentation_layer(x, training=True), y), num_parallel_calls=autotune)
        return ds.batch(batch_size).prefetch(autotune)

    train_ds = make_dataset(train_df, training=True)
    val_ds = make_dataset(val_df, training=False)
    test_ds = make_dataset(test_df, training=False)

    return train_ds, val_ds, test_ds


def main():
    args = parse_args()
    keras.utils.set_random_seed(args.seed)

    root_dir = Path(__file__).resolve().parent
    data_dir = root_dir / "data"
    models_dir = root_dir / "models"
    reports_dir = root_dir / "reports"
    figures_dir = reports_dir / "figures"

    models_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    with open(models_dir / "class_names.json", "r", encoding="utf-8") as f:
        class_names = json.load(f)
    print("Classes:", class_names)

    use_aug = not args.no_augmentation
    print(f"Data augmentation: {'Enabled (safe rotation + zoom)' if use_aug else 'Disabled'}")

    train_ds, val_ds, test_ds = get_data_loaders(
        data_dir, batch_size=args.batch_size, use_augmentation=use_aug, seed=args.seed
    )

    print("Building model architecture...")
    model = build_cnn(input_shape=(224, 224, 3), num_classes=len(class_names), dropout_rate=args.dropout)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    best_model_path = models_dir / "best_brain_tumor_cnn.keras"
    last_model_path = models_dir / "last_brain_tumor_cnn.keras"
    history_csv_path = reports_dir / "training_history.csv"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
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

    print(f"\nStarting model training for up to {args.epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(str(last_model_path))
    print(f"Saved final model to: {last_model_path}")

    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(history_csv_path, index=False)
    print(f"Saved training history to: {history_csv_path}")

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs_range = range(1, len(history_df) + 1)
    axes[0].plot(epochs_range, history_df["accuracy"], label="Training", marker="o")
    axes[0].plot(epochs_range, history_df["val_accuracy"], label="Validation", marker="s")
    axes[0].set_title("Model Accuracy across Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].plot(epochs_range, history_df["loss"], label="Training", marker="o")
    axes[1].plot(epochs_range, history_df["val_loss"], label="Validation", marker="s")
    axes[1].set_title("Model Loss across Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    training_curves_path = figures_dir / "training_curves.png"
    plt.savefig(training_curves_path, dpi=150)
    plt.close()
    print(f"Saved training curves to: {training_curves_path}")

    # Final evaluation on the unseen test split using the best model
    print("\n--- Evaluating Best Model on Held-out Test Set ---")
    best_model = keras.models.load_model(str(best_model_path))
    test_loss, test_acc = best_model.evaluate(test_ds, verbose=1)

    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    predictions = best_model.predict(test_ds, verbose=1)
    y_pred = np.argmax(predictions, axis=1)

    print(f"\nFinal Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")

    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    report_text = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print("\nClassification Report:\n", report_text)

    # Save metrics JSON and text report
    metrics_summary = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "classification_report": report_dict,
        "epochs_trained": len(history_df),
        "best_epoch": int(np.argmin(history_df["val_loss"]) + 1),
        "best_val_loss": float(np.min(history_df["val_loss"])),
        "best_val_accuracy": float(history_df["val_accuracy"].iloc[np.argmin(history_df["val_loss"])]),
    }
    with open(reports_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    with open(figures_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    # Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    ax.set_title(f"Test Confusion Matrix (Accuracy: {test_acc * 100:.1f}%)")
    plt.tight_layout()
    cm_path = figures_dir / "test_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to: {cm_path}")
    print("\nTraining and evaluation successfully completed!")


if __name__ == "__main__":
    main()
