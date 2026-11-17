"""Fine-tune the existing 6-convolution-block brain MRI CNN checkpoint.

Run from the project root:
    python continue_training.py --epochs 12 --batch-size 16 --lr 0.00001

This saves a NEW checkpoint and does not overwrite the current model.
Requires train_cnn_improved.py beside this file.
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight

from train_cnn_improved import (
    load_class_names,
    load_split_csv,
    make_dataset,
    save_evaluation,
)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune existing brain MRI CNN")
    parser.add_argument("--model", default="models/experimental_best_brain_tumor_cnn.keras")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    keras.utils.set_random_seed(args.seed)
    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    models_dir = root / "models"
    reports_dir = root / "reports"
    figures_dir = reports_dir / "figures"
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    names = load_class_names(models_dir)
    train_df = load_split_csv(data_dir / "train_split.csv", root, data_dir, names)
    val_df = load_split_csv(data_dir / "val_split.csv", root, data_dir, names)
    test_df = load_split_csv(data_dir / "test_split.csv", root, data_dir, names)

    train_ds = make_dataset(train_df, args.batch_size, training=True,
                            use_augmentation=True, seed=args.seed)
    val_ds = make_dataset(val_df, args.batch_size, training=False,
                          use_augmentation=False, seed=args.seed)
    test_ds = make_dataset(test_df, args.batch_size, training=False,
                           use_augmentation=False, seed=args.seed)

    source_model = Path(args.model)
    if not source_model.is_absolute():
        source_model = root / source_model
    if not source_model.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {source_model}")

    model = keras.models.load_model(str(source_model), compile=False)
    if tuple(model.input_shape[1:]) != (224, 224, 3):
        raise ValueError(f"Unexpected input shape: {model.input_shape}")
    if int(model.output_shape[-1]) != len(names):
        raise ValueError("Model output count does not match models/class_names.json")

    y = train_df["label"].to_numpy(dtype=np.int32)
    weights = compute_class_weight(class_weight="balanced",
                                   classes=np.arange(len(names)), y=y)
    class_weights = {int(i): float(w) for i, w in enumerate(weights)}
    print("Continuing from:", source_model)
    print("Class order:", list(enumerate(names)))
    print("Class weights:", class_weights)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    best_path = models_dir / "experimental_finetuned_best_brain_tumor_cnn.keras"
    last_path = models_dir / "experimental_finetuned_last_brain_tumor_cnn.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(best_path), monitor="val_loss",
                                        mode="min", save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", mode="min",
                                      patience=args.patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min",
                                          factor=0.5, patience=2,
                                          min_lr=1e-7, verbose=1),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )
    model.save(last_path)
    print("Saved last fine-tuned checkpoint:", last_path)
    print("Best checkpoint:", best_path)
    print("Epochs completed:", len(history.history.get("loss", [])))

    best_model = keras.models.load_model(str(best_path), compile=False)
    best_model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    save_evaluation(best_model, test_ds, test_df, names,
                    reports_dir, figures_dir, "finetuned")
    print("Fine-tuning and held-out test evaluation completed.")


if __name__ == "__main__":
    main()
