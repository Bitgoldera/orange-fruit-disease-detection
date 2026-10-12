"""Baseline EfficientNet-B0 transfer-learning training script.

Expected processed layout:
processed/train/<class>/*.jpg
processed/val/<class>/*.jpg
"""
from __future__ import annotations

from pathlib import Path
import yaml
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> keras.Model:
    h, w = cfg["image_size"]
    num_classes = cfg["num_classes"]

    backbone = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(h, w, 3),
        pooling="avg",
    )
    backbone.trainable = False

    inputs = keras.Input(shape=(h, w, 3))
    x = inputs
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.08)(x)
    x = layers.RandomZoom(0.10)(x)
    x = backbone(x, training=False)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def main() -> None:
    cfg = load_config()
    train_dir = Path("data/processed/train")
    val_dir = Path("data/processed/val")
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError("Create train/ and val/ folders under data/processed first.")

    tf.keras.utils.set_random_seed(cfg["seed"])
    img_size = tuple(cfg["image_size"])

    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=cfg["batch_size"],
        label_mode="int",
        class_names=cfg["class_names"],
        shuffle=True,
        seed=cfg["seed"],
    )
    val_ds = keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=img_size,
        batch_size=cfg["batch_size"],
        label_mode="int",
        class_names=cfg["class_names"],
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    model = build_model(cfg)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg["initial_learning_rate"]),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2),
        keras.callbacks.ModelCheckpoint("models/efficientnet_b0_best.keras", monitor="val_loss", save_best_only=True),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=cfg["initial_epochs"], callbacks=callbacks)
    model.save("models/efficientnet_b0_stage1.keras")
    print("Saved baseline model to models/efficientnet_b0_stage1.keras")


if __name__ == "__main__":
    Path("models").mkdir(parents=True, exist_ok=True)
    main()
