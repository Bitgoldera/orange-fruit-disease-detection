import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from tensorflow import keras
from tensorflow.keras import layers

# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed_pilot500"
MODEL_DIR = ROOT / "models" / "pilot500"
REPORT_DIR = ROOT / "reports" / "pilot500"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

with open(ROOT / "config.yaml", encoding="utf-8") as f:
    import yaml
    cfg = yaml.safe_load(f)

CLASS_NAMES = cfg["class_names"]
NUM_CLASSES = cfg["num_classes"]
IMG_SIZE = tuple(cfg["image_size"])

# Use a smaller batch for safer fine-tuning on the RTX 4050.
BATCH_SIZE = min(cfg["batch_size"], 16)
EPOCHS = cfg["fine_tune_epochs"]
LEARNING_RATE = cfg["fine_tune_learning_rate"]

tf.keras.utils.set_random_seed(cfg["seed"])


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

def load_dataset(split, shuffle=False):
    path = DATA_DIR / split

    if not path.exists():
        raise FileNotFoundError(f"Dataset folder not found: {path}")

    return keras.utils.image_dataset_from_directory(
        path,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        class_names=CLASS_NAMES,
        shuffle=shuffle,
        seed=cfg["seed"],
    ).prefetch(tf.data.AUTOTUNE)


print("Loading training and validation datasets...")

train_ds = load_dataset("train", shuffle=True)
val_ds = load_dataset("val")
test_ds = load_dataset("test")

print("\nClasses:", CLASS_NAMES)
print("Batch size:", BATCH_SIZE)
print("Fine-tuning epochs:", EPOCHS)


# --------------------------------------------------
# Load the existing baseline checkpoint
# --------------------------------------------------

best_baseline = MODEL_DIR / "efficientnet_b0_best.keras"
stage1_model = MODEL_DIR / "efficientnet_b0_stage1.keras"

if best_baseline.exists():
    checkpoint = best_baseline
elif stage1_model.exists():
    checkpoint = stage1_model
else:
    raise FileNotFoundError(
        "No saved baseline found. Expected one of:\n"
        f"{best_baseline}\n{stage1_model}"
    )

print(f"\nLoading baseline: {checkpoint}")

model = keras.models.load_model(checkpoint)


# --------------------------------------------------
# Unfreeze the deeper EfficientNet layers
# --------------------------------------------------

backbones = [
    layer for layer in model.layers
    if isinstance(layer, keras.Model)
    and "efficientnet" in layer.name.lower()
]

if not backbones:
    raise RuntimeError(
        "Could not find the EfficientNet backbone in the saved model. "
        f"Model layers: {[layer.name for layer in model.layers]}"
    )

backbone = backbones[0]
backbone.trainable = True

# Fine-tune approximately the final 20% of backbone layers.
fine_tune_from = int(len(backbone.layers) * 0.80)

for index, layer in enumerate(backbone.layers):
    layer.trainable = index >= fine_tune_from

    # Keep Batch Normalization stable during fine-tuning.
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

print("Backbone:", backbone.name)
print("Total backbone layers:", len(backbone.layers))
print("Fine-tuning from layer:", fine_tune_from)


# --------------------------------------------------
# Calculate class weights from actual training counts
# --------------------------------------------------

class_counts = {}

for class_name in CLASS_NAMES:
    class_dir = DATA_DIR / "train" / class_name

    image_count = sum(
        1 for p in class_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
        }
    )

    if image_count == 0:
        raise ValueError(f"No training images found for {class_name}")

    class_counts[class_name] = image_count

total_images = sum(class_counts.values())

class_weights = {
    index: total_images / (NUM_CLASSES * class_counts[name])
    for index, name in enumerate(CLASS_NAMES)
}

print("\nTraining counts and class weights:")

for index, name in enumerate(CLASS_NAMES):
    print(
        f"{name:12s}: {class_counts[name]:3d} images, "
        f"weight={class_weights[index]:.3f}"
    )


# --------------------------------------------------
# Compile for fine-tuning
# --------------------------------------------------

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="accuracy")
    ],
)


# --------------------------------------------------
# Train and save the best fine-tuned checkpoint
# --------------------------------------------------

fine_tuned_path = MODEL_DIR / "efficientnet_b0_best_finetuned.keras"

callbacks = [
    keras.callbacks.ModelCheckpoint(
        str(fine_tuned_path),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),
    keras.callbacks.CSVLogger(
        str(REPORT_DIR / "finetune_history.csv")
    ),
]

print("\nStarting fine-tuning on the available GPU...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
)


# --------------------------------------------------
# Restore best fine-tuned model and save final model
# --------------------------------------------------

if fine_tuned_path.exists():
    model = keras.models.load_model(fine_tuned_path)

final_model_path = MODEL_DIR / "orange_disease_efficientnet_b0_final.keras"
model.save(final_model_path)

with open(MODEL_DIR / "class_names.json", "w", encoding="utf-8") as f:
    json.dump(CLASS_NAMES, f, indent=2)

print(f"\nFinal model saved to: {final_model_path}")


# --------------------------------------------------
# Plot fine-tuning curves
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["accuracy"], label="Training")
axes[0].plot(history.history["val_accuracy"], label="Validation")
axes[0].set_title("Fine-tuning Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

axes[1].plot(history.history["loss"], label="Training")
axes[1].plot(history.history["val_loss"], label="Validation")
axes[1].set_title("Fine-tuning Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.savefig(REPORT_DIR / "finetune_curves.png", dpi=150)
plt.close()


# --------------------------------------------------
# Final test evaluation
# --------------------------------------------------

print("\nEvaluating on the untouched test set...")

test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

y_true = np.concatenate([
    labels.numpy() for _, labels in test_ds
])

probabilities = model.predict(test_ds, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

report_text = classification_report(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES)),
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0,
)

print("\nClassification report:")
print(report_text)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES)),
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES,
)

display.plot(xticks_rotation=45, values_format="d")
plt.title("Orange Disease Test Confusion Matrix")
plt.tight_layout()
plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=150)
plt.close()

with open(REPORT_DIR / "test_metrics.txt", "w", encoding="utf-8") as f:
    f.write(f"Test loss: {test_loss:.6f}\n")
    f.write(f"Test accuracy: {test_accuracy:.6f}\n\n")
    f.write(report_text)

pd.DataFrame(
    classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
).transpose().to_csv(
    REPORT_DIR / "classification_report.csv"
)

print("\n========================================")
print(f"Test loss:     {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
print("========================================")
print(f"Reports saved under: {REPORT_DIR}")
print("Fine-tuning complete.")
