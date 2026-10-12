from pathlib import Path
from collections import Counter
import json

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/orange_gate"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42
CLASS_NAMES = ["Not_Orange", "Orange"]
TARGET_PRECISION = 0.95

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Help avoid grabbing all GPU memory immediately.
for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

keras.utils.set_random_seed(SEED)

if not DATA_DIR.exists():
    raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")

for split in ["train", "val", "test"]:
    for name in CLASS_NAMES:
        folder = DATA_DIR / split / name
        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")

AUTOTUNE = tf.data.AUTOTUNE


def load_split(split, shuffle=False):
    return keras.utils.image_dataset_from_directory(
        DATA_DIR / split,
        class_names=CLASS_NAMES,
        labels="inferred",
        label_mode="binary",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    ).prefetch(AUTOTUNE)


train_ds = load_split("train", shuffle=True)
val_ds = load_split("val")
test_ds = load_split("test")

# Balanced class weights help compensate for the unequal class counts.
class_counts = {
    0: len(list((DATA_DIR / "train" / "Not_Orange").glob("*"))),
    1: len(list((DATA_DIR / "train" / "Orange").glob("*"))),
}
total = sum(class_counts.values())

class_weights = {
    class_id: total / (2.0 * count)
    for class_id, count in class_counts.items()
    if count > 0
}

print("\nClass mapping:", {0: "Not_Orange", 1: "Orange"})
print("Training counts:", class_counts)
print("Class weights:", class_weights)
print("GPU devices:", tf.config.list_physical_devices("GPU"))

# ---------------------------
# Stage 1: train the new head
# ---------------------------
augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.04),
        layers.RandomZoom(0.08),
    ],
    name="augmentation",
)

backbone = keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(*IMAGE_SIZE, 3),
)
backbone.trainable = False

inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
x = augmentation(inputs)
x = backbone(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.30)(x)
outputs = layers.Dense(1, activation="sigmoid", name="orange_probability")(x)

model = keras.Model(inputs, outputs, name="orange_gate")

def compile_model(net, learning_rate):
    net.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )


stage1_path = MODEL_DIR / "orange_gate_stage1_best.keras"
stage2_path = MODEL_DIR / "orange_gate_finetuned_best.keras"
final_path = MODEL_DIR / "orange_gate_final.keras"

def callbacks_for(checkpoint, csv_name):
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            mode="min",
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            mode="min",
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            str(REPORT_DIR / csv_name)
        ),
    ]


compile_model(model, 1e-3)

print("\n========== STAGE 1: TRAIN CLASSIFICATION HEAD ==========")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    class_weight=class_weights,
    callbacks=callbacks_for(stage1_path, "orange_gate_stage1_history.csv"),
)

# ---------------------------
# Stage 2: fine-tune backbone
# ---------------------------
print("\n========== STAGE 2: FINE-TUNE ==========")
model = keras.models.load_model(stage1_path)
backbone = model.get_layer("efficientnetb0")

backbone.trainable = True

# Freeze everything first, then unfreeze approximately the final quarter.
for layer in backbone.layers:
    layer.trainable = False

backbone.trainable = True
number_to_unfreeze = max(1, int(len(backbone.layers) * 0.25))

for layer in backbone.layers[-number_to_unfreeze:]:
    if not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True

print(
    f"Backbone layers: {len(backbone.layers)}; "
    f"last {number_to_unfreeze} layers selected for fine-tuning "
    "(BatchNormalization layers remain frozen)."
)

compile_model(model, 1e-5)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=12,
    class_weight=class_weights,
    callbacks=callbacks_for(stage2_path, "orange_gate_stage2_history.csv"),
)

# Pick the checkpoint with the lower validation loss.
stage1_model = keras.models.load_model(stage1_path)
stage1_val = stage1_model.evaluate(val_ds, verbose=0, return_dict=True)

stage2_model = keras.models.load_model(stage2_path)
stage2_val = stage2_model.evaluate(val_ds, verbose=0, return_dict=True)

if stage2_val["loss"] < stage1_val["loss"]:
    model = stage2_model
    selected_stage = "fine_tuned"
else:
    model = stage1_model
    selected_stage = "stage1"

model.save(final_path)

print("\nSelected checkpoint:", selected_stage)
print("Stage 1 validation loss:", round(float(stage1_val["loss"]), 5))
print("Fine-tuned validation loss:", round(float(stage2_val["loss"]), 5))
print("Final model saved to:", final_path)


# ---------------------------
# Select a threshold on VAL
# ---------------------------
def collect_labels(dataset):
    return np.concatenate(
        [labels.numpy().reshape(-1) for _, labels in dataset]
    ).astype(int)


def calculate_metrics(y_true, probabilities, threshold):
    y_pred = (probabilities >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "total": int(len(y_true)),
    }


val_labels = collect_labels(val_ds)
val_probabilities = model.predict(val_ds, verbose=0).reshape(-1)

candidates = sorted(
    set(float(p) for p in val_probabilities) | {0.0, 1.000001}
)

precision_candidates = []
f1_candidates = []

for threshold in candidates:
    metrics = calculate_metrics(
        val_labels, val_probabilities, threshold
    )

    # Don't choose a threshold that classifies every image as not-orange.
    if metrics["true_positive"] + metrics["false_positive"] > 0:
        f1_candidates.append(metrics)

        if metrics["precision"] >= TARGET_PRECISION:
            precision_candidates.append(metrics)

if precision_candidates:
    chosen = max(
        precision_candidates,
        key=lambda m: (m["recall"], m["precision"], -m["threshold"]),
    )
    threshold_method = f"maximum recall with validation precision >= {TARGET_PRECISION}"
elif f1_candidates:
    chosen = max(f1_candidates, key=lambda m: m["f1"])
    threshold_method = "best validation F1; target precision not reached"
else:
    chosen = {
        **calculate_metrics(val_labels, val_probabilities, 0.5),
    }
    threshold_method = "fallback threshold 0.5"

chosen_threshold = chosen["threshold"]

threshold_config = {
    "class_names": CLASS_NAMES,
    "orange_class_index": 1,
    "threshold": chosen_threshold,
    "selection_method": threshold_method,
    "validation_metrics_at_threshold": chosen,
}

threshold_path = MODEL_DIR / "orange_gate_threshold.json"
threshold_path.write_text(
    json.dumps(threshold_config, indent=2),
    encoding="utf-8",
)

print("\n========== VALIDATION THRESHOLD ==========")
print("Selection method:", threshold_method)
print("Chosen Orange threshold:", round(chosen_threshold, 5))
print("Validation metrics:", json.dumps(chosen, indent=2))


# ---------------------------
# Final evaluation on TEST
# ---------------------------
test_labels = collect_labels(test_ds)
test_probabilities = model.predict(test_ds, verbose=0).reshape(-1)

test_at_threshold = calculate_metrics(
    test_labels, test_probabilities, chosen_threshold
)
test_at_half = calculate_metrics(
    test_labels, test_probabilities, 0.5
)

test_loss_and_metrics = model.evaluate(
    test_ds, verbose=0, return_dict=True
)

report = {
    "selected_checkpoint": selected_stage,
    "test_metrics_at_validation_selected_threshold": test_at_threshold,
    "test_metrics_at_threshold_0_5": test_at_half,
    "test_loss_and_keras_metrics": {
        key: float(value)
        for key, value in test_loss_and_metrics.items()
    },
    "warning": (
        "Preliminary evaluation on a small dataset. "
        "Real-world performance on vegetables, non-fruit objects, "
        "and unfamiliar image sources has not yet been established."
    ),
}

report_path = REPORT_DIR / "orange_gate_test_metrics.json"
report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print("\n========== TEST RESULTS ==========")
print("Metrics at selected threshold:")
print(json.dumps(test_at_threshold, indent=2))
print("\nMetrics at threshold 0.5:")
print(json.dumps(test_at_half, indent=2))
print("\nKeras test metrics:")
print(json.dumps(report["test_loss_and_keras_metrics"], indent=2))
print("\nFinal binary model:", final_path)
print("Threshold configuration:", threshold_path)
print("Test report:", report_path)
