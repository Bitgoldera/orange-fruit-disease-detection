from pathlib import Path
import csv
import gc
import json

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageDraw
import yaml
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path.cwd()
TEST_DIR = ROOT / "data/processed/val"

MODELS = {
    "existing": ROOT / "models/orange_disease_efficientnet_b0_final.keras",
    "pilot500": ROOT / "models/pilot500/orange_disease_efficientnet_b0_final.keras",
}

OUT = ROOT / "reports/pilot500"
OUT.mkdir(parents=True, exist_ok=True)

with (ROOT / "config.yaml").open(encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

class_names = cfg["class_names"]
img_size = tuple(cfg["image_size"])
exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

for name, path in MODELS.items():
    if not path.is_file():
        raise FileNotFoundError(f"Model not found ({name}): {path}")

# Enable gradual GPU allocation where supported.
for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

# Collect test images in a deterministic order.
paths = []
y_true = []

for label in class_names:
    class_dir = TEST_DIR / label
    if not class_dir.is_dir():
        raise FileNotFoundError(f"Missing test class folder: {class_dir}")

    for path in sorted(class_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in exts:
            paths.append(path)
            y_true.append(class_names.index(label))

if len(paths) != 91:
    raise RuntimeError(
        f"Expected 91 validation images, found {len(paths)}. Stopping for safety."
    )

# Use the same preprocessing for both models.
images = []
for path in paths:
    with Image.open(path) as im:
        im = im.convert("RGB").resize(
            (img_size[1], img_size[0]), Image.Resampling.BILINEAR
        )
        images.append(np.asarray(im, dtype=np.float32))

images = np.stack(images)
y_true = np.asarray(y_true, dtype=np.int64)


def predict_model(path):
    print(f"\nLoading {path.name}")
    model = keras.models.load_model(path)
    outputs = []

    for start in range(0, len(images), 16):
        batch = images[start:start + 16]
        outputs.append(model.predict(batch, verbose=0))

    probs = np.concatenate(outputs, axis=0)
    del model
    keras.backend.clear_session()
    gc.collect()
    return probs


probs_existing = predict_model(MODELS["existing"])
probs_pilot = predict_model(MODELS["pilot500"])

pred_existing = probs_existing.argmax(axis=1)
pred_pilot = probs_pilot.argmax(axis=1)

records = []
status_counts = {
    "both_correct": 0,
    "pilot_fixed": 0,
    "pilot_regression": 0,
    "both_wrong": 0,
}

for i, path in enumerate(paths):
    truth = y_true[i]
    old_pred = pred_existing[i]
    new_pred = pred_pilot[i]

    old_correct = old_pred == truth
    new_correct = new_pred == truth

    if old_correct and new_correct:
        status = "both_correct"
    elif not old_correct and new_correct:
        status = "pilot_fixed"
    elif old_correct and not new_correct:
        status = "pilot_regression"
    else:
        status = "both_wrong"

    status_counts[status] += 1

    records.append({
        "file": path.name,
        "true_label": class_names[truth],
        "existing_prediction": class_names[old_pred],
        "existing_confidence": round(float(probs_existing[i, old_pred]), 6),
        "pilot_prediction": class_names[new_pred],
        "pilot_confidence": round(float(probs_pilot[i, new_pred]), 6),
        "status": status,
        "path": str(path),
    })

csv_path = OUT / "existing_vs_pilot500_val_predictions.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
    writer.writeheader()
    writer.writerows(records)

print("\n=== COMPARISON ON THE SAME 91 VALIDATION IMAGES ===")
print("Existing model accuracy:", f"{np.mean(pred_existing == y_true):.2%}")
print("Pilot model accuracy:   ", f"{np.mean(pred_pilot == y_true):.2%}")

print("\nPrediction changes:")
for name, count in status_counts.items():
    print(f"  {name:20s}: {count}")

for name, predictions in [
    ("EXISTING MODEL", pred_existing),
    ("PILOT MODEL", pred_pilot),
]:
    print(f"\n=== {name}: CLASSIFICATION REPORT ===")
    print(classification_report(
        y_true,
        predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    ))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(
        y_true, predictions, labels=list(range(len(class_names)))
    ))

# Create a visual sheet of every test image where either model was wrong.
mistakes = [r for r in records if r["status"] != "both_correct"]
cell_w, cell_h = 440, 155
cols = 2
rows = max(1, (len(mistakes) + cols - 1) // cols)
sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
draw = ImageDraw.Draw(sheet)

for i, record in enumerate(mistakes):
    x = (i % cols) * cell_w
    y = (i // cols) * cell_h

    with Image.open(record["path"]) as im:
        thumb = im.convert("RGB")
        thumb.thumbnail((125, 125))

    sheet.paste(thumb, (x + 5, y + 5))
    old = record["existing_prediction"]
    new = record["pilot_prediction"]

    lines = [
        f"True: {record['true_label']}",
        f"Existing: {old} ({record['existing_confidence']:.1%})",
        f"Pilot: {new} ({record['pilot_confidence']:.1%})",
        f"Status: {record['status']}",
        record["file"][:43],
    ]

    for j, line in enumerate(lines):
        draw.text((x + 140, y + 8 + j * 24), line, fill="black")

sheet_path = OUT / "existing_vs_pilot500_val_mistakes.jpg"
sheet.save(sheet_path, quality=92)

summary_path = OUT / "existing_vs_pilot500_val_summary.json"
summary_path.write_text(json.dumps({
    "test_image_count": len(paths),
    "existing_accuracy": float(np.mean(pred_existing == y_true)),
    "pilot_accuracy": float(np.mean(pred_pilot == y_true)),
    "prediction_status_counts": status_counts,
}, indent=2), encoding="utf-8")

print("\nFiles saved:")
print(csv_path)
print(sheet_path)
print(summary_path)
print("\nNo training, models, or dataset splits were modified.")
