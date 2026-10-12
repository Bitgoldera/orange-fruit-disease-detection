from pathlib import Path
import csv
import yaml
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path.cwd()
OUT = ROOT / "reports/greening_bg_sensitivity.jpg"
CSV = ROOT / "reports/greening_bg_sensitivity.csv"

images = [
    ("greening_00002.jpg",
     ROOT / "data/processed/val/Greening/greening_00002.jpg",
     (0.02, 0.04, 0.96, 1.00)),
    ("greening_00005.jpeg",
     ROOT / "data/processed/val/Greening/greening_00005.jpeg",
     (0.00, 0.06, 0.76, 0.99)),
]

models = {
    "Original": ROOT / "models/orange_disease_efficientnet_b0_final.keras",
    "Melanose-only": ROOT / "models/melanose100/orange_disease_efficientnet_b0_final.keras",
}

with (ROOT / "config.yaml").open(encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

classes = cfg["class_names"]
img_h, img_w = cfg["image_size"]
greening_idx = classes.index("Greening")
melanose_idx = classes.index("Melanose")

for _, path, _ in images:
    if not path.exists():
        raise FileNotFoundError(path)
for path in models.values():
    if not path.exists():
        raise FileNotFoundError(path)

def make_blurred_background(path, ellipse):
    with Image.open(path) as opened:
        original = opened.convert("RGB")

    w, h = original.size
    blurred = original.filter(
        ImageFilter.GaussianBlur(radius=max(8, min(w, h) * 0.035))
    )

    # Approximate fruit mask; feather the border to reduce a hard seam.
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    x1, y1, x2, y2 = ellipse
    draw.ellipse((x1*w, y1*h, x2*w, y2*h), fill=255)
    mask = mask.filter(
        ImageFilter.GaussianBlur(radius=max(3, min(w, h) * 0.025))
    )

    return original, Image.composite(original, blurred, mask)

rows = []
diagnostic_images = {}

for name, path, ellipse in images:
    original, blurred_bg = make_blurred_background(path, ellipse)
    diagnostic_images[name] = (original, blurred_bg)

    for model_name, model_path in models.items():
        model = keras.models.load_model(model_path)

        for variant_name, picture in [
            ("original", original),
            ("background_blurred", blurred_bg),
        ]:
            resized = picture.resize(
                (img_w, img_h), Image.Resampling.BILINEAR
            )
            array = np.asarray(resized, dtype=np.float32)[None, ...]
            probs = model.predict(array, verbose=0)[0]
            pred_idx = int(np.argmax(probs))

            row = {
                "image": name,
                "model": model_name,
                "variant": variant_name,
                "prediction": classes[pred_idx],
                "top_confidence": float(probs[pred_idx]),
                "greening_score": float(probs[greening_idx]),
                "melanose_score": float(probs[melanose_idx]),
            }
            rows.append(row)

            print(
                f"{name:22s} | {model_name:13s} | {variant_name:18s} "
                f"| pred={classes[pred_idx]:10s} "
                f"| Greening={probs[greening_idx]:.3f} "
                f"| Melanose={probs[melanose_idx]:.3f}"
            )

        del model
        keras.backend.clear_session()

with CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

# Save the input variants so the approximate masks can be reviewed.
tw, th = 340, 270
sheet = Image.new("RGB", (tw * 2, th * 2), "white")
draw = ImageDraw.Draw(sheet)

for r, (name, (original, blurred)) in enumerate(diagnostic_images.items()):
    for c, (label, picture) in enumerate([
        ("Original", original),
        ("Background blurred (approximate fruit mask)", blurred),
    ]):
        x, y = c * tw, r * th
        picture.thumbnail((tw - 20, th - 45))
        sheet.paste(picture, (x + 10, y + 25))
        draw.text((x + 10, y + 5), f"{name} — {label}", fill="black")

OUT.parent.mkdir(parents=True, exist_ok=True)
sheet.save(OUT, quality=92)

print("\nDiagnostic image:", OUT)
print("Scores CSV:", CSV)
print("No models, training data, or dataset splits were changed.")
print("Note: fruit masks are approximate; interpret cautiously.")
