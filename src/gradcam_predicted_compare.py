from pathlib import Path
import gc
import yaml
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageDraw
from matplotlib import colormaps

ROOT = Path.cwd()

MODEL_PATHS = {
    "Original model": ROOT / "models/orange_disease_efficientnet_b0_final.keras",
    "Melanose-only model": ROOT / "models/melanose100/orange_disease_efficientnet_b0_final.keras",
}

IMAGE_PATHS = [
    ROOT / "data/processed/val/Greening/greening_00002.jpg",
    ROOT / "data/processed/val/Greening/greening_00005.jpeg",
]

OUT = ROOT / "reports/greening_predicted_gradcam_compare.jpg"

with (ROOT / "config.yaml").open(encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

classes = cfg["class_names"]
img_h, img_w = cfg["image_size"]
greening_idx = classes.index("Greening")

for path in list(MODEL_PATHS.values()) + IMAGE_PATHS:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")

for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass


def load_rgb(path):
    with Image.open(path) as im:
        return im.convert("RGB").resize(
            (img_w, img_h), Image.Resampling.BILINEAR
        )


def gradcam_for_image(model, backbone, feature_model, before, after, path):
    base = load_rgb(path)
    array = np.asarray(base, dtype=np.float32)[None, ...]

    # Run the same inference-time augmentation layers as the saved model.
    x = tf.convert_to_tensor(array)
    for layer in before:
        x = layer(x, training=False)

    with tf.GradientTape() as tape:
        features = feature_model(x, training=False)
        pooled = tf.reduce_mean(features, axis=(1, 2))

        scores = pooled
        for layer in after:
            scores = layer(scores, training=False)

        # Explain the model's highest-scoring predicted class.
        target_score = tf.reduce_max(scores)

    gradients = tape.gradient(target_score, features)
    if gradients is None:
        raise RuntimeError(f"Gradients unavailable for {path.name}")

    weights = tf.reduce_mean(gradients, axis=(1, 2))
    heat = tf.reduce_sum(features * weights[:, None, None, :], axis=-1)[0]
    heat = tf.nn.relu(heat)

    maximum = float(tf.reduce_max(heat).numpy())
    if maximum > 1e-12:
        heat = heat / maximum

    heat = heat.numpy()
    heat_image = Image.fromarray(
        np.uint8(np.clip(heat, 0, 1) * 255)
    ).resize((img_w, img_h), Image.Resampling.BILINEAR)

    heat_array = np.asarray(heat_image, dtype=np.float32) / 255.0
    colour = colormaps["jet"](heat_array)[..., :3] * 255.0
    original = np.asarray(base, dtype=np.float32)

    overlay = np.uint8(np.clip(
        0.55 * original + 0.45 * colour, 0, 255
    ))

    probabilities = scores.numpy()[0]
    predicted_idx = int(np.argmax(probabilities))

    return (
        base,
        Image.fromarray(overlay),
        classes[predicted_idx],
        float(probabilities[predicted_idx]),
        float(probabilities[greening_idx]),
    )


results = {}

for model_name, model_path in MODEL_PATHS.items():
    print(f"\nLoading {model_name}: {model_path}")
    model = keras.models.load_model(model_path)

    backbone_matches = [
        (i, layer)
        for i, layer in enumerate(model.layers)
        if isinstance(layer, keras.Model)
        and "efficientnet" in layer.name.lower()
    ]
    if not backbone_matches:
        raise RuntimeError(
            f"Could not locate EfficientNet backbone in {model_name}"
        )

    backbone_idx, backbone = backbone_matches[0]

    try:
        conv_layer = backbone.get_layer("top_activation")
    except ValueError:
        try:
            conv_layer = backbone.get_layer("top_conv")
        except ValueError as exc:
            raise RuntimeError(
                f"Could not locate final convolution layer in {model_name}"
            ) from exc

    feature_model = keras.Model(
        inputs=backbone.inputs,
        outputs=conv_layer.output,
    )

    before = model.layers[1:backbone_idx]
    after = model.layers[backbone_idx + 1:]

    for path in IMAGE_PATHS:
        result = gradcam_for_image(
            model, backbone, feature_model, before, after, path
        )
        results[(model_name, path.name)] = result

        _, _, predicted, confidence, greening_score = result
        print(
            f"  {path.name}: predicted={predicted} "
            f"({confidence:.1%}), Greening score={greening_score:.1%}"
        )

    del feature_model, backbone, model
    keras.backend.clear_session()
    gc.collect()

# Create a 2-row, 3-column comparison image.
columns = ["Original image", "Original model Grad-CAM",
           "Melanose-only model Grad-CAM"]
tile_w, tile_h = 340, 315
sheet = Image.new("RGB", (3 * tile_w, 2 * tile_h), "white")
draw = ImageDraw.Draw(sheet)

for col, label in enumerate(columns):
    draw.text((col * tile_w + 10, 8), label, fill="black")

for row, path in enumerate(IMAGE_PATHS):
    y = row * tile_h
    original = load_rgb(path)
    sheet.paste(original.resize((300, 225)), (10, y + 32))

    draw.text(
        (10, y + 263),
        f"True label: Greening | {path.name}",
        fill="black",
    )

    for col, model_name in [(1, "Original model"),
                            (2, "Melanose-only model")]:
        _, overlay, predicted, confidence, score = results[
            (model_name, path.name)
        ]

        x = col * tile_w
        sheet.paste(overlay.resize((300, 225)), (x + 10, y + 32))
        draw.text(
            (x + 10, y + 263),
            f"Predicted: {predicted} ({confidence:.1%})",
            fill="black",
        )
        draw.text(
            (x + 10, y + 282),
            f"Greening score: {score:.1%}",
            fill="black",
        )

OUT.parent.mkdir(parents=True, exist_ok=True)
sheet.save(OUT, quality=93)

print("\nGrad-CAM comparison saved:", OUT)
print("No model weights or dataset files were changed.")
