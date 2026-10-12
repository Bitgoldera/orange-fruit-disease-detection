import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "orange_disease_efficientnet_b0_final.keras"
CLASS_PATH = ROOT / "models" / "class_names.json"
IMAGE_SIZE = (224, 224)


def main():
    parser = argparse.ArgumentParser(
        description="Predict orange fruit disease from an image."
    )
    parser.add_argument("image", help="Path to the orange image")
    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    with CLASS_PATH.open(encoding="utf-8") as f:
        classes = json.load(f)

    model = tf.keras.models.load_model(MODEL_PATH)

    with Image.open(image_path) as img:
        image = img.convert("RGB").resize(IMAGE_SIZE)

    array = np.asarray(image, dtype=np.float32)[None, ...]

    scores = model.predict(array, verbose=0)[0]
    ranking = np.argsort(scores)[::-1]

    print("\n=== ORANGE DISEASE PREDICTION ===")
    print(f"Image: {image_path.name}")
    print(f"Predicted class: {classes[ranking[0]]}")
    print(f"Model score: {scores[ranking[0]] * 100:.2f}%")

    print("\nClass scores:")
    for index in ranking:
        print(f"{classes[index]:12s}: {scores[index] * 100:.2f}%")


if __name__ == "__main__":
    main()
