from pathlib import Path
import shutil
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = PROJECT_ROOT / "data" / "raw" / "dataset_1"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "orange_5_class"

# Final classes for our project
CLASS_MAP = {
    "healthy": "Healthy",
    "fresh": "Healthy",
    "canker": "Canker",
    "black spot": "Black_Spot",
    "black_spot": "Black_Spot",
    "blackspot": "Black_Spot",
    "greening": "Greening",
    "hlb": "Greening",
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def detect_class(path: Path):
    """
    Check folder names from deepest to highest level
    and map them to our final class names.
    """
    for folder in reversed(path.parent.parts):
        folder_name = normalize(folder)

        for key, target_class in CLASS_MAP.items():
            if key in folder_name:
                return target_class

    return None


def main():
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{SOURCE_DIR}\n"
            "Run extract_dataset.py first."
        )

    # Create final folders
    for class_name in [
        "Healthy",
        "Canker",
        "Black_Spot",
        "Greening",
        "Melanose",
    ]:
        (OUTPUT_DIR / class_name).mkdir(parents=True, exist_ok=True)

    counts = {
        "Healthy": 0,
        "Canker": 0,
        "Black_Spot": 0,
        "Greening": 0,
        "Melanose": 0,
    }

    skipped = 0

    for image_path in SOURCE_DIR.rglob("*"):
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        target_class = detect_class(image_path)

        # Skip Scab and anything we don't recognize
        if target_class is None:
            skipped += 1
            continue

        # Unique filename to avoid overwriting
        new_name = (
            f"{target_class.lower()}_"
            f"{counts[target_class] + 1:05d}"
            f"{image_path.suffix.lower()}"
        )

        destination = OUTPUT_DIR / target_class / new_name
        shutil.copy2(image_path, destination)

        counts[target_class] += 1

    print("\nDataset preparation complete.")
    print("--------------------------------")
    for class_name, count in counts.items():
        print(f"{class_name:12s}: {count}")
    print("--------------------------------")
    print(f"Skipped files/images: {skipped}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
