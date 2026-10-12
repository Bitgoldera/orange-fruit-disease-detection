from pathlib import Path
import shutil
import hashlib

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = PROJECT_ROOT / "data" / "raw" / "dataset_2"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "orange_5_class"
    / "Melanose"
)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}


def file_hash(path: Path) -> str:
    """Return SHA-256 hash of an image."""
    sha = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)

    return sha.hexdigest()


def main():
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Dataset 2 not found: {SOURCE_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_files = []

    # Collect images from train/valid/test
    for split in ["train", "valid", "test"]:
        image_dir = SOURCE_DIR / split / "images"

        if not image_dir.exists():
            print(f"Warning: {image_dir} not found")
            continue

        for image_path in image_dir.iterdir():
            if (
                image_path.is_file()
                and image_path.suffix.lower() in IMAGE_EXTENSIONS
            ):
                image_files.append(image_path)

    print(f"Images found: {len(image_files)}")

    seen_hashes = set()
    copied = 0
    duplicates = 0

    for image_path in image_files:
        image_hash = file_hash(image_path)

        if image_hash in seen_hashes:
            duplicates += 1
            continue

        seen_hashes.add(image_hash)

        new_name = f"melanose_{copied + 1:05d}{image_path.suffix.lower()}"
        destination = OUTPUT_DIR / new_name

        shutil.copy2(image_path, destination)
        copied += 1

    print("\nMelanose merge complete.")
    print("--------------------------------")
    print(f"Images found      : {len(image_files)}")
    print(f"Images copied     : {copied}")
    print(f"Exact duplicates  : {duplicates}")
    print(f"Output directory  : {OUTPUT_DIR}")
    print("--------------------------------")


if __name__ == "__main__":
    main()
