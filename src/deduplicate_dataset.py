from pathlib import Path
from collections import defaultdict
import hashlib
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "orange_5_class"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "orange_5_class_clean"
)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff"
}


def sha256_file(path: Path) -> str:
    sha = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)

    return sha.hexdigest()


def main():
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Source dataset not found: {SOURCE_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    hashes = set()

    total = 0
    copied = 0
    duplicates = 0

    for class_dir in sorted(SOURCE_DIR.iterdir()):

        if not class_dir.is_dir():
            continue

        output_class_dir = OUTPUT_DIR / class_dir.name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        for image_path in sorted(class_dir.rglob("*")):

            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            total += 1

            file_hash = sha256_file(image_path)

            if file_hash in hashes:
                duplicates += 1
                continue

            hashes.add(file_hash)

            destination = (
                output_class_dir / image_path.name
            )

            # Avoid filename collision
            counter = 1

            while destination.exists():
                destination = (
                    output_class_dir
                    / f"{image_path.stem}_{counter}"
                    f"{image_path.suffix}"
                )
                counter += 1

            shutil.copy2(image_path, destination)

            copied += 1

    print("\n====================================")
    print("DEDUPLICATION COMPLETE")
    print("====================================")
    print(f"Original images : {total}")
    print(f"Unique images   : {copied}")
    print(f"Duplicates      : {duplicates}")
    print("------------------------------------")
    print(f"Output: {OUTPUT_DIR}")

    print("\nClass counts:")

    for class_dir in sorted(OUTPUT_DIR.iterdir()):
        if class_dir.is_dir():
            count = len([
                p for p in class_dir.iterdir()
                if p.is_file()
            ])

            print(f"  {class_dir.name:12s}: {count}")


if __name__ == "__main__":
    main()
