from pathlib import Path
from collections import defaultdict

from PIL import Image
import hashlib

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "orange_5_class"
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
    hashes = defaultdict(list)

    for class_dir in DATASET_DIR.iterdir():
        if not class_dir.is_dir():
            continue

        for image_path in class_dir.rglob("*"):
            if (
                image_path.is_file()
                and image_path.suffix.lower() in IMAGE_EXTENSIONS
            ):
                file_hash = sha256_file(image_path)
                hashes[file_hash].append(image_path)

    duplicate_groups = {
        h: files for h, files in hashes.items()
        if len(files) > 1
    }

    same_class_groups = 0
    cross_class_groups = 0
    duplicate_files = 0

    print("\n===================================")
    print("DUPLICATE ANALYSIS")
    print("===================================")

    for file_hash, files in duplicate_groups.items():

        classes = {
            path.parent.name
            for path in files
        }

        duplicate_files += len(files) - 1

        if len(classes) == 1:
            same_class_groups += 1
        else:
            cross_class_groups += 1

            print("\n⚠ CROSS-CLASS DUPLICATE")
            for path in files:
                print(f"  {path.relative_to(DATASET_DIR)}")

    print("\n-----------------------------------")
    print(f"Duplicate groups       : {len(duplicate_groups)}")
    print(f"Extra duplicate files  : {duplicate_files}")
    print(f"Same-class groups      : {same_class_groups}")
    print(f"Cross-class groups     : {cross_class_groups}")
    print("-----------------------------------")

    if cross_class_groups == 0:
        print("\nGood: no exact duplicate has conflicting labels.")
    else:
        print(
            "\nWARNING: Some identical images have different labels."
            "\nThese need to be resolved before training."
        )


if __name__ == "__main__":
    main()
