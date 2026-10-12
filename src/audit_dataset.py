from pathlib import Path
from collections import Counter
import hashlib
import csv

from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "orange_5_class"
)

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

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
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_DIR}"
        )

    class_dirs = sorted(
        [p for p in DATASET_DIR.iterdir() if p.is_dir()]
    )

    class_counts = Counter()
    hash_to_files = {}
    corrupted = []
    image_records = []

    total_images = 0

    for class_dir in class_dirs:
        class_name = class_dir.name

        for image_path in class_dir.rglob("*"):
            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            total_images += 1
            class_counts[class_name] += 1

            # Check image integrity
            try:
                with Image.open(image_path) as img:
                    img.verify()

                # Open again to read properties
                with Image.open(image_path) as img:
                    width, height = img.size
                    mode = img.mode

            except (UnidentifiedImageError, OSError, ValueError) as e:
                corrupted.append(
                    {
                        "file": str(image_path),
                        "error": str(e)
                    }
                )
                continue

            # Duplicate detection
            file_hash = sha256_file(image_path)

            hash_to_files.setdefault(file_hash, []).append(
                str(image_path)
            )

            image_records.append({
                "class": class_name,
                "file": str(image_path.relative_to(DATASET_DIR)),
                "width": width,
                "height": height,
                "mode": mode,
                "sha256": file_hash
            })

    # Find duplicates
    duplicate_groups = {
        h: files
        for h, files in hash_to_files.items()
        if len(files) > 1
    }

    # Dimension statistics
    dimensions = Counter(
        (r["width"], r["height"])
        for r in image_records
    )

    # Print report
    print("\n======================================")
    print("ORANGE DATASET AUDIT")
    print("======================================")

    print(f"\nDataset location:")
    print(DATASET_DIR)

    print(f"\nTotal images: {total_images}")

    print("\nClass distribution:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name:12s}: {count}")

    print(f"\nCorrupted images: {len(corrupted)}")
    print(f"Duplicate groups: {len(duplicate_groups)}")

    print("\nMost common image dimensions:")
    for (w, h), count in dimensions.most_common(10):
        print(f"  {w} x {h}: {count}")

    # Save image report
    image_report = REPORT_DIR / "image_audit.csv"

    with image_report.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class",
                "file",
                "width",
                "height",
                "mode",
                "sha256"
            ]
        )

        writer.writeheader()
        writer.writerows(image_records)

    # Save corrupted report
    corrupted_report = REPORT_DIR / "corrupted_images.csv"

    with corrupted_report.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "error"]
        )

        writer.writeheader()
        writer.writerows(corrupted)

    # Save duplicate report
    duplicate_report = REPORT_DIR / "duplicates.txt"

    with duplicate_report.open(
        "w",
        encoding="utf-8"
    ) as f:
        for hash_value, files in duplicate_groups.items():
            f.write(f"SHA256: {hash_value}\n")

            for file in files:
                f.write(f"  {file}\n")

            f.write("\n")

    print("\nReports created:")
    print(f"  {image_report}")
    print(f"  {corrupted_report}")
    print(f"  {duplicate_report}")

    print("\nAudit complete.")


if __name__ == "__main__":
    main()
