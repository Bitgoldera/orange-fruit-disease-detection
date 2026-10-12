from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from collections import Counter
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "data/raw/orange_gate_dataset.zip"
SOURCE_SPLITS = ROOT / "data/splits"
OUTPUT_DIR = ROOT / "data/orange_gate"
REPORT_DIR = ROOT / "reports"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}

SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}
ZIP_SPLITS = {"train": "train", "val": "valid", "test": "test"}


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def is_image(name):
    return PurePosixPath(name).suffix.lower() in IMAGE_EXTENSIONS


def read_class_ids(text, label_name):
    ids = set()

    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue

        try:
            class_id = int(fields[0])
        except ValueError as exc:
            raise ValueError(
                f"Invalid YOLO class ID in {label_name}: {line}"
            ) from exc

        if class_id not in range(6):
            raise ValueError(
                f"Unexpected class ID {class_id} in {label_name}"
            )

        ids.add(class_id)

    return ids


def main():
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Dataset ZIP not found: {ZIP_PATH}")

    records = []
    excluded_mixed = 0

    # Existing orange images from our original five-class dataset.
    for split in SPLIT_ORDER:
        split_dir = SOURCE_SPLITS / split

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            for path in sorted(class_dir.iterdir()):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                records.append({
                    "split": split,
                    "label": "Orange",
                    "kind": "file",
                    "source": str(path),
                    "name": path.name,
                    "suffix": path.suffix.lower(),
                    "sha": sha256_bytes(path.read_bytes()),
                })

    with ZipFile(ZIP_PATH) as archive:
        names = set(archive.namelist())

        # Use YOLO labels to identify fruit images.
        for split, zip_split in ZIP_SPLITS.items():
            prefix = f"{zip_split}/images/"

            image_names = sorted(
                name for name in names
                if name.startswith(prefix) and is_image(name)
            )

            for member in image_names:
                image_path = PurePosixPath(member)
                label_member = (
                    f"{zip_split}/labels/{image_path.stem}.txt"
                )

                if label_member not in names:
                    raise ValueError(f"Missing label file: {label_member}")

                label_text = archive.read(label_member).decode(
                    "utf-8", errors="strict"
                ).strip()

                class_ids = read_class_ids(label_text, label_member)

                # Exclude empty/unlabelled examples.
                if not class_ids:
                    continue

                # Class 4 is Orange in this dataset.
                if class_ids == {4}:
                    target_label = "Orange"
                elif 4 not in class_ids:
                    target_label = "Not_Orange"
                else:
                    excluded_mixed += 1
                    continue

                image_bytes = archive.read(member)

                records.append({
                    "split": split,
                    "label": target_label,
                    "kind": "zip",
                    "source": member,
                    "name": image_path.name,
                    "suffix": image_path.suffix.lower(),
                    "sha": sha256_bytes(image_bytes),
                })

        # Prefer train, then validation, then test if exact copies exist.
        records.sort(
            key=lambda r: (
                SPLIT_ORDER[r["split"]],
                r["label"],
                r["source"],
            )
        )

        seen = {}
        unique_records = []
        duplicates_skipped = 0

        for record in records:
            digest = record["sha"]

            if digest in seen:
                previous = seen[digest]

                if previous["label"] != record["label"]:
                    raise ValueError(
                        "The same image has conflicting binary labels:\n"
                        f"{previous['source']} -> {previous['label']}\n"
                        f"{record['source']} -> {record['label']}"
                    )

                duplicates_skipped += 1
                continue

            seen[digest] = record
            unique_records.append(record)

        # Rebuild only our generated binary dataset.
        if OUTPUT_DIR.exists():
            shutil.rmtree(OUTPUT_DIR)

        for split in SPLIT_ORDER:
            for label in ["Orange", "Not_Orange"]:
                (OUTPUT_DIR / split / label).mkdir(
                    parents=True,
                    exist_ok=True,
                )

        counts = Counter()
        copied = 0

        for record in unique_records:
            key = (record["split"], record["label"])
            counts[key] += 1

            filename = (
                f"{record['label'].lower()}_"
                f"{counts[key]:05d}{record['suffix']}"
            )

            destination = (
                OUTPUT_DIR / record["split"]
                / record["label"] / filename
            )

            if record["kind"] == "file":
                shutil.copy2(record["source"], destination)
            else:
                with archive.open(record["source"]) as src:
                    with destination.open("wb") as dst:
                        shutil.copyfileobj(src, dst)

            copied += 1

    summary = {
        "class_names": ["Not_Orange", "Orange"],
        "counts": {
            split: {
                label: counts[(split, label)]
                for label in ["Orange", "Not_Orange"]
            }
            for split in SPLIT_ORDER
        },
        "duplicates_skipped": duplicates_skipped,
        "mixed_images_excluded": excluded_mixed,
        "images_copied": copied,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORT_DIR / "orange_gate_dataset_summary.json"
    report_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nORANGE GATE DATASET BUILT")
    print("=" * 38)

    for split in SPLIT_ORDER:
        print(f"\n{split.upper()}")
        for label in ["Orange", "Not_Orange"]:
            print(f"  {label:12s}: {counts[(split, label)]}")

    print("\nImages copied:", copied)
    print("Exact duplicates skipped:", duplicates_skipped)
    print("Mixed images excluded:", excluded_mixed)
    print("Output:", OUTPUT_DIR)
    print("Report:", report_path)


if __name__ == "__main__":
    main()

