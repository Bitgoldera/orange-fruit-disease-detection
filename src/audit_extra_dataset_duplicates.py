from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from collections import Counter, defaultdict
import hashlib
import io

ROOT = Path(".")
EXTRA = ROOT / "data/raw/extra_datasets"
SPLITS = ROOT / "data/splits"

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}

LABELS = {
    "blackspot": "Black_Spot",
    "black-spot": "Black_Spot",
    "black spot": "Black_Spot",
    "canker": "Canker",
    "citrus-canker": "Canker",
    "grenning": "Greening",
    "greening": "Greening",
    "fresh": "Healthy",
    "healthy": "Healthy",
    "melanose": "Melanose",
    "scab": "Scab",
}


def label_from_parts(parts):
    for part in reversed(parts):
        key = part.strip().lower()
        if key in LABELS:
            return LABELS[key]
    return None


def digest(data):
    return hashlib.sha256(data).hexdigest()


records = []
counts = Counter()


def add_record(data, source, split, label, name):
    if label is None:
        return

    record = {
        "hash": digest(data),
        "source": source,
        "split": split,
        "label": label,
        "name": name,
    }
    records.append(record)
    counts[(source, split, label)] += 1


# Existing disease dataset.
for split in ["train", "val", "test"]:
    for path in sorted((SPLITS / split).glob("*/*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue

        add_record(
            path.read_bytes(),
            "existing",
            split,
            path.parent.name,
            str(path),
        )

# The two directly image-based archives.
archives = [
    ("archive1", EXTRA / "archive (1).zip"),
    ("archive2", EXTRA / "archive (2).zip"),
]

for source_name, archive_path in archives:
    print(f"Reading {archive_path.name}...")
    with ZipFile(archive_path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue

            path = PurePosixPath(info.filename)
            if path.suffix.lower() not in IMAGE_EXTS:
                continue

            parts = path.parts
            label = label_from_parts(parts[:-1])

            split = "unsplit"
            if parts and parts[0].lower() in {"train", "test", "valid", "val"}:
                split = {"valid": "val"}.get(parts[0].lower(), parts[0].lower())

            add_record(
                z.read(info),
                source_name,
                split,
                label,
                info.filename,
            )

# The Citrus archive is nested inside another ZIP.
citrus_outer = next(EXTRA.glob("A Citrus*.zip"))

print("Reading nested Citrus archive...")
with ZipFile(citrus_outer) as outer:
    nested = [
        info for info in outer.infolist()
        if not info.is_dir()
        and PurePosixPath(info.filename).suffix.lower() == ".zip"
    ]

    for nested_info in nested:
        # Avoid extracting the nested dataset to disk.
        content = outer.read(nested_info)
        with ZipFile(io.BytesIO(content)) as inner:
            for info in inner.infolist():
                if info.is_dir():
                    continue

                path = PurePosixPath(info.filename)
                if path.suffix.lower() not in IMAGE_EXTS:
                    continue

                parts = path.parts

                # Only fruit photographs are candidates for disease training.
                if not any(part.lower() == "fruits" for part in parts):
                    continue

                label = label_from_parts(parts[:-1])
                add_record(
                    inner.read(info),
                    "citrus_fruits",
                    "unsplit",
                    label,
                    info.filename,
                )

print("\nIMAGE COUNTS READ")
for (source, split, label), count in sorted(counts.items()):
    print(f"{source:15s} {split:8s} {label:12s} {count:5d}")

by_hash = defaultdict(list)
for record in records:
    by_hash[record["hash"]].append(record)

duplicate_groups = [
    group for group in by_hash.values()
    if len(group) > 1
]

print("\nEXACT DUPLICATE AUDIT")
print("Recognized images:", len(records))
print("Unique SHA-256 hashes:", len(by_hash))
print("Duplicate groups:", len(duplicate_groups))

# Prioritize duplicates involving the existing validation/test data.
duplicate_groups.sort(
    key=lambda group: (
        not any(
            r["source"] == "existing" and r["split"] in {"val", "test"}
            for r in group
        ),
        len(group),
    ),
    reverse=True,
)

shown = 0
for group in duplicate_groups:
    if shown >= 100:
        break

    # Print groups spanning sources/splits or labels.
    origins = {(r["source"], r["split"]) for r in group}
    labels = {r["label"] for r in group}

    if len(origins) < 2 and len(labels) < 2:
        continue

    print("\nGROUP:")
    if len(labels) > 1:
        print("  WARNING: conflicting labels:", sorted(labels))

    for r in group:
        print(
            f"  {r['source']:15s} {r['split']:8s} "
            f"{r['label']:12s} {r['name']}"
        )
    shown += 1

print("\nAudit complete. No images, labels, splits, or models were changed.")
