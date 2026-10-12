from pathlib import Path
from collections import Counter, defaultdict
from io import BytesIO
import csv
import hashlib

import numpy as np
from PIL import Image

ROOT = Path(".")
BIG = ROOT / "data/raw/extracted_big_dataset/Orange_Citrus_Disease_Dataset"
SPLITS = ROOT / "data/splits"
POOL = ROOT / "data/extra_train_pool"
REPORTS = ROOT / "reports"

CLASS_FOLDERS = {
    "Citrus_Canker": "Canker",
    "Citrus_Melanose": "Melanose",
    "Healthy": "Healthy",
}
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# DCT-based perceptual hash, matching the earlier audit approach.
n = 32
x = np.arange(n)
k = x[:, None]
dct = np.cos(np.pi * (2 * x + 1) * k / (2 * n))
dct[0] *= np.sqrt(1 / n)
dct[1:] *= np.sqrt(2 / n)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def phash(data):
    with Image.open(BytesIO(data)) as im:
        arr = np.asarray(
            im.convert("L").resize((n, n)), dtype=np.float32
        )
    coeff = dct @ arr @ dct.T
    low = coeff[:8, :8].flatten()[1:]
    value = 0
    for bit in low > np.median(low):
        value = (value << 1) | int(bit)
    return value


def image_files(folder):
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTS
    )


# Hash existing split files and staged extra-training files.
split_hashes = defaultdict(list)
pool_hashes = defaultdict(list)
holdout_hashes = []

print("Indexing existing splits and staging pool...")

for split in ("train", "val", "test"):
    for path in image_files(SPLITS / split):
        label = path.parent.name
        digest = sha(path.read_bytes())
        split_hashes[digest].append((split, label, str(path)))

        if split in ("val", "test"):
            data = path.read_bytes()
            holdout_hashes.append((phash(data), split, label, str(path)))

for path in image_files(POOL):
    label = path.parent.name
    pool_hashes[sha(path.read_bytes())].append((label, str(path)))

print("Existing unique hashes:", len(split_hashes))
print("Staging pool unique hashes:", len(pool_hashes))
print("Validation/test images for near-duplicate screening:",
      len(holdout_hashes))

# Inspect the large dataset.
rows = []
print("\nAuditing large dataset images...")

for folder, label in CLASS_FOLDERS.items():
    paths = image_files(BIG / folder)
    print(f"Reading {label}: {len(paths)} images")

    for path in paths:
        data = path.read_bytes()
        digest = sha(data)
        image_hash = phash(data)

        matches = split_hashes.get(digest, [])
        pool_matches = pool_hashes.get(digest, [])

        split_conflict = any(existing_label != label
                             for _, existing_label, _ in matches)
        pool_conflict = any(existing_label != label
                            for existing_label, _ in pool_matches)

        row = {
            "label": label,
            "path": str(path),
            "sha256": digest,
            "status": "",
            "matching_split": "",
            "matching_path": "",
            "near_distance": "",
            "near_holdout_path": "",
        }

        if split_conflict:
            row["status"] = "LABEL_CONFLICT_EXISTING_SPLIT"
            row["matching_path"] = matches[0][2]
        elif matches:
            row["status"] = "EXACT_MATCH_EXISTING_SPLIT"
            row["matching_split"] = ",".join(
                sorted({s for s, _, _ in matches})
            )
            row["matching_path"] = matches[0][2]
        elif pool_conflict:
            row["status"] = "LABEL_CONFLICT_EXTRA_POOL"
            row["matching_path"] = pool_matches[0][1]
        elif pool_matches:
            row["status"] = "EXACT_MATCH_EXTRA_POOL"
            row["matching_path"] = pool_matches[0][1]
        else:
            nearest = None
            for holdout_hash, split, holdout_label, holdout_path in holdout_hashes:
                distance = (image_hash ^ holdout_hash).bit_count()
                if nearest is None or distance < nearest[0]:
                    nearest = (distance, split, holdout_label, holdout_path)

            if nearest and nearest[0] <= 5:
                row["status"] = "NEAR_VALIDATION_TEST"
                row["near_distance"] = nearest[0]
                row["near_holdout_path"] = nearest[3]
            else:
                row["status"] = "PROVISIONAL_CANDIDATE"

        rows.append(row)

# Deduplicate potential candidates inside the large dataset itself.
candidate_groups = defaultdict(list)
for row in rows:
    if row["status"] == "PROVISIONAL_CANDIDATE":
        candidate_groups[row["sha256"]].append(row)

for digest, group in candidate_groups.items():
    labels = {row["label"] for row in group}

    if len(labels) > 1:
        for row in group:
            row["status"] = "LABEL_CONFLICT_INSIDE_BIG_DATASET"
    else:
        group.sort(key=lambda row: row["path"])
        for duplicate in group[1:]:
            duplicate["status"] = "DUPLICATE_INSIDE_BIG_DATASET"

REPORTS.mkdir(parents=True, exist_ok=True)
csv_path = REPORTS / "big_dataset_overlap_audit.csv"

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("\n=== OVERALL AUDIT RESULTS ===")
overall = Counter(row["status"] for row in rows)
for status, count in sorted(overall.items()):
    print(f"{status:38s} {count:5d}")

print("\n=== RESULTS BY CLASS ===")
for label in CLASS_FOLDERS.values():
    print(f"\n{label}")
    counts = Counter(
        row["status"] for row in rows if row["label"] == label
    )
    for status, count in sorted(counts.items()):
        print(f"  {status:36s} {count:5d}")

print("\nAudit report:", csv_path)
print("No images were moved, copied, or deleted.")
print("No model or dataset split was modified.")
