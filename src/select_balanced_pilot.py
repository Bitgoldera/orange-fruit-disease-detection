from pathlib import Path
from collections import Counter
from io import BytesIO
import csv
import hashlib
import random
import re
import shutil

import numpy as np
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(".")
SPLITS = ROOT / "data/splits"
POOL = ROOT / "data/extra_train_pool"
BIG = ROOT / "data/raw/extracted_big_dataset/Orange_Citrus_Disease_Dataset"
AUDIT = ROOT / "reports/big_dataset_overlap_audit.csv"
OUT = ROOT / "data/extra_train_pilot_500"
REPORTS = ROOT / "reports"

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SEED = 42
MAX_PHASH_DISTANCE = 5

POOL_QUOTA = {
    "Black_Spot": 100,
    "Canker": 50,
    "Greening": 100,
    "Healthy": 50,
    "Melanose": 0,
}
BIG_QUOTA = {
    "Black_Spot": 0,
    "Canker": 50,
    "Greening": 0,
    "Healthy": 50,
    "Melanose": 100,
}

rng = random.Random(SEED)

# DCT-based 64-bit perceptual hash.
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
            im.convert("L").resize((n, n)),
            dtype=np.float32,
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


if not AUDIT.exists():
    raise SystemExit(f"Audit report missing: {AUDIT}")

if OUT.exists() and any(OUT.iterdir()):
    raise SystemExit(
        f"{OUT} already contains files. Nothing was overwritten."
    )

# Exact hashes from all current splits are forbidden in the pilot.
existing_hashes = set()
holdout_phashes = []

for split in ("train", "val", "test"):
    for path in image_files(SPLITS / split):
        data = path.read_bytes()
        existing_hashes.add(sha(data))
        if split in ("val", "test"):
            holdout_phashes.append(phash(data))

selected = []
selected_hashes = set()
selected_phashes = []
stats = Counter()


def check_candidate(path):
    """Return a usable record, or None if duplicate/near-duplicate."""
    try:
        data = path.read_bytes()
        digest = sha(data)

        if digest in existing_hashes or digest in selected_hashes:
            stats["exact_duplicate_skipped"] += 1
            return None

        image_hash = phash(data)

        if any(
            (image_hash ^ h).bit_count() <= MAX_PHASH_DISTANCE
            for h in holdout_phashes
        ):
            stats["near_validation_test_skipped"] += 1
            return None

        if any(
            (image_hash ^ h).bit_count() <= MAX_PHASH_DISTANCE
            for h in selected_phashes
        ):
            stats["near_selected_image_skipped"] += 1
            return None

        return {
            "path": path,
            "sha256": digest,
            "phash": image_hash,
        }
    except Exception:
        stats["unreadable_skipped"] += 1
        return None


def select_from_paths(paths, label, source, quota):
    paths = list(paths)
    rng.shuffle(paths)
    initial = len(selected)
    found = 0

    for path in paths:
        stats[f"{source}_examined"] += 1
        record = check_candidate(path)
        if record is None:
            continue

        record["label"] = label
        record["source"] = source
        selected.append(record)
        selected_hashes.add(record["sha256"])
        selected_phashes.append(record["phash"])
        found += 1

        if found == quota:
            break

    if found != quota:
        raise RuntimeError(
            f"Could only select {found}/{quota} for {label} from {source}. "
            "No pilot files were copied; inspect the available candidates."
        )

    print(f"{label:12s} {source:12s}: {found} selected")


# First choose from the previously screened staging pool.
for label, quota in POOL_QUOTA.items():
    if quota:
        select_from_paths(
            image_files(POOL / label), label, "pool", quota
        )

# Then choose only candidates marked PROVISIONAL_CANDIDATE by the audit.
big_candidates = {label: [] for label in BIG_QUOTA}

with AUDIT.open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["status"] != "PROVISIONAL_CANDIDATE":
            continue
        if row["label"] in big_candidates:
            big_candidates[row["label"]].append(Path(row["path"]))

for label, quota in BIG_QUOTA.items():
    if quota:
        select_from_paths(
            big_candidates[label], label, "big_dataset", quota
        )

# Only write files once all 500 candidates have passed screening.
if len(selected) != 500:
    raise RuntimeError(
        f"Expected 500 selections, got {len(selected)}. No files copied."
    )

OUT.mkdir(parents=True, exist_ok=True)
for label in sorted(set(POOL_QUOTA) | set(BIG_QUOTA)):
    (OUT / label).mkdir(parents=True, exist_ok=True)

manifest = REPORTS / "extra_train_pilot_500_manifest.csv"
REPORTS.mkdir(parents=True, exist_ok=True)

name_counts = Counter()
with manifest.open("w", newline="", encoding="utf-8") as f:
    fields = [
        "label", "source", "output_path", "original_path", "sha256"
    ]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for record in selected:
        label = record["label"]
        name_counts[label] += 1
        original = record["path"]
        filename = (
            f"{record['source']}_{name_counts[label]:03d}_"
            f"{re.sub(r'[^A-Za-z0-9_-]+', '_', original.stem)[:45]}"
            f"{original.suffix.lower()}"
        )
        dest = OUT / label / filename
        shutil.copy2(original, dest)

        writer.writerow({
            "label": label,
            "source": record["source"],
            "output_path": str(dest),
            "original_path": str(original),
            "sha256": record["sha256"],
        })

# Make a review sheet showing 12 randomly selected pilot images per class.
cards = []
for label in sorted(set(POOL_QUOTA) | set(BIG_QUOTA)):
    paths = image_files(OUT / label)
    rng.shuffle(paths)
    cards.extend((label, p) for p in paths[:12])

tile_w, tile_h = 180, 155
columns = 6
rows = (len(cards) + columns - 1) // columns
sheet = Image.new("RGB", (columns * tile_w, rows * tile_h), "white")
draw = ImageDraw.Draw(sheet)

for i, (label, path) in enumerate(cards):
    px = (i % columns) * tile_w
    py = (i // columns) * tile_h
    with Image.open(path) as im:
        thumb = ImageOps.fit(im.convert("RGB"), (160, 110))
    sheet.paste(thumb, (px + 10, py + 5))
    draw.text((px + 8, py + 120), label, fill="black")
    draw.text((px + 8, py + 137), path.name[:22], fill="black")

contact = REPORTS / "extra_train_pilot_500_contact_sheet.jpg"
sheet.save(contact, quality=92)

print("\nPILOT CREATED")
print("Folder:", OUT)
print("Manifest:", manifest)
print("Contact sheet:", contact)
print("\nClass counts:")
for label in sorted(set(POOL_QUOTA) | set(BIG_QUOTA)):
    print(f"  {label:12s}: {len(image_files(OUT / label))}")
print("\nScreening counts:")
for name, count in sorted(stats.items()):
    print(f"  {name}: {count}")
print("\nThe original splits and models were not modified.")
