from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import re

import numpy as np
from PIL import Image

ROOT = Path(".")
EXTRA = ROOT / "data/raw/extra_datasets"
SPLITS = ROOT / "data/splits"
OUTPUT = ROOT / "data/extra_train_pool"
REPORTS = ROOT / "reports"

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASSES = {"Black_Spot", "Canker", "Greening", "Healthy", "Melanose"}

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
}

# The same DCT-based perceptual hash used in the earlier audit.
n = 32
x = np.arange(n)
k = x[:, None]
dct = np.cos(np.pi * (2 * x + 1) * k / (2 * n))
dct[0] *= np.sqrt(1 / n)
dct[1:] *= np.sqrt(2 / n)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def phash(data):
    with Image.open(io.BytesIO(data)) as im:
        arr = np.asarray(
            im.convert("L").resize((n, n)),
            dtype=np.float32,
        )
    coeff = dct @ arr @ dct.T
    low = coeff[:8, :8].flatten()[1:]
    value = 0
    for bit in (low > np.median(low)):
        value = (value << 1) | int(bit)
    return value


def label_from_parts(parts):
    for part in reversed(parts):
        label = LABELS.get(part.strip().lower())
        if label:
            return label
    return None


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")[:70]


if OUTPUT.exists() and any(OUTPUT.iterdir()):
    raise SystemExit(
        f"{OUTPUT} already contains files. "
        "Nothing was changed; move or rename the previous staging pool first."
    )

# Hash every existing image; validation/test hashes and perceptual hashes
# are kept separately to screen candidate training data.
existing_hashes = set()
holdout_phashes = []

for split in ["train", "val", "test"]:
    for path in sorted((SPLITS / split).glob("*/*")):
        if not path.is_file() or path.suffix.lower() not in EXTS:
            continue

        data = path.read_bytes()
        existing_hashes.add(sha256(data))

        if split in {"val", "test"}:
            holdout_phashes.append((phash(data), str(path)))

print("Existing image hashes:", len(existing_hashes))
print("Validation/test images used for near-duplicate screening:",
      len(holdout_phashes))

# Keep the source member name, rather than extracting whole archives.
records = []
stats = defaultdict(Counter)
conflicting_source_hashes = set()


def consider(source, member, label, data, suffix):
    stats[source]["labeled_images_scanned"] += 1

    digest = sha256(data)
    if digest in existing_hashes:
        stats[source]["exact_matches_existing_splits"] += 1
        return

    image_hash = phash(data)
    nearest = None

    for holdout_hash, holdout_path in holdout_phashes:
        distance = (image_hash ^ holdout_hash).bit_count()
        if distance <= 5:
            nearest = (distance, holdout_path)
            break

    if nearest is not None:
        stats[source]["near_holdout_candidates_skipped"] += 1
        near_rows.append({
            "source": source,
            "member": member,
            "label": label,
            "distance": nearest[0],
            "holdout_path": nearest[1],
        })
        return

    records.append({
        "source": source,
        "member": member,
        "label": label,
        "sha": digest,
        "phash": image_hash,
        "suffix": suffix,
    })


near_rows = []

# Source 1: use the publisher-provided TRAIN split only.
archive1 = EXTRA / "archive (1).zip"
with ZipFile(archive1) as z:
    for info in z.infolist():
        if info.is_dir():
            continue

        p = PurePosixPath(info.filename)
        if p.suffix.lower() not in EXTS:
            continue

        parts = p.parts
        if not parts or parts[0].lower() != "train":
            continue

        label = label_from_parts(parts[:-1])
        if label not in CLASSES:
            stats["archive1"]["unsupported_or_unrecognized"] += 1
            continue

        consider("archive1", info.filename, label, z.read(info), p.suffix.lower())

# Source 2: folders have class names but no declared train/validation/test split.
archive2 = EXTRA / "archive (2).zip"
with ZipFile(archive2) as z:
    for info in z.infolist():
        if info.is_dir():
            continue

        p = PurePosixPath(info.filename)
        if p.suffix.lower() not in EXTS:
            continue

        label = label_from_parts(p.parts[:-1])
        if label not in CLASSES:
            stats["archive2"]["unsupported_or_unrecognized"] += 1
            continue

        consider("archive2", info.filename, label, z.read(info), p.suffix.lower())

# Source 3: read only Citrus/Fruits images from the nested ZIP.
citrus_outer_path = next(EXTRA.glob("A Citrus*.zip"))

with ZipFile(citrus_outer_path) as outer:
    nested = [
        i for i in outer.infolist()
        if not i.is_dir()
        and PurePosixPath(i.filename).suffix.lower() == ".zip"
    ]

    if len(nested) != 1:
        raise RuntimeError(
            f"Expected one nested Citrus ZIP, found {len(nested)}."
        )

    with ZipFile(io.BytesIO(outer.read(nested[0]))) as inner:
        for info in inner.infolist():
            if info.is_dir():
                continue

            p = PurePosixPath(info.filename)
            if p.suffix.lower() not in EXTS:
                continue

            parts = p.parts
            fruit_indices = [
                i for i, part in enumerate(parts)
                if part.lower() == "fruits"
            ]
            if not fruit_indices:
                continue

            label = label_from_parts(parts[:-1])
            if label not in CLASSES:
                stats["citrus_fruits"]["unsupported_or_unrecognized"] += 1
                continue

            consider(
                "citrus_fruits",
                info.filename,
                label,
                inner.read(info),
                p.suffix.lower(),
            )

# Deduplicate candidate images across all new sources.
groups = defaultdict(list)
for record in records:
    groups[record["sha"]].append(record)

accepted = []

# Prefer the source with an explicit published training split, then archive2,
# then the nested Citrus fruit dataset.
priority = {"archive1": 0, "archive2": 1, "citrus_fruits": 2}

for digest, group in groups.items():
    labels = {r["label"] for r in group}

    if len(labels) != 1:
        conflicting_source_hashes.add(digest)
        for r in group:
            stats[r["source"]]["conflicting_hash_group_skipped"] += 1
        continue

    group.sort(key=lambda r: (priority[r["source"]], r["member"]))
    accepted.append(group[0])

    if len(group) > 1:
        for r in group[1:]:
            stats[r["source"]]["duplicate_of_another_extra_source"] += 1

accepted.sort(key=lambda r: (r["label"], r["source"], r["member"]))

# Save accepted candidates into staging, never into the live training split.
OUTPUT.mkdir(parents=True, exist_ok=True)
for label in sorted(CLASSES):
    (OUTPUT / label).mkdir(parents=True, exist_ok=True)

# Open archives once and copy only selected members.
with ZipFile(archive1) as z1, ZipFile(archive2) as z2:
    with ZipFile(citrus_outer_path) as outer:
        nested_info = next(
            i for i in outer.infolist()
            if not i.is_dir()
            and PurePosixPath(i.filename).suffix.lower() == ".zip"
        )

        with ZipFile(io.BytesIO(outer.read(nested_info))) as citrus:
            handles = {
                "archive1": z1,
                "archive2": z2,
                "citrus_fruits": citrus,
            }

            manifest_path = REPORTS / "extra_train_pool_manifest.csv"
            REPORTS.mkdir(parents=True, exist_ok=True)

            with manifest_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "label", "output_path", "source",
                        "source_member", "sha256",
                    ],
                )
                writer.writeheader()

                label_counts = Counter()
                for r in accepted:
                    label_counts[r["label"]] += 1
                    filename = (
                        f"{r['source']}_{label_counts[r['label']]:05d}_"
                        f"{safe_name(PurePosixPath(r['member']).stem)}"
                        f"{r['suffix']}"
                    )

                    dest = OUTPUT / r["label"] / filename
                    data = handles[r["source"]].read(r["member"])

                    # Guard against source files changing between passes.
                    if sha256(data) != r["sha"]:
                        raise RuntimeError(
                            f"Hash changed while reading {r['source']}: {r['member']}"
                        )

                    dest.write_bytes(data)
                    writer.writerow({
                        "label": r["label"],
                        "output_path": str(dest),
                        "source": r["source"],
                        "source_member": r["member"],
                        "sha256": r["sha"],
                    })
                    stats[r["source"]]["copied_to_staging"] += 1

near_csv = REPORTS / "extra_train_near_holdout_skipped.csv"
with near_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["source", "member", "label", "distance", "holdout_path"],
    )
    writer.writeheader()
    writer.writerows(near_rows)

summary = {
    "staging_folder": str(OUTPUT),
    "candidate_groups_with_label_conflicts": len(conflicting_source_hashes),
    "near_holdout_candidate_images_skipped": len(near_rows),
    "counts_by_source": {k: dict(v) for k, v in stats.items()},
    "counts_by_class": {
        label: sum(1 for r in accepted if r["label"] == label)
        for label in sorted(CLASSES)
    },
    "notes": [
        "Only archive1/train was imported from archive1.",
        "Validation and test folders were not changed.",
        "Exact overlaps with any existing split were excluded.",
        "Perceptual-hash distance <= 5 from validation/test was excluded.",
        "This staging pool has not been merged into data/splits/train.",
        "The 7.26 GB Orange fruits archive was not used in this pass.",
    ],
}

summary_path = REPORTS / "extra_train_pool_summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("\nSTAGING POOL CREATED")
print("Folder:", OUTPUT)
print("\nCounts by class:")
for label, count in summary["counts_by_class"].items():
    print(f"  {label:12s}: {count}")

print("\nCounts by source:")
for source, source_stats in sorted(stats.items()):
    print(f"\n{source}")
    for key, value in sorted(source_stats.items()):
        print(f"  {key}: {value}")

print("\nNear-holdout candidates skipped:", len(near_rows))
print("Cross-source conflicting hash groups skipped:",
      len(conflicting_source_hashes))
print("Manifest:", manifest_path)
print("Near-holdout review list:", near_csv)
print("Summary:", summary_path)
print("\nNo current train/validation/test images or models were modified.")
