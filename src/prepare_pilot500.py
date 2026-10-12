from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import shutil
import yaml

ROOT = Path.cwd()
PROCESSED = ROOT / "data/processed"
SPLITS = ROOT / "data/splits"
PILOT = ROOT / "data/extra_train_pilot_500"
TARGET = ROOT / "data/processed_pilot500"

TRAIN_SCRIPT = ROOT / "src/train.py"
FINETUNE_SCRIPT = ROOT / "src/finetune.py"
NEW_TRAIN_SCRIPT = ROOT / "src/train_pilot500.py"
NEW_FINETUNE_SCRIPT = ROOT / "src/finetune_pilot500.py"

MODEL_OUT = ROOT / "models/pilot500"
REPORT_OUT = ROOT / "reports/pilot500"

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def files(folder):
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTS
    )


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory(base):
    result = defaultdict(dict)
    for split in ("train", "val", "test"):
        split_dir = base / split
        if not split_dir.is_dir():
            raise RuntimeError(f"Missing directory: {split_dir}")

        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            result[split][class_dir.name] = {
                digest(p) for p in files(class_dir)
            }
    return result


# Refuse to overwrite any previous experiment.
outputs = [
    TARGET, MODEL_OUT, REPORT_OUT,
    NEW_TRAIN_SCRIPT, NEW_FINETUNE_SCRIPT,
]
existing = [str(p) for p in outputs if p.exists()]
if existing:
    raise SystemExit(
        "Stopped to avoid overwriting existing files/directories:\n"
        + "\n".join(existing)
    )

for path in (PROCESSED, SPLITS, PILOT):
    if not path.is_dir():
        raise SystemExit(f"Required directory missing: {path}")

with (ROOT / "config.yaml").open(encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

class_names = cfg["class_names"]
if len(class_names) != 5 or set(class_names) != {
    "Black_Spot", "Canker", "Greening", "Healthy", "Melanose"
}:
    raise SystemExit(f"Unexpected config class names: {class_names}")

# Confirm data/processed matches the current split dataset exactly,
# class by class, before making a copy.
processed_idx = inventory(PROCESSED)
splits_idx = inventory(SPLITS)

differences = []
for split in ("train", "val", "test"):
    for label in class_names:
        a = processed_idx[split].get(label, set())
        b = splits_idx[split].get(label, set())
        if a != b:
            differences.append(
                f"{split}/{label}: processed={len(a)}, splits={len(b)}, "
                f"hashes only in processed={len(a-b)}, "
                f"hashes only in splits={len(b-a)}"
            )

if differences:
    print("DATASET MISMATCH: no experiment files were created.")
    print("\n".join(differences))
    raise SystemExit(
        "Investigate data/processed versus data/splits before training."
    )

# Verify 100 pilot images per class and no exact overlaps.
all_processed_hashes = set()
for split in processed_idx.values():
    for hashes in split.values():
        all_processed_hashes.update(hashes)

pilot_paths = {}
pilot_hashes = set()

for label in class_names:
    paths = files(PILOT / label)
    print(f"Pilot {label}: {len(paths)}")

    if len(paths) != 100:
        raise SystemExit(
            f"Expected 100 pilot images for {label}; found {len(paths)}."
        )

    pilot_paths[label] = paths

    for p in paths:
        h = digest(p)
        if h in all_processed_hashes:
            raise SystemExit(
                f"Pilot image exactly matches processed data: {p}"
            )
        if h in pilot_hashes:
            raise SystemExit(f"Exact duplicate within pilot: {p}")
        pilot_hashes.add(h)

# Prepare isolated versions of the scripts in memory, and verify each
# change before writing anything.
train_text = TRAIN_SCRIPT.read_text(encoding="utf-8")
train_replacements = [
    ('Path("data/processed/train")',
     'Path("data/processed_pilot500/train")'),
    ('Path("data/processed/val")',
     'Path("data/processed_pilot500/val")'),
    ('keras.callbacks.ModelCheckpoint("models/efficientnet_b0_best.keras"',
     'keras.callbacks.ModelCheckpoint("models/pilot500/efficientnet_b0_best.keras"'),
    ('model.save("models/efficientnet_b0_stage1.keras")',
     'model.save("models/pilot500/efficientnet_b0_stage1.keras")'),
    ('Saved baseline model to models/efficientnet_b0_stage1.keras',
     'Saved pilot baseline model to models/pilot500/efficientnet_b0_stage1.keras'),
    ('Path("models").mkdir(parents=True, exist_ok=True)',
     'Path("models/pilot500").mkdir(parents=True, exist_ok=True)'),
]

for old, new in train_replacements:
    if train_text.count(old) != 1:
        raise SystemExit(
            f"Expected one occurrence in train.py of: {old}"
        )
    train_text = train_text.replace(old, new, 1)

finetune_text = FINETUNE_SCRIPT.read_text(encoding="utf-8")
finetune_replacements = [
    ('DATA_DIR = ROOT / "data" / "processed"',
     'DATA_DIR = ROOT / "data" / "processed_pilot500"'),
    ('MODEL_DIR = ROOT / "models"',
     'MODEL_DIR = ROOT / "models" / "pilot500"'),
    ('REPORT_DIR = ROOT / "reports"',
     'REPORT_DIR = ROOT / "reports" / "pilot500"'),
]

for old, new in finetune_replacements:
    if finetune_text.count(old) != 1:
        raise SystemExit(
            f"Expected one occurrence in finetune.py of: {old}"
        )
    finetune_text = finetune_text.replace(old, new, 1)

# Create the copy, then add only the selected pilot images to its train split.
try:
    shutil.copytree(PROCESSED, TARGET)

    for label, paths in pilot_paths.items():
        destination = TARGET / "train" / label
        if not destination.is_dir():
            raise RuntimeError(f"Missing class directory: {destination}")

        for i, source in enumerate(paths, start=1):
            dest = destination / f"pilot500_{i:03d}_{source.name}"
            if dest.exists():
                raise RuntimeError(f"Filename collision: {dest}")
            shutil.copy2(source, dest)

    MODEL_OUT.mkdir(parents=True)
    REPORT_OUT.mkdir(parents=True)
    NEW_TRAIN_SCRIPT.write_text(train_text, encoding="utf-8")
    NEW_FINETUNE_SCRIPT.write_text(finetune_text, encoding="utf-8")

except Exception:
    # Remove only outputs that this run was allowed to create.
    for p in [NEW_TRAIN_SCRIPT, NEW_FINETUNE_SCRIPT]:
        if p.exists():
            p.unlink()
    for p in [REPORT_OUT, MODEL_OUT, TARGET]:
        if p.exists():
            shutil.rmtree(p)
    raise

print("\nISOLATED DATASET READY")
for split in ("train", "val", "test"):
    print(f"\n{split.upper()}:")
    for label in class_names:
        count = len(files(TARGET / split / label))
        print(f"  {label:12s}: {count}")

print("\nExperimental scripts:")
print(" ", NEW_TRAIN_SCRIPT.relative_to(ROOT))
print(" ", NEW_FINETUNE_SCRIPT.relative_to(ROOT))
print("Experimental model directory:", MODEL_OUT.relative_to(ROOT))
print("Experimental reports directory:", REPORT_OUT.relative_to(ROOT))
print("\nOriginal dataset and models were not modified.")
