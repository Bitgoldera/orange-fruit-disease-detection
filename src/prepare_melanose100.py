from pathlib import Path
import hashlib
import shutil
import yaml

ROOT = Path.cwd()
PROCESSED = ROOT / "data/processed"
SPLITS = ROOT / "data/splits"
PILOT = ROOT / "data/extra_train_pilot_500/Melanose"
TARGET = ROOT / "data/processed_melanose100"

TRAIN_SCRIPT = ROOT / "src/train.py"
FINETUNE_SCRIPT = ROOT / "src/finetune.py"
COMPARE_SCRIPT = ROOT / "src/compare_pilot500_val.py"

NEW_TRAIN = ROOT / "src/train_melanose100.py"
NEW_FINETUNE = ROOT / "src/finetune_melanose100.py"
NEW_COMPARE = ROOT / "src/compare_melanose100_val.py"

MODEL_DIR = ROOT / "models/melanose100"
REPORT_DIR = ROOT / "reports/melanose100"

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def images(folder):
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTS
    )


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# Never overwrite an earlier experiment.
outputs = [
    TARGET, MODEL_DIR, REPORT_DIR,
    NEW_TRAIN, NEW_FINETUNE, NEW_COMPARE,
]
existing = [str(p) for p in outputs if p.exists()]
if existing:
    raise SystemExit(
        "Stopped: these outputs already exist. Nothing overwritten:\n"
        + "\n".join(existing)
    )

for path in (PROCESSED, SPLITS, PILOT):
    if not path.is_dir():
        raise SystemExit(f"Required directory missing: {path}")

with (ROOT / "config.yaml").open(encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

classes = cfg["class_names"]
if set(classes) != {
    "Black_Spot", "Canker", "Greening", "Healthy", "Melanose"
}:
    raise SystemExit(f"Unexpected class names: {classes}")

# Verify processed data matches the cleaned splits.
for split in ("train", "val", "test"):
    for label in classes:
        a = {sha(p) for p in images(PROCESSED / split / label)}
        b = {sha(p) for p in images(SPLITS / split / label)}
        if a != b:
            raise SystemExit(
                f"Dataset mismatch at {split}/{label}; "
                "nothing was prepared."
            )

# Verify the 100 selected Melanose images are distinct from the
# original data and from one another by exact file hash.
new_images = images(PILOT)
if len(new_images) != 100:
    raise SystemExit(
        f"Expected 100 Melanose pilot images, found {len(new_images)}"
    )

existing_hashes = {
    sha(p)
    for split in ("train", "val", "test")
    for label in classes
    for p in images(PROCESSED / split / label)
}

new_hashes = []
for path in new_images:
    h = sha(path)
    if h in existing_hashes:
        raise SystemExit(f"Duplicate of existing data: {path}")
    new_hashes.append(h)

if len(set(new_hashes)) != 100:
    raise SystemExit("Duplicate images found within the Melanose selection.")

# Prepare experimental scripts in memory and verify replacements.
train_text = TRAIN_SCRIPT.read_text(encoding="utf-8")

train_replacements = [
    ('Path("data/processed/train")',
     'Path("data/processed_melanose100/train")'),
    ('Path("data/processed/val")',
     'Path("data/processed_melanose100/val")'),
    ('models/efficientnet_b0_best.keras',
     'models/melanose100/efficientnet_b0_best.keras'),
    ('model.save("models/efficientnet_b0_stage1.keras")',
     'model.save("models/melanose100/efficientnet_b0_stage1.keras")'),
    ('Saved baseline model to models/efficientnet_b0_stage1.keras',
     'Saved Melanose-only baseline model to models/melanose100/efficientnet_b0_stage1.keras'),
    ('Path("models").mkdir(parents=True, exist_ok=True)',
     'Path("models/melanose100").mkdir(parents=True, exist_ok=True)'),
]

for old, new in train_replacements:
    if train_text.count(old) != 1:
        raise SystemExit(f"Unexpected train.py contents near: {old}")
    train_text = train_text.replace(old, new, 1)

finetune_text = FINETUNE_SCRIPT.read_text(encoding="utf-8")

fine_replacements = [
    ('DATA_DIR = ROOT / "data" / "processed"',
     'DATA_DIR = ROOT / "data" / "processed_melanose100"'),
    ('MODEL_DIR = ROOT / "models"',
     'MODEL_DIR = ROOT / "models" / "melanose100"'),
    ('REPORT_DIR = ROOT / "reports"',
     'REPORT_DIR = ROOT / "reports" / "melanose100"'),
    ('test_ds = load_dataset("test")',
     '# Test evaluation is intentionally deferred until model selection.'),
]

for old, new in fine_replacements:
    if finetune_text.count(old) != 1:
        raise SystemExit(f"Unexpected finetune.py contents near: {old}")
    finetune_text = finetune_text.replace(old, new, 1)

# Remove the final test-evaluation/reporting block so this experiment
# does not evaluate the test set while we are still selecting models.
marker = "# Final test evaluation"
if finetune_text.count(marker) != 1:
    raise SystemExit("Could not safely locate the final test evaluation block.")

finetune_text = (
    finetune_text[:finetune_text.index(marker)]
    + "# Test-set evaluation is deferred until a model is selected.\n"
    + 'print("\\nMelanose-only training complete. Test evaluation deferred.")\n'
)

compare_text = COMPARE_SCRIPT.read_text(encoding="utf-8")
compare_replacements = [
    ('models/pilot500/', 'models/melanose100/'),
    ('reports/pilot500', 'reports/melanose100'),
    ('pilot500', 'melanose100'),
    ('pilot_prediction', 'melanose100_prediction'),
    ('pilot_confidence', 'melanose100_confidence'),
    ('pilot_fixed', 'melanose100_fixed'),
    ('pilot_regression', 'melanose100_regression'),
    ('PILOT MODEL', 'MELANOSE-ONLY MODEL'),
    ('Pilot model accuracy', 'Melanose-only model accuracy'),
    ('pred_pilot', 'pred_melanose100'),
    ('probs_pilot', 'probs_melanose100'),
]

for old, new in compare_replacements:
    compare_text = compare_text.replace(old, new)

# Remove any chance of writing files before copying has succeeded.
try:
    shutil.copytree(PROCESSED, TARGET)

    for i, source in enumerate(new_images, start=1):
        dest = TARGET / "train" / "Melanose" / (
            f"melanose100_{i:03d}_{source.name}"
        )
        if dest.exists():
            raise RuntimeError(f"Filename collision: {dest}")
        shutil.copy2(source, dest)

    MODEL_DIR.mkdir(parents=True)
    REPORT_DIR.mkdir(parents=True)

    NEW_TRAIN.write_text(train_text, encoding="utf-8")
    NEW_FINETUNE.write_text(finetune_text, encoding="utf-8")
    NEW_COMPARE.write_text(compare_text, encoding="utf-8")

except Exception:
    for p in (NEW_TRAIN, NEW_FINETUNE, NEW_COMPARE):
        if p.exists():
            p.unlink()
    for p in (REPORT_DIR, MODEL_DIR, TARGET):
        if p.exists():
            shutil.rmtree(p)
    raise

print("\nMELANOSE-ONLY EXPERIMENT READY")
for split in ("train", "val", "test"):
    print(f"\n{split.upper()}")
    for label in classes:
        count = len(images(TARGET / split / label))
        print(f"  {label:12s}: {count}")

print("\nNew scripts:")
print(" ", NEW_TRAIN.relative_to(ROOT))
print(" ", NEW_FINETUNE.relative_to(ROOT))
print(" ", NEW_COMPARE.relative_to(ROOT))
print("\nModels:", MODEL_DIR.relative_to(ROOT))
print("Reports:", REPORT_DIR.relative_to(ROOT))
print("\nOriginal dataset and models remain unchanged.")
