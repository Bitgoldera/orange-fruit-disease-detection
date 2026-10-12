from pathlib import Path
import shutil
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "orange_5_class_clean"
)

REPORT = PROJECT_ROOT / "reports" / "quality_flags.csv"

REVIEW_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "quality_review"
)

REVIEW_DIR.mkdir(parents=True, exist_ok=True)

with REPORT.open("r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

for i, row in enumerate(rows, start=1):
    source = DATASET_DIR / row["file"]

    if not source.exists():
        print(f"Missing: {source}")
        continue

    destination = (
        REVIEW_DIR
        / f"{i:02d}_{source.name}"
    )

    shutil.copy2(source, destination)

print(f"Copied {len(rows)} images to:")
print(REVIEW_DIR)
