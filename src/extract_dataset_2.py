from pathlib import Path
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ZIP_FILE = PROJECT_ROOT / "data" / "raw" / "orange_melanose.zip"
EXTRACT_DIR = PROJECT_ROOT / "data" / "raw" / "dataset_2"

EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

if not ZIP_FILE.exists():
    raise FileNotFoundError(f"ZIP not found: {ZIP_FILE}")

with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)

print("Dataset 2 extracted successfully!")
print(f"Location: {EXTRACT_DIR}")
