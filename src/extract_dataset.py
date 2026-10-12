from pathlib import Path
import zipfile

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ZIP_FILE = PROJECT_ROOT / "data" / "raw" / "orange_dataset_1.zip"
EXTRACT_DIR = PROJECT_ROOT / "data" / "raw" / "dataset_1"

# Create extraction directory
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

# Check ZIP exists
if not ZIP_FILE.exists():
    raise FileNotFoundError(f"ZIP file not found: {ZIP_FILE}")

# Extract ZIP
with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)

print("Dataset extracted successfully!")
print(f"Location: {EXTRACT_DIR}")
