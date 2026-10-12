from pathlib import Path
import csv

from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "orange_5_class_clean"
)

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".webp", ".tif", ".tiff"
}

# Images below this shortest-side size will be flagged for inspection.
MIN_SHORT_SIDE = 160

# Very unusual aspect ratios will also be flagged.
MAX_ASPECT_RATIO = 2.5


def main():
    flagged = []
    total = 0

    for class_dir in sorted(DATASET_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        for image_path in sorted(class_dir.iterdir()):
            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            total += 1

            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    img.verify()

                short_side = min(width, height)
                long_side = max(width, height)
                aspect_ratio = long_side / short_side

                reasons = []

                if short_side < MIN_SHORT_SIDE:
                    reasons.append(
                        f"small_short_side({short_side})"
                    )

                if aspect_ratio > MAX_ASPECT_RATIO:
                    reasons.append(
                        f"extreme_aspect_ratio({aspect_ratio:.2f})"
                    )

                if reasons:
                    flagged.append({
                        "class": class_dir.name,
                        "file": str(
                            image_path.relative_to(DATASET_DIR)
                        ),
                        "width": width,
                        "height": height,
                        "aspect_ratio": round(aspect_ratio, 3),
                        "reason": "; ".join(reasons)
                    })

            except (UnidentifiedImageError, OSError, ValueError) as e:
                flagged.append({
                    "class": class_dir.name,
                    "file": str(
                        image_path.relative_to(DATASET_DIR)
                    ),
                    "width": "",
                    "height": "",
                    "aspect_ratio": "",
                    "reason": f"invalid_image: {e}"
                })

    report_path = REPORT_DIR / "quality_flags.csv"

    with report_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class",
                "file",
                "width",
                "height",
                "aspect_ratio",
                "reason"
            ]
        )
        writer.writeheader()
        writer.writerows(flagged)

    print("\n====================================")
    print("IMAGE QUALITY CHECK")
    print("====================================")
    print(f"Total images checked : {total}")
    print(f"Flagged for review   : {len(flagged)}")
    print("------------------------------------")
    print(f"Report: {report_path}")

    if flagged:
        print("\nFlagged images:")
        for item in flagged:
            print(
                f"{item['class']:12s} | "
                f"{item['width']}x{item['height']} | "
                f"{item['file']} | "
                f"{item['reason']}"
            )
    else:
        print("\nNo images were flagged.")


if __name__ == "__main__":
    main()
