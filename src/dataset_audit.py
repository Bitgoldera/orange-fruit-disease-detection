"""Audit a folder-organized image dataset before model training.

Expected layout:
DATA_ROOT/
  healthy/
  canker/
  black_spot/
  greening/
  melanose/

The script reports counts, file types, corrupt/unreadable files, and basic image sizes.
"""
from __future__ import annotations

from pathlib import Path
from collections import Counter
from PIL import Image
import pandas as pd

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def audit(root: Path) -> pd.DataFrame:
    rows = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() not in VALID_EXTS:
                continue
            row = {
                "path": str(path),
                "class": class_dir.name,
                "ok": False,
                "width": None,
                "height": None,
                "mode": None,
            }
            try:
                with Image.open(path) as img:
                    img.verify()
                with Image.open(path) as img:
                    row["width"], row["height"], row["mode"] = img.size[0], img.size[1], img.mode
                row["ok"] = True
            except Exception:
                pass
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, default=Path("reports/dataset_audit.csv"))
    args = parser.parse_args()

    df = audit(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print("\nClass counts:")
    print(df["class"].value_counts(dropna=False).to_string())
    print(f"\nReadable: {int(df['ok'].sum())}/{len(df)}")
    if len(df):
        print("\nImage modes:")
        print(df.loc[df.ok, "mode"].value_counts().to_string())
        print("\nImage size summary:")
        print(df.loc[df.ok, ["width", "height"]].describe().round(1).to_string())
    print(f"\nSaved audit to {args.out}")


if __name__ == "__main__":
    main()
