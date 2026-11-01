#!/usr/bin/env python3
"""Safely archive optional experiments and stale app variants.

Default action is a dry run. Use --apply to move the listed items into a
 timestamped archive folder. Nothing is permanently deleted. Dataset folders,
 the active app.py, production model files, config and core reports stay put.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

# Keep app.py as the stable entry point. Preserve two feature-rich alternatives
# (multilingual and treatment/PDF) until their features are consolidated.
ROOT_VARIANTS = [
    "app_augrobust.py",
    "app_before_gradcam.py",
    "app_full.py",
    "app_premium.py",
    "app_redesigned.py",
    "app_refined.py",
]

EXPERIMENTAL_SCRIPTS = [
    "analyze_duplicates.py",
    "audit_big_dataset.py",
    "audit_extra_dataset_duplicates.py",
    "build_extra_train_pool.py",
    "compare_augrobust_test.py",
    "compare_augrobust_val.py",
    "compare_melanose100_val.py",
    "compare_pilot500.py",
    "compare_pilot500_val.py",
    "extract_dataset_2.py",
    "finetune_augrobust.py",
    "finetune_melanose100.py",
    "finetune_pilot500.py",
    "gradcam_greening_compare.py",
    "gradcam_predicted_compare.py",
    "greening_bg_sensitivity.py",
    "merge_melanose.py",
    "prepare_melanose100.py",
    "prepare_pilot500.py",
    "select_balanced_pilot.py",
    "train_augrobust.py",
    "train_melanose100.py",
    "train_pilot500.py",
]

OPTIONAL_REPORTS = [
    "big_dataset_overlap_audit.csv",
    "canker_audit_01.png",
    "canker_audit_02.png",
    "canker_audit_03.png",
    "canker_audit_04.png",
    "cross_split_duplicate_audit.png",
    "extra_train_near_holdout_skipped.csv",
    "extra_train_pilot_500_manifest.csv",
    "extra_train_pool_manifest.csv",
    "extra_train_pool_summary.json",
    "greening_bg_sensitivity.csv",
]
OPTIONAL_DIRS = [
    Path("reports/augrobust"),
    Path("reports/melanose100"),
    Path("reports/pilot500"),
    Path("models/augrobust"),
    Path("models/melanose100"),
    Path("models/pilot500"),
]


def planned_paths() -> list[Path]:
    paths: set[Path] = set()
    paths.update(ROOT / name for name in ROOT_VARIANTS)
    paths.update(ROOT / "src" / name for name in EXPERIMENTAL_SCRIPTS)
    paths.update(ROOT / "reports" / name for name in OPTIONAL_REPORTS)
    for rel_dir in OPTIONAL_DIRS:
        path = ROOT / rel_dir
        if path.exists():
            paths.add(path)
    return sorted((p for p in paths if p.exists()), key=lambda p: str(p).lower())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true",
        help="Move optional files into a timestamped archive (default is preview only).",
    )
    args = parser.parse_args()
    items = planned_paths()

    if not items:
        print("No optional files found to archive. Project unchanged.")
        return

    archive_root = ROOT / "archive" / f"cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print("Active entry point: app.py")
    print("Kept feature variants: app_multilingual.py and app_treatment_report.py (if present)")
    print("Kept production models, data folders, config, requirements and core reports.")
    print(f"Optional items identified: {len(items)}")
    for path in items:
        print(" -", path.relative_to(ROOT))

    if not args.apply:
        print("\nDRY RUN ONLY — no files were moved.")
        print("Review the list, then run: python src/cleanup_project.py --apply")
        return

    moved = 0
    for path in items:
        relative = path.relative_to(ROOT)
        destination = archive_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination = destination.with_name(destination.name + "_copy")
        shutil.move(str(path), str(destination))
        moved += 1

    print(f"\nArchived {moved} items under: {archive_root.relative_to(ROOT)}")
    print("Nothing was permanently deleted. Keep the archive until the app is tested.")


if __name__ == "__main__":
    main()
