from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wheel_filter.config import load_config, output_path
from wheel_filter.data import build_manifest, validate_manifest
from wheel_filter.utils import ensure_dir, save_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    reports_dir = ensure_dir(output_path(cfg, "reports") / "audit")

    manifest = build_manifest(cfg)
    checks = validate_manifest(manifest)

    by_dataset = (
        manifest.groupby(["dataset_name", "label_name"])
        .size()
        .reset_index(name="samples")
        .sort_values(["dataset_name", "label_name"])
    )
    by_vehicle = (
        manifest.groupby(["vehicle_type", "label_name"])
        .size()
        .reset_index(name="samples")
        .sort_values(["vehicle_type", "label_name"])
    )
    quality = (
        manifest.groupby(["quality_flags", "label_name"], dropna=False)
        .size()
        .reset_index(name="samples")
        .sort_values(["quality_flags", "label_name"])
    )

    manifest.to_csv(reports_dir / "audit_manifest_preview.csv", index=False)
    by_dataset.to_csv(reports_dir / "counts_by_dataset.csv", index=False)
    by_vehicle.to_csv(reports_dir / "counts_by_vehicle_type.csv", index=False)
    quality.to_csv(reports_dir / "counts_by_quality_flags.csv", index=False)
    save_json(checks, reports_dir / "manifest_checks.json")

    print("Audit complete")
    print(f"Rows: {checks['rows']}")
    print(f"Labels: {checks['label_counts']}")
    print(f"Unique source images: {checks['unique_source_images']}")
    print(f"Tiny crops: {checks['tiny_crops']}, huge crops: {checks['huge_crops']}")
    print(f"Missing/corrupt: {checks['missing_files']}/{checks['corrupt_images']}")
    print(f"Duplicate label conflict groups: {checks['duplicate_label_conflict_groups']}")
    print(f"Reports: {reports_dir}")
    return 0 if checks["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

