from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wheel_filter.config import load_config
from wheel_filter.data import (
    build_manifest,
    make_grouped_splits,
    split_summary,
    validate_manifest,
    write_processed_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest = build_manifest(cfg)
    checks = validate_manifest(manifest)
    if not checks["ok"]:
        print("Manifest validation failed before split:")
        print(checks)
        return 2

    split_df = make_grouped_splits(manifest, cfg)
    outputs = write_processed_outputs(split_df, cfg, with_splits=True)
    summary = split_summary(split_df)

    print("Prepared split manifest")
    print(summary.to_string(index=False))
    print("Outputs:")
    for key, value in outputs.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
