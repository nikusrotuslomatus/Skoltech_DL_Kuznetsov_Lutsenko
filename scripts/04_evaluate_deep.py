from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wheel_filter.metrics import compute_binary_metrics
from wheel_filter.utils import load_json, save_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics.json"
    predictions_path = run_dir / "test_predictions.csv"
    if not metrics_path.exists() or not predictions_path.exists():
        raise FileNotFoundError("Expected metrics.json and test_predictions.csv in run-dir")

    metrics = load_json(metrics_path)
    threshold = float(metrics["threshold"])
    pred = pd.read_csv(predictions_path)
    recomputed = compute_binary_metrics(pred["label"], pred["y_prob"], threshold=threshold)
    save_json(recomputed, run_dir / "test_metrics_recomputed.json")
    print("Recomputed test metrics")
    print(recomputed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
