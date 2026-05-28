from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wheel_filter.config import load_config, output_path
from wheel_filter.utils import ensure_dir
from wheel_filter.visualization import make_error_grid, plot_confusion_matrix, plot_roc_pr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    pred_path = Path(args.predictions)
    out_dir = ensure_dir(Path(args.out_dir) if args.out_dir else output_path(cfg, "reports") / pred_path.parent.name)
    df = pd.read_csv(pred_path)

    plot_confusion_matrix(df["label"], df["y_pred"], out_dir / "confusion_matrix.png")
    plot_roc_pr(df["label"], df["y_prob"], out_dir)

    fp = df.index[(df["label"] == 0) & (df["y_pred"] == 1)]
    fn = df.index[(df["label"] == 1) & (df["y_pred"] == 0)]
    tp = df.index[(df["label"] == 1) & (df["y_pred"] == 1)]
    tn = df.index[(df["label"] == 0) & (df["y_pred"] == 0)]
    make_error_grid(df, fp, out_dir / "false_positives.jpg", "False positives")
    make_error_grid(df, fn, out_dir / "false_negatives.jpg", "False negatives")
    make_error_grid(df, tp, out_dir / "true_positives.jpg", "True positives")
    make_error_grid(df, tn, out_dir / "true_negatives.jpg", "True negatives")

    print(f"Visualizations written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

