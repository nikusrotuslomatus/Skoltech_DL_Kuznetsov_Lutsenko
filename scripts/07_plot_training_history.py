from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wheel_filter.utils import ensure_dir


DEFAULT_RUNS = [
    "artifacts/deep/mobilenet_v3_large",
    "artifacts/deep/efficientnet_b0",
]


def load_histories(run_dirs: list[Path]) -> list[tuple[str, pd.DataFrame]]:
    histories = []
    for run_dir in run_dirs:
        history_path = run_dir / "history.csv"
        if not history_path.exists():
            print(f"Skipping missing history: {history_path}")
            continue
        histories.append((run_dir.name, pd.read_csv(history_path)))
    return histories


def plot_histories(histories: list[tuple[str, pd.DataFrame]], out_path: Path) -> None:
    if not histories:
        raise FileNotFoundError("No history.csv files were found")

    ensure_dir(out_path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for name, history in histories:
        axes[0].plot(history["epoch"], history["train_loss"], marker="o", label=f"{name} train")
        axes[0].plot(history["epoch"], history["validation_loss"], marker="o", label=f"{name} val")
        axes[1].plot(
            history["epoch"],
            history["validation_f1_at_0_5"],
            marker="o",
            label=f"{name} F1",
        )
        axes[1].plot(
            history["epoch"],
            history["validation_recall_at_0_5"],
            marker="o",
            label=f"{name} recall",
        )

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Validation quality at threshold 0.5")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.75, 0.95)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    parser.add_argument("--out", default="reports/training_history.png")
    args = parser.parse_args()

    run_dirs = [Path(p) if Path(p).is_absolute() else PROJECT_ROOT / p for p in args.runs]
    out_path = Path(args.out) if Path(args.out).is_absolute() else PROJECT_ROOT / args.out
    histories = load_histories(run_dirs)
    plot_histories(histories, out_path)
    print(f"Training history plot written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
