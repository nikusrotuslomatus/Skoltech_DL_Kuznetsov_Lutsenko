from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, command: list[str], continue_on_error: bool = False) -> None:
    print("=" * 100, flush=True)
    print(f"[START] {name}", flush=True)
    print(" ".join(command), flush=True)
    start = time.time()
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"[FAILED] {name} exit_code={result.returncode} seconds={elapsed:.1f}", flush=True)
        if not continue_on_error:
            raise SystemExit(result.returncode)
    else:
        print(f"[DONE] {name} seconds={elapsed:.1f}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the full wheel-candidate filtering pipeline end to end."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mobilenet_v3_large"],
        help="Deep models to train sequentially.",
    )
    parser.add_argument("--config", default=None, help="Optional config path.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use.")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-sklearn", action="store_true")
    parser.add_argument("--skip-deep", action="store_true")
    parser.add_argument("--skip-visualize", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    py = args.python
    config_args = ["--config", args.config] if args.config else []

    print("Full pipeline runner")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python: {py}")
    print(f"Models: {args.models}")

    if not args.skip_audit:
        run_step(
            "dataset audit",
            [py, "scripts/01_audit_dataset.py", *config_args],
            continue_on_error=args.continue_on_error,
        )

    if not args.skip_prepare:
        run_step(
            "prepare manifest and grouped splits",
            [py, "scripts/02_prepare_manifest.py", *config_args],
            continue_on_error=args.continue_on_error,
        )

    if not args.skip_sklearn:
        run_step(
            "train sklearn sanity baseline",
            [py, "scripts/03_train_sklearn_baseline.py", *config_args],
            continue_on_error=args.continue_on_error,
        )
        if not args.skip_visualize:
            run_step(
                "visualize sklearn baseline",
                [
                    py,
                    "scripts/05_visualize_results.py",
                    "--predictions",
                    "artifacts/sklearn_baseline/test_predictions.csv",
                    "--out-dir",
                    "reports/sklearn_baseline",
                    *config_args,
                ],
                continue_on_error=args.continue_on_error,
            )

    if not args.skip_deep:
        for model in args.models:
            run_step(
                f"train deep model: {model}",
                [py, "scripts/03_train_deep.py", "--model", model, *config_args],
                continue_on_error=args.continue_on_error,
            )
            run_step(
                f"evaluate deep model: {model}",
                [py, "scripts/04_evaluate_deep.py", "--run-dir", f"artifacts/deep/{model}", *config_args],
                continue_on_error=args.continue_on_error,
            )
            if not args.skip_visualize:
                run_step(
                    f"visualize deep model: {model}",
                    [
                        py,
                        "scripts/05_visualize_results.py",
                        "--predictions",
                        f"artifacts/deep/{model}/test_predictions.csv",
                        "--out-dir",
                        f"reports/deep_{model}",
                        *config_args,
                    ],
                    continue_on_error=args.continue_on_error,
                )

        if not args.skip_visualize:
            run_step(
                "plot deep training history",
                [
                    py,
                    "scripts/07_plot_training_history.py",
                    "--runs",
                    *[f"artifacts/deep/{m}" for m in args.models],
                ],
                continue_on_error=args.continue_on_error,
            )

    print("=" * 100)
    print("Full pipeline finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
