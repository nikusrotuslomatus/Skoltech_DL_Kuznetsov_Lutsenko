from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wheel_filter.config import load_config, output_path
from wheel_filter.data import load_split_manifest
from wheel_filter.metrics import choose_threshold_for_recall, compute_binary_metrics
from wheel_filter.utils import ensure_dir, save_json, seed_everything


def square_pad(image: Image.Image, fill: int = 0) -> Image.Image:
    width, height = image.size
    side = max(width, height)
    left = (side - width) // 2
    top = (side - height) // 2
    right = side - width - left
    bottom = side - height - top
    return ImageOps.expand(image, border=(left, top, right, bottom), fill=fill)


def extract_features(paths: list[str], size: int = 64) -> np.ndarray:
    features = []
    for path in paths:
        image = Image.open(path)
        image = ImageOps.exif_transpose(image).convert("RGB")
        padded = square_pad(image).resize((size, size), Image.Resampling.BILINEAR)
        arr = np.asarray(padded).astype(np.float32) / 255.0

        gray = arr.mean(axis=2)
        small = np.asarray(Image.fromarray((gray * 255).astype(np.uint8)).resize((32, 32), Image.Resampling.BILINEAR)).astype(np.float32) / 255.0
        hist_parts = []
        for channel in range(3):
            hist, _ = np.histogram(arr[:, :, channel], bins=16, range=(0.0, 1.0), density=True)
            hist_parts.append(hist.astype(np.float32))
        stats = np.array(
            [
                arr.mean(),
                arr.std(),
                gray.mean(),
                gray.std(),
                image.size[0] / max(1, image.size[1]),
                min(image.size) / max(1, max(image.size)),
            ],
            dtype=np.float32,
        )
        features.append(np.concatenate([small.reshape(-1), *hist_parts, stats]))
    return np.vstack(features)


def predict_and_save(model, df: pd.DataFrame, out_path: Path, threshold: float) -> dict:
    x = extract_features(df["crop_abs_path"].tolist())
    y = df["label"].astype(int).to_numpy()
    prob = model.predict_proba(x)[:, 1]
    pred = (prob >= threshold).astype(int)
    out = df[["sample_id", "split", "label", "label_name", "crop_abs_path", "crop_rel_path", "source_relative_path"]].copy()
    out["y_prob"] = prob
    out["y_pred"] = pred
    out.to_csv(out_path, index=False)
    return compute_binary_metrics(y, prob, threshold=threshold)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg["project"]["seed"]))
    manifest = load_split_manifest(cfg)

    out_dir = ensure_dir(output_path(cfg, "artifacts") / "sklearn_baseline")

    train = manifest[manifest["split"] == "train"].reset_index(drop=True)
    val = manifest[manifest["split"] == "validation"].reset_index(drop=True)
    test = manifest[manifest["split"] == "test"].reset_index(drop=True)

    x_train = extract_features(train["crop_abs_path"].tolist())
    y_train = train["label"].astype(int).to_numpy()
    x_val = extract_features(val["crop_abs_path"].tolist())
    y_val = val["label"].astype(int).to_numpy()

    majority = DummyClassifier(strategy="most_frequent")
    majority.fit(x_train, y_train)
    majority_prob = majority.predict_proba(x_val)[:, 1]
    majority_metrics = compute_binary_metrics(y_val, majority_prob, threshold=0.5)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1500,
            solver="lbfgs",
            random_state=int(cfg["project"]["seed"]),
        ),
    )
    model.fit(x_train, y_train)

    val_prob = model.predict_proba(x_val)[:, 1]
    threshold, val_metrics = choose_threshold_for_recall(
        y_val,
        val_prob,
        min_valid_recall=float(cfg["training"]["threshold_target_recall"]),
    )

    test_metrics = predict_and_save(model, test, out_dir / "test_predictions.csv", threshold)
    val_saved_metrics = predict_and_save(model, val, out_dir / "validation_predictions.csv", threshold)

    metrics = {
        "model": "sklearn_logistic_regression_image_features",
        "threshold": threshold,
        "majority_validation": majority_metrics,
        "validation": val_saved_metrics,
        "test": test_metrics,
    }
    save_json(metrics, out_dir / "metrics.json")
    joblib.dump(model, out_dir / "model.joblib")

    print("Sklearn baseline complete")
    print(f"Threshold: {threshold:.4f}")
    print("Validation:", val_saved_metrics)
    print("Test:", test_metrics)
    print(f"Artifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

