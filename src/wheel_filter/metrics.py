from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_binary_metrics(y_true: Any, y_prob: Any, threshold: float = 0.5) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    out: dict[str, Any] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "valid_precision": float(precision),
        "valid_recall": float(recall),
        "valid_f1": float(f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist(),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))

    return out


def choose_threshold_for_recall(
    y_true: Any,
    y_prob: Any,
    min_valid_recall: float = 0.90,
    low: float = 0.01,
    high: float = 0.99,
    steps: int = 197,
) -> tuple[float, dict[str, Any]]:
    thresholds = np.linspace(low, high, steps)
    candidates: list[tuple[float, dict[str, Any]]] = []

    for threshold in thresholds:
        metrics = compute_binary_metrics(y_true, y_prob, threshold=float(threshold))
        if metrics["valid_recall"] >= min_valid_recall:
            candidates.append((float(threshold), metrics))

    if not candidates:
        all_metrics = [
            (float(t), compute_binary_metrics(y_true, y_prob, threshold=float(t)))
            for t in thresholds
        ]
        return max(all_metrics, key=lambda x: x[1]["valid_f1"])

    return max(candidates, key=lambda x: (x[1]["valid_f1"], x[1]["valid_precision"]))

