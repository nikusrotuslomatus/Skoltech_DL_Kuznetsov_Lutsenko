from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix

from .utils import ensure_dir


def plot_confusion_matrix(y_true, y_pred, out_path: str | Path) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["invalid", "valid"])
    ax.set_yticks([0, 1], ["invalid", "valid"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_roc_pr(y_true, y_prob, out_dir: str | Path) -> None:
    out_dir = ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curve.png", dpi=160)
    plt.close(fig)


def make_error_grid(
    df: pd.DataFrame,
    rows: Iterable[int],
    out_path: str | Path,
    title: str,
    max_items: int = 24,
    thumb_size: tuple[int, int] = (180, 180),
) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    selected = df.loc[list(rows)].head(max_items)
    if selected.empty:
        canvas = Image.new("RGB", (480, 120), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 12), f"{title}: no samples", fill="black")
        canvas.save(out_path)
        return

    cols = 4
    text_h = 70
    cell_w = thumb_size[0] + 20
    cell_h = thumb_size[1] + text_h + 20
    grid_rows = int(np.ceil(len(selected) / cols))
    canvas = Image.new("RGB", (cols * cell_w, grid_rows * cell_h + 30), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill="black")

    for n, (_, row) in enumerate(selected.iterrows()):
        col = n % cols
        grid_row = n // cols
        x = col * cell_w + 10
        y = grid_row * cell_h + 35
        try:
            image = Image.open(row["crop_abs_path"])
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail(thumb_size, Image.Resampling.LANCZOS)
            thumb = Image.new("RGB", thumb_size, "white")
            thumb.paste(image, ((thumb_size[0] - image.width) // 2, (thumb_size[1] - image.height) // 2))
            canvas.paste(thumb, (x, y))
        except Exception as exc:
            draw.rectangle([x, y, x + thumb_size[0], y + thumb_size[1]], outline="red")
            draw.text((x + 5, y + 5), str(exc)[:60], fill="red")
        text = f"true={row['label']} pred={row['y_pred']} p={row['y_prob']:.3f}\n{row['sample_id']}"
        draw.text((x, y + thumb_size[1] + 5), text, fill="black")

    canvas.save(out_path, quality=95)

