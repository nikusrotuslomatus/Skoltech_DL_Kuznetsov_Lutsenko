from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wheel_filter.config import load_config
from wheel_filter.models import create_model
from wheel_filter.transforms import build_transforms
from wheel_filter.utils import ensure_dir, load_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Image file or folder of jpg/png crops")
    parser.add_argument("--out", default="predictions.csv")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for deep prediction") from exc

    checkpoint_path = Path(args.checkpoint)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt.get("cfg") or load_config()
    model_name = ckpt["model_name"]
    threshold = args.threshold
    metrics_path = checkpoint_path.parent / "metrics.json"
    if threshold is None and metrics_path.exists():
        threshold = float(load_json(metrics_path)["threshold"])
    if threshold is None:
        threshold = 0.5

    model = create_model(model_name, num_classes=2, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    transform = build_transforms(cfg, train=False)

    input_path = Path(args.input)
    if input_path.is_dir():
        paths = sorted([p for p in input_path.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    else:
        paths = [input_path]

    rows = []
    with torch.no_grad():
        for path in paths:
            image = Image.open(path)
            image = ImageOps.exif_transpose(image).convert("RGB")
            x = transform(image).unsqueeze(0)
            prob = torch.softmax(model(x), dim=1)[0, 1].item()
            rows.append(
                {
                    "path": str(path),
                    "prob_valid": prob,
                    "prediction": int(prob >= threshold),
                    "threshold": threshold,
                }
            )

    out_path = Path(args.out)
    ensure_dir(out_path.parent if out_path.parent != Path("") else Path("."))
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Predictions written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

