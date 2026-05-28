from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wheel_filter.config import load_config, output_path
from wheel_filter.data import load_split_manifest
from wheel_filter.models import create_model
from wheel_filter.transforms import build_transforms
from wheel_filter.utils import ensure_dir, load_json, save_json


def image_paths(manifest: pd.DataFrame, split: str, limit: int) -> list[Path]:
    rows = manifest[manifest["split"] == split].head(limit)
    paths = [Path(p) for p in rows["crop_abs_path"].tolist()]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Some manifest paths do not exist. Regenerate data/processed/manifest_splits.csv "
            f"on this machine. First missing path: {missing[0]}"
        )
    return paths


def make_batches(paths: list[Path], transform, batch_size: int):
    import torch

    tensors = []
    for path in paths:
        image = Image.open(path)
        image = ImageOps.exif_transpose(image).convert("RGB")
        tensors.append(transform(image))

    for start in range(0, len(tensors), batch_size):
        yield torch.stack(tensors[start : start + batch_size])


def synchronize(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/deep/mobilenet_v3_large/best.pt")
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    args = parser.parse_args()

    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for latency benchmarking") from exc

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = load_config(args.config)
    checkpoint_cfg = checkpoint.get("cfg") or {}
    if "preprocessing" in checkpoint_cfg:
        cfg["preprocessing"] = checkpoint_cfg["preprocessing"]
    transform = build_transforms(cfg, train=False)

    manifest = load_split_manifest(cfg)
    paths = image_paths(manifest, args.split, args.samples)
    batches = list(make_batches(paths, transform, args.batch_size))

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = create_model(checkpoint["model_name"], num_classes=2, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in batches[: args.warmup]:
            _ = model(batch.to(device))
        synchronize(device)

        start = time.perf_counter()
        seen = 0
        for batch in batches:
            _ = model(batch.to(device))
            seen += int(batch.shape[0])
        synchronize(device)
        elapsed = time.perf_counter() - start

    result = {
        "model_name": checkpoint["model_name"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_mb": round(checkpoint_path.stat().st_size / 1024 / 1024, 3),
        "device": str(device),
        "split": args.split,
        "samples": int(seen),
        "batch_size": int(args.batch_size),
        "seconds_total": float(elapsed),
        "milliseconds_per_crop": float(elapsed * 1000.0 / max(1, seen)),
    }

    reports_dir = ensure_dir(output_path(cfg, "reports"))
    out_path = reports_dir / f"latency_{checkpoint['model_name']}_{device.type}.json"
    save_json(result, out_path)
    print(load_json(out_path))
    print(f"Latency report written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
