from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wheel_filter.config import load_config, output_path
from wheel_filter.data import WheelCandidateTorchDataset, load_split_manifest
from wheel_filter.metrics import choose_threshold_for_recall, compute_binary_metrics
from wheel_filter.models import create_model
from wheel_filter.transforms import build_transforms
from wheel_filter.utils import ensure_dir, save_json, seed_everything


def require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader
    except Exception as exc:
        raise RuntimeError(
            "PyTorch/torchvision are required. Install them first, then rerun this script."
        ) from exc
    return torch, DataLoader


def run_epoch(model, loader, criterion, optimizer, device, train: bool, scaler=None):
    torch, _ = require_torch()
    model.train(train)
    total_loss = 0.0
    total = 0
    probs = []
    labels = []
    sample_ids = []

    for images, y, ids in loader:
        images = images.to(device)
        y = y.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
                logits = model(images)
                loss = criterion(logits, y)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            with torch.no_grad():
                logits = model(images)
                loss = criterion(logits, y)

        batch = int(y.shape[0])
        total_loss += float(loss.item()) * batch
        total += batch
        prob = torch.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy()
        probs.extend(prob.tolist())
        labels.extend(y.detach().cpu().numpy().astype(int).tolist())
        sample_ids.extend(list(ids))

    return {
        "loss": total_loss / max(1, total),
        "y_true": np.asarray(labels, dtype=int),
        "y_prob": np.asarray(probs, dtype=float),
        "sample_ids": sample_ids,
    }


def save_predictions(df: pd.DataFrame, sample_ids, y_true, y_prob, threshold: float, path: Path) -> None:
    pred_df = pd.DataFrame({"sample_id": sample_ids, "label": y_true, "y_prob": y_prob})
    pred_df["y_pred"] = (pred_df["y_prob"] >= threshold).astype(int)
    out = pred_df.merge(
        df[["sample_id", "split", "label_name", "crop_abs_path", "crop_rel_path", "source_relative_path"]],
        on="sample_id",
        how="left",
    )
    out.to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    torch, DataLoader = require_torch()
    cfg = load_config(args.config)
    seed_everything(int(cfg["project"]["seed"]))

    model_name = args.model or cfg["training"]["model_name"]
    if model_name in cfg.get("models", {}):
        model_cfg = cfg["models"][model_name]
        cfg["training"]["batch_size"] = int(model_cfg.get("batch_size", cfg["training"]["batch_size"]))
        cfg["training"]["learning_rate"] = float(model_cfg.get("learning_rate", cfg["training"]["learning_rate"]))
        cfg["training"]["epochs"] = int(model_cfg.get("epochs", cfg["training"]["epochs"]))

    run_dir = ensure_dir(output_path(cfg, "artifacts") / "deep" / model_name)
    manifest = load_split_manifest(cfg)

    train_tf = build_transforms(cfg, train=True)
    eval_tf = build_transforms(cfg, train=False)
    train_ds = WheelCandidateTorchDataset(manifest, "train", transform=train_tf)
    val_ds = WheelCandidateTorchDataset(manifest, "validation", transform=eval_tf)
    test_ds = WheelCandidateTorchDataset(manifest, "test", transform=eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["training"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(model_name, num_classes=2, pretrained=bool(cfg["training"]["pretrained"]))
    model.to(device)

    y_train = manifest[manifest["split"] == "train"]["label"].astype(int).to_numpy()
    counts = np.bincount(y_train, minlength=2).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["training"]["epochs"]))
    use_amp = bool(cfg["training"]["mixed_precision"]) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    best_f1 = -1.0
    best_epoch = -1
    stale = 0
    history = []
    best_path = run_dir / "best.pt"

    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        start = time.time()
        train_out = run_epoch(model, train_loader, criterion, optimizer, device, train=True, scaler=scaler)
        val_out = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        val_metrics = compute_binary_metrics(val_out["y_true"], val_out["y_prob"], threshold=0.5)
        row = {
            "epoch": epoch,
            "train_loss": train_out["loss"],
            "validation_loss": val_out["loss"],
            "validation_f1_at_0_5": val_metrics["valid_f1"],
            "validation_recall_at_0_5": val_metrics["valid_recall"],
            "seconds": time.time() - start,
        }
        history.append(row)
        print(row)

        if val_metrics["valid_f1"] > best_f1:
            best_f1 = val_metrics["valid_f1"]
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_name": model_name,
                    "model_state": model.state_dict(),
                    "cfg": cfg,
                    "epoch": epoch,
                    "best_f1_at_0_5": best_f1,
                },
                best_path,
            )
        else:
            stale += 1
            if stale >= int(cfg["training"]["early_stopping_patience"]):
                print("Early stopping")
                break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    val_out = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
    threshold, val_threshold_metrics = choose_threshold_for_recall(
        val_out["y_true"],
        val_out["y_prob"],
        min_valid_recall=float(cfg["training"]["threshold_target_recall"]),
    )
    test_out = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    test_metrics = compute_binary_metrics(test_out["y_true"], test_out["y_prob"], threshold=threshold)

    val_df = manifest[manifest["split"] == "validation"]
    test_df = manifest[manifest["split"] == "test"]
    save_predictions(val_df, val_out["sample_ids"], val_out["y_true"], val_out["y_prob"], threshold, run_dir / "validation_predictions.csv")
    save_predictions(test_df, test_out["sample_ids"], test_out["y_true"], test_out["y_prob"], threshold, run_dir / "test_predictions.csv")

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    save_json(
        {
            "model_name": model_name,
            "best_epoch": best_epoch,
            "threshold": threshold,
            "validation": val_threshold_metrics,
            "test": test_metrics,
            "device": str(device),
        },
        run_dir / "metrics.json",
    )
    print(f"Deep training complete: {run_dir}")
    print("Test:", test_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

