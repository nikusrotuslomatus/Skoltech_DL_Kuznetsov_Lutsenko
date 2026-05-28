from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import GroupShuffleSplit

from .config import resolve_path
from .utils import ensure_dir, save_json


CLASS_NAMES = {
    0: "invalid_candidate",
    1: "valid_wheel_candidate",
}


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_crop_file(dataset_root: Path, crop_path: str, label: int) -> Path:
    raw = Path(str(crop_path))
    if raw.is_absolute() and raw.exists():
        return raw

    rel = str(crop_path).replace("\\", "/")
    direct = dataset_root / rel
    if direct.exists():
        return direct

    fallback = dataset_root / CLASS_NAMES[label] / raw.name
    return fallback


def read_export_csv(dataset_name: str, dataset_root: Path) -> pd.DataFrame:
    csv_path = dataset_root / "labels.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    rows: list[dict[str, Any]] = []

    for row_idx, row in df.iterrows():
        label = int(row["label"])
        crop_file = resolve_crop_file(dataset_root, str(row["crop_path"]), label)
        rel_crop_path = (
            crop_file.relative_to(dataset_root).as_posix()
            if crop_file.exists()
            else str(row["crop_path"]).replace("\\", "/")
        )
        sample_id = f"{dataset_name}_{row_idx:06d}"

        item = row.to_dict()
        item.update(
            {
                "sample_id": sample_id,
                "dataset_name": dataset_name,
                "dataset_root": str(dataset_root),
                "crop_rel_path": rel_crop_path,
                "crop_abs_path": str(crop_file),
                "label": label,
                "label_name": CLASS_NAMES[label],
                "row_idx": int(row_idx),
            }
        )
        rows.append(item)

    return pd.DataFrame(rows)


def add_image_metadata(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    warn_min_side = int(cfg["data"]["quality"]["warn_min_side_px"])
    warn_max_side = int(cfg["data"]["quality"]["warn_max_side_px"])

    records = []
    for _, row in df.iterrows():
        path = Path(row["crop_abs_path"])
        meta = {
            "file_exists": path.exists(),
            "image_width": np.nan,
            "image_height": np.nan,
            "aspect_ratio": np.nan,
            "file_size_bytes": np.nan,
            "md5": "",
            "is_tiny": False,
            "is_huge": False,
            "is_corrupt": False,
            "quality_flags": "",
        }

        if path.exists():
            try:
                with Image.open(path) as img:
                    img = ImageOps.exif_transpose(img)
                    width, height = img.size
                meta["image_width"] = int(width)
                meta["image_height"] = int(height)
                meta["aspect_ratio"] = float(width / max(1, height))
                meta["file_size_bytes"] = int(path.stat().st_size)
                meta["md5"] = md5_file(path)
                meta["is_tiny"] = bool(width < warn_min_side or height < warn_min_side)
                meta["is_huge"] = bool(width > warn_max_side or height > warn_max_side)
            except Exception:
                meta["is_corrupt"] = True

        flags = []
        if not meta["file_exists"]:
            flags.append("missing_file")
        if meta["is_corrupt"]:
            flags.append("corrupt")
        if meta["is_tiny"]:
            flags.append("tiny_crop")
        if meta["is_huge"]:
            flags.append("huge_crop")
        meta["quality_flags"] = "|".join(flags)
        records.append(meta)

    meta_df = pd.DataFrame(records)
    return pd.concat([df.reset_index(drop=True), meta_df], axis=1)


def build_manifest(cfg: dict[str, Any]) -> pd.DataFrame:
    root = Path(cfg["_project_root"])
    frames = []
    for ds in cfg["data"]["datasets"]:
        dataset_root = resolve_path(ds["root"], root)
        frames.append(read_export_csv(ds["name"], dataset_root))
    manifest = pd.concat(frames, ignore_index=True)
    manifest = add_image_metadata(manifest, cfg)
    return manifest


def validate_manifest(df: pd.DataFrame) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    checks["rows"] = int(len(df))
    checks["label_counts"] = {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()}
    checks["missing_files"] = int((~df["file_exists"]).sum())
    checks["corrupt_images"] = int(df["is_corrupt"].sum())
    checks["tiny_crops"] = int(df["is_tiny"].sum())
    checks["huge_crops"] = int(df["is_huge"].sum())
    checks["unique_source_images"] = int(df["source_relative_path"].nunique())

    duplicate_groups = df[df["md5"].astype(str) != ""].groupby("md5")
    duplicate_count = 0
    duplicate_label_conflicts = 0
    for _, group in duplicate_groups:
        if len(group) > 1:
            duplicate_count += int(len(group))
            if group["label"].nunique() > 1:
                duplicate_label_conflicts += 1
    checks["duplicate_files_total"] = duplicate_count
    checks["duplicate_label_conflict_groups"] = duplicate_label_conflicts

    path_label_mismatches = 0
    for _, row in df.iterrows():
        expected = CLASS_NAMES[int(row["label"])]
        rel = str(row["crop_rel_path"]).replace("\\", "/")
        if not rel.startswith(expected + "/"):
            path_label_mismatches += 1
    checks["path_label_mismatches"] = path_label_mismatches
    checks["ok"] = all(
        checks[k] == 0
        for k in [
            "missing_files",
            "corrupt_images",
            "duplicate_label_conflict_groups",
            "path_label_mismatches",
        ]
    )
    return checks


def _split_score(df: pd.DataFrame, split_col: str, ratios: dict[str, float]) -> float:
    total = len(df)
    overall_valid = float(df["label"].mean())
    score = 0.0
    for split, target_ratio in ratios.items():
        part = df[df[split_col] == split]
        if part.empty or part["label"].nunique() < 2:
            return 1e9
        size_ratio = len(part) / total
        valid_ratio = float(part["label"].mean())
        score += abs(size_ratio - target_ratio) * 4.0
        score += abs(valid_ratio - overall_valid) * 2.0
    return float(score)


def make_grouped_splits(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    split_cfg = cfg["data"]["split"]
    ratios = {
        "train": float(split_cfg["train"]),
        "validation": float(split_cfg["validation"]),
        "test": float(split_cfg["test"]),
    }
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0: {ratios}")

    group_col = cfg["data"]["group_col"]
    seed = int(cfg["project"]["seed"])
    iterations = int(split_cfg.get("search_iterations", 300))

    best_df = None
    best_score = float("inf")
    df = df.reset_index(drop=True).copy()
    groups = df[group_col].astype(str).values

    for i in range(iterations):
        random_state = seed + i
        first = GroupShuffleSplit(n_splits=1, test_size=ratios["test"], random_state=random_state)
        train_val_idx, test_idx = next(first.split(df, df["label"], groups))
        train_val = df.iloc[train_val_idx].copy()
        test = df.iloc[test_idx].copy()

        relative_val_size = ratios["validation"] / (ratios["train"] + ratios["validation"])
        second = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=random_state + 100_000)
        tv_groups = train_val[group_col].astype(str).values
        train_idx, val_idx = next(second.split(train_val, train_val["label"], tv_groups))

        candidate = df.copy()
        candidate["split"] = ""
        candidate.loc[train_val.iloc[train_idx].index, "split"] = "train"
        candidate.loc[train_val.iloc[val_idx].index, "split"] = "validation"
        candidate.loc[test.index, "split"] = "test"

        score = _split_score(candidate, "split", ratios)
        if score < best_score:
            best_score = score
            best_df = candidate

    if best_df is None:
        raise RuntimeError("Could not create grouped splits")

    # Hard leakage check.
    group_splits = best_df.groupby(group_col)["split"].nunique()
    leaked = group_splits[group_splits > 1]
    if not leaked.empty:
        raise RuntimeError(f"Group leakage detected: {leaked.head().to_dict()}")

    return best_df


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("split", dropna=False)
        .agg(
            samples=("sample_id", "count"),
            valid=("label", "sum"),
            source_images=("source_relative_path", "nunique"),
            tiny=("is_tiny", "sum"),
            huge=("is_huge", "sum"),
        )
        .assign(invalid=lambda x: x["samples"] - x["valid"])
        .assign(valid_share=lambda x: x["valid"] / x["samples"])
        .reset_index()
    )


def write_processed_outputs(df: pd.DataFrame, cfg: dict[str, Any], with_splits: bool) -> dict[str, str]:
    out_dir = resolve_path(cfg["data"]["output_dir"], Path(cfg["_project_root"]))
    ensure_dir(out_dir)

    manifest_path = out_dir / (cfg["data"]["split_manifest_name"] if with_splits else cfg["data"]["manifest_name"])
    df.to_csv(manifest_path, index=False)
    outputs = {"manifest": str(manifest_path)}

    checks = validate_manifest(df)
    save_json(checks, out_dir / "manifest_checks.json")
    outputs["checks"] = str(out_dir / "manifest_checks.json")

    if with_splits:
        summary = split_summary(df)
        summary_path = out_dir / cfg["data"]["split_summary_name"]
        summary.to_csv(summary_path, index=False)
        outputs["split_summary"] = str(summary_path)

        leakage = df.groupby(cfg["data"]["group_col"])["split"].nunique().max()
        save_json({"max_splits_per_group": int(leakage)}, out_dir / "split_leakage_check.json")
        outputs["split_leakage_check"] = str(out_dir / "split_leakage_check.json")

    return outputs


def load_split_manifest(cfg: dict[str, Any]) -> pd.DataFrame:
    out_dir = resolve_path(cfg["data"]["output_dir"], Path(cfg["_project_root"]))
    manifest_path = out_dir / cfg["data"]["split_manifest_name"]
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run scripts/02_prepare_manifest.py first: {manifest_path}")
    return pd.read_csv(manifest_path)


class WheelCandidateTorchDataset:
    def __init__(self, manifest: pd.DataFrame, split: str, transform: Any | None = None) -> None:
        try:
            import torch  # noqa: F401
        except Exception as exc:
            raise RuntimeError("PyTorch is required for WheelCandidateTorchDataset") from exc

        self.df = manifest[manifest["split"] == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[Any, int, str]:
        row = self.df.iloc[idx]
        image = Image.open(row["crop_abs_path"])
        image = ImageOps.exif_transpose(image).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(row["label"])
        return image, label, str(row["sample_id"])
