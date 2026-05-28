from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    root = project_root()
    cfg_path = Path(path) if path else root / "configs" / "default.yaml"
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_project_root"] = str(root)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def resolve_path(value: str | Path, base: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return ((base or project_root()) / path).resolve()


def output_path(cfg: dict[str, Any], key: str) -> Path:
    root = Path(cfg["_project_root"])
    if key == "processed":
        return resolve_path(cfg["data"]["output_dir"], root)
    if key == "artifacts":
        return resolve_path(cfg["outputs"]["artifacts_dir"], root)
    if key == "reports":
        return resolve_path(cfg["outputs"]["reports_dir"], root)
    raise KeyError(key)
