"""Normalize YOLO data.yaml paths for Ultralytics (path must be dataset root, not cwd)."""

from __future__ import annotations

from pathlib import Path

import yaml


def prepare_yolo_data_yaml(yaml_path: Path) -> Path:
    """Return a data.yaml with absolute dataset root in ``path`` (writes ``.yolo_data.yaml``)."""
    yaml_path = yaml_path.resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"data.yaml 不存在: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    dataset_root = yaml_path.parent.resolve()
    path_val = str(data.get("path", ".") or ".").strip()
    if path_val in (".", "./", ""):
        data["path"] = _posix_path(dataset_root)
    else:
        p = Path(path_val)
        if not p.is_absolute():
            data["path"] = _posix_path((dataset_root / p).resolve())
        else:
            data["path"] = _posix_path(p.resolve())

    if "valid" in data and "val" not in data:
        data["val"] = data.pop("valid")

    prepared = dataset_root / ".yolo_data.yaml"
    with prepared.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return prepared


def _posix_path(path: Path) -> str:
    return str(path).replace("\\", "/")
