"""Helpers to locate training checkpoints for resume."""

from __future__ import annotations

from pathlib import Path

from .config import ROOT_DIR


def rel_checkpoint_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR)).replace("\\", "/")
    except ValueError:
        return str(path)


def find_rfdetr_resume_checkpoint(output: Path) -> Path | None:
    if not output.is_dir():
        return None
    candidates: list[Path] = [
        output / "last.ckpt",
        output / "checkpoint_best_total.pth",
    ]
    candidates.extend(sorted(output.glob("last*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True))
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.is_file():
            return path
    return None


def find_yolo_resume_checkpoint(output: Path) -> Path | None:
    last_pt = output / "train" / "weights" / "last.pt"
    if last_pt.is_file():
        return last_pt
    return None


def find_resume_checkpoint(output: Path, model_type: str) -> Path | None:
    if model_type == "yolo":
        return find_yolo_resume_checkpoint(output)
    return find_rfdetr_resume_checkpoint(output)


def resolve_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path
