from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from sqlalchemy.orm import Session

from ..config import ROOT_DIR
from ..db_models import ModelRecord

EXPORT_DIR = ROOT_DIR / "data" / "exports"


def export_model(
    db: Session,
    model_id: str,
    export_format: str = "onnx",
    imgsz: int = 640,
) -> dict[str, Any]:
    """导出模型为 ONNX 等格式，当前优先支持 YOLO。"""
    model = db.query(ModelRecord).filter(ModelRecord.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="模型不存在")

    fmt = export_format.lower()
    if fmt not in ("onnx", "torchscript", "engine"):
        raise HTTPException(status_code=400, detail=f"不支持的导出格式: {export_format}")

    ckpt = ROOT_DIR / model.file_path
    if not ckpt.exists():
        raise HTTPException(status_code=400, detail="模型权重文件不存在")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    export_id = uuid.uuid4().hex[:8]
    out_dir = EXPORT_DIR / model_id / export_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if model.model_type == "yolo":
        return _export_yolo(model, ckpt, out_dir, fmt, imgsz)
    if model.model_type == "rf-detr":
        raise HTTPException(
            status_code=400,
            detail="RF-DETR 暂不支持在线导出，请使用训练产出 checkpoint 配合官方工具链转换",
        )
    raise HTTPException(status_code=400, detail=f"未知模型类型: {model.model_type}")


def _export_yolo(
    model: ModelRecord,
    ckpt: Path,
    out_dir: Path,
    fmt: str,
    imgsz: int,
) -> dict[str, Any]:
    from ultralytics import YOLO

    yolo = YOLO(str(ckpt))
    export_path = yolo.export(format=fmt, imgsz=imgsz)
    export_path = Path(export_path)
    if export_path.exists() and export_path.parent != out_dir:
        dest = out_dir / export_path.name
        shutil.copy2(export_path, dest)
        export_path = dest

    rel = export_path.relative_to(ROOT_DIR).as_posix()
    return {
        "model_id": model.id,
        "format": fmt,
        "path": rel,
        "download_url": f"/api/platform/models/{model.id}/exports/{export_path.name}",
        "size_bytes": export_path.stat().st_size if export_path.exists() else 0,
    }
