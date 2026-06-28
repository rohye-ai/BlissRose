from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from sqlalchemy.orm import Session

from ..config import ROOT_DIR
from ..db_models import DatasetRecord, ModelRecord


def evaluate_model_on_dataset(
    db: Session,
    model_id: str,
    dataset_id: str | None = None,
    split: str = "valid",
) -> dict[str, Any]:
    """在验证集上评估模型，YOLO 使用 ultralytics val，RF-DETR 返回基础统计。"""
    model = db.query(ModelRecord).filter(ModelRecord.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="模型不存在")

    ckpt = ROOT_DIR / model.file_path
    if not ckpt.exists():
        raise HTTPException(status_code=400, detail="模型权重文件不存在")

    if model.model_type == "yolo":
        return _evaluate_yolo(model, db, dataset_id, split)
    return _evaluate_rfdetr(model, db, dataset_id, split)


def _resolve_dataset(db: Session, dataset_id: str | None) -> DatasetRecord | None:
    if not dataset_id:
        return None
    rec = db.query(DatasetRecord).filter(DatasetRecord.id == dataset_id).first()
    if not rec:
        raise HTTPException(status_code=404, detail="数据集不存在")
    return rec


def _evaluate_yolo(
    model: ModelRecord,
    db: Session,
    dataset_id: str | None,
    split: str,
) -> dict[str, Any]:
    from ultralytics import YOLO

    dataset = _resolve_dataset(db, dataset_id)
    ckpt = ROOT_DIR / model.file_path
    yolo = YOLO(str(ckpt))

    data_yaml = None
    if dataset and dataset.data_yaml:
        data_yaml = str(ROOT_DIR / dataset.data_yaml)
        if not Path(data_yaml).exists():
            data_yaml = str(ROOT_DIR / dataset.path / "data.yaml")

    metrics: dict[str, Any] = {"model_type": "yolo", "split": split}
    if data_yaml and Path(data_yaml).exists():
        result = yolo.val(data=data_yaml, split=split, verbose=False)
        if hasattr(result, "box"):
            box = result.box
            metrics.update(
                {
                    "map50": round(float(getattr(box, "map50", 0) or 0), 4),
                    "map50_95": round(float(getattr(box, "map", 0) or 0), 4),
                    "precision": round(float(getattr(box, "mp", 0) or 0), 4),
                    "recall": round(float(getattr(box, "mr", 0) or 0), 4),
                }
            )
        metrics["data_yaml"] = data_yaml
    else:
        metrics["message"] = "未指定有效 data.yaml，仅返回模型信息"
        metrics["class_names"] = json.loads(model.class_names or "[]")

    _save_model_metrics(db, model.id, metrics)
    return {"model_id": model.id, "metrics": metrics}


def _evaluate_rfdetr(
    model: ModelRecord,
    db: Session,
    dataset_id: str | None,
    split: str,
) -> dict[str, Any]:
    dataset = _resolve_dataset(db, dataset_id)
    metrics: dict[str, Any] = {
        "model_type": "rf-detr",
        "split": split,
        "message": "RF-DETR 离线评估需结合训练日志；此处返回数据集规模统计",
    }
    if dataset:
        metrics["dataset_id"] = dataset.id
        metrics["train_count"] = dataset.train_count
        metrics["valid_count"] = dataset.valid_count
        metrics["test_count"] = dataset.test_count
        metrics["class_names"] = json.loads(dataset.class_names or "[]")

    job_metrics_path = None

    _save_model_metrics(db, model.id, metrics)
    return {"model_id": model.id, "metrics": metrics}


def _save_model_metrics(db: Session, model_id: str, metrics: dict[str, Any]) -> None:
    rec = db.query(ModelRecord).filter(ModelRecord.id == model_id).first()
    if rec:
        rec.metrics_json = json.dumps(metrics, ensure_ascii=False)
        db.commit()
