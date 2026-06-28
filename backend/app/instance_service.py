from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

from fastapi import HTTPException
from sqlalchemy.orm import Session

from .audit import apply_create_audit, apply_update_audit, dt_to_str
from .db_models import AppSetting, InferenceInstanceRecord, ModelRecord, TrainingJobRecord
from .schemas import InferenceInstanceConfig, ModelSize

MODEL_SIZE_KEYS = ("nano", "small", "medium", "large", "base")


def infer_model_size(path_or_name: str) -> str:
    lower = (path_or_name or "").lower()
    for key in MODEL_SIZE_KEYS:
        if key in lower:
            return key
    return "medium"


def _json_loads(raw: str, default: Any = None) -> Any:
    if not raw:
        return default if default is not None else []
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default if default is not None else []


def record_to_config(rec: InferenceInstanceRecord) -> InferenceInstanceConfig:
    size_val = rec.size or "medium"
    try:
        size = ModelSize(size_val)
    except ValueError:
        size = ModelSize.MEDIUM
    return InferenceInstanceConfig(
        id=rec.id,
        name=rec.name,
        enabled=rec.enabled,
        model_type=rec.model_type or "rf-detr",
        model_id=rec.model_id or "",
        device_ids=_json_loads(rec.device_ids, []),
        device_id=_json_loads(rec.device_ids, [""])[0] if _json_loads(rec.device_ids, []) else "",
        size=size,
        checkpoint=rec.checkpoint or "",
        gpu_ids=_json_loads(rec.gpu_ids, [0]),
        confidence=rec.confidence,
        resolution=rec.resolution,
        optimize_inference=rec.optimize_inference,
        class_names=_json_loads(rec.class_names, []),
        created_by=rec.created_by or "",
        updated_by=rec.updated_by or "",
        created_at=dt_to_str(rec.created_at),
        updated_at=dt_to_str(rec.updated_at),
    )


def list_inference_instances(db: Session) -> list[InferenceInstanceConfig]:
    rows = db.query(InferenceInstanceRecord).order_by(InferenceInstanceRecord.created_at).all()
    return [record_to_config(r) for r in rows]


def get_inference_instance(db: Session, instance_id: str) -> InferenceInstanceConfig | None:
    rec = db.query(InferenceInstanceRecord).filter(InferenceInstanceRecord.id == instance_id).first()
    return record_to_config(rec) if rec else None


def resolve_model_fields(db: Session, body: dict[str, Any]) -> dict[str, Any]:
    from .platform_service import resolve_model_for_instance

    model_id = body.get("model_id", "")
    if not model_id:
        return body
    rec = resolve_model_for_instance(model_id, db)
    assert rec is not None
    body["checkpoint"] = rec.file_path
    body["model_type"] = rec.model_type
    body["size"] = rec.size or infer_model_size(rec.file_path or rec.name)
    names = _json_loads(rec.class_names, [])
    if names:
        body["class_names"] = names
    return body


def create_inference_instance(db: Session, body: dict[str, Any], username: str = "") -> InferenceInstanceConfig:
    from .config import config_store

    body = resolve_model_fields(db, dict(body))
    if not body.get("model_id"):
        raise HTTPException(status_code=400, detail="请选择已上传的模型")
    if not body.get("checkpoint"):
        raise HTTPException(status_code=400, detail="所选模型权重无效")

    cfg = config_store.get()
    audit = apply_create_audit(username)
    instance_id = body.get("id") or uuid.uuid4().hex[:8]
    device_ids = body.get("device_ids") or ([body["device_id"]] if body.get("device_id") else [])
    size_val = body.get("size") or cfg.model.size
    if isinstance(size_val, ModelSize):
        size_val = size_val.value

    rec = InferenceInstanceRecord(
        id=instance_id,
        name=body.get("name") or f"实例-{uuid.uuid4().hex[:4]}",
        enabled=bool(body.get("enabled", True)),
        model_id=body.get("model_id", ""),
        model_type=body.get("model_type", "rf-detr"),
        device_ids=json.dumps(device_ids, ensure_ascii=False),
        size=str(size_val),
        checkpoint=body.get("checkpoint", ""),
        gpu_ids=json.dumps(body.get("gpu_ids") or [0]),
        confidence=float(body.get("confidence", cfg.model.confidence)),
        resolution=int(body.get("resolution", cfg.model.resolution)),
        optimize_inference=bool(body.get("optimize_inference", cfg.model.optimize_inference)),
        class_names=json.dumps(body.get("class_names") or list(cfg.model.class_names), ensure_ascii=False),
        created_by=audit["created_by"],
        updated_by=audit["updated_by"],
        created_at=audit["created_at"],
        updated_at=audit["updated_at"],
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    from .config import config_store

    config_store.reload()
    return record_to_config(rec)


def update_inference_instance(
    db: Session, instance_id: str, body: dict[str, Any], username: str = ""
) -> InferenceInstanceConfig:
    rec = db.query(InferenceInstanceRecord).filter(InferenceInstanceRecord.id == instance_id).first()
    if not rec:
        raise HTTPException(status_code=404, detail=f"实例不存在: {instance_id}")

    body = resolve_model_fields(db, dict(body))
    if not body.get("model_id"):
        raise HTTPException(status_code=400, detail="请选择已上传的模型")
    if not body.get("checkpoint"):
        raise HTTPException(status_code=400, detail="所选模型权重无效")

    audit = apply_update_audit(username)
    device_ids = body.get("device_ids") or ([body["device_id"]] if body.get("device_id") else [])
    size_val = body.get("size", rec.size)
    if isinstance(size_val, ModelSize):
        size_val = size_val.value

    rec.name = body.get("name", rec.name)
    rec.enabled = bool(body.get("enabled", rec.enabled))
    rec.model_id = body.get("model_id", rec.model_id)
    rec.model_type = body.get("model_type", rec.model_type)
    rec.device_ids = json.dumps(device_ids, ensure_ascii=False)
    rec.size = str(size_val)
    rec.checkpoint = body.get("checkpoint", rec.checkpoint)
    rec.gpu_ids = json.dumps(body.get("gpu_ids") or _json_loads(rec.gpu_ids, [0]))
    rec.confidence = float(body.get("confidence", rec.confidence))
    rec.resolution = int(body.get("resolution", rec.resolution))
    rec.optimize_inference = bool(body.get("optimize_inference", rec.optimize_inference))
    if "class_names" in body:
        rec.class_names = json.dumps(body["class_names"], ensure_ascii=False)
    rec.updated_by = audit["updated_by"]
    rec.updated_at = audit["updated_at"]
    db.commit()
    db.refresh(rec)
    from .config import config_store

    config_store.reload()
    return record_to_config(rec)


def delete_inference_instance(db: Session, instance_id: str) -> None:
    from .config import config_store

    cfg = config_store.get()
    if instance_id == cfg.default_instance_id:
        raise HTTPException(status_code=400, detail="不能删除默认推理实例")
    rec = db.query(InferenceInstanceRecord).filter(InferenceInstanceRecord.id == instance_id).first()
    if not rec:
        raise HTTPException(status_code=404, detail=f"实例不存在: {instance_id}")
    db.delete(rec)
    db.commit()
    from .config import config_store

    config_store.reload()


def instance_references_model(db: Session, model_id: str, checkpoint: str = "") -> bool:
    q = db.query(InferenceInstanceRecord).filter(InferenceInstanceRecord.model_id == model_id)
    if q.count():
        return True
    if checkpoint:
        norm = checkpoint.replace("\\", "/")
        for rec in db.query(InferenceInstanceRecord).all():
            if rec.checkpoint and rec.checkpoint.replace("\\", "/") == norm:
                return True
    return False


def migrate_instances_from_json(db: Session) -> None:
    """一次性：将 app_config JSON 中的 inference_instances 迁入 DB 表。"""
    if db.query(InferenceInstanceRecord).count() > 0:
        return
    row = db.query(AppSetting).filter(AppSetting.key == "app_config").first()
    if not row:
        return
    try:
        data = json.loads(row.value)
    except json.JSONDecodeError:
        return
    instances_raw = data.get("inference_instances") or []
    if not instances_raw:
        return
    for item in instances_raw:
        inst_id = item.get("id") or uuid.uuid4().hex[:8]
        device_ids = item.get("device_ids") or ([item["device_id"]] if item.get("device_id") else [])
        size_val = item.get("size", "medium")
        if isinstance(size_val, dict):
            size_val = size_val.get("value", "medium")
        rec = InferenceInstanceRecord(
            id=inst_id,
            name=item.get("name", "推理实例"),
            enabled=bool(item.get("enabled", True)),
            model_id=item.get("model_id", ""),
            model_type=item.get("model_type", "rf-detr"),
            device_ids=json.dumps(device_ids, ensure_ascii=False),
            size=str(size_val),
            checkpoint=item.get("checkpoint", ""),
            gpu_ids=json.dumps(item.get("gpu_ids") or [0]),
            confidence=float(item.get("confidence", 0.5)),
            resolution=int(item.get("resolution", 576)),
            optimize_inference=bool(item.get("optimize_inference", True)),
            class_names=json.dumps(item.get("class_names") or [], ensure_ascii=False),
            created_by=item.get("created_by", ""),
            updated_by=item.get("updated_by", ""),
        )
        db.add(rec)
    data.pop("inference_instances", None)
    row.value = json.dumps(data, ensure_ascii=False)
    db.commit()
