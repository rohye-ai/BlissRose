from __future__ import annotations

import io
import json
import shutil
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import cv2
import httpx
import yaml
from fastapi import HTTPException, UploadFile
from PIL import Image
from sqlalchemy.orm import Session

from .audit import apply_create_audit, apply_update_audit, dt_to_str
from .config import ROOT_DIR, config_store
from .db_models import (
    AlertRecord,
    DatasetRecord,
    DeviceRecord,
    ModelLineage,
    ModelRecord,
    TrainingJobRecord,
)
from .model_manager import infer_model_size
from .schemas import (
    AlertRecordOut,
    DatasetRecordOut,
    DetectionItem,
    DeviceRecordOut,
    ModelRecordOut,
    RoiRegion,
    TrainState,
    TrainingJobOut,
)

MODELS_DIR = ROOT_DIR / "models"
DATASETS_DIR = ROOT_DIR / "datasets"
TRAIN_OUTPUT_BASE = ROOT_DIR / "outputs" / "train"
ALERTS_DIR = ROOT_DIR / "data" / "results" / "alerts"


def ensure_platform_dirs() -> None:
    for d in (MODELS_DIR, DATASETS_DIR, TRAIN_OUTPUT_BASE, ALERTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _json_loads(raw: str, default: Any = None) -> Any:
    if not raw:
        return default if default is not None else []
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default if default is not None else []


def _dt_str(dt: datetime | None) -> str:
    return dt_to_str(dt)


def _rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _count_images(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(1 for f in folder.rglob("*") if f.suffix.lower() in exts)


def _parse_class_names_from_yaml(data_yaml: Path) -> list[str]:
    if not data_yaml.exists():
        return []
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    names = data.get("names")
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=lambda x: int(x) if str(x).isdigit() else x)]
    if isinstance(names, list):
        return [str(n) for n in names]
    return []


def _model_in_use(db: Session, model_id: str) -> bool:
    rec = db.query(ModelRecord).filter(ModelRecord.id == model_id).first()
    cfg = config_store.get()
    for inst in cfg.inference_instances:
        if inst.model_id == model_id:
            return True
        if rec and inst.checkpoint and inst.checkpoint.replace("\\", "/") == rec.file_path.replace("\\", "/"):
            return True
    jobs = db.query(TrainingJobRecord).filter(
        TrainingJobRecord.model_id == model_id,
        TrainingJobRecord.state.in_(["pending", "running"]),
    ).count()
    return jobs > 0


def model_to_out(db: Session, rec: ModelRecord) -> ModelRecordOut:
    return ModelRecordOut(
        id=rec.id,
        name=rec.name,
        model_type=rec.model_type,
        file_path=rec.file_path,
        class_names=_json_loads(rec.class_names, []),
        parent_id=rec.parent_id,
        source=rec.source,
        version=rec.version,
        in_use=_model_in_use(db, rec.id),
        uploaded_by=rec.uploaded_by or "",
        created_at=_dt_str(rec.created_at),
    )


def dataset_to_out(rec: DatasetRecord) -> DatasetRecordOut:
    return DatasetRecordOut(
        id=rec.id,
        name=rec.name,
        path=rec.path,
        data_yaml=rec.data_yaml,
        class_names=_json_loads(rec.class_names, []),
        train_count=rec.train_count,
        valid_count=rec.valid_count,
        test_count=rec.test_count,
        uploaded_by=rec.uploaded_by or "",
        created_at=_dt_str(rec.created_at),
    )


def device_to_out(rec: DeviceRecord, analysis_running: bool = False) -> DeviceRecordOut:
    roi_raw = _json_loads(rec.roi, [])
    roi = [RoiRegion(**r) for r in roi_raw if isinstance(r, dict)]
    return DeviceRecordOut(
        id=rec.id,
        name=rec.name,
        device_type=rec.device_type,
        source=rec.source,
        poll_interval=rec.poll_interval,
        roi=roi,
        enabled=rec.enabled,
        analysis_running=analysis_running,
        created_by=rec.created_by or "",
        updated_by=rec.updated_by or "",
        created_at=_dt_str(rec.created_at),
        updated_at=_dt_str(rec.updated_at),
    )


def alert_to_out(db: Session, rec: AlertRecord) -> AlertRecordOut:
    device = db.query(DeviceRecord).filter(DeviceRecord.id == rec.device_id).first()
    cfg = config_store.get()
    inst_cfg = next((i for i in cfg.inference_instances if i.id == rec.instance_id), None)
    dets_raw = _json_loads(rec.detections, [])
    detections = [DetectionItem(**d) for d in dets_raw if isinstance(d, dict)]
    rel = rec.image_path.replace("\\", "/")
    return AlertRecordOut(
        id=rec.id,
        device_id=rec.device_id,
        device_name=device.name if device else rec.device_id,
        instance_id=rec.instance_id,
        instance_name=inst_cfg.name if inst_cfg else rec.instance_id,
        image_url=f"/static/alerts/{Path(rec.image_path).name}" if rec.image_path else "",
        detections=detections,
        max_confidence=rec.max_confidence,
        alert_at=_dt_str(rec.alert_at),
    )


def training_job_to_out(db: Session, job: TrainingJobRecord) -> TrainingJobOut:
    model = db.query(ModelRecord).filter(ModelRecord.id == job.model_id).first()
    dataset = db.query(DatasetRecord).filter(DatasetRecord.id == job.dataset_id).first()
    return TrainingJobOut(
        id=job.id,
        name=job.name,
        model_id=job.model_id,
        model_name=model.name if model else job.model_id,
        dataset_id=job.dataset_id,
        dataset_name=dataset.name if dataset else job.dataset_id,
        output_dir=job.output_dir,
        state=job.state,
        epochs=job.epochs,
        batch_size=job.batch_size,
        grad_accum_steps=job.grad_accum_steps,
        lr=job.lr,
        gpu_ids=_json_loads(job.gpu_ids, [0]),
        checkpoint_path=job.checkpoint_path,
        deployed_model_id=job.deployed_model_id,
        message=job.message,
        created_by=job.created_by or "",
        updated_by=job.updated_by or "",
        created_at=_dt_str(job.created_at),
        updated_at=_dt_str(job.updated_at),
        completed_at=_dt_str(job.completed_at) if job.completed_at else None,
    )


def rebuild_model_lineage(db: Session, model_id: str, parent_id: str | None) -> None:
    db.query(ModelLineage).filter(ModelLineage.model_id == model_id).delete()
    db.add(ModelLineage(model_id=model_id, ancestor_id=model_id, depth=0))
    if not parent_id:
        return
    ancestors = (
        db.query(ModelLineage)
        .filter(ModelLineage.model_id == parent_id)
        .order_by(ModelLineage.depth)
        .all()
    )
    for anc in ancestors:
        db.add(ModelLineage(model_id=model_id, ancestor_id=anc.ancestor_id, depth=anc.depth + 1))


def get_model_lineage(db: Session, model_id: str) -> list[dict[str, Any]]:
    rows = (
        db.query(ModelLineage, ModelRecord)
        .join(ModelRecord, ModelLineage.ancestor_id == ModelRecord.id)
        .filter(ModelLineage.model_id == model_id)
        .order_by(ModelLineage.depth)
        .all()
    )
    return [
        {
            "depth": lin.depth,
            "model_id": m.id,
            "name": m.name,
            "model_type": m.model_type,
            "version": m.version,
            "source": m.source,
            "created_at": _dt_str(m.created_at),
        }
        for lin, m in rows
    ]


def resolve_model_for_instance(model_id: str, db: Session) -> ModelRecord | None:
    if not model_id:
        return None
    rec = db.query(ModelRecord).filter(ModelRecord.id == model_id).first()
    if not rec:
        raise HTTPException(status_code=404, detail=f"模型不存在: {model_id}")
    return rec


def apply_model_to_instance_config(body: dict[str, Any], db: Session) -> dict[str, Any]:
    model_id = body.get("model_id", "")
    if not model_id:
        return body
    rec = resolve_model_for_instance(model_id, db)
    assert rec is not None
    body["checkpoint"] = rec.file_path
    body["size"] = rec.size or infer_model_size(rec.file_path or rec.name)
    names = _json_loads(rec.class_names, [])
    if names:
        body["class_names"] = names
    return body


async def upload_model(
    db: Session,
    file: UploadFile,
    name: str,
    model_type: str,
    class_names: list[str],
    uploaded_by: str = "",
) -> ModelRecordOut:
    if model_type not in ("yolo", "rf-detr"):
        raise HTTPException(status_code=400, detail="model_type 必须为 yolo 或 rf-detr")
    model_id = uuid.uuid4().hex[:8]
    dest_dir = MODELS_DIR / model_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = file.filename or "weights.bin"
    dest_path = dest_dir / filename
    content = await file.read()
    dest_path.write_bytes(content)

    inferred_size = infer_model_size(filename)

    rec = ModelRecord(
        id=model_id,
        name=name or filename,
        model_type=model_type,
        file_path=_rel_path(dest_path),
        size=inferred_size,
        class_names=json.dumps(class_names, ensure_ascii=False),
        source="upload",
        version="v1",
        uploaded_by=uploaded_by or "",
    )
    db.add(rec)
    rebuild_model_lineage(db, model_id, None)
    db.commit()
    db.refresh(rec)
    return model_to_out(db, rec)


def delete_model(db: Session, model_id: str) -> None:
    rec = db.query(ModelRecord).filter(ModelRecord.id == model_id).first()
    if not rec:
        raise HTTPException(status_code=404, detail="模型不存在")
    if _model_in_use(db, model_id):
        raise HTTPException(status_code=400, detail="模型已被推理实例或训练任务引用，无法删除")
    children = db.query(ModelRecord).filter(ModelRecord.parent_id == model_id).count()
    if children:
        raise HTTPException(status_code=400, detail="存在派生模型，无法删除")
    cfg = config_store.get()
    for inst in cfg.inference_instances:
        if inst.checkpoint and inst.checkpoint.replace("\\", "/") == rec.file_path.replace("\\", "/"):
            raise HTTPException(status_code=400, detail="模型权重已被推理实例引用，无法删除")

    model_dir = ROOT_DIR / rec.file_path
    if model_dir.parent.name == model_id and model_dir.parent.parent.name == "models":
        shutil.rmtree(model_dir.parent, ignore_errors=True)
    elif model_dir.exists():
        model_dir.unlink(missing_ok=True)

    db.query(ModelLineage).filter(
        (ModelLineage.model_id == model_id) | (ModelLineage.ancestor_id == model_id)
    ).delete()
    db.delete(rec)
    db.commit()


async def upload_dataset_zip(db: Session, file: UploadFile, name: str, uploaded_by: str = "") -> DatasetRecordOut:
    dataset_id = uuid.uuid4().hex[:8]
    dest_dir = DATASETS_DIR / dataset_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "upload.zip"
    content = await file.read()
    zip_path.write_bytes(content)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
    except zipfile.BadZipFile as exc:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="无效的 ZIP 文件") from exc
    finally:
        zip_path.unlink(missing_ok=True)

    data_yaml = dest_dir / "data.yaml"
    if not data_yaml.exists():
        for candidate in dest_dir.rglob("data.yaml"):
            data_yaml = candidate
            break
    if not data_yaml.exists():
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="ZIP 中缺少 data.yaml")

    for sub in ("train", "valid", "test"):
        if not (dest_dir / sub).is_dir() and not any(dest_dir.rglob(sub)):
            pass
    train_count = _count_images(dest_dir / "train")
    valid_count = _count_images(dest_dir / "valid")
    test_count = _count_images(dest_dir / "test")
    class_names = _parse_class_names_from_yaml(data_yaml)

    rec = DatasetRecord(
        id=dataset_id,
        name=name or f"数据集-{dataset_id[:4]}",
        path=_rel_path(dest_dir),
        data_yaml=_rel_path(data_yaml),
        class_names=json.dumps(class_names, ensure_ascii=False),
        train_count=train_count,
        valid_count=valid_count,
        test_count=test_count,
        uploaded_by=uploaded_by or "",
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return dataset_to_out(rec)


def delete_dataset(db: Session, dataset_id: str) -> None:
    rec = db.query(DatasetRecord).filter(DatasetRecord.id == dataset_id).first()
    if not rec:
        raise HTTPException(status_code=404, detail="数据集不存在")
    active = db.query(TrainingJobRecord).filter(
        TrainingJobRecord.dataset_id == dataset_id,
        TrainingJobRecord.state.in_(["pending", "running"]),
    ).count()
    if active:
        raise HTTPException(status_code=400, detail="数据集正在被训练任务使用")
    dest = ROOT_DIR / rec.path
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    db.delete(rec)
    db.commit()


def browse_dataset(db: Session, dataset_id: str, split: str = "train", page: int = 1, page_size: int = 20) -> dict[str, Any]:
    rec = db.query(DatasetRecord).filter(DatasetRecord.id == dataset_id).first()
    if not rec:
        raise HTTPException(status_code=404, detail="数据集不存在")
    if split not in ("train", "valid", "test"):
        raise HTTPException(status_code=400, detail="split 必须为 train/valid/test")
    all_items = _collect_dataset_images(ROOT_DIR / rec.path, split, dataset_id)
    total = len(all_items)
    start = (page - 1) * page_size
    slice_items = all_items[start : start + page_size]
    return {"split": split, "total": total, "page": page, "page_size": page_size, "items": slice_items}


def _split_images_dir(dataset_path: Path, split: str) -> Path:
    base = dataset_path / split
    return base / "images" if (base / "images").is_dir() else base


def _label_file_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    if image_path.parent.name == "images":
        return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    return image_path.parent / "labels" / f"{image_path.stem}.txt"


def _collect_dataset_images(dataset_path: Path, split: str, dataset_id: str) -> list[dict[str, Any]]:
    images_dir = _split_images_dir(dataset_path, split)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(f for f in images_dir.rglob("*") if f.suffix.lower() in exts)
    items: list[dict[str, Any]] = []
    for img in images:
        rel = _rel_path(img)
        label_file = _label_file_for_image(img)
        items.append({
            "path": rel,
            "name": img.name,
            "url": f"/api/platform/datasets/{dataset_id}/file?path={quote(rel, safe='')}",
            "has_labels": label_file.is_file() and label_file.stat().st_size > 0,
        })
    return items


def list_dataset_images(db: Session, dataset_id: str, split: str = "train") -> dict[str, Any]:
    rec = db.query(DatasetRecord).filter(DatasetRecord.id == dataset_id).first()
    if not rec:
        raise HTTPException(status_code=404, detail="数据集不存在")
    if split not in ("train", "valid", "test"):
        raise HTTPException(status_code=400, detail="split 必须为 train/valid/test")
    items = _collect_dataset_images(ROOT_DIR / rec.path, split, dataset_id)
    labeled = sum(1 for i in items if i.get("has_labels"))
    return {
        "dataset_id": dataset_id,
        "dataset_name": rec.name,
        "split": split,
        "total": len(items),
        "labeled_count": labeled,
        "class_names": _json_loads(rec.class_names, []),
        "items": items,
    }


def _parse_yolo_label_file(label_path: Path, class_names: list[str]) -> list[dict[str, Any]]:
    if not label_path.is_file():
        return []
    annotations: list[dict[str, Any]] = []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            cx, cy, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        except ValueError:
            continue
        name = class_names[cid] if 0 <= cid < len(class_names) else f"class_{cid}"
        annotations.append({
            "class_id": cid,
            "class_name": name,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
        })
    return annotations


def get_image_labels(db: Session, dataset_id: str, image_path: str) -> dict[str, Any]:
    rec = db.query(DatasetRecord).filter(DatasetRecord.id == dataset_id).first()
    if not rec:
        raise HTTPException(status_code=404, detail="数据集不存在")
    target = resolve_dataset_file(dataset_id, image_path)
    if target.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        raise HTTPException(status_code=400, detail="不是有效的图片路径")
    class_names = _json_loads(rec.class_names, [])
    label_path = _label_file_for_image(target)
    annotations = _parse_yolo_label_file(label_path, class_names)
    return {
        "path": image_path,
        "name": target.name,
        "has_labels": bool(annotations),
        "label_path": _rel_path(label_path) if label_path.is_file() else "",
        "class_names": class_names,
        "annotations": annotations,
    }


def resolve_dataset_file(dataset_id: str, path: str) -> Path:
    rec_path = (ROOT_DIR / "datasets" / dataset_id).resolve()
    target = (ROOT_DIR / path).resolve()
    if not str(target).startswith(str(rec_path)):
        raise HTTPException(status_code=403, detail="非法路径")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return target


def create_device(db: Session, body: dict[str, Any], username: str = "") -> DeviceRecordOut:
    device_id = uuid.uuid4().hex[:8]
    roi = body.get("roi") or []
    audit = apply_create_audit(username)
    rec = DeviceRecord(
        id=device_id,
        name=body.get("name") or f"设备-{device_id[:4]}",
        device_type=body.get("device_type", "video"),
        source=body.get("source", ""),
        poll_interval=int(body.get("poll_interval", 5)),
        roi=json.dumps(roi, ensure_ascii=False),
        enabled=bool(body.get("enabled", True)),
        created_by=audit["created_by"],
        updated_by=audit["updated_by"],
        created_at=audit["created_at"],
        updated_at=audit["updated_at"],
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return device_to_out(rec)


def update_device(db: Session, device_id: str, body: dict[str, Any], username: str = "") -> DeviceRecordOut:
    rec = db.query(DeviceRecord).filter(DeviceRecord.id == device_id).first()
    if not rec:
        raise HTTPException(status_code=404, detail="设备不存在")
    if "name" in body:
        rec.name = body["name"]
    if "device_type" in body:
        rec.device_type = body["device_type"]
    if "source" in body:
        rec.source = body["source"]
    if "poll_interval" in body:
        rec.poll_interval = int(body["poll_interval"])
    if "roi" in body:
        validated = []
        for item in body["roi"] or []:
            if isinstance(item, dict):
                validated.append(RoiRegion(**item).model_dump())
        rec.roi = json.dumps(validated, ensure_ascii=False)
    if "enabled" in body:
        rec.enabled = bool(body["enabled"])
    audit = apply_update_audit(username)
    rec.updated_at = audit["updated_at"]
    rec.updated_by = audit["updated_by"]
    db.commit()
    db.refresh(rec)
    from .analysis_worker import analysis_worker

    running = analysis_worker.is_device_running(device_id)
    return device_to_out(rec, analysis_running=running)


def delete_device(db: Session, device_id: str) -> None:
    from .analysis_worker import analysis_worker

    if analysis_worker.is_device_running(device_id):
        raise HTTPException(status_code=400, detail="设备正在分析中，请先停止")
    cfg = config_store.get()
    for inst in cfg.inference_instances:
        bound = list(inst.device_ids or [])
        if inst.device_id and inst.device_id not in bound:
            bound.append(inst.device_id)
        if device_id in bound:
            raise HTTPException(status_code=400, detail="设备已被推理实例绑定，无法删除")
    rec = db.query(DeviceRecord).filter(DeviceRecord.id == device_id).first()
    if not rec:
        raise HTTPException(status_code=404, detail="设备不存在")
    db.delete(rec)
    db.commit()


def _resize_preview_image(image: Image.Image, max_side: int = 1280) -> Image.Image:
    w, h = image.size
    if max(w, h) <= max_side:
        return image
    scale = max_side / max(w, h)
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def _fetch_image_from_url(url: str) -> Image.Image:
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")


def _capture_video_frame(source: str) -> Image.Image:
    cap_source: str | int = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail=f"无法连接视频源: {source}")
    try:
        frame = None
        for _ in range(15):
            ok, candidate = cap.read()
            if ok and candidate is not None:
                frame = candidate
                break
        if frame is None:
            raise HTTPException(status_code=400, detail="无法从视频源读取画面，请检查地址或稍后重试")
        rgb = frame[:, :, ::-1]
        return Image.fromarray(rgb)
    finally:
        cap.release()


def capture_device_preview(db: Session, device_id: str) -> bytes:
    rec = db.query(DeviceRecord).filter(DeviceRecord.id == device_id).first()
    if not rec:
        raise HTTPException(status_code=404, detail="设备不存在")
    source = (rec.source or "").strip()
    if not source:
        raise HTTPException(status_code=400, detail="设备源地址未配置")

    try:
        if rec.device_type == "image":
            image = _fetch_image_from_url(source)
        else:
            image = _capture_video_frame(source)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"获取画面失败: {exc}") from exc

    image = _resize_preview_image(image)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def create_alert(
    db: Session,
    device_id: str,
    instance_id: str,
    image_bytes: bytes,
    detections: list[DetectionItem],
    max_confidence: float,
) -> AlertRecord:
    alert_id = uuid.uuid4().hex[:8]
    ALERTS_DIR.mkdir(parents=True, exist_ok=True)
    img_path = ALERTS_DIR / f"{alert_id}.jpg"
    img_path.write_bytes(image_bytes)
    rec = AlertRecord(
        id=alert_id,
        device_id=device_id,
        instance_id=instance_id,
        image_path=_rel_path(img_path),
        detections=json.dumps([d.model_dump() for d in detections], ensure_ascii=False),
        max_confidence=max_confidence,
        alert_at=datetime.utcnow(),
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return rec


def list_alerts(db: Session, page: int = 1, page_size: int = 20, device_id: str | None = None) -> dict[str, Any]:
    q = db.query(AlertRecord)
    if device_id:
        q = q.filter(AlertRecord.device_id == device_id)
    total = q.count()
    rows = q.order_by(AlertRecord.alert_at.desc()).offset((page - 1) * page_size).limit(page_size).all()
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": [alert_to_out(db, r) for r in rows],
    }


def delete_alert(db: Session, alert_id: str) -> None:
    rec = db.query(AlertRecord).filter(AlertRecord.id == alert_id).first()
    if not rec:
        raise HTTPException(status_code=404, detail="报警记录不存在")
    if rec.image_path:
        img = ROOT_DIR / rec.image_path.replace("\\", "/")
        if not img.is_file():
            img = ALERTS_DIR / Path(rec.image_path).name
        if img.is_file():
            try:
                img.unlink()
            except OSError:
                pass
    db.delete(rec)
    db.commit()


def get_training_job_log(db: Session, job_id: str) -> dict[str, Any]:
    from .training_worker import training_worker

    job = db.query(TrainingJobRecord).filter(TrainingJobRecord.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="训练任务不存在")
    log_path = ROOT_DIR / job.output_dir / "train.log"
    content = ""
    if log_path.is_file():
        try:
            content = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            content = f"读取日志失败: {exc}"
    live = training_worker.current_job_id == job_id and training_worker.state == TrainState.RUNNING
    tail = training_worker.progress.get("log_tail", []) if live else []
    if live and tail:
        live_text = "\n".join(tail)
        if not content.endswith(live_text):
            content = (content + "\n" + live_text).strip() if content else live_text
    return {
        "job_id": job_id,
        "state": job.state,
        "live": live,
        "log_path": str(log_path.relative_to(ROOT_DIR)).replace("\\", "/") if log_path.is_file() else "",
        "content": content or ("暂无日志" if job.state == "pending" else ""),
        "message": job.message,
        "progress": training_worker.progress if live else {},
    }


def create_training_job(db: Session, body: dict[str, Any], username: str = "") -> TrainingJobOut:
    model_id = body.get("model_id")
    dataset_id = body.get("dataset_id")
    if not model_id or not dataset_id:
        raise HTTPException(status_code=400, detail="请选择模型和数据集")
    model = db.query(ModelRecord).filter(ModelRecord.id == model_id).first()
    dataset = db.query(DatasetRecord).filter(DatasetRecord.id == dataset_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="模型不存在")
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")
    if model.model_type != "rf-detr":
        raise HTTPException(status_code=400, detail="当前仅支持 RF-DETR 模型训练")

    job_id = uuid.uuid4().hex[:8]
    output_dir = TRAIN_OUTPUT_BASE / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    gpu_ids = body.get("gpu_ids") or [0]
    audit = apply_create_audit(username)
    job = TrainingJobRecord(
        id=job_id,
        name=body.get("name") or f"训练-{job_id[:4]}",
        model_id=model_id,
        dataset_id=dataset_id,
        output_dir=_rel_path(output_dir),
        state="pending",
        epochs=int(body.get("epochs", 50)),
        batch_size=int(body.get("batch_size", 4)),
        grad_accum_steps=int(body.get("grad_accum_steps", 4)),
        lr=float(body.get("lr", 1e-4)),
        gpu_ids=json.dumps(gpu_ids),
        created_by=audit["created_by"],
        updated_by=audit["updated_by"],
        created_at=audit["created_at"],
        updated_at=audit["updated_at"],
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return training_job_to_out(db, job)


def deploy_training_job(db: Session, job_id: str, uploaded_by: str = "") -> ModelRecordOut:
    job = db.query(TrainingJobRecord).filter(TrainingJobRecord.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="训练任务不存在")
    if job.state != "completed":
        raise HTTPException(status_code=400, detail="仅已完成的训练任务可部署")
    if job.deployed_model_id:
        existing = db.query(ModelRecord).filter(ModelRecord.id == job.deployed_model_id).first()
        if existing:
            return model_to_out(db, existing)

    checkpoint = job.checkpoint_path
    ckpt_path = ROOT_DIR / checkpoint if checkpoint else None
    if not ckpt_path or not ckpt_path.exists():
        output = ROOT_DIR / job.output_dir
        for name in ("checkpoint_best_regular.pth", "checkpoint_best_total.pth", "best.pt"):
            candidate = output / name
            if candidate.exists():
                ckpt_path = candidate
                checkpoint = _rel_path(candidate)
                break
    if not ckpt_path or not ckpt_path.exists():
        raise HTTPException(status_code=400, detail="未找到训练产出权重文件")

    source_model = db.query(ModelRecord).filter(ModelRecord.id == job.model_id).first()
    model_id = uuid.uuid4().hex[:8]
    dest_dir = MODELS_DIR / model_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / ckpt_path.name
    shutil.copy2(ckpt_path, dest_file)

    sibling_count = db.query(ModelRecord).filter(ModelRecord.parent_id == job.model_id).count()
    version = f"v{sibling_count + 1}"
    class_names = source_model.class_names if source_model else "[]"

    rec = ModelRecord(
        id=model_id,
        name=f"{job.name}-部署",
        model_type=source_model.model_type if source_model else "rf-detr",
        file_path=_rel_path(dest_file),
        size=source_model.size if source_model else infer_model_size(str(ckpt_path)),
        class_names=class_names,
        parent_id=job.model_id,
        source="deploy",
        version=version,
        uploaded_by=uploaded_by or "",
    )
    db.add(rec)
    rebuild_model_lineage(db, model_id, job.model_id)
    job.deployed_model_id = model_id
    job.checkpoint_path = checkpoint
    audit = apply_update_audit(uploaded_by or "")
    job.updated_at = audit["updated_at"]
    job.updated_by = audit["updated_by"]
    db.commit()
    db.refresh(rec)
    return model_to_out(db, rec)
