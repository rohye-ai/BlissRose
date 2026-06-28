from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .config import ROOT_DIR
from .database import SessionLocal
from .events import broadcast
from .platform_service import set_model_stage
from .schemas import BatchInferUrlRequest, ModelEvaluateRequest, ModelExportRequest, ModelRecordOut, ModelStageUpdate
from .services.evaluation_service import evaluate_model_on_dataset
from .services.export_service import export_model
from .services.inference_service import batch_infer_images, batch_infer_urls

router = APIRouter(prefix="/api/platform", tags=["ml-capabilities"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _resolve_instance_id(instance_id: str | None) -> str:
    from .config import config_store

    cfg = config_store.get()
    return instance_id or cfg.default_instance_id


@router.post("/infer/batch/urls")
async def api_batch_infer_urls(body: BatchInferUrlRequest) -> dict[str, Any]:
    target = _resolve_instance_id(body.instance_id)
    try:
        result = await batch_infer_urls(body.urls, target, body.confidence, body.annotate)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await broadcast("batch_inference_done", {"instance_id": target, "success": result["success"]})
    return result


@router.post("/infer/batch/images")
async def api_batch_infer_images(
    files: list[UploadFile] = File(...),
    instance_id: str | None = Form(None),
    confidence: float | None = Form(None),
    annotate: bool = Form(True),
) -> dict[str, Any]:
    target = _resolve_instance_id(instance_id)
    payload: list[tuple[str, bytes]] = []
    for f in files:
        content = await f.read()
        payload.append((f.filename or "upload", content))
    try:
        result = await batch_infer_images(payload, target, confidence, annotate)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await broadcast("batch_inference_done", {"instance_id": target, "success": result["success"]})
    return result


@router.post("/models/{model_id}/evaluate")
def api_evaluate_model(
    model_id: str,
    body: ModelEvaluateRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    return evaluate_model_on_dataset(db, model_id, body.dataset_id, body.split)


@router.post("/models/{model_id}/export")
def api_export_model(
    model_id: str,
    body: ModelExportRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    return export_model(db, model_id, body.format, body.imgsz)


@router.get("/models/{model_id}/exports/{filename}")
def api_download_export(model_id: str, filename: str) -> FileResponse:
    path = ROOT_DIR / "data" / "exports" / model_id
    file_path = path / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="导出文件不存在")
    return FileResponse(path=str(file_path), filename=filename)


@router.patch("/models/{model_id}/stage", response_model=ModelRecordOut)
def api_set_model_stage(
    model_id: str,
    body: ModelStageUpdate,
    db: Session = Depends(get_db),
) -> ModelRecordOut:
    return set_model_stage(db, model_id, body.stage.value)
