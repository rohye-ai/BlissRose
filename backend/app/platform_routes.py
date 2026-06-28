from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session

from .database import SessionLocal
from .db_models import DatasetRecord, DeviceRecord, ModelRecord, TrainingJobRecord
from .platform_service import (
    approve_dataset_review,
    browse_dataset,
    capture_device_preview,
    create_device,
    create_training_job,
    dataset_to_out,
    delete_dataset,
    delete_alert,
    delete_device,
    delete_image_labels,
    delete_model,
    deploy_training_job,
    device_to_out,
    find_next_unlabeled_image,
    get_image_labels,
    get_model_lineage,
    get_training_job_log,
    list_alerts,
    list_dataset_images,
    model_to_out,
    pre_annotate_image,
    reject_dataset_review,
    resolve_dataset_file,
    review_dataset_image,
    save_image_labels,
    submit_dataset_review,
    sync_dataset_images,
    training_job_to_out,
    update_dataset_class_names,
    update_device,
    update_training_job,
    upload_dataset_zip,
    upload_model,
)
from .schemas import DatasetImageReviewRequest, DatasetReviewRequest, DeviceRecordOut, ModelRecordOut, PreAnnotateRequest, SaveLabelsRequest, WebhookConfigUpdate
from .webhook_service import get_webhook_config, test_webhook_channel, update_webhook_config

router = APIRouter(prefix="/api/platform", tags=["platform"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/models")
def list_models(db: Session = Depends(get_db)) -> dict[str, Any]:
    rows = db.query(ModelRecord).order_by(ModelRecord.created_at.desc()).all()
    return {"items": [model_to_out(db, r) for r in rows]}


def _current_username(request: Request) -> str:
    return getattr(request.state, "username", "") or ""


@router.post("/models/upload", response_model=ModelRecordOut)
async def api_upload_model(
    request: Request,
    file: UploadFile = File(...),
    name: str = Form(""),
    model_type: str = Form("rf-detr"),
    class_names: str = Form(""),
    db: Session = Depends(get_db),
):
    names = [s.strip() for s in class_names.split(",") if s.strip()]
    return await upload_model(db, file, name, model_type, names, _current_username(request))


@router.delete("/models/{model_id}")
def api_delete_model(model_id: str, db: Session = Depends(get_db)) -> dict[str, str]:
    delete_model(db, model_id)
    return {"message": "模型已删除"}


@router.get("/models/{model_id}/lineage")
def api_model_lineage(model_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    rec = db.query(ModelRecord).filter(ModelRecord.id == model_id).first()
    if not rec:
        raise HTTPException(status_code=404, detail="模型不存在")
    return {"model_id": model_id, "lineage": get_model_lineage(db, model_id)}


@router.get("/datasets")
def list_datasets(db: Session = Depends(get_db)) -> dict[str, Any]:
    rows = db.query(DatasetRecord).order_by(DatasetRecord.created_at.desc()).all()
    for rec in rows:
        if (getattr(rec, "total_count", 0) or 0) <= 0:
            sync_dataset_images(db, rec.id)
            db.refresh(rec)
    return {"items": [dataset_to_out(r) for r in rows]}


@router.post("/datasets/upload")
async def api_upload_dataset(
    request: Request,
    file: UploadFile = File(...),
    name: str = Form(""),
    class_names: str = Form(""),
    db: Session = Depends(get_db),
):
    names = [s.strip() for s in class_names.split(",") if s.strip()]
    return await upload_dataset_zip(db, file, name, _current_username(request), names or None)


@router.delete("/datasets/{dataset_id}")
def api_delete_dataset(dataset_id: str, db: Session = Depends(get_db)) -> dict[str, str]:
    delete_dataset(db, dataset_id)
    return {"message": "数据集已删除"}


@router.get("/datasets/{dataset_id}/browse")
def api_browse_dataset(
    dataset_id: str,
    split: str = Query("train"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    return browse_dataset(db, dataset_id, split, page, page_size)


@router.get("/datasets/{dataset_id}/images")
def api_list_dataset_images(
    dataset_id: str,
    split: str = Query("train"),
    status: str = Query("all"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    search: str = Query(""),
    db: Session = Depends(get_db),
):
    return list_dataset_images(db, dataset_id, split, status, page, page_size, search)


@router.get("/datasets/{dataset_id}/images/next-unlabeled")
def api_next_unlabeled_image(
    dataset_id: str,
    split: str = Query("train"),
    path: str = Query(""),
    page_size: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    return find_next_unlabeled_image(db, dataset_id, split, path, page_size)


@router.get("/datasets/{dataset_id}/labels")
def api_get_image_labels(
    dataset_id: str,
    path: str = Query(...),
    db: Session = Depends(get_db),
):
    return get_image_labels(db, dataset_id, path)


@router.put("/datasets/{dataset_id}/labels")
def api_save_image_labels(
    dataset_id: str,
    path: str = Query(...),
    body: SaveLabelsRequest = ...,
    db: Session = Depends(get_db),
):
    anns = [a.model_dump() for a in body.annotations]
    return save_image_labels(db, dataset_id, path, anns, body.class_names)


@router.delete("/datasets/{dataset_id}/labels")
def api_delete_image_labels(
    dataset_id: str,
    path: str = Query(...),
    db: Session = Depends(get_db),
):
    return delete_image_labels(db, dataset_id, path)


@router.post("/datasets/{dataset_id}/pre-annotate")
def api_pre_annotate(
    dataset_id: str,
    path: str = Query(...),
    body: PreAnnotateRequest = ...,
    db: Session = Depends(get_db),
):
    return pre_annotate_image(
        db, dataset_id, path, body.instance_id, body.confidence, body.save
    )


@router.put("/datasets/{dataset_id}/classes")
def api_update_dataset_classes(
    dataset_id: str,
    body: dict[str, Any],
    db: Session = Depends(get_db),
):
    names = body.get("class_names") or []
    return update_dataset_class_names(db, dataset_id, [str(n) for n in names])


@router.post("/datasets/{dataset_id}/review/submit")
def api_submit_dataset_review(dataset_id: str, db: Session = Depends(get_db)):
    return submit_dataset_review(db, dataset_id)


@router.post("/datasets/{dataset_id}/review/approve")
def api_approve_dataset_review(
    request: Request,
    dataset_id: str,
    db: Session = Depends(get_db),
):
    return approve_dataset_review(db, dataset_id, _current_username(request))


@router.post("/datasets/{dataset_id}/review/reject")
def api_reject_dataset_review(
    dataset_id: str,
    body: DatasetReviewRequest,
    db: Session = Depends(get_db),
):
    return reject_dataset_review(db, dataset_id, body.message)


@router.post("/datasets/{dataset_id}/images/review")
def api_review_dataset_image(
    request: Request,
    dataset_id: str,
    path: str = Query(...),
    body: DatasetImageReviewRequest = ...,
    db: Session = Depends(get_db),
):
    return review_dataset_image(db, dataset_id, path, body.action, _current_username(request))


@router.get("/datasets/{dataset_id}/file")
def api_dataset_file(dataset_id: str, path: str = Query(...)):
    target = resolve_dataset_file(dataset_id, path)
    return FileResponse(target)


@router.get("/devices")
def list_devices(db: Session = Depends(get_db)) -> dict[str, Any]:
    from .analysis_worker import analysis_worker

    rows = db.query(DeviceRecord).order_by(DeviceRecord.created_at.desc()).all()
    return {
        "items": [
            device_to_out(r, analysis_running=analysis_worker.is_device_running(r.id)) for r in rows
        ]
    }


@router.post("/devices", response_model=DeviceRecordOut)
def api_create_device(request: Request, body: dict[str, Any], db: Session = Depends(get_db)):
    if body.get("device_type") not in ("video", "image"):
        raise HTTPException(status_code=400, detail="device_type 必须为 video 或 image")
    return create_device(db, body, _current_username(request))


@router.put("/devices/{device_id}", response_model=DeviceRecordOut)
def api_update_device(device_id: str, request: Request, body: dict[str, Any], db: Session = Depends(get_db)):
    return update_device(db, device_id, body, _current_username(request))


@router.delete("/devices/{device_id}")
def api_delete_device(device_id: str, db: Session = Depends(get_db)) -> dict[str, str]:
    delete_device(db, device_id)
    return {"message": "设备已删除"}


@router.get("/devices/{device_id}/preview")
def api_device_preview(device_id: str, db: Session = Depends(get_db)):
    data = capture_device_preview(db, device_id)
    return Response(content=data, media_type="image/jpeg")


@router.get("/alerts")
def api_list_alerts(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    device_id: str | None = Query(None),
    db: Session = Depends(get_db),
):
    return list_alerts(db, page, page_size, device_id)


@router.delete("/alerts/{alert_id}")
def api_delete_alert(alert_id: str, db: Session = Depends(get_db)) -> dict[str, str]:
    delete_alert(db, alert_id)
    return {"message": "报警记录已删除"}


@router.get("/training-jobs")
def list_training_jobs(db: Session = Depends(get_db)) -> dict[str, Any]:
    rows = db.query(TrainingJobRecord).order_by(TrainingJobRecord.created_at.desc()).all()
    return {"items": [training_job_to_out(db, r) for r in rows]}


@router.get("/training-jobs/queue")
def api_training_queue() -> dict[str, Any]:
    from .training_worker import training_worker

    return training_worker.list_queue()


@router.post("/training-jobs")
def api_create_training_job(request: Request, body: dict[str, Any], db: Session = Depends(get_db)):
    return create_training_job(db, body, _current_username(request))


@router.put("/training-jobs/{job_id}")
def api_update_training_job(
    job_id: str,
    request: Request,
    body: dict[str, Any],
    db: Session = Depends(get_db),
):
    return update_training_job(db, job_id, body, _current_username(request))


@router.post("/training-jobs/{job_id}/start")
def api_start_training_job(
    job_id: str,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from .training_worker import training_worker

    resume = True if body is None else body.get("resume", True)
    try:
        training_worker.start(job_id, resume=bool(resume))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"message": training_worker.message, "queue": training_worker.list_queue()}


@router.post("/training-jobs/{job_id}/deploy", response_model=ModelRecordOut)
def api_deploy_training(job_id: str, request: Request, db: Session = Depends(get_db)):
    return deploy_training_job(db, job_id, _current_username(request))


@router.get("/training-jobs/{job_id}/log")
def api_training_job_log(job_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    return get_training_job_log(db, job_id)


@router.post("/training-jobs/{job_id}/stop")
def api_stop_training_job(job_id: str) -> dict[str, str]:
    from .training_worker import training_worker

    try:
        training_worker.stop(job_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"message": training_worker.message}


@router.get("/analysis/status")
def analysis_status() -> dict[str, Any]:
    from .analysis_worker import analysis_worker

    return {"tasks": analysis_worker.list_status()}


@router.get("/webhooks/config")
def api_get_webhook_config() -> dict[str, Any]:
    return get_webhook_config()


@router.put("/webhooks/config")
def api_update_webhook_config(body: WebhookConfigUpdate) -> dict[str, Any]:
    return update_webhook_config(body.model_dump(mode="json"))


@router.post("/webhooks/test")
def api_test_webhook(body: dict[str, Any]) -> dict[str, Any]:
    channel = body.get("channel") or body
    try:
        return test_webhook_channel(channel)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
