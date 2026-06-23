from __future__ import annotations



import asyncio

import io

import json

import uuid

from pathlib import Path

from typing import Any



import httpx

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect

from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import HTMLResponse, StreamingResponse

from fastapi.staticfiles import StaticFiles

from PIL import Image



from .audit import apply_create_audit_iso, apply_update_audit_iso
from .auth import decode_token
from .config import ROOT_DIR, config_store
from .database import SessionLocal
from .db_models import User
from .gpu_monitor import get_gpu_info
from .middleware import auth_http_middleware
from .model_manager import model_manager
from .rbac_routes import admin_router, router as auth_router
from .schemas import AppConfig, InferUrlRequest, InferenceInstanceConfig, StatusResponse
from .seed import init_database
from .platform_routes import router as platform_router
from .platform_service import apply_model_to_instance_config, ensure_platform_dirs
from .analysis_worker import analysis_worker
from .training_worker import training_worker

from .video_worker import mjpeg_generator, video_worker



app = FastAPI(

    title="RF-DETR Platform",

    description="基于 RF-DETR 的目标检测推理与训练平台（Transformer 端到端检测，无需 NMS）",

    version="1.1.0",

)



app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],

)


@app.middleware("http")
async def auth_middleware(request, call_next):
    return await auth_http_middleware(request, call_next)


app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(platform_router)



FRONTEND_DIR = ROOT_DIR / "frontend"

STATIC_DIR = ROOT_DIR / "data" / "results"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "alerts").mkdir(parents=True, exist_ok=True)



app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if FRONTEND_DIR.exists():

    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR)), name="assets")



_ws_clients: set[WebSocket] = set()





async def broadcast(event: str, data: dict[str, Any]) -> None:

    message = json.dumps({"event": event, "data": data}, ensure_ascii=False)

    dead: list[WebSocket] = []

    for ws in list(_ws_clients):

        try:

            await ws.send_text(message)

        except Exception:

            dead.append(ws)

    for ws in dead:

        _ws_clients.discard(ws)





def _status() -> StatusResponse:

    cfg = config_store.get()
    default = None
    try:
        default = model_manager.get_default()
    except KeyError:
        pass

    return StatusResponse(

        model_state=default.state if default else model_manager.state,

        model_message=default.message if default else model_manager.message,

        video_state=video_worker.state,

        video_message=video_worker.message,

        train_state=training_worker.state,

        train_message=training_worker.message,

        train_progress=training_worker.progress,

        config=cfg,

        device=default.device if default else model_manager.device,

        last_inference_ms=default.last_inference_ms if default else model_manager.last_inference_ms,

        video_fps=video_worker.fps if video_worker.fps > 0 else None,

        gpus=get_gpu_info(),

        instances=model_manager.list_status(),

    )





@app.on_event("startup")

async def startup() -> None:

    db = SessionLocal()

    try:

        init_database(db)

    finally:

        db.close()

    config_store.reload()

    config_store.ensure_dirs()

    ensure_platform_dirs()

    model_manager.reload_config()





@app.get("/login", response_class=HTMLResponse)

async def login_page() -> HTMLResponse:

    html_path = FRONTEND_DIR / "login.html"

    if not html_path.exists():

        raise HTTPException(status_code=404, detail="frontend/login.html not found")

    return HTMLResponse(

        html_path.read_text(encoding="utf-8"),

        headers={"Cache-Control": "no-cache"},

    )





@app.get("/", response_class=HTMLResponse)

async def index() -> HTMLResponse:

    html_path = FRONTEND_DIR / "index.html"

    if not html_path.exists():

        raise HTTPException(status_code=404, detail="frontend/index.html not found")

    return HTMLResponse(

        html_path.read_text(encoding="utf-8"),

        headers={"Cache-Control": "no-cache"},

    )





@app.get("/api/status", response_model=StatusResponse)

async def get_status() -> StatusResponse:

    return _status()





@app.get("/api/gpu")

async def get_gpu_status() -> dict[str, Any]:

    gpus = get_gpu_info()

    return {"count": len(gpus), "gpus": [g.model_dump() for g in gpus]}





@app.get("/api/config", response_model=AppConfig)

async def get_config() -> AppConfig:

    return config_store.get()





@app.put("/api/config", response_model=AppConfig)

async def update_config(patch: dict[str, Any]) -> AppConfig:

    updated = config_store.update(patch)

    model_manager.reload_config()

    await broadcast("config_updated", updated.model_dump())

    return updated





@app.get("/api/instances")

async def list_instances() -> dict[str, Any]:

    return {"instances": [s.model_dump() for s in model_manager.list_status()]}





@app.post("/api/instances")

async def create_instance(request: Request, body: dict[str, Any] | None = None) -> dict[str, Any]:

    body = body or {}
    username = getattr(request.state, "username", "") or ""

    db = SessionLocal()
    try:
        body = apply_model_to_instance_config(dict(body), db)
    finally:
        db.close()

    if not body.get("model_id"):
        raise HTTPException(status_code=400, detail="请选择已上传的模型")
    if not body.get("checkpoint"):
        raise HTTPException(status_code=400, detail="所选模型权重无效")

    cfg = config_store.get()
    audit = apply_create_audit_iso(username)

    new_cfg = InferenceInstanceConfig(

        id=body.get("id") or uuid.uuid4().hex[:8],

        name=body.get("name") or f"实例-{uuid.uuid4().hex[:4]}",

        model_id=body.get("model_id", ""),

        device_id=body.get("device_id", ""),

        device_ids=body.get("device_ids") or ([body["device_id"]] if body.get("device_id") else []),

        size=body.get("size") or cfg.model.size,

        checkpoint=body.get("checkpoint", ""),

        gpu_ids=body.get("gpu_ids") or [0],

        confidence=float(body.get("confidence", cfg.model.confidence)),

        resolution=int(body.get("resolution", cfg.model.resolution)),

        optimize_inference=bool(body.get("optimize_inference", cfg.model.optimize_inference)),

        class_names=body.get("class_names") or list(cfg.model.class_names),

        created_by=audit["created_by"],

        updated_by=audit["updated_by"],

        created_at=audit["created_at"],

        updated_at=audit["updated_at"],

    )

    instances = [i.model_dump(mode="json") for i in cfg.inference_instances]

    instances.append(new_cfg.model_dump(mode="json"))

    updated = config_store.update({"inference_instances": instances})

    model_manager.reload_config()

    await broadcast("instances_updated", updated.model_dump())

    return {"instance": new_cfg.model_dump(), "config": updated.model_dump()}





@app.put("/api/instances/{instance_id}")

async def update_instance(instance_id: str, request: Request, body: dict[str, Any]) -> dict[str, Any]:

    username = getattr(request.state, "username", "") or ""
    audit = apply_update_audit_iso(username)

    db = SessionLocal()
    try:
        body = apply_model_to_instance_config(dict(body), db)
    finally:
        db.close()

    if not body.get("model_id"):
        raise HTTPException(status_code=400, detail="请选择已上传的模型")
    if not body.get("checkpoint"):
        raise HTTPException(status_code=400, detail="所选模型权重无效")

    cfg = config_store.get()

    found = False

    instances: list[dict[str, Any]] = []

    for item in cfg.inference_instances:

        data = item.model_dump(mode="json")

        if item.id == instance_id:

            data.update(body)

            data["id"] = instance_id

            data["updated_at"] = audit["updated_at"]

            data["updated_by"] = audit["updated_by"]

            if not data.get("created_at"):

                data["created_at"] = audit["updated_at"]

            if not data.get("created_by"):

                data["created_by"] = username

            ids = data.get("device_ids") or []
            if not ids and data.get("device_id"):
                ids = [data["device_id"]]
            data["device_ids"] = ids
            data["device_id"] = ids[0] if ids else ""

            found = True

        instances.append(data)

    if not found:

        raise HTTPException(status_code=404, detail=f"实例不存在: {instance_id}")

    updated = config_store.update({"inference_instances": instances})

    model_manager.reload_config()

    await broadcast("instances_updated", updated.model_dump())

    return {"config": updated.model_dump()}





@app.delete("/api/instances/{instance_id}")

async def delete_instance(instance_id: str) -> dict[str, str]:

    cfg = config_store.get()

    if instance_id == cfg.default_instance_id:

        raise HTTPException(status_code=400, detail="不能删除默认推理实例")

    if not any(i.id == instance_id for i in cfg.inference_instances):

        raise HTTPException(status_code=404, detail=f"实例不存在: {instance_id}")

    model_manager.remove_instance(instance_id)

    instances = [i.model_dump(mode="json") for i in cfg.inference_instances if i.id != instance_id]

    config_store.update({"inference_instances": instances})

    model_manager.reload_config()

    await broadcast("instances_updated", config_store.get().model_dump())

    return {"message": f"已删除实例 {instance_id}"}





@app.post("/api/instances/{instance_id}/start")

async def start_instance(instance_id: str) -> dict[str, str]:

    try:

        await asyncio.to_thread(model_manager.start, instance_id)

        await asyncio.to_thread(model_manager.warmup, instance_id, 2)

    except Exception as exc:

        raise HTTPException(status_code=500, detail=str(exc)) from exc

    inst = model_manager.get(instance_id)

    await broadcast("instance_started", inst.status().model_dump())

    return {"message": inst.message}





@app.post("/api/instances/{instance_id}/stop")

async def stop_instance(instance_id: str) -> dict[str, str]:

    analysis_worker.stop(instance_id)

    if video_worker.state.value == "running":

        video_cfg = config_store.get().video

        if video_cfg.instance_id == instance_id:

            video_worker.stop()

    model_manager.stop(instance_id)

    inst = model_manager.get(instance_id)

    await broadcast("instance_stopped", inst.status().model_dump())

    return {"message": inst.message}





@app.post("/api/instances/{instance_id}/warmup")

async def warmup_instance(instance_id: str) -> dict[str, str]:

    try:

        await asyncio.to_thread(model_manager.warmup, instance_id, 3)

    except Exception as exc:

        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"message": "实例预热完成"}


@app.post("/api/instances/{instance_id}/analysis/start")
async def start_instance_analysis(instance_id: str) -> dict[str, str]:
    try:
        result = await asyncio.to_thread(analysis_worker.start, instance_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await broadcast("analysis_started", {"instance_id": instance_id})
    return result


@app.post("/api/instances/{instance_id}/analysis/stop")
async def stop_instance_analysis(instance_id: str) -> dict[str, str]:
    result = analysis_worker.stop(instance_id)
    await broadcast("analysis_stopped", {"instance_id": instance_id})
    return result


@app.post("/api/model/start")

async def start_model() -> dict[str, str]:

    cfg = config_store.get()

    return await start_instance(cfg.default_instance_id)





@app.post("/api/model/stop")

async def stop_model() -> dict[str, str]:

    cfg = config_store.get()

    if video_worker.state.value == "running":

        video_worker.stop()

    return await stop_instance(cfg.default_instance_id)





@app.post("/api/model/warmup")

async def warmup_model() -> dict[str, str]:

    cfg = config_store.get()

    return await warmup_instance(cfg.default_instance_id)





def _resolve_instance_id(instance_id: str | None) -> str:

    cfg = config_store.get()

    return instance_id or cfg.default_instance_id





@app.post("/api/infer/image")

async def infer_image(

    file: UploadFile = File(...),

    confidence: float | None = Form(None),

    instance_id: str | None = Form(None),

):

    target = _resolve_instance_id(instance_id)

    if not model_manager.is_ready(target):

        raise HTTPException(status_code=400, detail=f"推理实例 {target} 未就绪，请先启动")

    content = await file.read()

    try:

        image = Image.open(io.BytesIO(content)).convert("RGB")

    except Exception as exc:

        raise HTTPException(status_code=400, detail=f"无效图片: {exc}") from exc



    result = await asyncio.to_thread(

        model_manager.get(target).predict_pil,

        image,

        confidence,

        True,

    )

    result.source = file.filename or "upload"

    await broadcast("inference_done", result.model_dump())

    return result





@app.post("/api/infer/url")

async def infer_url(body: InferUrlRequest):

    target = _resolve_instance_id(body.instance_id)

    if not model_manager.is_ready(target):

        raise HTTPException(status_code=400, detail=f"推理实例 {target} 未就绪，请先启动")

    try:

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:

            resp = await client.get(body.url)

            resp.raise_for_status()

            image = Image.open(io.BytesIO(resp.content)).convert("RGB")

    except Exception as exc:

        raise HTTPException(status_code=400, detail=f"无法下载图片: {exc}") from exc



    result = await asyncio.to_thread(

        model_manager.get(target).predict_pil,

        image,

        body.confidence,

        True,

    )

    result.source = body.url

    await broadcast("inference_done", result.model_dump())

    return result





@app.post("/api/video/start")

async def start_video() -> dict[str, str]:

    try:

        video_worker.start()

    except Exception as exc:

        raise HTTPException(status_code=400, detail=str(exc)) from exc

    await broadcast("video_started", _status().model_dump())

    return {"message": video_worker.message}





@app.post("/api/video/stop")

async def stop_video() -> dict[str, str]:

    video_worker.stop()

    await broadcast("video_stopped", _status().model_dump())

    return {"message": video_worker.message}





@app.get("/api/video/mjpeg")

async def video_mjpeg() -> StreamingResponse:

    return StreamingResponse(

        mjpeg_generator(),

        media_type="multipart/x-mixed-replace; boundary=frame",

    )





@app.get("/api/video/latest")

async def video_latest() -> dict[str, Any]:

    return {

        "state": video_worker.state.value,

        "fps": video_worker.fps,

        "result": video_worker.latest_result,

    }





@app.post("/api/train/start")
async def start_training(request: Request, body: dict[str, Any] | None = None) -> dict[str, str]:
    job_id = (body or {}).get("job_id")
    username = getattr(request.state, "username", "") or ""
    if job_id:
        from .db_models import TrainingJobRecord
        from .audit import apply_update_audit

        db = SessionLocal()
        try:
            job = db.query(TrainingJobRecord).filter(TrainingJobRecord.id == job_id).first()
            if job:
                audit = apply_update_audit(username)
                job.updated_at = audit["updated_at"]
                job.updated_by = audit["updated_by"]
                db.commit()
        finally:
            db.close()
    try:
        training_worker.start(job_id)

    except Exception as exc:

        raise HTTPException(status_code=400, detail=str(exc)) from exc

    await broadcast("train_started", _status().model_dump())

    return {"message": training_worker.message, "job_id": training_worker.current_job_id}





@app.post("/api/train/stop")
async def stop_training(body: dict[str, Any] | None = None) -> dict[str, str]:
    job_id = (body or {}).get("job_id")
    try:
        training_worker.stop(job_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await broadcast("train_stopped", _status().model_dump())
    return {"message": training_worker.message}





@app.get("/api/train/status")

async def train_status() -> dict[str, Any]:

    return {

        "state": training_worker.state.value,

        "message": training_worker.message,

        "progress": training_worker.progress,

    }





@app.websocket("/ws/events")

async def ws_events(websocket: WebSocket, token: str | None = Query(None)) -> None:

    token = token or websocket.query_params.get("token")

    if not token:

        return

    try:

        payload = decode_token(token)

        user_id = int(payload.get("sub", 0))

        db = SessionLocal()

        try:

            user = db.query(User).filter(User.id == user_id, User.is_active.is_(True)).first()

            if not user:

                return

        finally:

            db.close()

    except Exception:

        return

    await websocket.accept()

    _ws_clients.add(websocket)

    try:

        await websocket.send_text(json.dumps({"event": "connected", "data": _status().model_dump()}, ensure_ascii=False))

        while True:

            await websocket.receive_text()

    except WebSocketDisconnect:

        pass

    finally:

        _ws_clients.discard(websocket)

