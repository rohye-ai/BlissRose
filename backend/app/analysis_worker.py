from __future__ import annotations

import io
import threading
import time
from typing import Any

import cv2
import httpx
from PIL import Image

from .config import config_store
from .database import SessionLocal
from .db_models import DeviceRecord
from .model_manager import model_manager
from .platform_service import create_alert
from .schemas import DetectionItem, InferenceInstanceConfig, RoiRegion


def _bbox_in_roi(bbox: list[float], roi: RoiRegion, img_w: int, img_h: int) -> bool:
    cx = (bbox[0] + bbox[2]) / 2 / img_w
    cy = (bbox[1] + bbox[3]) / 2 / img_h
    return roi.x <= cx <= roi.x + roi.w and roi.y <= cy <= roi.y + roi.h


def _filter_by_roi(detections: list[DetectionItem], rois: list[RoiRegion], img_w: int, img_h: int) -> list[DetectionItem]:
    if not rois:
        return detections
    return [d for d in detections if any(_bbox_in_roi(d.bbox, r, img_w, img_h) for r in rois)]


def _parse_rois(raw: str) -> list[RoiRegion]:
    import json

    try:
        data = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []
    return [RoiRegion(**r) for r in data if isinstance(r, dict)]


def resolve_instance_device_ids(inst_cfg: InferenceInstanceConfig) -> list[str]:
    if inst_cfg.device_ids:
        return list(inst_cfg.device_ids)
    if inst_cfg.device_id:
        return [inst_cfg.device_id]
    return []


class InstanceAnalysisTask:
    """Round-robin analysis across multiple devices bound to one inference instance."""

    def __init__(self, instance_id: str, devices: list[DeviceRecord]) -> None:
        self.instance_id = instance_id
        self.devices = devices
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.message = "未启动"
        self._last_fetch: dict[str, float] = {}
        self._video_caps: dict[str, cv2.VideoCapture] = {}

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        if not model_manager.is_ready(self.instance_id):
            raise RuntimeError("推理实例未就绪，请先启动实例")
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"analysis-inst-{self.instance_id}",
            daemon=True,
        )
        self._thread.start()
        self.message = "分析运行中"

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._thread = None
        for cap in self._video_caps.values():
            try:
                cap.release()
            except Exception:
                pass
        self._video_caps.clear()
        self.message = "已停止"

    def _maybe_alert(self, device: DeviceRecord, result, image: Image.Image, rois: list[RoiRegion]) -> None:
        from .instance_service import get_inference_instance

        db = SessionLocal()
        try:
            inst_cfg = get_inference_instance(db, self.instance_id)
        finally:
            db.close()
        threshold = inst_cfg.confidence if inst_cfg else 0.5
        filtered = _filter_by_roi(result.detections, rois, image.width, image.height)
        qualifying = [d for d in filtered if d.confidence >= threshold]
        if not qualifying:
            return
        max_conf = max(d.confidence for d in qualifying)
        buf = io.BytesIO()
        inst = model_manager.get(self.instance_id)
        annotated = inst._annotate(image, None, qualifying)
        if result.image_base64:
            try:
                import base64

                raw = result.image_base64.split(",", 1)[-1]
                buf.write(base64.b64decode(raw))
            except Exception:
                annotated.save(buf, format="JPEG", quality=90)
        else:
            annotated.save(buf, format="JPEG", quality=90)
        db = SessionLocal()
        try:
            create_alert(db, device.id, self.instance_id, buf.getvalue(), qualifying, max_conf)
        finally:
            db.close()

    def _fetch_image(self, device: DeviceRecord) -> Image.Image:
        url = device.source.strip()
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")

    def _fetch_video_frame(self, device: DeviceRecord) -> Image.Image | None:
        source = device.source
        cap_source: str | int = int(source) if source.isdigit() else source
        cap = self._video_caps.get(device.id)
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            cap = cv2.VideoCapture(cap_source)
            self._video_caps[device.id] = cap
        if not cap.isOpened():
            return None
        ok, frame = cap.read()
        if not ok:
            cap.release()
            self._video_caps.pop(device.id, None)
            return None
        rgb = frame[:, :, ::-1]
        return Image.fromarray(rgb)

    def _process_device(self, device: DeviceRecord) -> None:
        rois = _parse_rois(device.roi)
        if device.device_type == "video":
            image = self._fetch_video_frame(device)
            if image is None:
                raise RuntimeError(f"无法读取视频帧: {device.source}")
            import numpy as np

            frame = np.array(image)[:, :, ::-1]
            result, _ = model_manager.get(self.instance_id).predict_numpy(frame)
            self._maybe_alert(device, result, image, rois)
        else:
            image = self._fetch_image(device)
            result = model_manager.get(self.instance_id).predict_pil(image, annotate=True)
            self._maybe_alert(device, result, image, rois)

    def _run_loop(self) -> None:
        video_cfg = config_store.get().video
        while not self._stop.is_set():
            if not self.devices:
                self.message = "无绑定设备"
                time.sleep(1)
                continue

            now = time.time()
            processed = False
            for device in self.devices:
                if self._stop.is_set():
                    break
                if device.device_type == "image":
                    interval = max(1, device.poll_interval)
                else:
                    interval = 1.0 / max(1, video_cfg.fps_limit)
                last = self._last_fetch.get(device.id, 0.0)
                if now - last < interval:
                    continue
                try:
                    self._process_device(device)
                    self._last_fetch[device.id] = time.time()
                    self.message = f"轮询分析中 · 最近: {device.name}"
                    processed = True
                except Exception as exc:
                    self.message = f"[{device.name}] {exc}"

            if not processed:
                time.sleep(0.2)

        for cap in self._video_caps.values():
            try:
                cap.release()
            except Exception:
                pass
        self._video_caps.clear()


class AnalysisWorker:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tasks: dict[str, InstanceAnalysisTask] = {}

    def _device_in_running_task(self, device_id: str, exclude_instance: str | None = None) -> str | None:
        for iid, task in self._tasks.items():
            if exclude_instance and iid == exclude_instance:
                continue
            if task.running and any(d.id == device_id for d in task.devices):
                return iid
        return None

    def is_device_running(self, device_id: str) -> bool:
        with self._lock:
            return self._device_in_running_task(device_id) is not None

    def is_instance_running(self, instance_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(instance_id)
            return task.running if task else False

    def list_status(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "instance_id": iid,
                    "device_ids": [d.id for d in t.devices],
                    "device_id": t.devices[0].id if t.devices else "",
                    "running": t.running,
                    "message": t.message,
                }
                for iid, t in self._tasks.items()
            ]

    def start(self, instance_id: str) -> dict[str, str]:
        from .instance_service import get_inference_instance

        db = SessionLocal()
        try:
            inst_cfg = get_inference_instance(db, instance_id)
        finally:
            db.close()
        if not inst_cfg:
            raise RuntimeError(f"实例不存在: {instance_id}")
        device_ids = resolve_instance_device_ids(inst_cfg)
        if not device_ids:
            raise RuntimeError("实例未绑定设备，请先选择至少一个设备")
        if not model_manager.is_ready(instance_id):
            raise RuntimeError("请先启动推理实例")

        db = SessionLocal()
        try:
            devices: list[DeviceRecord] = []
            for did in device_ids:
                device = db.query(DeviceRecord).filter(DeviceRecord.id == did).first()
                if not device:
                    raise RuntimeError(f"设备不存在: {did}")
                if not device.enabled:
                    raise RuntimeError(f"设备 [{device.name}] 已禁用")
                devices.append(device)
        finally:
            db.close()

        with self._lock:
            existing = self._tasks.get(instance_id)
            if existing and existing.running:
                return {"message": "分析已在运行"}

            for device in devices:
                other = self._device_in_running_task(device.id, exclude_instance=instance_id)
                if other:
                    raise RuntimeError(f"设备 [{device.name}] 已被实例 {other} 占用分析")

            task = InstanceAnalysisTask(instance_id, devices)
            task.start()
            self._tasks[instance_id] = task
            names = "、".join(d.name for d in devices)
            return {"message": f"已启动轮询分析: {names}"}

    def stop(self, instance_id: str) -> dict[str, str]:
        with self._lock:
            task = self._tasks.get(instance_id)
            if task:
                task.stop()
                del self._tasks[instance_id]
        return {"message": "分析已停止"}


analysis_worker = AnalysisWorker()
