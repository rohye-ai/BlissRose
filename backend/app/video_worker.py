from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import cv2
import numpy as np

from .config import config_store
from .model_manager import model_manager
from .schemas import VideoState


class VideoStreamWorker:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._state = VideoState.STOPPED
        self._message = "视频流未启动"
        self._latest_jpeg: bytes | None = None
        self._latest_result: dict[str, Any] | None = None
        self._fps = 0.0
        self._frame_count = 0
        self._last_fps_time = time.time()

    @property
    def state(self) -> VideoState:
        return self._state

    @property
    def message(self) -> str:
        return self._message

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def latest_result(self) -> dict[str, Any] | None:
        return self._latest_result

    def get_latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def start(self) -> None:
        cfg = config_store.get().video
        with self._lock:
            if self._state == VideoState.RUNNING:
                return
            if not model_manager.is_ready(cfg.instance_id):
                raise RuntimeError("请先启动对应推理实例后再开启视频流")
            self._stop_event.clear()
            self._state = VideoState.RUNNING
            self._message = "视频流运行中"
            self._thread = threading.Thread(target=self._run, name="video-worker", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = None
        with self._lock:
            thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=5)
        with self._lock:
            self._thread = None
            self._state = VideoState.STOPPED
            self._message = "视频流已停止"
            self._latest_jpeg = None
            self._latest_result = None
            self._fps = 0.0

    def _parse_source(self, source: str) -> str | int:
        if source.isdigit():
            return int(source)
        return source

    def _run(self) -> None:
        cfg = config_store.get().video
        source = self._parse_source(cfg.source)
        cap: cv2.VideoCapture | None = None
        skip_counter = 0

        while not self._stop_event.is_set():
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(source)
                if not cap.isOpened():
                    with self._lock:
                        self._state = VideoState.ERROR
                        self._message = f"无法连接视频源: {cfg.source}"
                    time.sleep(cfg.reconnect_delay)
                    continue
                with self._lock:
                    self._state = VideoState.RUNNING
                    self._message = "视频流运行中"

            ok, frame = cap.read()
            if not ok:
                cap.release()
                cap = None
                with self._lock:
                    self._message = "视频流中断，正在重连..."
                time.sleep(cfg.reconnect_delay)
                continue

            if cfg.skip_frames > 0:
                skip_counter += 1
                if skip_counter <= cfg.skip_frames:
                    continue
                skip_counter = 0

            try:
                result, annotated = model_manager.get(cfg.instance_id).predict_numpy(frame)
                ok_enc, jpeg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if ok_enc:
                    with self._lock:
                        self._latest_jpeg = jpeg.tobytes()
                        self._latest_result = {
                            "count": result.count,
                            "inference_ms": result.inference_ms,
                            "detections": [d.model_dump() for d in result.detections],
                        }
                self._update_fps()
            except Exception as exc:
                with self._lock:
                    self._message = f"推理错误: {exc}"

            if cfg.fps_limit > 0:
                time.sleep(1.0 / cfg.fps_limit)

        if cap is not None:
            cap.release()

    def _update_fps(self) -> None:
        self._frame_count += 1
        now = time.time()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_fps_time = now


video_worker = VideoStreamWorker()


async def mjpeg_generator():
    boundary = b"--frame"
    while True:
        if video_worker.state != VideoState.RUNNING:
            await asyncio.sleep(0.2)
            continue
        jpeg = video_worker.get_latest_jpeg()
        if jpeg:
            yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        await asyncio.sleep(0.03)
