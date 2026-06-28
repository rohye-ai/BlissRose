from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import supervision as sv
from PIL import Image

from ..config import ROOT_DIR
from .base import EngineConfig, InferenceEngine


def _resolve_checkpoint(checkpoint: str) -> Path:
    path = Path(checkpoint)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def _primary_device(gpu_ids: list[int]) -> str:
    if not gpu_ids:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return f"cuda:{gpu_ids[0]}"


class YOLOEngine(InferenceEngine):
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self._model: Any | None = None
        self._device = ""
        self._message = "未加载"

    @property
    def device(self) -> str:
        return self._device

    @property
    def message(self) -> str:
        return self._message

    def load(self) -> None:
        from ultralytics import YOLO

        cfg = self.config
        if not cfg.checkpoint:
            raise ValueError("YOLO 模型需要指定权重文件")
        checkpoint = _resolve_checkpoint(cfg.checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"权重文件不存在: {checkpoint}")

        device = _primary_device(cfg.gpu_ids)
        self._model = YOLO(str(checkpoint))
        try:
            weights_label = str(checkpoint.relative_to(ROOT_DIR))
        except ValueError:
            weights_label = str(checkpoint)
        gpu_label = ",".join(str(g) for g in cfg.gpu_ids) if cfg.gpu_ids else "auto"
        self._device = device
        self._message = f"YOLO 已就绪 ({weights_label}) | GPU [{gpu_label}]"

    def unload(self) -> None:
        self._model = None
        self._device = ""
        self._message = "已卸载"

    def predict(self, image: Image.Image, threshold: float | None = None) -> sv.Detections:
        if self._model is None:
            raise RuntimeError("YOLO 模型未加载")
        thr = threshold if threshold is not None else self.config.confidence
        imgsz = self.config.resolution or 640
        results = self._model.predict(
            source=image,
            conf=thr,
            imgsz=imgsz,
            device=self._device,
            verbose=False,
        )
        if not results:
            return sv.Detections.empty()
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return sv.Detections.empty()
        xyxy = result.boxes.xyxy.cpu().numpy()
        confidence = result.boxes.conf.cpu().numpy()
        class_id = result.boxes.cls.cpu().numpy().astype(int)
        return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)

    def warmup(self, repeats: int = 3) -> None:
        dummy = Image.new("RGB", (640, 480), color=(128, 128, 128))
        for _ in range(repeats):
            self.predict(dummy)
