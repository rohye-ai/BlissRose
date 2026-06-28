from __future__ import annotations

from pathlib import Path
from typing import Any

import supervision as sv
from PIL import Image

from ..config import ROOT_DIR
from .base import EngineConfig, InferenceEngine

MODEL_CLASS_MAP = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
    "base": "RFDETRBase",
}


def _primary_device(gpu_ids: list[int]) -> str:
    if not gpu_ids:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return f"cuda:{gpu_ids[0]}"


def _resolve_checkpoint(checkpoint: str) -> Path:
    path = Path(checkpoint)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


class RFDETREngine(InferenceEngine):
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
        import rfdetr

        cfg = self.config
        class_name = MODEL_CLASS_MAP.get(cfg.size, "RFDETRMedium")
        ModelClass = getattr(rfdetr, class_name)
        kwargs: dict[str, Any] = {"device": _primary_device(cfg.gpu_ids)}
        weights_label = "COCO 预训练"
        if cfg.checkpoint:
            checkpoint = _resolve_checkpoint(cfg.checkpoint)
            if not checkpoint.exists():
                raise FileNotFoundError(f"权重文件不存在: {checkpoint}")
            kwargs["pretrain_weights"] = str(checkpoint)
            try:
                weights_label = str(checkpoint.relative_to(ROOT_DIR))
            except ValueError:
                weights_label = str(checkpoint)

        model = ModelClass(**kwargs)
        if cfg.optimize_inference and hasattr(model, "optimize_for_inference"):
            try:
                model.optimize_for_inference()
            except Exception:
                pass

        if hasattr(model, "model") and hasattr(model.model, "device"):
            device = str(model.model.device)
        else:
            device = _primary_device(cfg.gpu_ids)

        gpu_label = ",".join(str(g) for g in cfg.gpu_ids) if cfg.gpu_ids else "auto"
        self._model = model
        self._device = device
        self._message = f"RF-DETR 已就绪 ({weights_label}) | GPU [{gpu_label}]"

    def unload(self) -> None:
        self._model = None
        self._device = ""
        self._message = "已卸载"

    def predict(self, image: Image.Image, threshold: float | None = None) -> sv.Detections:
        if self._model is None:
            raise RuntimeError("RF-DETR 模型未加载")
        thr = threshold if threshold is not None else self.config.confidence
        predict_kwargs: dict[str, Any] = {"threshold": thr}
        if self.config.resolution:
            predict_kwargs["resolution"] = self.config.resolution
        result = self._model.predict(image, **predict_kwargs)
        if result is None:
            return sv.Detections.empty()
        return result

    def warmup(self, repeats: int = 3) -> None:
        dummy = Image.new("RGB", (640, 480), color=(128, 128, 128))
        for _ in range(repeats):
            self.predict(dummy)
