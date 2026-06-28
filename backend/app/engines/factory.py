from __future__ import annotations

from .base import EngineConfig, InferenceEngine
from .rfdetr_engine import RFDETREngine
from .yolo_engine import YOLOEngine


def create_engine(config: EngineConfig) -> InferenceEngine:
    model_type = (config.model_type or "rf-detr").lower().replace("_", "-")
    if model_type == "yolo":
        return YOLOEngine(config)
    if model_type in ("rf-detr", "rfdetr"):
        return RFDETREngine(config)
    raise ValueError(f"不支持的模型类型: {config.model_type}")
