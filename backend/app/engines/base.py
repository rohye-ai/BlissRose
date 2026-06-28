from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import supervision as sv
from PIL import Image


@dataclass
class EngineConfig:
    model_type: str = "rf-detr"
    size: str = "medium"
    checkpoint: str = ""
    gpu_ids: list[int] = field(default_factory=lambda: [0])
    confidence: float = 0.5
    resolution: int = 576
    optimize_inference: bool = True
    class_names: list[str] = field(default_factory=list)


class InferenceEngine(ABC):
    """可插拔推理引擎抽象，统一 RF-DETR / YOLO 等后端。"""

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def unload(self) -> None:
        ...

    @abstractmethod
    def predict(self, image: Image.Image, threshold: float | None = None) -> sv.Detections:
        ...

    @property
    @abstractmethod
    def device(self) -> str:
        ...

    @property
    @abstractmethod
    def message(self) -> str:
        ...

    @abstractmethod
    def warmup(self, repeats: int = 3) -> None:
        ...

    def optimize(self) -> None:
        """可选推理优化，子类按需实现。"""
        return
