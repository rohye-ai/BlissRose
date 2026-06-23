from __future__ import annotations

import base64
import io
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import supervision as sv
from PIL import Image

from .config import ROOT_DIR, config_store
from .schemas import (
    DetectionItem,
    InferenceInstanceConfig,
    InferenceResult,
    InstanceStatus,
    ModelSize,
    ModelState,
)

MODEL_CLASS_MAP = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
    "base": "RFDETRBase",
}


def infer_model_size(path_or_name: str) -> str:
    """从文件名/路径推断 RF-DETR 架构规格，无法识别时默认 medium。"""
    lower = (path_or_name or "").lower()
    for key in MODEL_CLASS_MAP:
        if key in lower:
            return key
    return "medium"


def _import_model_class(size: str):
    import rfdetr

    class_name = MODEL_CLASS_MAP.get(size, "RFDETRMedium")
    return getattr(rfdetr, class_name)


def _resolve_checkpoint_path(checkpoint: str) -> Path:
    path = Path(checkpoint)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def _resolve_class_names(class_ids: np.ndarray, custom_names: list[str]) -> list[str]:
    if custom_names:
        return [custom_names[cid] if 0 <= cid < len(custom_names) else f"class_{cid}" for cid in class_ids]
    try:
        from rfdetr.assets.coco_classes import COCO_CLASSES

        return [COCO_CLASSES[cid] if 0 <= cid < len(COCO_CLASSES) else f"class_{cid}" for cid in class_ids]
    except Exception:
        return [f"class_{cid}" for cid in class_ids]


def _primary_device(gpu_ids: list[int]) -> str:
    if not gpu_ids:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return f"cuda:{gpu_ids[0]}"


class ModelInstance:
    def __init__(self, config: InferenceInstanceConfig) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._model: Any | None = None
        self._state = ModelState.STOPPED
        self._message = "未启动"
        self._device = ""
        self._last_inference_ms: float | None = None
        self._box_annotator = sv.BoxAnnotator()
        self._label_annotator = sv.LabelAnnotator()

    @property
    def id(self) -> str:
        return self.config.id

    @property
    def state(self) -> ModelState:
        return self._state

    @property
    def message(self) -> str:
        return self._message

    @property
    def device(self) -> str:
        return self._device

    @property
    def last_inference_ms(self) -> float | None:
        return self._last_inference_ms

    def is_ready(self) -> bool:
        return self._state == ModelState.READY and self._model is not None

    def update_config(self, config: InferenceInstanceConfig) -> None:
        with self._lock:
            if self._state == ModelState.READY:
                raise RuntimeError("请先停止实例再修改配置")
            self.config = config

    def status(self) -> InstanceStatus:
        return InstanceStatus(
            id=self.config.id,
            name=self.config.name,
            state=self._state,
            message=self._message,
            device=self._device,
            gpu_ids=list(self.config.gpu_ids),
            checkpoint=self.config.checkpoint,
            last_inference_ms=self._last_inference_ms,
        )

    def start(self) -> None:
        with self._lock:
            if self._state == ModelState.LOADING:
                raise RuntimeError("实例正在加载中")
            if self._state == ModelState.READY:
                return
            self._state = ModelState.LOADING
            self._message = "正在加载模型..."

        try:
            cfg = self.config
            ModelClass = _import_model_class(cfg.size.value)
            kwargs: dict[str, Any] = {"device": _primary_device(cfg.gpu_ids)}
            weights_label = "COCO 预训练"
            if cfg.checkpoint:
                checkpoint = _resolve_checkpoint_path(cfg.checkpoint)
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
            with self._lock:
                self._model = model
                self._state = ModelState.READY
                self._device = device
                self._message = f"模型已就绪 ({weights_label}) | GPU [{gpu_label}]"
        except Exception as exc:
            with self._lock:
                self._model = None
                self._state = ModelState.ERROR
                self._message = f"加载失败: {exc}"
            raise

    def stop(self) -> None:
        with self._lock:
            self._model = None
            self._state = ModelState.STOPPED
            self._message = "已停止"
            self._device = ""
            self._last_inference_ms = None

    def warmup(self, repeats: int = 3) -> None:
        if not self.is_ready():
            raise RuntimeError("实例未就绪")
        dummy = Image.new("RGB", (640, 480), color=(128, 128, 128))
        for _ in range(repeats):
            self._predict_internal(dummy)

    def predict_pil(
        self,
        image: Image.Image,
        confidence: float | None = None,
        annotate: bool = True,
    ) -> InferenceResult:
        if not self.is_ready():
            raise RuntimeError(f"实例 {self.config.name} 未就绪")
        cfg = self.config
        threshold = confidence if confidence is not None else cfg.confidence
        start = time.perf_counter()
        detections = self._predict_internal(image, threshold=threshold)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._last_inference_ms = elapsed_ms

        items: list[DetectionItem] = []
        if detections is not None and len(detections) > 0:
            names = _resolve_class_names(detections.class_id, cfg.class_names)
            for i in range(len(detections)):
                x1, y1, x2, y2 = detections.xyxy[i].tolist()
                conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
                cid = int(detections.class_id[i])
                items.append(
                    DetectionItem(
                        class_id=cid,
                        class_name=names[i],
                        confidence=conf,
                        bbox=[x1, y1, x2, y2],
                    )
                )

        image_b64 = None
        if annotate:
            annotated = self._annotate(image, detections, items)
            image_b64 = _pil_to_base64(annotated)

        return InferenceResult(
            detections=items,
            count=len(items),
            inference_ms=elapsed_ms,
            image_base64=image_b64,
            instance_id=self.config.id,
        )

    def predict_numpy(self, frame_bgr: np.ndarray, **kwargs: Any) -> tuple[InferenceResult, np.ndarray]:
        rgb = frame_bgr[:, :, ::-1]
        image = Image.fromarray(rgb)
        threshold = kwargs.get("confidence")
        thr = threshold if threshold is not None else self.config.confidence
        start = time.perf_counter()
        detections = self._predict_internal(image, threshold=thr)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._last_inference_ms = elapsed_ms

        items: list[DetectionItem] = []
        if detections is not None and len(detections) > 0:
            names = _resolve_class_names(detections.class_id, self.config.class_names)
            for i in range(len(detections)):
                x1, y1, x2, y2 = detections.xyxy[i].tolist()
                conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
                cid = int(detections.class_id[i])
                items.append(
                    DetectionItem(class_id=cid, class_name=names[i], confidence=conf, bbox=[x1, y1, x2, y2])
                )

        annotated = self._annotate(image, detections, items)
        annotated_bgr = np.array(annotated)[:, :, ::-1]
        result = InferenceResult(
            detections=items,
            count=len(items),
            inference_ms=elapsed_ms,
            instance_id=self.config.id,
        )
        return result, annotated_bgr

    def _predict_internal(self, image: Image.Image, threshold: float | None = None):
        cfg = self.config
        thr = threshold if threshold is not None else cfg.confidence
        with self._lock:
            if self._model is None:
                raise RuntimeError("模型未加载")
            model = self._model
        predict_kwargs: dict[str, Any] = {"threshold": thr}
        if cfg.resolution:
            predict_kwargs["resolution"] = cfg.resolution
        return model.predict(image, **predict_kwargs)

    def _annotate(self, image: Image.Image, detections: sv.Detections | None, items: list[DetectionItem]) -> Image.Image:
        canvas = image.copy()
        if detections is None or len(detections) == 0:
            return canvas
        labels = [f"{d.class_name} {d.confidence:.2f}" for d in items]
        canvas = self._box_annotator.annotate(canvas, detections)
        canvas = self._label_annotator.annotate(canvas, detections, labels)
        return canvas


class InstanceManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._instances: dict[str, ModelInstance] = {}
        self._sync_from_config()

    def _sync_from_config(self) -> None:
        cfg = config_store.get()
        known = set(self._instances)
        configured = {item.id: item for item in cfg.inference_instances}

        for instance_id, item in configured.items():
            if instance_id in self._instances:
                inst = self._instances[instance_id]
                if inst.state != ModelState.READY:
                    inst.config = item
            else:
                self._instances[instance_id] = ModelInstance(item)

        for stale_id in known - set(configured):
            inst = self._instances.pop(stale_id, None)
            if inst and inst.state == ModelState.READY:
                inst.stop()

    def reload_config(self) -> None:
        with self._lock:
            self._sync_from_config()

    def list_status(self) -> list[InstanceStatus]:
        with self._lock:
            self._sync_from_config()
            return [inst.status() for inst in self._instances.values()]

    def get(self, instance_id: str | None = None) -> ModelInstance:
        cfg = config_store.get()
        target_id = instance_id or cfg.default_instance_id
        with self._lock:
            self._sync_from_config()
            inst = self._instances.get(target_id)
            if inst is None:
                raise KeyError(f"推理实例不存在: {target_id}")
            return inst

    def get_default(self) -> ModelInstance:
        return self.get(None)

    def add_instance(self, config: InferenceInstanceConfig | None = None) -> InferenceInstanceConfig:
        if config is None:
            base = config_store.get().model
            config = InferenceInstanceConfig(
                id=uuid.uuid4().hex[:8],
                name=f"实例-{uuid.uuid4().hex[:4]}",
                size=base.size,
                checkpoint=base.checkpoint,
                gpu_ids=[0],
                confidence=base.confidence,
                resolution=base.resolution,
                optimize_inference=base.optimize_inference,
                class_names=list(base.class_names),
            )
        with self._lock:
            self._instances[config.id] = ModelInstance(config)
        return config

    def remove_instance(self, instance_id: str) -> None:
        cfg = config_store.get()
        if instance_id == cfg.default_instance_id:
            raise RuntimeError("不能删除默认推理实例")
        with self._lock:
            inst = self._instances.pop(instance_id, None)
            if inst:
                inst.stop()

    def start(self, instance_id: str | None = None) -> ModelInstance:
        inst = self.get(instance_id)
        inst.start()
        return inst

    def stop(self, instance_id: str | None = None) -> None:
        if instance_id:
            self.get(instance_id).stop()
            return
        with self._lock:
            for inst in self._instances.values():
                if inst.state == ModelState.READY:
                    inst.stop()

    def warmup(self, instance_id: str | None = None, repeats: int = 3) -> None:
        self.get(instance_id).warmup(repeats)

    def is_ready(self, instance_id: str | None = None) -> bool:
        return self.get(instance_id).is_ready()

    @property
    def state(self) -> ModelState:
        try:
            return self.get_default().state
        except KeyError:
            return ModelState.STOPPED

    @property
    def message(self) -> str:
        try:
            return self.get_default().message
        except KeyError:
            return "无默认实例"

    @property
    def device(self) -> str:
        try:
            return self.get_default().device
        except KeyError:
            return ""

    @property
    def last_inference_ms(self) -> float | None:
        try:
            return self.get_default().last_inference_ms
        except KeyError:
            return None

    def predict_pil(self, image: Image.Image, confidence: float | None = None, annotate: bool = True) -> InferenceResult:
        return self.get_default().predict_pil(image, confidence, annotate)

    def predict_numpy(self, frame_bgr: np.ndarray, **kwargs: Any) -> tuple[InferenceResult, np.ndarray]:
        return self.get_default().predict_numpy(frame_bgr, **kwargs)


def _pil_to_base64(image: Image.Image, fmt: str = "JPEG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{encoded}"


instance_manager = InstanceManager()
model_manager = instance_manager
