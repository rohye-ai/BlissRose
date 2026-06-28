from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ModelSize(str, Enum):
    NANO = "nano"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    BASE = "base"


class ModelState(str, Enum):
    STOPPED = "stopped"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


class VideoState(str, Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    ERROR = "error"


class TrainState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPING = "stopping"


class ModelConfig(BaseModel):
    size: ModelSize = ModelSize.MEDIUM
    checkpoint: str = ""
    confidence: float = Field(0.5, ge=0.01, le=0.99)
    resolution: int = Field(576, ge=320, le=880)
    optimize_inference: bool = True
    device: str = "auto"
    class_names: list[str] = Field(default_factory=list)


class ModelType(str, Enum):
    YOLO = "yolo"
    RF_DETR = "rf-detr"


class DeviceType(str, Enum):
    VIDEO = "video"
    IMAGE = "image"


class ModelStage(str, Enum):
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class InferenceInstanceConfig(BaseModel):
    """Single inference instance: own weights, GPUs, and thresholds."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = "推理实例"
    enabled: bool = True
    model_type: str = "rf-detr"
    model_id: str = ""
    device_id: str = ""
    device_ids: list[str] = Field(default_factory=list)
    size: ModelSize = ModelSize.MEDIUM
    checkpoint: str = ""
    gpu_ids: list[int] = Field(default_factory=lambda: [0])
    confidence: float = Field(0.5, ge=0.01, le=0.99)
    resolution: int = Field(576, ge=320, le=880)
    optimize_inference: bool = True
    class_names: list[str] = Field(default_factory=list)
    created_by: str = ""
    updated_by: str = ""
    created_at: str = ""
    updated_at: str = ""

    @field_validator("gpu_ids", mode="before")
    @classmethod
    def normalize_gpu_ids(cls, value: Any) -> list[int]:
        if value is None or value == "":
            return [0]
        if isinstance(value, int):
            return [value]
        if isinstance(value, str):
            parts = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
            return [int(p) for p in parts] if parts else [0]
        return [int(v) for v in value]

    @field_validator("device_ids", mode="before")
    @classmethod
    def normalize_device_ids(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            parts = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
            return parts
        return [str(v) for v in value if v]

    def model_post_init(self, __context: Any) -> None:
        if not self.device_ids and self.device_id:
            object.__setattr__(self, "device_ids", [self.device_id])
        elif self.device_ids and not self.device_id:
            object.__setattr__(self, "device_id", self.device_ids[0])


class VideoConfig(BaseModel):
    source: str = "0"
    fps_limit: int = Field(15, ge=1, le=60)
    skip_frames: int = Field(0, ge=0, le=30)
    reconnect_delay: int = Field(3, ge=1, le=60)
    instance_id: str = "default"


class TrainingConfig(BaseModel):
    dataset_dir: str = ""
    output_dir: str = "outputs/train"
    epochs: int = Field(50, ge=1, le=1000)
    batch_size: int = Field(4, ge=1, le=64)
    grad_accum_steps: int = Field(4, ge=1, le=32)
    lr: float = Field(1e-4, gt=0)
    resume: bool = False
    gpu_ids: list[int] = Field(default_factory=lambda: [0])

    @field_validator("gpu_ids", mode="before")
    @classmethod
    def normalize_gpu_ids(cls, value: Any) -> list[int]:
        if value is None or value == "":
            return [0]
        if isinstance(value, int):
            return [value]
        if isinstance(value, str):
            parts = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
            return [int(p) for p in parts] if parts else [0]
        return [int(v) for v in value]


class AppConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference_instances: list[InferenceInstanceConfig] = Field(default_factory=list)
    default_instance_id: str = "default"


class GpuInfo(BaseModel):
    index: int
    name: str
    memory_total_mb: float
    memory_used_mb: float
    memory_free_mb: float
    utilization_gpu: float = 0.0
    temperature_c: float = 0.0
    torch_allocated_mb: float = 0.0


class InstanceStatus(BaseModel):
    id: str
    name: str
    state: ModelState
    message: str = ""
    device: str = ""
    gpu_ids: list[int] = Field(default_factory=list)
    checkpoint: str = ""
    model_type: str = "rf-detr"
    last_inference_ms: float | None = None


class DetectionItem(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]


class InferenceResult(BaseModel):
    detections: list[DetectionItem]
    count: int
    inference_ms: float
    image_base64: str | None = None
    source: str = ""
    instance_id: str = ""


class InferUrlRequest(BaseModel):
    url: str
    confidence: float | None = None
    instance_id: str | None = None


class StatusResponse(BaseModel):
    model_state: ModelState
    model_message: str = ""
    video_state: VideoState
    video_message: str = ""
    train_state: TrainState
    train_message: str = ""
    train_progress: dict[str, Any] = Field(default_factory=dict)
    config: AppConfig
    device: str = ""
    last_inference_ms: float | None = None
    video_fps: float | None = None
    gpus: list[GpuInfo] = Field(default_factory=list)
    instances: list[InstanceStatus] = Field(default_factory=list)


class RoiRegion(BaseModel):
    x: float = Field(0, ge=0, le=1)
    y: float = Field(0, ge=0, le=1)
    w: float = Field(1, ge=0, le=1)
    h: float = Field(1, ge=0, le=1)


class ModelRecordOut(BaseModel):
    id: str
    name: str
    model_type: str
    file_path: str
    class_names: list[str] = Field(default_factory=list)
    parent_id: str | None = None
    source: str = "upload"
    version: str = "v1"
    stage: str = "staging"
    metrics: dict[str, Any] = Field(default_factory=dict)
    in_use: bool = False
    uploaded_by: str = ""
    created_at: str = ""


class BatchInferUrlRequest(BaseModel):
    urls: list[str] = Field(..., min_length=1, max_length=50)
    instance_id: str | None = None
    confidence: float | None = None
    annotate: bool = True


class ModelStageUpdate(BaseModel):
    stage: ModelStage


class ModelEvaluateRequest(BaseModel):
    dataset_id: str | None = None
    split: str = "valid"


class ModelExportRequest(BaseModel):
    format: str = "onnx"
    imgsz: int = Field(640, ge=320, le=1280)


class DatasetRecordOut(BaseModel):
    id: str
    name: str
    path: str
    data_yaml: str
    format: str = "yolo"
    review_status: str = "draft"
    total_count: int = 0
    labeled_count: int = 0
    unlabeled_count: int = 0
    approved_count: int = 0
    train_ready: bool = False
    class_names: list[str] = Field(default_factory=list)
    train_count: int = 0
    valid_count: int = 0
    test_count: int = 0
    uploaded_by: str = ""
    created_at: str = ""


class DatasetReviewRequest(BaseModel):
    message: str = ""


class DatasetImageReviewRequest(BaseModel):
    action: str = Field(..., pattern="^(approve|reject)$")


class DeviceRecordOut(BaseModel):
    id: str
    name: str
    device_type: str
    source: str
    poll_interval: int = 5
    roi: list[RoiRegion] = Field(default_factory=list)
    enabled: bool = True
    analysis_running: bool = False
    created_by: str = ""
    updated_by: str = ""
    created_at: str = ""
    updated_at: str = ""


class AlertRecordOut(BaseModel):
    id: str
    device_id: str
    device_name: str = ""
    instance_id: str
    instance_name: str = ""
    image_url: str = ""
    detections: list[DetectionItem] = Field(default_factory=list)
    max_confidence: float = 0.0
    alert_at: str = ""


class TrainingJobOut(BaseModel):
    id: str
    name: str
    model_id: str
    model_name: str = ""
    dataset_id: str
    dataset_name: str = ""
    output_dir: str
    state: str
    epochs: int
    batch_size: int
    grad_accum_steps: int
    lr: float
    gpu_ids: list[int] = Field(default_factory=list)
    checkpoint_path: str = ""
    can_resume: bool = False
    deployed_model_id: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    message: str = ""
    created_by: str = ""
    updated_by: str = ""
    created_at: str = ""
    updated_at: str = ""
    completed_at: str | None = None


class AnalysisStatus(BaseModel):
    instance_id: str
    device_id: str
    running: bool = False
    message: str = ""


class YoloAnnotation(BaseModel):
    class_id: int = Field(0, ge=0)
    class_name: str = ""
    cx: float = Field(..., ge=0, le=1)
    cy: float = Field(..., ge=0, le=1)
    w: float = Field(..., ge=0, le=1)
    h: float = Field(..., ge=0, le=1)


class SaveLabelsRequest(BaseModel):
    annotations: list[YoloAnnotation] = Field(default_factory=list)
    class_names: list[str] | None = None


class PreAnnotateRequest(BaseModel):
    instance_id: str
    confidence: float | None = None
    save: bool = False


class WebhookChannel(BaseModel):
    type: str = "generic"  # generic | dingtalk | wecom
    url: str
    secret: str = ""
    enabled: bool = True
    headers: dict[str, str] = Field(default_factory=dict)
    timeout: float = 10.0


class WebhookConfigUpdate(BaseModel):
    enabled: bool = False
    min_confidence: float = Field(0.0, ge=0, le=1)
    device_ids: list[str] = Field(default_factory=list)
    channels: list[WebhookChannel] = Field(default_factory=list)
