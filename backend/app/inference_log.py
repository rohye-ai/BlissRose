"""Structured inference logging for instance predictions."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING

from .config import ROOT_DIR

if TYPE_CHECKING:
    from .schemas import DetectionItem

_LOG_PATH = ROOT_DIR / "data" / "logs" / "inference.log"
_logger: logging.Logger | None = None


def _get_logger() -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger

    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("blissrose.inference")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = RotatingFileHandler(
            _LOG_PATH,
            maxBytes=20 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)

        console = logging.StreamHandler()
        console.setFormatter(handler.formatter)
        logger.addHandler(console)

    _logger = logger
    return logger


def _format_detections(detections: list[DetectionItem]) -> str:
    if not detections:
        return "无目标"
    parts: list[str] = []
    for d in detections:
        bbox = d.bbox
        if len(bbox) >= 4:
            parts.append(
                f"{d.class_name} conf={d.confidence:.3f} "
                f"bbox=[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}]"
            )
        else:
            parts.append(f"{d.class_name} conf={d.confidence:.3f}")
    return " | ".join(parts)


def log_inference(
    *,
    instance_id: str,
    instance_name: str = "",
    source: str = "",
    source_type: str = "",
    device_name: str = "",
    inference_ms: float,
    count: int,
    detections: list[DetectionItem] | None = None,
) -> None:
    """Log one inference: source address, results, and elapsed time."""
    type_label = source_type or "image"
    name_part = f" instance={instance_name or instance_id}"
    device_part = f" device={device_name}" if device_name else ""
    source_part = source or "(unknown)"
    det_text = _format_detections(detections or [])
    msg = (
        f"{name_part}{device_part} type={type_label} source={source_part} "
        f"ms={inference_ms:.1f} count={count} result={det_text}"
    )
    _get_logger().info(msg.strip())
