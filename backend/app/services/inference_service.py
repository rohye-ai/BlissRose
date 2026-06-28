from __future__ import annotations

import asyncio
import io
from typing import Any

import httpx
from PIL import Image

from ..model_manager import model_manager
from ..schemas import InferenceResult


async def _infer_one(
    image: Image.Image,
    instance_id: str,
    confidence: float | None,
    annotate: bool,
    source: str,
) -> InferenceResult:
    result = await asyncio.to_thread(
        model_manager.get(instance_id).predict_pil,
        image,
        confidence,
        annotate,
    )
    result.source = source
    return result


async def batch_infer_images(
    files: list[tuple[str, bytes]],
    instance_id: str,
    confidence: float | None = None,
    annotate: bool = True,
) -> dict[str, Any]:
    if not model_manager.is_ready(instance_id):
        raise RuntimeError(f"推理实例 {instance_id} 未就绪，请先启动")

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    total_ms = 0.0

    for filename, content in files:
        try:
            image = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception as exc:
            errors.append({"source": filename, "error": f"无效图片: {exc}"})
            continue
        try:
            result = await _infer_one(image, instance_id, confidence, annotate, filename)
            total_ms += result.inference_ms
            results.append(result.model_dump())
        except Exception as exc:
            errors.append({"source": filename, "error": str(exc)})

    return {
        "instance_id": instance_id,
        "total": len(files),
        "success": len(results),
        "failed": len(errors),
        "total_inference_ms": round(total_ms, 2),
        "results": results,
        "errors": errors,
    }


async def batch_infer_urls(
    urls: list[str],
    instance_id: str,
    confidence: float | None = None,
    annotate: bool = True,
) -> dict[str, Any]:
    if not model_manager.is_ready(instance_id):
        raise RuntimeError(f"推理实例 {instance_id} 未就绪，请先启动")

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    total_ms = 0.0

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        for url in urls:
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                image = Image.open(io.BytesIO(resp.content)).convert("RGB")
            except Exception as exc:
                errors.append({"source": url, "error": f"无法下载: {exc}"})
                continue
            try:
                result = await _infer_one(image, instance_id, confidence, annotate, url)
                total_ms += result.inference_ms
                results.append(result.model_dump())
            except Exception as exc:
                errors.append({"source": url, "error": str(exc)})

    return {
        "instance_id": instance_id,
        "total": len(urls),
        "success": len(results),
        "failed": len(errors),
        "total_inference_ms": round(total_ms, 2),
        "results": results,
        "errors": errors,
    }
