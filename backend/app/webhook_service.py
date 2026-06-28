from __future__ import annotations

import hashlib
import hmac
import base64
import json
import logging
import threading
import time
import urllib.parse
from datetime import datetime
from typing import Any

import httpx

from .config import ROOT_DIR
from .database import SessionLocal
from .db_models import AppSetting, DeviceRecord

logger = logging.getLogger(__name__)

WEBHOOK_SETTING_KEY = "webhook_config"

DEFAULT_WEBHOOK_CONFIG: dict[str, Any] = {
    "enabled": False,
    "min_confidence": 0.0,
    "device_ids": [],
    "channels": [],
}


def _read_webhook_config() -> dict[str, Any]:
    db = SessionLocal()
    try:
        row = db.query(AppSetting).filter(AppSetting.key == WEBHOOK_SETTING_KEY).first()
        if row:
            data = json.loads(row.value)
            merged = dict(DEFAULT_WEBHOOK_CONFIG)
            merged.update(data)
            return merged
    except Exception:
        pass
    finally:
        db.close()
    return dict(DEFAULT_WEBHOOK_CONFIG)


def _write_webhook_config(data: dict[str, Any]) -> dict[str, Any]:
    db = SessionLocal()
    try:
        row = db.query(AppSetting).filter(AppSetting.key == WEBHOOK_SETTING_KEY).first()
        payload = json.dumps(data, ensure_ascii=False)
        if row:
            row.value = payload
            row.updated_at = datetime.utcnow()
        else:
            db.add(AppSetting(key=WEBHOOK_SETTING_KEY, value=payload))
        db.commit()
        return data
    finally:
        db.close()


def get_webhook_config() -> dict[str, Any]:
    return _read_webhook_config()


def update_webhook_config(patch: dict[str, Any]) -> dict[str, Any]:
    current = _read_webhook_config()
    if "channels" in patch and isinstance(patch["channels"], list):
        current["channels"] = patch["channels"]
    for key in ("enabled", "min_confidence", "device_ids"):
        if key in patch:
            current[key] = patch[key]
    return _write_webhook_config(current)


def _dingtalk_signed_url(webhook_url: str, secret: str) -> str:
    timestamp = str(round(time.time() * 1000))
    string_to_sign = f"{timestamp}\n{secret}"
    sign = urllib.parse.quote_plus(
        base64.b64encode(hmac.new(secret.encode(), string_to_sign.encode(), hashlib.sha256).digest()).decode()
    )
    sep = "&" if "?" in webhook_url else "?"
    return f"{webhook_url}{sep}timestamp={timestamp}&sign={sign}"


def _build_payload(channel: dict[str, Any], alert: dict[str, Any]) -> dict[str, Any]:
    ctype = channel.get("type", "generic")
    title = alert.get("title", "视觉分析报警")
    text_lines = [
        f"### {title}",
        f"- 设备: {alert.get('device_name', alert.get('device_id', ''))}",
        f"- 实例: {alert.get('instance_name', alert.get('instance_id', ''))}",
        f"- 最高置信度: {alert.get('max_confidence', 0):.2f}",
        f"- 检测数: {alert.get('detection_count', 0)}",
        f"- 时间: {alert.get('alert_at', '')}",
    ]
    for det in alert.get("detections", [])[:5]:
        text_lines.append(f"  · {det.get('class_name', '')} {det.get('confidence', 0):.2f}")
    markdown_text = "\n".join(text_lines)

    if ctype == "dingtalk":
        return {"msgtype": "markdown", "markdown": {"title": title, "text": markdown_text}}
    if ctype == "wecom":
        return {"msgtype": "markdown", "markdown": {"content": markdown_text}}
    return {
        "event": "alert_created",
        "alert_id": alert.get("alert_id"),
        "device_id": alert.get("device_id"),
        "instance_id": alert.get("instance_id"),
        "max_confidence": alert.get("max_confidence"),
        "detection_count": alert.get("detection_count"),
        "detections": alert.get("detections", []),
        "image_url": alert.get("image_url", ""),
        "alert_at": alert.get("alert_at", ""),
    }


def _post_webhook(channel: dict[str, Any], payload: dict[str, Any]) -> None:
    url = channel.get("url", "").strip()
    if not url:
        return
    if channel.get("type") == "dingtalk" and channel.get("secret"):
        url = _dingtalk_signed_url(url, channel["secret"])
    headers = {"Content-Type": "application/json"}
    custom_headers = channel.get("headers") or {}
    if isinstance(custom_headers, dict):
        headers.update({str(k): str(v) for k, v in custom_headers.items()})
    timeout = float(channel.get("timeout", 10))
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()


def dispatch_alert_webhooks(
    alert_id: str,
    device_id: str,
    instance_id: str,
    instance_name: str,
    max_confidence: float,
    detections: list[dict[str, Any]],
    alert_at: str,
) -> None:
    cfg = _read_webhook_config()
    if not cfg.get("enabled"):
        return
    if max_confidence < float(cfg.get("min_confidence", 0)):
        return
    filter_devices = cfg.get("device_ids") or []
    if filter_devices and device_id not in filter_devices:
        return

    db = SessionLocal()
    try:
        device = db.query(DeviceRecord).filter(DeviceRecord.id == device_id).first()
        device_name = device.name if device else device_id
    finally:
        db.close()

    alert_payload = {
        "alert_id": alert_id,
        "title": f"【报警】{device_name}",
        "device_id": device_id,
        "device_name": device_name,
        "instance_id": instance_id,
        "instance_name": instance_name,
        "max_confidence": max_confidence,
        "detection_count": len(detections),
        "detections": detections,
        "image_url": f"/static/alerts/{alert_id}.jpg",
        "alert_at": alert_at,
    }

    channels = [c for c in (cfg.get("channels") or []) if c.get("enabled", True) and c.get("url")]
    if not channels:
        return

    def _run() -> None:
        for channel in channels:
            try:
                payload = _build_payload(channel, alert_payload)
                _post_webhook(channel, payload)
            except Exception as exc:
                logger.warning("Webhook 推送失败 [%s]: %s", channel.get("type"), exc)

    threading.Thread(target=_run, name=f"webhook-{alert_id}", daemon=True).start()


def test_webhook_channel(channel: dict[str, Any]) -> dict[str, Any]:
    sample = {
        "alert_id": "test",
        "title": "Webhook 测试消息",
        "device_id": "demo",
        "device_name": "测试设备",
        "instance_id": "default",
        "instance_name": "测试实例",
        "max_confidence": 0.95,
        "detection_count": 1,
        "detections": [{"class_name": "test", "confidence": 0.95, "bbox": [0, 0, 100, 100]}],
        "image_url": "",
        "alert_at": datetime.utcnow().isoformat(),
    }
    payload = _build_payload(channel, sample)
    _post_webhook(channel, payload)
    return {"ok": True, "message": "测试消息已发送"}
