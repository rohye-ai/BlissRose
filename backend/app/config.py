from __future__ import annotations

import json
import threading
from datetime import datetime
from typing import Any

import yaml

from .database import SessionLocal
from .db_models import AppSetting
from .schemas import AppConfig, InferenceInstanceConfig, ModelConfig, TrainingConfig, VideoConfig

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "default.yaml"
USER_CONFIG_PATH = ROOT_DIR / "config" / "user.yaml"


class ConfigStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._config = self._load()

    def _read_setting(self, key: str) -> dict[str, Any]:
        db = SessionLocal()
        try:
            row = db.query(AppSetting).filter(AppSetting.key == key).first()
            if row:
                return json.loads(row.value)
        except Exception:
            pass
        finally:
            db.close()
        return {}

    def _write_setting(self, key: str, data: dict[str, Any]) -> None:
        db = SessionLocal()
        try:
            row = db.query(AppSetting).filter(AppSetting.key == key).first()
            payload = json.dumps(data, ensure_ascii=False)
            if row:
                row.value = payload
                row.updated_at = datetime.utcnow()
            else:
                db.add(AppSetting(key=key, value=payload))
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def _load_yaml_defaults(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if DEFAULT_CONFIG_PATH.exists():
            with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        return data

    def _load(self) -> AppConfig:
        db_data = self._read_setting("app_config")
        if not db_data:
            yaml_data = self._load_yaml_defaults()
            if USER_CONFIG_PATH.exists():
                with USER_CONFIG_PATH.open("r", encoding="utf-8") as f:
                    user = yaml.safe_load(f) or {}
                    db_data = _deep_merge(yaml_data, user)
            else:
                db_data = yaml_data
            db_data = {k: v for k, v in db_data.items() if k not in ("server", "paths")}
            if db_data:
                self._write_setting("app_config", db_data)
        else:
            defaults = self._load_yaml_defaults()
            defaults = {k: v for k, v in defaults.items() if k not in ("server", "paths")}
            db_data = _deep_merge(defaults, db_data)

        model = ModelConfig(**(db_data.get("model") or {}))
        instances, default_id = self._load_instances_from_db(model, db_data)
        if not any(i.id == default_id for i in instances) and instances:
            default_id = instances[0].id

        return AppConfig(
            model=model,
            video=VideoConfig(**(db_data.get("video") or {})),
            training=TrainingConfig(**(db_data.get("training") or {})),
            inference_instances=instances,
            default_instance_id=default_id,
        )

    def _load_instances_from_db(self, model: ModelConfig, db_data: dict[str, Any]) -> tuple[list[InferenceInstanceConfig], str]:
        from .database import Base, engine
        from .instance_service import list_inference_instances, migrate_instances_from_json
        from .db_models import InferenceInstanceRecord

        Base.metadata.create_all(bind=engine)
        db = SessionLocal()
        try:
            migrate_instances_from_json(db)
            instances = list_inference_instances(db)
            default_id = db_data.get("default_instance_id") or "default"
            if not instances:
                rec = InferenceInstanceRecord(
                    id=default_id if default_id != "default" else "default",
                    name="默认推理实例",
                    size=model.size.value,
                    checkpoint=model.checkpoint,
                    gpu_ids=json.dumps(_parse_gpu_ids_from_device(model.device)),
                    confidence=model.confidence,
                    resolution=model.resolution,
                    optimize_inference=model.optimize_inference,
                    class_names=json.dumps(model.class_names, ensure_ascii=False),
                )
                db.add(rec)
                db.commit()
                instances = list_inference_instances(db)
                default_id = instances[0].id if instances else "default"
            return instances, default_id
        finally:
            db.close()

    def get(self) -> AppConfig:
        with self._lock:
            return self._config.model_copy(deep=True)

    def reload(self) -> AppConfig:
        with self._lock:
            self._config = self._load()
            return self._config.model_copy(deep=True)

    def update(self, patch: dict[str, Any]) -> AppConfig:
        with self._lock:
            # inference_instances 已入库，忽略 JSON 中的实例 patch
            patch = {k: v for k, v in patch.items() if k != "inference_instances"}
            current = self._config.model_dump(mode="json")
            merged = _deep_merge(current, patch)
            self._config = AppConfig(**merged)
            payload = self._config.model_dump(mode="json")
            payload.pop("inference_instances", None)
            self._write_setting("app_config", payload)
            return self._config.model_copy(deep=True)

    @staticmethod
    def get_server_config() -> dict[str, Any]:
        db = SessionLocal()
        try:
            row = db.query(AppSetting).filter(AppSetting.key == "server").first()
            if row:
                return json.loads(row.value)
        except Exception:
            pass
        finally:
            db.close()
        if DEFAULT_CONFIG_PATH.exists():
            with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                return data.get("server") or {"host": "0.0.0.0", "port": 8080}
        return {"host": "0.0.0.0", "port": 8080}

    @staticmethod
    def ensure_dirs() -> None:
        for rel in ("data/uploads", "data/results", "data/logs", "data/results/alerts", "outputs/train", "models", "datasets"):
            (ROOT_DIR / rel).mkdir(parents=True, exist_ok=True)


def _parse_gpu_ids_from_device(device: str) -> list[int]:
    device = (device or "auto").strip().lower()
    if device.startswith("cuda:"):
        try:
            return [int(device.split(":", 1)[1])]
        except ValueError:
            return [0]
    return [0]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


config_store = ConfigStore()
