from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy.orm import Session

from .auth import hash_password
from .database import Base, engine
from .db_models import AppSetting, Menu, Role, User

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "default.yaml"
USER_CONFIG_PATH = ROOT_DIR / "config" / "user.yaml"

SYSTEM_MENUS = [
    {"key": "dashboard", "label": "总览", "icon": "📊", "sort_order": 1},
    {"key": "models", "label": "模型管理", "icon": "🧠", "sort_order": 2},
    {"key": "datasets", "label": "数据集", "icon": "📁", "sort_order": 3},
    {"key": "devices", "label": "设备管理", "icon": "📡", "sort_order": 4},
    {"key": "gpu", "label": "推理实例", "icon": "🖥️", "sort_order": 5},
    {"key": "alerts", "label": "报警管理", "icon": "🚨", "sort_order": 6},
    {"key": "infer", "label": "图片推理", "icon": "🖼️", "sort_order": 7},
    {"key": "video", "label": "视频流", "icon": "📹", "sort_order": 8},
    {"key": "train", "label": "模型训练", "icon": "🎯", "sort_order": 9},
    {"key": "settings", "label": "全局默认", "icon": "⚙️", "sort_order": 10},
    {"key": "api", "label": "API 文档", "icon": "📖", "sort_order": 11},
    {"key": "users", "label": "用户管理", "icon": "👤", "sort_order": 12},
    {"key": "roles", "label": "角色管理", "icon": "🔐", "sort_order": 13},
]

ALL_MENU_KEYS = [m["key"] for m in SYSTEM_MENUS]

DEFAULT_ROLES = [
    {
        "name": "admin",
        "description": "系统管理员，拥有全部权限",
        "menus": ALL_MENU_KEYS,
    },
    {
        "name": "operator",
        "description": "操作员，可执行推理与训练",
        "menus": ["dashboard", "models", "datasets", "devices", "gpu", "alerts", "infer", "video", "train", "api"],
    },
    {
        "name": "viewer",
        "description": "只读用户，仅可查看总览与 API",
        "menus": ["dashboard", "api"],
    },
]

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml_config() -> dict[str, Any]:
    data: dict[str, Any] = {}
    if DEFAULT_CONFIG_PATH.exists():
        with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    if USER_CONFIG_PATH.exists():
        with USER_CONFIG_PATH.open("r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
            data = _deep_merge(data, user)
    return data


def _migrate_platform_schema() -> None:
    """Add columns to existing SQLite tables without Alembic."""
    from sqlalchemy import inspect, text

    insp = inspect(engine)
    migrations = [
        ("models", "uploaded_by", "VARCHAR(64) DEFAULT ''"),
        ("datasets", "uploaded_by", "VARCHAR(64) DEFAULT ''"),
        ("users", "created_by", "VARCHAR(64) DEFAULT ''"),
        ("users", "updated_by", "VARCHAR(64) DEFAULT ''"),
        ("users", "updated_at", "DATETIME"),
        ("roles", "created_by", "VARCHAR(64) DEFAULT ''"),
        ("roles", "updated_by", "VARCHAR(64) DEFAULT ''"),
        ("roles", "updated_at", "DATETIME"),
        ("devices", "created_by", "VARCHAR(64) DEFAULT ''"),
        ("devices", "updated_by", "VARCHAR(64) DEFAULT ''"),
        ("devices", "updated_at", "DATETIME"),
        ("training_jobs", "created_by", "VARCHAR(64) DEFAULT ''"),
        ("training_jobs", "updated_by", "VARCHAR(64) DEFAULT ''"),
        ("training_jobs", "updated_at", "DATETIME"),
    ]
    with engine.connect() as conn:
        for table, column, col_type in migrations:
            if table not in insp.get_table_names():
                continue
            existing = {c["name"] for c in insp.get_columns(table)}
            if column in existing:
                continue
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"))
        # backfill updated_at from created_at where missing
        for table in ("users", "roles", "devices", "training_jobs"):
            if table not in insp.get_table_names():
                continue
            cols = {c["name"] for c in insp.get_columns(table)}
            if "updated_at" in cols and "created_at" in cols:
                conn.execute(text(
                    f"UPDATE {table} SET updated_at = created_at WHERE updated_at IS NULL"
                ))
        conn.commit()


def init_database(db: Session) -> None:
    Base.metadata.create_all(bind=engine)
    _migrate_platform_schema()
    (ROOT_DIR / "data").mkdir(parents=True, exist_ok=True)

    menu_by_key: dict[str, Menu] = {}
    for item in SYSTEM_MENUS:
        menu = db.query(Menu).filter(Menu.key == item["key"]).first()
        if not menu:
            menu = Menu(**item, is_system=True)
            db.add(menu)
        else:
            menu.label = item["label"]
            menu.icon = item["icon"]
            menu.sort_order = item["sort_order"]
        menu_by_key[item["key"]] = menu
    db.flush()

    admin_role = db.query(Role).filter(Role.name == "admin").first()
    operator_role = db.query(Role).filter(Role.name == "operator").first()
    if admin_role:
        admin_role.menus = [menu_by_key[k] for k in DEFAULT_ROLES[0]["menus"] if k in menu_by_key]
    if operator_role:
        operator_role.menus = [menu_by_key[k] for k in DEFAULT_ROLES[1]["menus"] if k in menu_by_key]

    for role_def in DEFAULT_ROLES:
        role = db.query(Role).filter(Role.name == role_def["name"]).first()
        if not role:
            role = Role(name=role_def["name"], description=role_def["description"])
            db.add(role)
            db.flush()
            role.menus = [menu_by_key[k] for k in role_def["menus"] if k in menu_by_key]
        else:
            role.description = role_def["description"]

    admin = db.query(User).filter(User.username == ADMIN_USERNAME).first()
    if not admin:
        admin_role = db.query(Role).filter(Role.name == "admin").first()
        admin = User(
            username=ADMIN_USERNAME,
            password_hash=hash_password(ADMIN_PASSWORD),
            display_name="系统管理员",
            is_active=True,
            is_superuser=True,
        )
        if admin_role:
            admin.roles = [admin_role]
        db.add(admin)

    if not db.query(AppSetting).filter(AppSetting.key == "app_config").first():
        yaml_cfg = _load_yaml_config()
        app_cfg = {k: v for k, v in yaml_cfg.items() if k not in ("server", "paths")}
        db.add(AppSetting(key="app_config", value=json.dumps(app_cfg, ensure_ascii=False)))

    if not db.query(AppSetting).filter(AppSetting.key == "server").first():
        yaml_cfg = _load_yaml_config()
        server = yaml_cfg.get("server") or {"host": "0.0.0.0", "port": 8080}
        db.add(AppSetting(key="server", value=json.dumps(server, ensure_ascii=False)))

    db.commit()
