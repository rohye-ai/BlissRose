from __future__ import annotations

from datetime import datetime


def utc_now() -> datetime:
    return datetime.utcnow()


def dt_to_str(dt: datetime | None) -> str:
    return dt.isoformat() if dt else ""


def apply_create_audit(username: str) -> dict[str, str]:
    now = utc_now()
    user = username or ""
    return {
        "created_at": now,
        "updated_at": now,
        "created_by": user,
        "updated_by": user,
    }


def apply_update_audit(username: str) -> dict:
    return {
        "updated_at": utc_now(),
        "updated_by": username or "",
    }


def apply_create_audit_iso(username: str) -> dict[str, str]:
    now = utc_now().isoformat()
    user = username or ""
    return {
        "created_at": now,
        "updated_at": now,
        "created_by": user,
        "updated_by": user,
    }


def apply_update_audit_iso(username: str) -> dict[str, str]:
    return {
        "updated_at": utc_now().isoformat(),
        "updated_by": username or "",
    }
