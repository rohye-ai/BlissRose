from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Annotated

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session, joinedload

from .database import get_db
from .db_models import Role, User

SECRET_KEY = os.environ.get("RFDETR_SECRET_KEY", "rf-detr-platform-dev-secret-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

security = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def create_access_token(user_id: int, username: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    payload = {"sub": str(user_id), "username": username, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="无效或已过期的令牌") from exc


def get_user_menu_keys(user: User) -> list[str]:
    if user.is_superuser:
        from .seed import ALL_MENU_KEYS

        return ALL_MENU_KEYS
    keys: set[str] = set()
    for role in user.roles:
        for menu in role.menus:
            keys.add(menu.key)
    return sorted(keys, key=lambda k: _menu_sort_key(k, user))


def _menu_sort_key(key: str, user: User) -> int:
    for role in user.roles:
        for menu in role.menus:
            if menu.key == key:
                return menu.sort_order
    return 999


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    db: Annotated[Session, Depends(get_db)],
) -> User:
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="未登录")
    payload = decode_token(credentials.credentials)
    user_id = int(payload.get("sub", 0))
    user = (
        db.query(User)
        .options(joinedload(User.roles).joinedload(Role.menus))
        .filter(User.id == user_id, User.is_active.is_(True))
        .first()
    )
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户不存在或已禁用")
    return user


async def get_current_superuser(user: Annotated[User, Depends(get_current_user)]) -> User:
    if not user.is_superuser:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="需要管理员权限")
    return user


def user_to_dict(user: User) -> dict:
    from .audit import dt_to_str

    return {
        "id": user.id,
        "username": user.username,
        "display_name": user.display_name or user.username,
        "is_active": user.is_active,
        "is_superuser": user.is_superuser,
        "roles": [{"id": r.id, "name": r.name, "description": r.description} for r in user.roles],
        "menus": get_user_menu_keys(user),
        "created_by": getattr(user, "created_by", "") or "",
        "updated_by": getattr(user, "updated_by", "") or "",
        "created_at": dt_to_str(getattr(user, "created_at", None)),
        "updated_at": dt_to_str(getattr(user, "updated_at", None)),
    }


def role_to_dict(role: Role) -> dict:
    from .audit import dt_to_str

    return {
        "id": role.id,
        "name": role.name,
        "description": role.description,
        "menu_keys": [m.key for m in role.menus],
        "menus": [{"id": m.id, "key": m.key, "label": m.label} for m in role.menus],
        "created_by": getattr(role, "created_by", "") or "",
        "updated_by": getattr(role, "updated_by", "") or "",
        "created_at": dt_to_str(getattr(role, "created_at", None)),
        "updated_at": dt_to_str(getattr(role, "updated_at", None)),
    }
