from __future__ import annotations

from fastapi import Request
from starlette.responses import JSONResponse

from .auth import decode_token
from .database import SessionLocal
from .db_models import User

PUBLIC_API_PREFIXES = (
    "/api/auth/login",
    "/docs",
    "/openapi.json",
    "/redoc",
)


async def auth_http_middleware(request: Request, call_next):
    """HTTP-only auth middleware. Do NOT use BaseHTTPMiddleware — it breaks WebSocket."""
    path = request.url.path
    if not path.startswith("/api/"):
        return await call_next(request)
    if any(path.startswith(p) for p in PUBLIC_API_PREFIXES):
        return await call_next(request)

    token = _extract_token(request)
    if not token:
        return JSONResponse(status_code=401, content={"detail": "未登录"})

    try:
        payload = decode_token(token)
        user_id = int(payload.get("sub", 0))
    except Exception:
        return JSONResponse(status_code=401, content={"detail": "无效或已过期的令牌"})

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id, User.is_active.is_(True)).first()
        if not user:
            return JSONResponse(status_code=401, content={"detail": "用户不存在或已禁用"})
        request.state.user_id = user.id
        request.state.username = user.username
    finally:
        db.close()

    return await call_next(request)


def _extract_token(request: Request) -> str | None:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    return request.query_params.get("token")
