from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, joinedload

from .audit import apply_create_audit, apply_update_audit
from .auth import (
    create_access_token,
    get_current_superuser,
    get_current_user,
    hash_password,
    role_to_dict,
    user_to_dict,
    verify_password,
)
from .database import get_db
from .db_models import Menu, Role, User

router = APIRouter(prefix="/api/auth", tags=["auth"])
admin_router = APIRouter(prefix="/api/admin", tags=["admin"])


class LoginRequest(BaseModel):
    username: str
    password: str


class UserCreateRequest(BaseModel):
    username: str = Field(min_length=2, max_length=64)
    password: str = Field(min_length=6, max_length=128)
    display_name: str = ""
    role_ids: list[int] = Field(default_factory=list)
    is_active: bool = True


class UserUpdateRequest(BaseModel):
    display_name: str | None = None
    password: str | None = Field(default=None, min_length=6, max_length=128)
    role_ids: list[int] | None = None
    is_active: bool | None = None


class RoleCreateRequest(BaseModel):
    name: str = Field(min_length=2, max_length=64)
    description: str = ""
    menu_keys: list[str] = Field(default_factory=list)


class RoleUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=64)
    description: str | None = None
    menu_keys: list[str] | None = None


@router.post("/login")
def login(body: LoginRequest, db: Annotated[Session, Depends(get_db)]) -> dict[str, Any]:
    user = (
        db.query(User)
        .options(joinedload(User.roles).joinedload(Role.menus))
        .filter(User.username == body.username)
        .first()
    )
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="账号已被禁用")
    token = create_access_token(user.id, user.username)
    return {"access_token": token, "token_type": "bearer", "user": user_to_dict(user)}


@router.get("/me")
def me(user: Annotated[User, Depends(get_current_user)]) -> dict[str, Any]:
    return user_to_dict(user)


@router.get("/menus")
def list_all_menus(
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(get_current_user)],
) -> dict[str, Any]:
    menus = db.query(Menu).order_by(Menu.sort_order).all()
    return {
        "menus": [
            {"id": m.id, "key": m.key, "label": m.label, "icon": m.icon, "sort_order": m.sort_order}
            for m in menus
        ]
    }


@admin_router.get("/users")
def list_users(
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(get_current_superuser)],
) -> dict[str, Any]:
    users = db.query(User).options(joinedload(User.roles)).order_by(User.created_at.desc()).all()
    return {"users": [user_to_dict(u) for u in users]}


@admin_router.post("/users")
def create_user(
    body: UserCreateRequest,
    db: Annotated[Session, Depends(get_db)],
    current: Annotated[User, Depends(get_current_superuser)],
) -> dict[str, Any]:
    if db.query(User).filter(User.username == body.username).first():
        raise HTTPException(status_code=400, detail="用户名已存在")
    roles = db.query(Role).filter(Role.id.in_(body.role_ids)).all() if body.role_ids else []
    audit = apply_create_audit(current.username)
    user = User(
        username=body.username,
        password_hash=hash_password(body.password),
        display_name=body.display_name or body.username,
        is_active=body.is_active,
        roles=roles,
        created_by=audit["created_by"],
        updated_by=audit["updated_by"],
        created_at=audit["created_at"],
        updated_at=audit["updated_at"],
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"user": user_to_dict(user)}


@admin_router.put("/users/{user_id}")
def update_user(
    user_id: int,
    body: UserUpdateRequest,
    db: Annotated[Session, Depends(get_db)],
    current: Annotated[User, Depends(get_current_superuser)],
) -> dict[str, Any]:
    user = db.query(User).options(joinedload(User.roles).joinedload(Role.menus)).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    if user.is_superuser and user.id != current.id and body.is_active is False:
        raise HTTPException(status_code=400, detail="不能禁用超级管理员")
    if body.display_name is not None:
        user.display_name = body.display_name
    if body.password:
        user.password_hash = hash_password(body.password)
    if body.is_active is not None:
        user.is_active = body.is_active
    if body.role_ids is not None:
        if user.is_superuser:
            raise HTTPException(status_code=400, detail="不能修改超级管理员的角色")
        user.roles = db.query(Role).filter(Role.id.in_(body.role_ids)).all()
    audit = apply_update_audit(current.username)
    user.updated_at = audit["updated_at"]
    user.updated_by = audit["updated_by"]
    db.commit()
    db.refresh(user)
    return {"user": user_to_dict(user)}


@admin_router.delete("/users/{user_id}")
def delete_user(
    user_id: int,
    db: Annotated[Session, Depends(get_db)],
    current: Annotated[User, Depends(get_current_superuser)],
) -> dict[str, str]:
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    if user.is_superuser:
        raise HTTPException(status_code=400, detail="不能删除超级管理员")
    if user.id == current.id:
        raise HTTPException(status_code=400, detail="不能删除当前登录用户")
    db.delete(user)
    db.commit()
    return {"message": "用户已删除"}


@admin_router.get("/roles")
def list_roles(
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(get_current_superuser)],
) -> dict[str, Any]:
    roles = db.query(Role).options(joinedload(Role.menus)).order_by(Role.created_at.desc()).all()
    return {"roles": [role_to_dict(r) for r in roles]}


@admin_router.post("/roles")
def create_role(
    body: RoleCreateRequest,
    db: Annotated[Session, Depends(get_db)],
    current: Annotated[User, Depends(get_current_superuser)],
) -> dict[str, Any]:
    if db.query(Role).filter(Role.name == body.name).first():
        raise HTTPException(status_code=400, detail="角色名已存在")
    if body.name == "admin":
        raise HTTPException(status_code=400, detail="不能创建名为 admin 的角色")
    menus = db.query(Menu).filter(Menu.key.in_(body.menu_keys)).all()
    audit = apply_create_audit(current.username)
    role = Role(
        name=body.name,
        description=body.description,
        menus=menus,
        created_by=audit["created_by"],
        updated_by=audit["updated_by"],
        created_at=audit["created_at"],
        updated_at=audit["updated_at"],
    )
    db.add(role)
    db.commit()
    db.refresh(role)
    return {"role": role_to_dict(role)}


@admin_router.put("/roles/{role_id}")
def update_role(
    role_id: int,
    body: RoleUpdateRequest,
    db: Annotated[Session, Depends(get_db)],
    current: Annotated[User, Depends(get_current_superuser)],
) -> dict[str, Any]:
    role = db.query(Role).options(joinedload(Role.menus)).filter(Role.id == role_id).first()
    if not role:
        raise HTTPException(status_code=404, detail="角色不存在")
    if role.name in ("admin", "operator", "viewer") and body.name is not None and body.name != role.name:
        raise HTTPException(status_code=400, detail="不能修改内置角色名称")
    if role.name == "admin":
        if body.menu_keys is not None:
            raise HTTPException(status_code=400, detail="admin 角色拥有全部菜单权限，无需修改")
    if body.name is not None:
        if body.name == "admin":
            raise HTTPException(status_code=400, detail="不能使用 admin 作为角色名")
        role.name = body.name
    if body.description is not None:
        role.description = body.description
    if body.menu_keys is not None:
        role.menus = db.query(Menu).filter(Menu.key.in_(body.menu_keys)).all()
    audit = apply_update_audit(current.username)
    role.updated_at = audit["updated_at"]
    role.updated_by = audit["updated_by"]
    db.commit()
    db.refresh(role)
    return {"role": role_to_dict(role)}


@admin_router.delete("/roles/{role_id}")
def delete_role(
    role_id: int,
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(get_current_superuser)],
) -> dict[str, str]:
    role = db.query(Role).filter(Role.id == role_id).first()
    if not role:
        raise HTTPException(status_code=404, detail="角色不存在")
    if role.name in ("admin", "operator", "viewer"):
        raise HTTPException(status_code=400, detail="不能删除内置角色")
    db.delete(role)
    db.commit()
    return {"message": "角色已删除"}
