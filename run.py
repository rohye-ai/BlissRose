#!/usr/bin/env python3
"""Launch RF-DETR Platform server."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))

import uvicorn


def load_server_config() -> dict:
    from app.config import config_store

    server = config_store.get_server_config()
    return {"host": server.get("host", "0.0.0.0"), "port": int(server.get("port", 8080))}


def main() -> None:
    cfg = load_server_config()
    uvicorn.run(
        "app.main:app",
        host=cfg["host"],
        port=cfg["port"],
        reload=False,
        app_dir=str(ROOT / "backend"),
    )


if __name__ == "__main__":
    main()
