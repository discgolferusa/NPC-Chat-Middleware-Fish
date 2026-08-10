"""Shared SQLite path for NPC memory and DM channel tables."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
_DEFAULT_MEMORY_DB = BASE_DIR / "npc_memory.db"


def resolve_memory_db(env: dict[str, str] | None = None) -> Path:
    """Resolve the single SQLite file used by NPC memory and DM tables.

    ``MEMORY_DB`` env (Compose / game_api.env) wins when set; otherwise
    ``<repo>/npc_memory.db``. Call again after loading ``game_api.env`` so
    file-based config applies to every module that imports this path.
    """
    environ = os.environ if env is None else env
    configured = (environ.get("MEMORY_DB") or "").strip().strip('"').strip("'")
    if configured:
        return Path(configured)
    return _DEFAULT_MEMORY_DB


# Import-time default (Compose sets MEMORY_DB before process start).
MEMORY_DB = resolve_memory_db()


def refresh_memory_db() -> Path:
    """Re-read ``MEMORY_DB`` from the environment and update this module."""
    global MEMORY_DB
    MEMORY_DB = resolve_memory_db()
    return MEMORY_DB
