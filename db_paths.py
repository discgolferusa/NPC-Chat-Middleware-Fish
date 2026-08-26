"""Back-compat shim — prefer ``memory_db`` for new code."""

from __future__ import annotations

from typing import Any

import memory_db


def __getattr__(name: str) -> Any:
    return getattr(memory_db, name)


def resolve_memory_db(env=None):
    return memory_db.resolve_memory_db(env)


def refresh_memory_db():
    return memory_db.refresh_memory_db()


def refresh_memory_db_config():
    return memory_db.refresh_memory_db_config()


def resolve_memory_db_config(env=None):
    return memory_db.resolve_memory_db_config(env)


def get_db():
    return memory_db.get_db()


def connect(config=None):
    return memory_db.connect(config)
