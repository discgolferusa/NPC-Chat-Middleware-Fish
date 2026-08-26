"""Functional: NPC memory and DM channel share the same memory DB config."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def test_dm_and_npc_tables_share_sqlite_memory_db(api, tmp_path, monkeypatch):
    client, server, headers = api
    shared = (tmp_path / "volume" / "npc_memory.db").resolve()
    shared.parent.mkdir(parents=True, exist_ok=True)

    for key in (
        "MEMORY_MYSQL_HOST",
        "MEMORY_MYSQL_USER",
        "MEMORY_MYSQL_PASSWORD",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MEMORY_DB", str(shared))

    import dm_channel
    import memory_db

    cfg = memory_db.refresh_memory_db_config()
    server.MEMORY_DB = memory_db.MEMORY_DB
    server.MEMORY_DB_CONFIG = cfg
    dm_channel.MEMORY_DB = server.MEMORY_DB

    server.init_memory_db()
    dm_channel.init_dm_tables()

    assert server.MEMORY_DB.resolve() == shared
    assert dm_channel.MEMORY_DB.resolve() == shared
    assert cfg.is_sqlite

    with sqlite3.connect(shared) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }

    assert "conversation_turns" in tables or "npc_memories" in tables
    assert "dm_messages" in tables
    assert "dm_pending_checks" in tables
    assert "dm_rulings" in tables

    repo_default = (Path(server.__file__).resolve().parent / "npc_memory.db").resolve()
    assert shared != repo_default
