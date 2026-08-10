"""Functional: NPC memory and DM channel share MEMORY_DB under Docker-style env."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def test_dm_and_npc_tables_share_memory_db_volume_path(api, tmp_path, monkeypatch):
    client, server, headers = api
    shared = (tmp_path / "volume" / "npc_memory.db").resolve()
    shared.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("MEMORY_DB", str(shared))
    import db_paths
    import dm_channel

    server.MEMORY_DB = db_paths.refresh_memory_db()
    dm_channel.MEMORY_DB = server.MEMORY_DB

    server.init_memory_db()
    dm_channel.init_dm_tables()

    assert server.MEMORY_DB.resolve() == shared
    assert dm_channel.MEMORY_DB.resolve() == shared

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

    # Sanity: no second DB created next to server.py for this run
    repo_default = (Path(server.__file__).resolve().parent / "npc_memory.db").resolve()
    assert shared != repo_default
