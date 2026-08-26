"""Functional MySQL memory DB (skipped unless MEMORY_MYSQL_* is set for tests)."""

from __future__ import annotations

import os
import uuid

import pytest

import memory_db


def _mysql_env_ready() -> str | None:
    if not (os.environ.get("MEMORY_MYSQL_HOST") or "").strip():
        return "Set MEMORY_MYSQL_HOST (and user/password/database) to run MySQL memory tests"
    if not (os.environ.get("MEMORY_MYSQL_USER") or "").strip():
        return "Set MEMORY_MYSQL_USER to run MySQL memory tests"
    return None


@pytest.mark.skipif(_mysql_env_ready() is not None, reason=_mysql_env_ready() or "mysql")
def test_mysql_init_and_roundtrip():
    cfg = memory_db.resolve_memory_db_config()
    assert cfg.is_mysql

    marker = f"test-memory-{uuid.uuid4().hex}"
    with memory_db.connect(cfg) as conn:
        memory_db.init_npc_memory_tables(conn)
        memory_db.init_dm_schema_tables(conn)
        conn.execute(
            """
            INSERT INTO npc_memories (campaign_id, npc_id, player_id, memory, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("camp_test", "npc_test", "player_test", marker, "2026-01-01T00:00:00Z"),
        )
        conn.commit()

    with memory_db.connect(cfg) as conn:
        rows = conn.execute(
            """
            SELECT memory FROM npc_memories
            WHERE campaign_id = ? AND npc_id = ? AND player_id = ? AND memory = ?
            """,
            ("camp_test", "npc_test", "player_test", marker),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["memory"] == marker
        conn.execute(
            """
            DELETE FROM npc_memories
            WHERE campaign_id = ? AND npc_id = ? AND player_id = ? AND memory = ?
            """,
            ("camp_test", "npc_test", "player_test", marker),
        )
        conn.commit()
