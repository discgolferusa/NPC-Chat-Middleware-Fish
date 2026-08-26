"""Unit tests for memory DB config resolution (MySQL vs SQLite)."""

from __future__ import annotations

import memory_db
import pytest


def test_resolve_sqlite_default_is_repo_file():
    cfg = memory_db.resolve_memory_db_config(env={})
    assert cfg.is_sqlite
    assert cfg.sqlite_path == memory_db.BASE_DIR / "npc_memory.db"


def test_resolve_sqlite_respects_memory_db_env(tmp_path):
    configured = tmp_path / "shared.db"
    cfg = memory_db.resolve_memory_db_config(env={"MEMORY_DB": str(configured)})
    assert cfg.is_sqlite
    assert cfg.sqlite_path == configured


def test_resolve_mysql_when_host_set():
    cfg = memory_db.resolve_memory_db_config(
        env={
            "MEMORY_MYSQL_HOST": "db.example",
            "MEMORY_MYSQL_PORT": "3307",
            "MEMORY_MYSQL_DATABASE": "npc_middleware",
            "MEMORY_MYSQL_USER": "mw",
            "MEMORY_MYSQL_PASSWORD": "secret",
        }
    )
    assert cfg.is_mysql
    assert cfg.mysql_host == "db.example"
    assert cfg.mysql_port == 3307
    assert cfg.mysql_database == "npc_middleware"
    assert cfg.mysql_user == "mw"
    assert cfg.mysql_password == "secret"


def test_resolve_mysql_requires_user():
    with pytest.raises(ValueError, match="MEMORY_MYSQL_USER"):
        memory_db.resolve_memory_db_config(env={"MEMORY_MYSQL_HOST": "localhost"})


def test_resolve_mysql_defaults_database_and_port():
    cfg = memory_db.resolve_memory_db_config(
        env={
            "MEMORY_MYSQL_HOST": "127.0.0.1",
            "MEMORY_MYSQL_USER": "mw",
            "MEMORY_MYSQL_PASSWORD": "x",
        }
    )
    assert cfg.mysql_port == 3306
    assert cfg.mysql_database == "npc_middleware"


def test_refresh_memory_db_config_updates_module(monkeypatch, tmp_path):
    configured = tmp_path / "volume.db"
    monkeypatch.setenv("MEMORY_DB", str(configured))
    for key in (
        "MEMORY_MYSQL_HOST",
        "MEMORY_MYSQL_USER",
        "MEMORY_MYSQL_PASSWORD",
    ):
        monkeypatch.delenv(key, raising=False)
    refreshed = memory_db.refresh_memory_db_config()
    assert refreshed.is_sqlite
    assert refreshed.sqlite_path == configured
    assert memory_db.MEMORY_DB == configured


def test_qmark_to_percent():
    assert memory_db._qmark_to_percent("SELECT * FROM t WHERE a = ? AND b = ?") == (
        "SELECT * FROM t WHERE a = %s AND b = %s"
    )


def test_sqlite_init_creates_tables(tmp_path):
    path = tmp_path / "init.db"
    cfg = memory_db.MemoryDbConfig(backend="sqlite", sqlite_path=path)
    with memory_db.connect(cfg) as conn:
        memory_db.init_npc_memory_tables(conn)
        memory_db.init_dm_schema_tables(conn)
        conn.commit()

    with memory_db.connect(cfg) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert "conversation_turns" in tables
    assert "npc_memories" in tables
    assert "dm_messages" in tables
    assert "dm_pending_checks" in tables
    assert "dm_rulings" in tables


_ALLOWED = frozenset({
    "conversation_turns",
    "npc_memories",
    "dm_messages",
    "dm_pending_checks",
    "dm_rulings",
})


def test_ensure_campaign_id_column_idempotent_sqlite(tmp_path):
    path = tmp_path / "campaign_idx.db"
    cfg = memory_db.MemoryDbConfig(backend="sqlite", sqlite_path=path)
    with memory_db.connect(cfg) as conn:
        memory_db.init_npc_memory_tables(conn)
        memory_db.ensure_campaign_id_column(
            conn, "npc_memories", _ALLOWED, backfill="greyhaven_intro"
        )
        memory_db.ensure_campaign_id_column(
            conn, "npc_memories", _ALLOWED, backfill="greyhaven_intro"
        )
        conn.commit()
        assert memory_db.index_exists(
            conn, "npc_memories", "idx_npc_memories_campaign_npc_player", _ALLOWED
        )


def test_ensure_campaign_id_mysql_create_index_has_no_if_not_exists(monkeypatch):
    """Oracle MySQL rejects CREATE INDEX IF NOT EXISTS; we must not emit it."""
    executed: list[str] = []

    class _FakeCfg:
        is_mysql = True
        is_sqlite = False

    class _FakeConn:
        config = _FakeCfg()

        def execute(self, sql, params=None):
            executed.append(sql)
            # First INFORMATION_SCHEMA.COLUMNS → no campaign_id
            # Then ALTER, UPDATE, INFORMATION_SCHEMA.STATISTICS → no index
            # Then CREATE INDEX
            class _Result:
                def fetchall(self_inner):
                    if "INFORMATION_SCHEMA.COLUMNS" in sql:
                        return []
                    if "INFORMATION_SCHEMA.STATISTICS" in sql:
                        return []
                    return []

            return _Result()

    memory_db.ensure_campaign_id_column(
        _FakeConn(),  # type: ignore[arg-type]
        "npc_memories",
        _ALLOWED,
        backfill="greyhaven_intro",
    )
    create_sql = next(s for s in executed if s.strip().upper().startswith("CREATE INDEX"))
    assert "IF NOT EXISTS" not in create_sql.upper()
    assert "idx_npc_memories_campaign_npc_player" in create_sql
