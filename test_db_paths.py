"""Unit tests for db_paths back-compat shim."""

from __future__ import annotations

import db_paths


def test_resolve_memory_db_default_is_repo_file():
    path = db_paths.resolve_memory_db(env={})
    assert path == db_paths.BASE_DIR / "npc_memory.db"


def test_resolve_memory_db_respects_env(tmp_path):
    configured = tmp_path / "shared.db"
    path = db_paths.resolve_memory_db(env={"MEMORY_DB": str(configured)})
    assert path == configured


def test_refresh_memory_db_updates_module(monkeypatch, tmp_path):
    configured = tmp_path / "volume.db"
    monkeypatch.setenv("MEMORY_DB", str(configured))
    for key in ("MEMORY_MYSQL_HOST", "MEMORY_MYSQL_USER", "MEMORY_MYSQL_PASSWORD"):
        monkeypatch.delenv(key, raising=False)
    refreshed = db_paths.refresh_memory_db()
    assert refreshed == configured
    assert db_paths.MEMORY_DB == configured
    import memory_db

    assert memory_db.MEMORY_DB == configured
