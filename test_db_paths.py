"""Unit tests for shared MEMORY_DB resolution."""

from __future__ import annotations

from pathlib import Path

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
    refreshed = db_paths.refresh_memory_db()
    assert refreshed == configured
    assert db_paths.MEMORY_DB == configured
