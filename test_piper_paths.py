"""Unit tests for Piper executable resolution (Windows + Linux/Docker)."""

from __future__ import annotations

from pathlib import Path

import pytest

from piper_paths import LEGACY_WINDOWS_PIPER_EXE, resolve_piper_exe


def test_resolve_piper_exe_prefers_env_when_file_exists(tmp_path):
    piper = tmp_path / "custom-piper"
    piper.write_text("", encoding="utf-8")

    resolved = resolve_piper_exe(
        env={"PIPER_EXE": str(piper)},
        which=lambda _name: None,
        legacy_windows=tmp_path / "missing-legacy.exe",
    )

    assert resolved == piper


def test_resolve_piper_exe_ignores_missing_env_and_uses_path(tmp_path):
    on_path = tmp_path / "piper"
    on_path.write_text("", encoding="utf-8")

    resolved = resolve_piper_exe(
        env={"PIPER_EXE": str(tmp_path / "does-not-exist")},
        which=lambda name: str(on_path) if name == "piper" else None,
        legacy_windows=tmp_path / "missing-legacy.exe",
    )

    assert resolved == on_path


def test_resolve_piper_exe_uses_legacy_windows_when_present(tmp_path, monkeypatch):
    legacy = tmp_path / "piper.exe"
    legacy.write_text("", encoding="utf-8")

    # Ensure PATH lookup fails so legacy is exercised.
    resolved = resolve_piper_exe(
        env={},
        which=lambda _name: None,
        legacy_windows=legacy,
    )

    assert resolved == legacy
    # Document the real constant still points at the historical Ryzen path.
    assert LEGACY_WINDOWS_PIPER_EXE == Path(r"C:\AI\npc-server\.venv\Scripts\piper.exe")


def test_resolve_piper_exe_raises_when_nothing_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="Piper executable not found"):
        resolve_piper_exe(
            env={},
            which=lambda _name: None,
            legacy_windows=tmp_path / "missing.exe",
        )


def test_legacy_windows_path_wins_when_no_env_and_file_exists(tmp_path):
    """Plan requirement: with no PIPER_EXE, existing Windows install still wins."""
    legacy = tmp_path / "Scripts" / "piper.exe"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("", encoding="utf-8")

    resolved = resolve_piper_exe(
        env={},
        which=lambda _name: None,
        legacy_windows=legacy,
    )
    assert resolved == legacy
