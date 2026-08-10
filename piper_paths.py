"""Resolve the Piper CLI path for Windows (dev) and Linux (Docker)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

# Historical Ryzen / local Windows install. Kept as a last-resort fallback so
# existing Funnel/dev setups keep working with zero env changes.
LEGACY_WINDOWS_PIPER_EXE = Path(r"C:\AI\npc-server\.venv\Scripts\piper.exe")


def resolve_piper_exe(
    *,
    env: dict[str, str] | None = None,
    which=shutil.which,
    legacy_windows: Path = LEGACY_WINDOWS_PIPER_EXE,
) -> Path:
    """Locate the Piper executable.

    Order:
    1. ``PIPER_EXE`` env (or game_api.env → os.environ) when the path exists
    2. ``piper`` / ``piper.exe`` on PATH
    3. Legacy Windows path if that file exists
    4. Raise FileNotFoundError with a clear message
    """
    environ = os.environ if env is None else env

    configured = (environ.get("PIPER_EXE") or "").strip().strip('"').strip("'")
    if configured:
        configured_path = Path(configured)
        if configured_path.is_file():
            return configured_path

    for candidate_name in ("piper", "piper.exe"):
        found = which(candidate_name)
        if found:
            found_path = Path(found)
            if found_path.is_file():
                return found_path

    if legacy_windows.is_file():
        return legacy_windows

    hints: list[str] = []
    if configured:
        hints.append(f"PIPER_EXE was set to {configured!r} but that file was not found")
    hints.append("install piper-tts (or set PIPER_EXE to the piper binary)")
    hints.append(f"legacy Windows fallback missing: {legacy_windows}")
    raise FileNotFoundError("Piper executable not found. " + "; ".join(hints))
