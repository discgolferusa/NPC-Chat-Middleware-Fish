"""Resolve Piper voice MODEL_CARD attribution text from VOICES_DIR."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_MODEL_CARD_NAMES = (
    "MODEL_CARD",
    "MODEL_CARD.md",
    "MODEL_CARD.txt",
    "modelcard",
    "modelcard.md",
    "modelcard.txt",
)


def try_read_model_card(directory: Path) -> str | None:
    """Return MODEL_CARD text from a directory, or None if missing/unreadable."""
    if not directory.is_dir():
        return None

    for name in _MODEL_CARD_NAMES:
        path = directory / name
        if not path.is_file():
            continue
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

    # Case-insensitive scan for MODEL_CARD* / modelcard* on case-sensitive filesystems.
    try:
        for path in directory.iterdir():
            if not path.is_file():
                continue
            lower = path.name.lower()
            if lower.startswith("model_card") or lower.startswith("modelcard"):
                try:
                    return path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
    except OSError:
        return None

    return None


def _cache_key(directory: Path) -> str:
    """Stable cache key; normcase helps Windows UNC case variants share one read."""
    return os.path.normcase(str(directory))


def _card_search_dirs(voice_folder: Path, voices_dir: Path) -> list[Path]:
    """
    Logical parents from voice_folder up through voices_dir (closest first).

    Does **not** call Path.resolve(). Resolving junctions/symlinks can land outside
    voices_dir and abort the walk before pack folders like ``licensed/`` are seen.
    Paths are built as ``voices_dir / relative_parts`` so the walk stays in-tree.
    """
    try:
        rel = voice_folder.relative_to(voices_dir)
    except ValueError:
        # Caller passed a folder not under voices_dir — still check both endpoints.
        if voice_folder == voices_dir:
            return [voices_dir]
        return [voice_folder, voices_dir]

    parts = rel.parts
    dirs: list[Path] = []
    for depth in range(len(parts), -1, -1):
        dirs.append(voices_dir.joinpath(*parts[:depth]) if depth else voices_dir)
    return dirs


def resolve_model_card_text(
    voice_folder: Path,
    voices_dir: Path,
    *,
    card_cache: dict[str, str | None] | None = None,
) -> tuple[str | None, bool]:
    """
    Find MODEL_CARD for a voice.

    Search order: voice folder, then each logical parent up to and including
    ``voices_dir`` (so a card in a licensed/pack parent applies to nested models).

    ``card_cache`` maps directory cache keys to stripped card text (or None when
    absent). Pass one shared dict from ``build_voice_attributions`` so sibling
    voices under the same pack do not re-stat/read the same MODEL_CARD on UNC shares.

    Returns ``(text, used_ancestor_fallback)``. The fallback flag is True when the
    card came from a parent of the voice folder (including the voices root).
    """
    cache = card_cache if card_cache is not None else {}

    for index, directory in enumerate(_card_search_dirs(voice_folder, voices_dir)):
        key = _cache_key(directory)
        if key not in cache:
            raw = try_read_model_card(directory)
            cache[key] = raw.strip() if raw and raw.strip() else None
        text = cache[key]
        if text:
            return text, index > 0
    return None, False


def build_voice_attributions(
    voices_dir: Path,
    voices: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Build attribution rows for each Piper *.onnx under voices_dir.

    Prefer a MODEL_CARD beside the model; otherwise walk logical parent folders up
    to the voices root (e.g. ``licensed/MODEL_CARD`` for models under ``licensed/...``).
    MODEL_CARD reads are cached per directory for the duration of this call.
    When ``voices`` is provided (inventory rows with relative ``path``), use that
    order; otherwise scan the tree for *.onnx.
    """
    if not voices_dir.exists():
        return []

    rows: list[dict[str, Any]] = []
    card_cache: dict[str, str | None] = {}

    if voices is None:
        rel_paths = [
            path.relative_to(voices_dir).as_posix()
            for path in sorted(voices_dir.rglob("*.onnx"), key=lambda p: str(p).lower())
        ]
    else:
        rel_paths = []
        seen: set[str] = set()
        for item in voices:
            rel = str(item.get("path") or "").replace("\\", "/").strip().lstrip("/")
            if not rel or rel.lower() in seen:
                continue
            seen.add(rel.lower())
            rel_paths.append(rel)

    for rel in rel_paths:
        folder = (voices_dir / rel).parent
        text, used_ancestor = resolve_model_card_text(
            folder,
            voices_dir,
            card_cache=card_cache,
        )
        if not text:
            text = "(No MODEL_CARD found for this voice.)"
            used_ancestor = False

        rows.append(
            {
                "voice_id": rel,
                "name": Path(rel).name,
                "attribution_text": text,
                # Kept for API compatibility: true when card is from any parent folder.
                "used_root_fallback": used_ancestor,
            }
        )

    return rows
