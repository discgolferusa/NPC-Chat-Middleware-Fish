"""Resolve Piper voice MODEL_CARD attribution text from VOICES_DIR."""

from __future__ import annotations

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
        except OSError:
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
                except OSError:
                    continue
    except OSError:
        return None

    return None


def build_voice_attributions(
    voices_dir: Path,
    voices: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Build attribution rows for each Piper *.onnx under voices_dir.

    Prefer a MODEL_CARD beside the model; otherwise use the voices root card.
    When ``voices`` is provided (inventory rows with relative ``path``), use that
    order; otherwise scan the tree for *.onnx.
    """
    if not voices_dir.exists():
        return []

    root_card = try_read_model_card(voices_dir)
    rows: list[dict[str, Any]] = []

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
        local = try_read_model_card(folder)
        if local and local.strip():
            text = local.strip()
            used_root = False
        elif root_card and root_card.strip():
            text = root_card.strip()
            used_root = True
        else:
            text = "(No MODEL_CARD found for this voice.)"
            used_root = True

        rows.append(
            {
                "voice_id": rel,
                "name": Path(rel).name,
                "attribution_text": text,
                "used_root_fallback": used_root,
            }
        )

    return rows
