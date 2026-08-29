"""
Admin-tunable prompt size caps for location and peer-NPC context, plus the
completion budget for spoken NPC replies.

Stored in ServerSettings/prompt_caps.yaml on the middleware host.
"""

from __future__ import annotations

import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

BASE_DIR = Path(__file__).resolve().parent
SERVER_SETTINGS_DIR = BASE_DIR / "ServerSettings"
PROMPT_CAPS_FILE = SERVER_SETTINGS_DIR / "prompt_caps.yaml"

DEFAULT_PROMPT_CAPS: dict[str, int] = {
    # Co-located NPC directory (name + role only)
    "directory_max_npcs": 8,
    "directory_line_max_chars": 80,
    # Peer public_facts (+ gender/pronouns) when the player asks about another NPC
    "peer_max_targets": 1,
    "peer_public_facts_max_chars": 650,
    # Location blob limits
    "location_summary_max_chars": 300,
    "location_events_max_items": 5,
    "location_places_max_items": 5,
    "location_rumors_max_items": 5,
    "location_tone_max_items": 5,
    "location_knowledge_rules_max_items": 5,
    # Completion budget for a spoken NPC reply. Too low and the model is cut
    # off mid-sentence, which reads as the NPC ending abruptly.
    "chat_reply_max_tokens": 512,
    # Reasoning models spend this same budget on chain-of-thought before any
    # visible text, so they need noticeably more room for the same reply length.
    "chat_reply_reasoning_max_tokens": 1536,
}

_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {"modified_at": None, "data": dict(DEFAULT_PROMPT_CAPS)}


def _as_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, parsed)


def normalize_prompt_caps(raw: dict[str, Any] | None) -> dict[str, int]:
    source = raw if isinstance(raw, dict) else {}
    normalized: dict[str, int] = {}
    for key, default in DEFAULT_PROMPT_CAPS.items():
        normalized[key] = _as_int(source.get(key, default), default)
    return normalized


def get_prompt_caps() -> dict[str, int]:
    """Load caps from disk (cached by mtime). Missing file → defaults."""
    SERVER_SETTINGS_DIR.mkdir(exist_ok=True)
    if not PROMPT_CAPS_FILE.exists():
        return dict(DEFAULT_PROMPT_CAPS)

    modified_at = PROMPT_CAPS_FILE.stat().st_mtime
    with _LOCK:
        cached = _CACHE
        if cached.get("modified_at") == modified_at and isinstance(cached.get("data"), dict):
            return deepcopy(cached["data"])

    with open(PROMPT_CAPS_FILE, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    data = normalize_prompt_caps(loaded if isinstance(loaded, dict) else {})
    with _LOCK:
        _CACHE["modified_at"] = modified_at
        _CACHE["data"] = data
    return deepcopy(data)


def save_prompt_caps(raw: dict[str, Any] | None) -> dict[str, int]:
    """Validate, write, and return normalized caps."""
    data = normalize_prompt_caps(raw)
    SERVER_SETTINGS_DIR.mkdir(exist_ok=True)
    payload = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    if not payload.endswith("\n"):
        payload += "\n"
    PROMPT_CAPS_FILE.write_text(payload, encoding="utf-8")
    with _LOCK:
        _CACHE["modified_at"] = PROMPT_CAPS_FILE.stat().st_mtime
        _CACHE["data"] = data
    return deepcopy(data)


def ensure_prompt_caps_file() -> None:
    if PROMPT_CAPS_FILE.exists():
        return
    save_prompt_caps(DEFAULT_PROMPT_CAPS)


ensure_prompt_caps_file()
