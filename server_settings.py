"""
Server-wide settings loaded from YAML in ServerSettings/.
"""

from __future__ import annotations

import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

BASE_DIR = Path(__file__).resolve().parent
SERVER_SETTINGS_DIR = BASE_DIR / "ServerSettings"
DEFAULT_LLM_RULES_FILE = SERVER_SETTINGS_DIR / "llm_rules.yaml"
DEFAULT_TTS_SETTINGS_FILE = SERVER_SETTINGS_DIR / "tts.yaml"

_DEFAULT_NARRATOR_VOICE_MODEL = r"C:\AI\Voices\en\en_US\amy\medium\en_US-amy-medium.onnx"
_DEFAULT_NARRATOR_FALLBACK_VOICE_MODEL = r"C:\AI\Voices\en\en_US\ryan\medium\en_US-ryan-medium.onnx"
_DEFAULT_NARRATOR_LENGTH_SCALE = 1.08

_CACHE_LOCK = threading.Lock()
_LLM_RULES_CACHE: dict[str, Any] = {}
_TTS_SETTINGS_CACHE: dict[str, Any] = {}


def format_llm_rules(raw: Any) -> str:
    """
    Format llm_rules from YAML (list[str], multiline string, or dict with llm_rules key).
    """
    if isinstance(raw, dict):
        raw = raw.get("llm_rules") or raw.get("rules")

    if raw is None:
        return ""

    lines: list[str] = []
    if isinstance(raw, str):
        lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    elif isinstance(raw, list):
        for item in raw:
            text = str(item).strip()
            if not text:
                continue
            if "\n" in text:
                lines.extend(line.strip() for line in text.splitlines() if line.strip())
            else:
                lines.append(text)

    if not lines:
        return ""

    normalized = []
    for line in lines:
        cleaned = line.lstrip("-").strip()
        if cleaned:
            normalized.append(f"- {cleaned}")

    return "\n".join(normalized)


def _load_yaml_cached(path: Path, cache: dict[str, Any], cache_key: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing settings file: {path}")

    modified_at = path.stat().st_mtime
    with _CACHE_LOCK:
        cached = cache.get(cache_key)
        if cached and cached.get("modified_at") == modified_at:
            return deepcopy(cached["data"])

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    with _CACHE_LOCK:
        cache[cache_key] = {"modified_at": modified_at, "data": data}

    return deepcopy(data)


def load_global_llm_rules(path: Path | None = None) -> str:
    """
    Load and format global LLM rules from ServerSettings/llm_rules.yaml.
    Reloads automatically when the file changes on disk.
    """
    rules_path = path or DEFAULT_LLM_RULES_FILE
    if not rules_path.exists():
        return ""

    data = _load_yaml_cached(rules_path, _LLM_RULES_CACHE, str(rules_path))
    return format_llm_rules(data)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_tts_settings(path: Path | None = None) -> dict[str, Any]:
    """
    Load TTS settings from ServerSettings/tts.yaml.
    Reloads automatically when the file changes on disk.
    """
    settings_path = path or DEFAULT_TTS_SETTINGS_FILE
    if not settings_path.exists():
        return {
            "narrator": {
                "voice_model": _DEFAULT_NARRATOR_VOICE_MODEL,
                "fallback_voice_model": _DEFAULT_NARRATOR_FALLBACK_VOICE_MODEL,
                "length_scale": _DEFAULT_NARRATOR_LENGTH_SCALE,
            }
        }

    return _load_yaml_cached(settings_path, _TTS_SETTINGS_CACHE, str(settings_path))


def get_narrator_tts_settings(path: Path | None = None) -> dict[str, Any]:
    """
    Resolved narrator voice settings for stage-direction TTS.
    """
    data = load_tts_settings(path)
    narrator = data.get("narrator") if isinstance(data.get("narrator"), dict) else {}

    voice_model = (narrator.get("voice_model") or _DEFAULT_NARRATOR_VOICE_MODEL).strip()
    fallback_voice_model = (
        narrator.get("fallback_voice_model") or _DEFAULT_NARRATOR_FALLBACK_VOICE_MODEL
    ).strip()
    length_scale = _as_float(narrator.get("length_scale"), _DEFAULT_NARRATOR_LENGTH_SCALE)

    resolved_model = Path(voice_model)
    if not resolved_model.exists() and fallback_voice_model:
        resolved_model = Path(fallback_voice_model)

    return {
        "voice_model": voice_model,
        "fallback_voice_model": fallback_voice_model,
        "length_scale": length_scale,
        "resolved_voice_model": str(resolved_model),
    }


def init_server_settings() -> None:
    SERVER_SETTINGS_DIR.mkdir(exist_ok=True)
    if DEFAULT_LLM_RULES_FILE.exists():
        rule_count = len((load_global_llm_rules() or "").splitlines())
        print(f"[SETTINGS] loaded {rule_count} global llm_rules from {DEFAULT_LLM_RULES_FILE.name}", flush=True)
    else:
        print(f"[SETTINGS] warning: missing {DEFAULT_LLM_RULES_FILE}", flush=True)

    narrator = get_narrator_tts_settings()
    print(
        f"[SETTINGS] narrator voice={narrator['resolved_voice_model']} "
        f"length_scale={narrator['length_scale']}",
        flush=True,
    )


init_server_settings()
