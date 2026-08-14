"""
Server-wide settings loaded from YAML in ServerSettings/.

Narrator TTS prefers the campaign owner's content-pack `tts` (per-user) when present;
otherwise falls back to middleware-local ServerSettings/tts.yaml.
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

# Relative to VOICES_DIR (Z:\voices locally / C:\AI\Voices on server).
_DEFAULT_NARRATOR_VOICE_MODEL = "en/en_US/Licensed/libritts/en_US-libritts-high.onnx"
_DEFAULT_NARRATOR_LENGTH_SCALE = 1.15
_DEFAULT_NARRATOR_NOISE_SCALE = 0.667
_DEFAULT_NARRATOR_NOISE_W = 0.8
_DEFAULT_NARRATOR_SENTENCE_SILENCE = 0.2
# Piper speaker_id_map entry "Dylan Howard" on en_US-libritts-high.
_DEFAULT_NARRATOR_SPEAKER_ID = 3
_DEFAULT_NARRATOR_SPEAKER_NAME = "Dylan Howard"
_DEFAULT_NARRATOR_PITCH_SEMITONES = 0.0
_DEFAULT_NARRATOR_FX = "none"

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


def _as_optional_int(value: Any) -> int | None:
    if value is None or (isinstance(value, str) and not str(value).strip()):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_voice_key(path: str) -> str:
    """Compare voice paths across absolute/relative and slash styles."""
    normalized = str(path or "").replace("\\", "/").strip().lower()
    for marker in ("/voices/",):
        idx = normalized.find(marker)
        if idx >= 0:
            normalized = normalized[idx + len(marker) :]
    return normalized.lstrip("/")


def _is_default_narrator_voice(path: str) -> bool:
    return _normalize_voice_key(path) == _normalize_voice_key(_DEFAULT_NARRATOR_VOICE_MODEL)


def _default_tts_settings() -> dict[str, Any]:
    return {
        "narrator": {
            "voice_model": _DEFAULT_NARRATOR_VOICE_MODEL,
            "length_scale": _DEFAULT_NARRATOR_LENGTH_SCALE,
            "noise_scale": _DEFAULT_NARRATOR_NOISE_SCALE,
            "noise_w": _DEFAULT_NARRATOR_NOISE_W,
            "sentence_silence": _DEFAULT_NARRATOR_SENTENCE_SILENCE,
            "speaker_id": _DEFAULT_NARRATOR_SPEAKER_ID,
            "speaker_name": _DEFAULT_NARRATOR_SPEAKER_NAME,
            "pitch_semitones": _DEFAULT_NARRATOR_PITCH_SEMITONES,
            "fx_preset": _DEFAULT_NARRATOR_FX,
        }
    }


def load_tts_settings(path: Path | None = None) -> dict[str, Any]:
    """
    Load TTS settings from ServerSettings/tts.yaml (middleware-local fallback).
    Reloads automatically when the file changes on disk.
    """
    settings_path = path or DEFAULT_TTS_SETTINGS_FILE
    if not settings_path.exists():
        return _default_tts_settings()

    return _load_yaml_cached(settings_path, _TTS_SETTINGS_CACHE, str(settings_path))


def _resolve_pack_tts() -> dict[str, Any] | None:
    """Prefer per-owner TTS from the active content pack (request owner + campaign)."""
    try:
        from content_pack import get_pack_tts
        from session_context import get_request_campaign, get_request_owner

        return get_pack_tts(get_request_owner(), get_request_campaign())
    except Exception:
        return None


def get_narrator_tts_settings(path: Path | None = None) -> dict[str, Any]:
    """
    Resolved narrator voice settings for stage-direction TTS.

    Prefers the campaign owner's content-pack `tts` when installed; otherwise
    middleware-local ServerSettings/tts.yaml (or built-in defaults).

    Narrator knobs mirror NPC TTS fields (voice_model, length/noise scales,
    sentence silence, speaker_id, pitch, fx). There is no fallback voice model.
    """
    pack_tts = _resolve_pack_tts()
    data = pack_tts if isinstance(pack_tts, dict) else load_tts_settings(path)
    narrator = data.get("narrator") if isinstance(data.get("narrator"), dict) else {}

    configured_voice = str(narrator.get("voice_model") or "").strip()
    voice_model = configured_voice or _DEFAULT_NARRATOR_VOICE_MODEL
    using_default_voice = (not configured_voice) or _is_default_narrator_voice(voice_model)

    length_scale = _as_float(
        narrator.get("length_scale", narrator.get("tts_length_scale")),
        _DEFAULT_NARRATOR_LENGTH_SCALE,
    )
    noise_scale = _as_float(
        narrator.get("noise_scale", narrator.get("tts_noise_scale")),
        _DEFAULT_NARRATOR_NOISE_SCALE,
    )
    noise_w = _as_float(
        narrator.get("noise_w", narrator.get("tts_noise_w")),
        _DEFAULT_NARRATOR_NOISE_W,
    )
    sentence_silence = _as_float(
        narrator.get("sentence_silence", narrator.get("tts_sentence_silence")),
        _DEFAULT_NARRATOR_SENTENCE_SILENCE,
    )
    speaker_id = _as_optional_int(
        narrator.get("speaker_id", narrator.get("tts_speaker_id"))
    )
    speaker_id_missing = speaker_id is None
    speaker_name = str(
        narrator.get("speaker_name") or narrator.get("tts_speaker_name") or ""
    ).strip()
    # LibriTTS Dylan Howard (id 3) only when on the default narrator voice.
    # Overridden single-speaker models (e.g. Amy) omit speaker_id → use 0.
    if speaker_id is None:
        speaker_id = (
            _DEFAULT_NARRATOR_SPEAKER_ID if using_default_voice else 0
        )
    # Never invent a display name for packs that only set speaker_id.
    if not speaker_name and speaker_id_missing and using_default_voice:
        speaker_name = _DEFAULT_NARRATOR_SPEAKER_NAME
    pitch_semitones = _as_float(
        narrator.get("pitch_semitones", narrator.get("tts_pitch_semitones")),
        _DEFAULT_NARRATOR_PITCH_SEMITONES,
    )
    fx_preset = str(
        narrator.get("fx_preset") or narrator.get("tts_fx") or _DEFAULT_NARRATOR_FX
    ).strip().lower() or _DEFAULT_NARRATOR_FX

    return {
        "voice_model": voice_model,
        "length_scale": length_scale,
        "noise_scale": noise_scale,
        "noise_w": noise_w,
        "sentence_silence": sentence_silence,
        "speaker_id": speaker_id,
        "speaker_name": speaker_name,
        "pitch_semitones": pitch_semitones,
        "fx_preset": fx_preset,
        "resolved_voice_model": voice_model,
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
        f"length_scale={narrator['length_scale']} "
        f"(pack TTS preferred when campaign content pack is installed)",
        flush=True,
    )


init_server_settings()
