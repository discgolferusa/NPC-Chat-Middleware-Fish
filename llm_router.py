"""
Multi-provider LLM routing: local llama.cpp (default/fallback) and Groq.

Requests are scoped by "purpose":
  - chat: spoken NPC dialogue turns (default, short completions).
  - content_creation: campaign document import / generation tooling
    (longer completions, its own selected model + provider mode).

Manual test (PowerShell):
  curl http://127.0.0.1:5000/llm/models
  curl http://127.0.0.1:5000/llm/status
  curl -X POST http://127.0.0.1:5000/llm/select -H "Content-Type: application/json" -d "{\"model_id\":\"local\",\"provider_mode\":\"local\"}"
  curl -X POST http://127.0.0.1:5000/llm/select -H "Content-Type: application/json" -d "{\"model_id\":\"groq-llama-4-scout\",\"provider_mode\":\"groq_with_local_fallback\"}"
  curl -X POST http://127.0.0.1:5000/llm/select -H "Content-Type: application/json" -d "{\"model_id\":\"groq-llama-3.3-70b\",\"provider_mode\":\"groq_with_local_fallback\",\"purpose\":\"content_creation\"}"
"""

from __future__ import annotations

import json
import os
import threading
import time
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

BASE_DIR = Path(__file__).resolve().parent

LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:8080/v1/chat/completions")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "npc")
LOCAL_LLM_TIMEOUT_SECONDS = int(os.getenv("LOCAL_LLM_TIMEOUT_SECONDS", "180"))
CONTENT_CREATION_LOCAL_TIMEOUT_SECONDS = int(
    os.getenv("CONTENT_CREATION_LOCAL_TIMEOUT_SECONDS", "600")
)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1/chat/completions")
GROQ_MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
GROQ_TIMEOUT_SECONDS = int(os.getenv("GROQ_TIMEOUT_SECONDS", "10"))
# content_creation sends large prompts and asks for long JSON completions;
# the chat default (10s) is far too short and causes cascading timeouts.
CONTENT_CREATION_GROQ_TIMEOUT_SECONDS = int(
    os.getenv("CONTENT_CREATION_GROQ_TIMEOUT_SECONDS", "120")
)

# Spoken NPC replies are short, but reasoning models spend completion tokens on
# chain-of-thought first. max_tokens=180 with gpt-oss/qwen3 often yields empty
# message.content (finish_reason=length) -> "I have nothing more to say."
DEFAULT_MAX_TOKENS = 180
REASONING_MAX_TOKENS = int(os.getenv("LLM_REASONING_MAX_TOKENS", "768"))

# content_creation purpose (campaign document import, generation tooling) needs
# much larger completions than spoken NPC dialogue, so it gets its own default.
CONTENT_CREATION_DEFAULT_MAX_TOKENS = int(os.getenv("LLM_CONTENT_CREATION_MAX_TOKENS", "4096"))

LLM_USAGE_FILE = BASE_DIR / "llm_usage.json"
LLM_RUNTIME_CONFIG_FILE = BASE_DIR / "llm_runtime_config.json"
LLM_SECRETS_FILE = BASE_DIR / "llm_secrets.env"

VALID_PROVIDER_MODES = frozenset({"local", "groq", "groq_with_local_fallback"})
REASONING_MODEL_MARKERS = ("gpt-oss", "qwen3")

PURPOSE_CHAT = "chat"
PURPOSE_CONTENT_CREATION = "content_creation"
VALID_PURPOSES = frozenset({PURPOSE_CHAT, PURPOSE_CONTENT_CREATION})
DEFAULT_PURPOSE = PURPOSE_CHAT

LLM_MODELS: dict[str, dict[str, Any]] = {
    "local": {
        "id": "local",
        "label": "Local llama.cpp",
        "provider": "local",
        "model": LOCAL_LLM_MODEL,
        "url": LOCAL_LLM_URL,
        "online": False,
        "limits": None,
    },
    "groq-llama-4-scout": {
        "id": "groq-llama-4-scout",
        "label": "Groq Llama 4 Scout 17B",
        "provider": "groq",
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "online": True,
        "limits": {"rpm": 30, "rpd": 1000, "tpm": 30000, "tpd": 500000},
    },
    "groq-llama-3.1-8b-instant": {
        "id": "groq-llama-3.1-8b-instant",
        "label": "Groq Llama 3.1 8B Instant",
        "provider": "groq",
        "model": "llama-3.1-8b-instant",
        "online": True,
        "limits": {"rpm": 30, "rpd": 14400, "tpm": 6000, "tpd": 500000},
    },
    "groq-qwen3-32b": {
        "id": "groq-qwen3-32b",
        "label": "Groq Qwen3 32B",
        "provider": "groq",
        "model": "qwen/qwen3-32b",
        "online": True,
        "limits": {"rpm": 60, "rpd": 1000, "tpm": 6000, "tpd": 500000},
    },
    "groq-llama-3.3-70b": {
        "id": "groq-llama-3.3-70b",
        "label": "Groq Llama 3.3 70B Versatile",
        "provider": "groq",
        "model": "llama-3.3-70b-versatile",
        "online": True,
        "limits": {"rpm": 30, "rpd": 1000, "tpm": 12000, "tpd": 100000},
    },
    "groq-gpt-oss-120b": {
        "id": "groq-gpt-oss-120b",
        "label": "Groq GPT-OSS 120B",
        "provider": "groq",
        "model": "openai/gpt-oss-120b",
        "online": True,
        "limits": {"rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200000},
    },
    "groq-gpt-oss-20b": {
        "id": "groq-gpt-oss-20b",
        "label": "Groq GPT-OSS 20B",
        "provider": "groq",
        "model": "openai/gpt-oss-20b",
        "online": True,
        "limits": {"rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200000},
    },
}

# --- chat purpose (default / backward-compatible globals) ---
CURRENT_LLM_MODEL_ID = os.getenv("CURRENT_LLM_MODEL_ID", "local")
LLM_PROVIDER_MODE = os.getenv("LLM_PROVIDER_MODE", "groq_with_local_fallback")

# --- content_creation purpose (campaign/document generation tooling) ---
CONTENT_CREATION_MODEL_ID = os.getenv("CONTENT_CREATION_LLM_MODEL_ID", CURRENT_LLM_MODEL_ID)
CONTENT_CREATION_PROVIDER_MODE = os.getenv("CONTENT_CREATION_LLM_PROVIDER_MODE", LLM_PROVIDER_MODE)

# Per-purpose runtime status. Mutated in place (never rebound) so callers never
# need `global` to update them.
LAST_ACTIVE_PROVIDER: dict[str, str] = {PURPOSE_CHAT: "local", PURPOSE_CONTENT_CREATION: "local"}
LAST_ACTIVE_MODEL_ID: dict[str, str] = {PURPOSE_CHAT: "local", PURPOSE_CONTENT_CREATION: "local"}
LAST_FALLBACK_REASON: dict[str, str | None] = {PURPOSE_CHAT: None, PURPOSE_CONTENT_CREATION: None}
LAST_LLM_ERROR: dict[str, str | None] = {PURPOSE_CHAT: None, PURPOSE_CONTENT_CREATION: None}

ONLINE_DISABLED_UNTIL: datetime | None = None

_STATE_LOCK = threading.Lock()
_USAGE_LOCK = threading.Lock()


class LLMProviderError(Exception):
    """Raised when strict online-only mode cannot complete a request."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_utc_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value).astimezone(timezone.utc)
    except ValueError:
        return None


def _normalize_purpose(purpose: str | None) -> str:
    if purpose in VALID_PURPOSES:
        return purpose
    return DEFAULT_PURPOSE


def _get_purpose_model_id(purpose: str) -> str:
    return CONTENT_CREATION_MODEL_ID if purpose == PURPOSE_CONTENT_CREATION else CURRENT_LLM_MODEL_ID


def _get_purpose_provider_mode(purpose: str) -> str:
    return CONTENT_CREATION_PROVIDER_MODE if purpose == PURPOSE_CONTENT_CREATION else LLM_PROVIDER_MODE


def _load_secrets_file() -> None:
    global GROQ_API_KEY

    if GROQ_API_KEY:
        return

    if not LLM_SECRETS_FILE.exists():
        return

    try:
        for line in LLM_SECRETS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key == "GROQ_API_KEY" and value:
                GROQ_API_KEY = value
                break
    except OSError as ex:
        print(f"[LLM] failed to read secrets file: {ex}", flush=True)


def _load_runtime_config() -> None:
    global CURRENT_LLM_MODEL_ID, LLM_PROVIDER_MODE
    global CONTENT_CREATION_MODEL_ID, CONTENT_CREATION_PROVIDER_MODE

    if not LLM_RUNTIME_CONFIG_FILE.exists():
        CURRENT_LLM_MODEL_ID = "local"
        if os.getenv("LLM_PROVIDER_MODE"):
            if LLM_PROVIDER_MODE not in VALID_PROVIDER_MODES:
                LLM_PROVIDER_MODE = "local"
        else:
            LLM_PROVIDER_MODE = "local"

        CONTENT_CREATION_MODEL_ID = os.getenv("CONTENT_CREATION_LLM_MODEL_ID", CURRENT_LLM_MODEL_ID)
        if CONTENT_CREATION_MODEL_ID not in LLM_MODELS:
            CONTENT_CREATION_MODEL_ID = "local"
        CONTENT_CREATION_PROVIDER_MODE = os.getenv("CONTENT_CREATION_LLM_PROVIDER_MODE", LLM_PROVIDER_MODE)
        if CONTENT_CREATION_PROVIDER_MODE not in VALID_PROVIDER_MODES:
            CONTENT_CREATION_PROVIDER_MODE = "local"
        return

    try:
        raw = json.loads(LLM_RUNTIME_CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as ex:
        print(f"[LLM] invalid runtime config, using defaults: {ex}", flush=True)
        CURRENT_LLM_MODEL_ID = "local"
        LLM_PROVIDER_MODE = "local"
        CONTENT_CREATION_MODEL_ID = "local"
        CONTENT_CREATION_PROVIDER_MODE = "local"
        return

    data = raw if isinstance(raw, dict) else {}
    purposes = data.get("purposes") if isinstance(data.get("purposes"), dict) else {}

    # Backward compat: pre-purpose runtime config files only had top-level
    # model_id/provider_mode. Treat those as the "chat" purpose.
    chat_cfg = purposes.get(PURPOSE_CHAT) if isinstance(purposes.get(PURPOSE_CHAT), dict) else {}
    chat_model_id = chat_cfg.get("model_id", data.get("model_id", CURRENT_LLM_MODEL_ID))
    chat_provider_mode = chat_cfg.get("provider_mode", data.get("provider_mode", LLM_PROVIDER_MODE))

    if chat_model_id in LLM_MODELS:
        CURRENT_LLM_MODEL_ID = chat_model_id
    elif CURRENT_LLM_MODEL_ID not in LLM_MODELS:
        CURRENT_LLM_MODEL_ID = "local"

    if chat_provider_mode in VALID_PROVIDER_MODES:
        LLM_PROVIDER_MODE = chat_provider_mode
    elif LLM_PROVIDER_MODE not in VALID_PROVIDER_MODES:
        LLM_PROVIDER_MODE = "local"

    creation_cfg = (
        purposes.get(PURPOSE_CONTENT_CREATION)
        if isinstance(purposes.get(PURPOSE_CONTENT_CREATION), dict)
        else {}
    )
    creation_model_id = creation_cfg.get("model_id", CONTENT_CREATION_MODEL_ID)
    creation_provider_mode = creation_cfg.get("provider_mode", CONTENT_CREATION_PROVIDER_MODE)

    if creation_model_id in LLM_MODELS:
        CONTENT_CREATION_MODEL_ID = creation_model_id
    elif CONTENT_CREATION_MODEL_ID not in LLM_MODELS:
        CONTENT_CREATION_MODEL_ID = "local"

    if creation_provider_mode in VALID_PROVIDER_MODES:
        CONTENT_CREATION_PROVIDER_MODE = creation_provider_mode
    elif CONTENT_CREATION_PROVIDER_MODE not in VALID_PROVIDER_MODES:
        CONTENT_CREATION_PROVIDER_MODE = "local"


def _save_runtime_config() -> None:
    payload = {
        # Legacy top-level keys kept for backward compatibility; mirror "chat".
        "model_id": CURRENT_LLM_MODEL_ID,
        "provider_mode": LLM_PROVIDER_MODE,
        "purposes": {
            PURPOSE_CHAT: {
                "model_id": CURRENT_LLM_MODEL_ID,
                "provider_mode": LLM_PROVIDER_MODE,
            },
            PURPOSE_CONTENT_CREATION: {
                "model_id": CONTENT_CREATION_MODEL_ID,
                "provider_mode": CONTENT_CREATION_PROVIDER_MODE,
            },
        },
        "updated_at": _utc_iso(_utc_now()),
    }
    try:
        LLM_RUNTIME_CONFIG_FILE.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
    except OSError as ex:
        print(f"[LLM] failed to save runtime config: {ex}", flush=True)


def _public_model_entry(model_id: str) -> dict[str, Any]:
    entry = deepcopy(LLM_MODELS[model_id])
    if model_id == "local":
        entry["model"] = LOCAL_LLM_MODEL
        entry["url"] = LOCAL_LLM_URL
    return entry


def _public_model_list() -> list[dict[str, Any]]:
    return [_public_model_entry(model_id) for model_id in LLM_MODELS]


def groq_is_configured() -> bool:
    return bool(GROQ_API_KEY)


def groq_is_available() -> bool:
    if not groq_is_configured():
        return False
    if ONLINE_DISABLED_UNTIL is not None and _utc_now() < ONLINE_DISABLED_UNTIL:
        return False
    return True


def _set_online_disabled(seconds: int, reason: str) -> None:
    global ONLINE_DISABLED_UNTIL

    ONLINE_DISABLED_UNTIL = _utc_now() + timedelta(seconds=max(1, seconds))
    print(
        f"[LLM] Groq disabled until {_utc_iso(ONLINE_DISABLED_UNTIL)} ({reason})",
        flush=True,
    )


def _parse_retry_after_seconds(response: requests.Response | None) -> int | None:
    if response is None:
        return None

    retry_after = response.headers.get("Retry-After")
    if not retry_after:
        return None

    try:
        return max(1, int(retry_after))
    except ValueError:
        return None


def _estimate_prompt_tokens(messages: list[dict[str, Any]]) -> int:
    all_prompt_text = "\n".join(str(m.get("content", "")) for m in messages)
    return max(1, len(all_prompt_text) // 4)


def _is_reasoning_model(model_id: str | None = None, model_name: str | None = None) -> bool:
    blob = f"{model_id or ''} {model_name or ''}".casefold()
    return any(marker in blob for marker in REASONING_MODEL_MARKERS)


def _effective_max_tokens(
    model_id: str | None,
    max_tokens: int | None,
    *,
    model_name: str | None = None,
    purpose: str = DEFAULT_PURPOSE,
) -> int:
    if max_tokens is not None:
        return max(1, int(max_tokens))
    if purpose == PURPOSE_CONTENT_CREATION:
        return CONTENT_CREATION_DEFAULT_MAX_TOKENS
    if _is_reasoning_model(model_id, model_name):
        return REASONING_MAX_TOKENS
    return DEFAULT_MAX_TOKENS


def _extract_chat_content(message: dict[str, Any] | None) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()
    return str(content).strip()


def _apply_groq_reasoning_options(payload: dict[str, Any], *, model_id: str, model_name: str) -> None:
    """Keep reasoning cheap so spoken dialogue still fits in max_tokens."""
    if not _is_reasoning_model(model_id, model_name):
        return
    name = (model_name or "").casefold()
    mid = (model_id or "").casefold()
    if "gpt-oss" in name or "gpt-oss" in mid:
        # Groq gpt-oss supports low|medium|high (default medium).
        payload.setdefault("reasoning_effort", "low")
    elif "qwen3" in name or "qwen3" in mid:
        # Groq qwen3 supports none|default|... — disable thinking for NPC turns.
        payload.setdefault("reasoning_effort", "none")


def _load_usage_records() -> list[dict[str, Any]]:
    if not LLM_USAGE_FILE.exists():
        return []

    try:
        data = json.loads(LLM_USAGE_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except (OSError, json.JSONDecodeError):
        pass

    return []


def _save_usage_records(records: list[dict[str, Any]]) -> None:
    cutoff = _utc_now() - timedelta(hours=24)
    pruned = []
    for record in records:
        ts = _parse_utc_iso(record.get("timestamp"))
        if ts is None or ts >= cutoff:
            pruned.append(record)

    try:
        LLM_USAGE_FILE.write_text(json.dumps(pruned, indent=2), encoding="utf-8")
    except OSError as ex:
        print(f"[LLM] failed to save usage ledger: {ex}", flush=True)


def record_llm_usage(
    *,
    provider: str,
    model_id: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    success: bool,
    fallback: bool = False,
    error: str | None = None,
    purpose: str = DEFAULT_PURPOSE,
) -> None:
    record = {
        "timestamp": _utc_iso(_utc_now()),
        "provider": provider,
        "model_id": model_id,
        "model": model,
        "purpose": _normalize_purpose(purpose),
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(total_tokens),
        "success": bool(success),
        "fallback": bool(fallback),
        "error": error,
    }

    with _USAGE_LOCK:
        records = _load_usage_records()
        records.append(record)
        _save_usage_records(records)


def _filter_usage_records(
    *,
    since: timedelta,
    model_id: str | None = None,
    provider: str | None = None,
    online_only: bool = False,
    purpose: str | None = None,
) -> list[dict[str, Any]]:
    cutoff = _utc_now() - since
    filtered = []

    for record in _load_usage_records():
        ts = _parse_utc_iso(record.get("timestamp"))
        if ts is None or ts < cutoff:
            continue
        if model_id is not None and record.get("model_id") != model_id:
            continue
        if provider is not None and record.get("provider") != provider:
            continue
        if online_only and record.get("provider") != "groq":
            continue
        if purpose is not None and record.get("purpose", DEFAULT_PURPOSE) != purpose:
            continue
        filtered.append(record)

    return filtered


def usage_requests_last_60s(
    model_id: str | None = None, provider: str | None = None, purpose: str | None = None
) -> int:
    records = _filter_usage_records(
        since=timedelta(seconds=60), model_id=model_id, provider=provider, online_only=True, purpose=purpose
    )
    return len(records)


def usage_tokens_last_60s(
    model_id: str | None = None, provider: str | None = None, purpose: str | None = None
) -> int:
    records = _filter_usage_records(
        since=timedelta(seconds=60), model_id=model_id, provider=provider, online_only=True, purpose=purpose
    )
    return sum(int(r.get("total_tokens", 0)) for r in records)


def usage_requests_last_24h(
    model_id: str | None = None, provider: str | None = None, purpose: str | None = None
) -> int:
    records = _filter_usage_records(
        since=timedelta(hours=24), model_id=model_id, provider=provider, online_only=True, purpose=purpose
    )
    return len(records)


def usage_tokens_last_24h(
    model_id: str | None = None, provider: str | None = None, purpose: str | None = None
) -> int:
    records = _filter_usage_records(
        since=timedelta(hours=24), model_id=model_id, provider=provider, online_only=True, purpose=purpose
    )
    return sum(int(r.get("total_tokens", 0)) for r in records)


def get_usage_summary(model_id: str, purpose: str | None = None) -> dict[str, int]:
    return {
        "requests_last_minute": usage_requests_last_60s(model_id=model_id, purpose=purpose),
        "tokens_last_minute": usage_tokens_last_60s(model_id=model_id, purpose=purpose),
        "requests_last_24h": usage_requests_last_24h(model_id=model_id, purpose=purpose),
        "tokens_last_24h": usage_tokens_last_24h(model_id=model_id, purpose=purpose),
    }


def _would_exceed_groq_limits(
    model_id: str,
    estimated_total_tokens: int,
    *,
    purpose: str = DEFAULT_PURPOSE,
) -> str | None:
    # Rate limits are enforced per model_id on the Groq account, shared across
    # purposes — do not filter by purpose here. Messages are purpose-aware so
    # chat failures do not tell operators to shorten an import document.
    model_entry = LLM_MODELS.get(model_id)
    if not model_entry or model_entry.get("provider") != "groq":
        return None

    limits = model_entry.get("limits") or {}
    label = model_entry.get("label") or model_id
    purpose = _normalize_purpose(purpose)
    model_slot = (
        "content creation model"
        if purpose == PURPOSE_CONTENT_CREATION
        else "chat model"
    )

    rpm = limits.get("rpm")
    if rpm is not None and usage_requests_last_60s(model_id=model_id) + 1 > rpm:
        return (
            f"Groq requests-per-minute limit reached for {label} "
            f"({rpm}/min). Wait about a minute, then retry."
        )

    rpd = limits.get("rpd")
    if rpd is not None and usage_requests_last_24h(model_id=model_id) + 1 > rpd:
        return (
            f"Groq requests-per-day limit reached for {label} "
            f"({rpd}/day). Try again tomorrow or switch the {model_slot} in Admin → LLM."
        )

    tpm = limits.get("tpm")
    if tpm is not None:
        used = usage_tokens_last_60s(model_id=model_id)
        if used + estimated_total_tokens > tpm:
            if purpose == PURPOSE_CONTENT_CREATION:
                action = (
                    "Wait about a minute, shorten the document, or switch content creation "
                    "to a higher-TPM model such as groq-llama-4-scout (Admin → LLM)."
                )
            else:
                action = (
                    "Wait about a minute, or switch the chat model in Admin → LLM "
                    "(this is not a document-import size issue)."
                )
            return (
                f"Groq tokens-per-minute budget too small for this request on {label} "
                f"(need ~{estimated_total_tokens} tokens, limit {tpm}/min, "
                f"~{used} used in the last minute). {action}"
            )

    tpd = limits.get("tpd")
    if tpd is not None and usage_tokens_last_24h(model_id=model_id) + estimated_total_tokens > tpd:
        return (
            f"Groq tokens-per-day limit reached for {label} "
            f"({tpd}/day). Try again tomorrow or switch the {model_slot} in Admin → LLM."
        )

    return None


# Rough tokens reserved for system prompt + outline instructions outside the document body.
_CONTENT_CREATION_PROMPT_OVERHEAD_TOKENS = 1_000


def content_creation_max_input_chars() -> int:
    """Chars of document body that fit the content_creation model's current budget.

    Returns 0 when a strict-Groq setup cannot fit a minimal import request.
    For local / groq_with_local_fallback, do NOT pre-truncate using Groq TPM —
    otherwise Groq limit failures fall back to local with a head-chopped document
    and NPC/location sections later in the file are invisible to the local model.
    """
    model_id = CONTENT_CREATION_MODEL_ID
    provider_mode = CONTENT_CREATION_PROVIDER_MODE
    if provider_mode == "local" or model_id == "local":
        return 120_000
    if provider_mode == "groq_with_local_fallback":
        # Keep the full document available for local fallback; Groq still has its
        # own preflight in generate_llm_response and will fall back when needed.
        return 120_000

    model_entry = LLM_MODELS.get(model_id) or {}
    limits = model_entry.get("limits") or {}
    tpm = limits.get("tpm")
    if tpm is None:
        return 120_000

    completion = CONTENT_CREATION_DEFAULT_MAX_TOKENS
    used = usage_tokens_last_60s(model_id=model_id)
    # Leave headroom for completion + fixed prompt overhead + recent usage.
    budget_tokens = int(tpm * 0.55) - completion - _CONTENT_CREATION_PROMPT_OVERHEAD_TOKENS - used
    if budget_tokens <= 0:
        return 0
    # Rough chars≈tokens*4 heuristic used elsewhere in this module.
    return min(120_000, budget_tokens * 4)


def call_local_chat_completions(
    messages: list[dict[str, Any]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout_seconds: int | None = None,
    **kwargs: Any,
) -> tuple[str, dict[str, int]]:
    effective_max = _effective_max_tokens("local", max_tokens, model_name=LOCAL_LLM_MODEL)
    payload: dict[str, Any] = {
        "model": LOCAL_LLM_MODEL,
        "messages": messages,
        "temperature": 0.80 if temperature is None else temperature,
        "top_p": kwargs.pop("top_p", 0.90),
        "max_tokens": effective_max,
        "stream": kwargs.pop("stream", False),
    }
    payload.update(kwargs)
    timeout = timeout_seconds if timeout_seconds is not None else LOCAL_LLM_TIMEOUT_SECONDS

    try:
        response = requests.post(LOCAL_LLM_URL, json=payload, timeout=timeout)
        response.raise_for_status()
    except requests.ConnectionError as ex:
        raise LLMProviderError(
            f"Local LLM is not reachable at {LOCAL_LLM_URL}. "
            "Start llama.cpp on that port, or set the content_creation / chat model to Groq "
            "in Admin → LLM (provider mode 'groq' or 'groq_with_local_fallback')."
        ) from ex
    except requests.Timeout as ex:
        raise LLMProviderError(
            f"Local LLM timed out at {LOCAL_LLM_URL} after {timeout}s."
        ) from ex

    data = response.json()
    message = (data.get("choices") or [{}])[0].get("message") or {}
    content = _extract_chat_content(message)

    usage = data.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or _estimate_prompt_tokens(messages))
    completion_tokens = int(usage.get("completion_tokens") or max(1, len(content) // 4))
    total_tokens = int(usage.get("total_tokens") or prompt_tokens + completion_tokens)

    if not content:
        finish = (data.get("choices") or [{}])[0].get("finish_reason")
        print(
            f"[LLM] local empty content finish={finish!r} "
            f"completion_tokens={completion_tokens} max_tokens={effective_max}",
            flush=True,
        )

    return content, {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _groq_timeout_seconds(
    *,
    purpose: str = DEFAULT_PURPOSE,
    model_id: str | None = None,
    model_name: str | None = None,
    timeout: int | None = None,
) -> int:
    if timeout is not None:
        return max(1, int(timeout))
    if purpose == PURPOSE_CONTENT_CREATION:
        return max(CONTENT_CREATION_GROQ_TIMEOUT_SECONDS, GROQ_TIMEOUT_SECONDS)
    if _is_reasoning_model(model_id, model_name):
        return max(GROQ_TIMEOUT_SECONDS, 30)
    return GROQ_TIMEOUT_SECONDS


def call_groq_chat_completions(
    messages: list[dict[str, Any]],
    model_name: str,
    *,
    model_id: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: int | None = None,
    purpose: str = DEFAULT_PURPOSE,
) -> tuple[str, dict[str, int], requests.Response]:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    effective_max = _effective_max_tokens(model_id, max_tokens, model_name=model_name, purpose=purpose)
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.80 if temperature is None else temperature,
        "max_tokens": effective_max,
    }
    _apply_groq_reasoning_options(payload, model_id=model_id or "", model_name=model_name)

    response = requests.post(
        GROQ_BASE_URL,
        headers=headers,
        json=payload,
        timeout=_groq_timeout_seconds(
            purpose=purpose,
            model_id=model_id,
            model_name=model_name,
            timeout=timeout,
        ),
    )

    if response.status_code >= 400:
        return "", {}, response

    data = response.json()
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = _extract_chat_content(message)
    usage = data.get("usage") or {}

    prompt_tokens = int(usage.get("prompt_tokens") or _estimate_prompt_tokens(messages))
    completion_tokens = int(usage.get("completion_tokens") or max(1, len(content) // 4))
    total_tokens = int(usage.get("total_tokens") or prompt_tokens + completion_tokens)

    if not content:
        print(
            f"[LLM] groq empty content model={model_name} finish={choice.get('finish_reason')!r} "
            f"completion_tokens={completion_tokens} max_tokens={effective_max} "
            f"reasoning_effort={payload.get('reasoning_effort')!r}",
            flush=True,
        )

    return content, {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }, response


def _effective_provider_mode(model_id: str, provider_mode: str) -> str:
    if model_id == "local" or LLM_MODELS.get(model_id, {}).get("provider") == "local":
        return "local"
    return provider_mode


def _set_active(provider: str, model_id: str, purpose: str = DEFAULT_PURPOSE) -> None:
    LAST_ACTIVE_PROVIDER[purpose] = provider
    LAST_ACTIVE_MODEL_ID[purpose] = model_id


def _local_timeout_for_purpose(purpose: str) -> int:
    if purpose == PURPOSE_CONTENT_CREATION:
        return CONTENT_CREATION_LOCAL_TIMEOUT_SECONDS
    return LOCAL_LLM_TIMEOUT_SECONDS


def _fallback_to_local(
    messages: list[dict[str, Any]],
    *,
    reason: str,
    selected_model_id: str,
    temperature: float | None,
    max_tokens: int | None,
    error: str | None = None,
    purpose: str = DEFAULT_PURPOSE,
    **kwargs: Any,
) -> str:
    LAST_FALLBACK_REASON[purpose] = reason
    LAST_LLM_ERROR[purpose] = error

    print(f"[LLM] falling back to local ({reason}) purpose={purpose}", flush=True)

    content, usage = call_local_chat_completions(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=_local_timeout_for_purpose(purpose),
        **kwargs,
    )
    _set_active("local", "local", purpose)

    record_llm_usage(
        provider="local",
        model_id="local",
        model=LOCAL_LLM_MODEL,
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        total_tokens=usage["total_tokens"],
        success=True,
        fallback=True,
        error=error,
        purpose=purpose,
    )

    return content


def generate_llm_response(
    messages: list[dict[str, Any]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    purpose: str = DEFAULT_PURPOSE,
    **kwargs: Any,
) -> str:
    purpose = _normalize_purpose(purpose)

    with _STATE_LOCK:
        selected_model_id = _get_purpose_model_id(purpose)
        provider_mode = _effective_provider_mode(selected_model_id, _get_purpose_provider_mode(purpose))

    selected_model = LLM_MODELS.get(selected_model_id, LLM_MODELS["local"])
    model_name_hint = selected_model.get("model") if isinstance(selected_model, dict) else None
    effective_max_tokens = _effective_max_tokens(
        selected_model_id,
        max_tokens,
        model_name=str(model_name_hint or ""),
        purpose=purpose,
    )
    estimated_prompt_tokens = _estimate_prompt_tokens(messages)
    estimated_total_tokens = estimated_prompt_tokens + effective_max_tokens

    if provider_mode == "local" or selected_model_id == "local":
        content, usage = call_local_chat_completions(
            messages,
            temperature=temperature,
            max_tokens=effective_max_tokens,
            timeout_seconds=_local_timeout_for_purpose(purpose),
            **kwargs,
        )
        _set_active("local", "local", purpose)
        LAST_FALLBACK_REASON[purpose] = None
        LAST_LLM_ERROR[purpose] = None

        record_llm_usage(
            provider="local",
            model_id="local",
            model=LOCAL_LLM_MODEL,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            success=True,
            fallback=False,
            purpose=purpose,
        )
        return content

    if not groq_is_configured():
        if provider_mode == "groq":
            LAST_LLM_ERROR[purpose] = "Groq API key is not configured"
            raise LLMProviderError(LAST_LLM_ERROR[purpose])
        return _fallback_to_local(
            messages,
            reason="groq_missing_api_key",
            selected_model_id=selected_model_id,
            temperature=temperature,
            max_tokens=effective_max_tokens,
            error="Groq API key is not configured",
            purpose=purpose,
            **kwargs,
        )

    if ONLINE_DISABLED_UNTIL is not None and _utc_now() < ONLINE_DISABLED_UNTIL:
        disabled_msg = f"Groq temporarily disabled until {_utc_iso(ONLINE_DISABLED_UNTIL)}"
        if provider_mode == "groq":
            LAST_LLM_ERROR[purpose] = disabled_msg
            raise LLMProviderError(LAST_LLM_ERROR[purpose])
        return _fallback_to_local(
            messages,
            reason="groq_disabled_until",
            selected_model_id=selected_model_id,
            temperature=temperature,
            max_tokens=effective_max_tokens,
            error=disabled_msg,
            purpose=purpose,
            **kwargs,
        )

    limit_reason = _would_exceed_groq_limits(
        selected_model_id,
        estimated_total_tokens,
        purpose=purpose,
    )
    if limit_reason:
        if provider_mode == "groq":
            LAST_LLM_ERROR[purpose] = limit_reason
            raise LLMProviderError(limit_reason)
        return _fallback_to_local(
            messages,
            reason=limit_reason,
            selected_model_id=selected_model_id,
            temperature=temperature,
            max_tokens=effective_max_tokens,
            error=limit_reason,
            purpose=purpose,
            **kwargs,
        )

    model_name = selected_model["model"]
    groq_timeout = _groq_timeout_seconds(
        purpose=purpose,
        model_id=selected_model_id,
        model_name=model_name,
    )

    try:
        content, usage, response = call_groq_chat_completions(
            messages,
            model_name,
            model_id=selected_model_id,
            temperature=temperature,
            max_tokens=effective_max_tokens,
            timeout=groq_timeout,
            purpose=purpose,
        )
    except requests.Timeout as ex:
        # Creation timeouts are expected for large docs; do not disable Groq for chat.
        if purpose == PURPOSE_CHAT:
            _set_online_disabled(300, "groq_timeout")
        if provider_mode == "groq":
            LAST_LLM_ERROR[purpose] = str(ex)
            record_llm_usage(
                provider="groq",
                model_id=selected_model_id,
                model=model_name,
                prompt_tokens=estimated_prompt_tokens,
                completion_tokens=0,
                total_tokens=estimated_prompt_tokens,
                success=False,
                fallback=False,
                error=str(ex),
                purpose=purpose,
            )
            raise LLMProviderError(f"Groq request timed out: {ex}") from ex
        return _fallback_to_local(
            messages,
            reason="groq_timeout",
            selected_model_id=selected_model_id,
            temperature=temperature,
            max_tokens=effective_max_tokens,
            error=str(ex),
            purpose=purpose,
            **kwargs,
        )
    except requests.RequestException as ex:
        if purpose == PURPOSE_CHAT:
            _set_online_disabled(300, "groq_exception")
        if provider_mode == "groq":
            LAST_LLM_ERROR[purpose] = str(ex)
            record_llm_usage(
                provider="groq",
                model_id=selected_model_id,
                model=model_name,
                prompt_tokens=estimated_prompt_tokens,
                completion_tokens=0,
                total_tokens=estimated_prompt_tokens,
                success=False,
                fallback=False,
                error=str(ex),
                purpose=purpose,
            )
            raise LLMProviderError(f"Groq request failed: {ex}") from ex
        return _fallback_to_local(
            messages,
            reason="groq_exception",
            selected_model_id=selected_model_id,
            temperature=temperature,
            max_tokens=effective_max_tokens,
            error=str(ex),
            purpose=purpose,
            **kwargs,
        )

    if response.status_code == 429:
        retry_seconds = _parse_retry_after_seconds(response) or 300
        _set_online_disabled(retry_seconds, "groq_rate_limited")
        error_text = response.text[:500]
        record_llm_usage(
            provider="groq",
            model_id=selected_model_id,
            model=model_name,
            prompt_tokens=estimated_prompt_tokens,
            completion_tokens=0,
            total_tokens=estimated_prompt_tokens,
            success=False,
            fallback=provider_mode != "groq",
            error=error_text,
            purpose=purpose,
        )
        if provider_mode == "groq":
            LAST_LLM_ERROR[purpose] = "Groq rate limited"
            raise LLMProviderError("Groq rate limited")
        return _fallback_to_local(
            messages,
            reason="groq_rate_limited",
            selected_model_id=selected_model_id,
            temperature=temperature,
            max_tokens=effective_max_tokens,
            error=error_text,
            purpose=purpose,
            **kwargs,
        )

    if response.status_code in (500, 502, 503, 504):
        # Transient server errors: disable globally for chat only. Creation
        # retries should not take chat offline for five minutes.
        if purpose == PURPOSE_CHAT:
            _set_online_disabled(300, "groq_http_error")
        error_text = response.text[:500]
        record_llm_usage(
            provider="groq",
            model_id=selected_model_id,
            model=model_name,
            prompt_tokens=estimated_prompt_tokens,
            completion_tokens=0,
            total_tokens=estimated_prompt_tokens,
            success=False,
            fallback=provider_mode != "groq",
            error=f"HTTP {response.status_code}: {error_text}",
            purpose=purpose,
        )
        if provider_mode == "groq":
            LAST_LLM_ERROR[purpose] = f"Groq HTTP {response.status_code}"
            raise LLMProviderError(LAST_LLM_ERROR[purpose])
        return _fallback_to_local(
            messages,
            reason="groq_http_error",
            selected_model_id=selected_model_id,
            temperature=temperature,
            max_tokens=effective_max_tokens,
            error=error_text,
            purpose=purpose,
            **kwargs,
        )

    if response.status_code >= 400:
        error_text = response.text[:500]
        record_llm_usage(
            provider="groq",
            model_id=selected_model_id,
            model=model_name,
            prompt_tokens=estimated_prompt_tokens,
            completion_tokens=0,
            total_tokens=estimated_prompt_tokens,
            success=False,
            fallback=provider_mode != "groq",
            error=f"HTTP {response.status_code}: {error_text}",
            purpose=purpose,
        )
        if provider_mode == "groq":
            LAST_LLM_ERROR[purpose] = f"Groq HTTP {response.status_code}"
            raise LLMProviderError(LAST_LLM_ERROR[purpose])
        return _fallback_to_local(
            messages,
            reason="groq_http_error",
            selected_model_id=selected_model_id,
            temperature=temperature,
            max_tokens=effective_max_tokens,
            error=error_text,
            purpose=purpose,
            **kwargs,
        )

    _set_active("groq", selected_model_id, purpose)
    LAST_FALLBACK_REASON[purpose] = None
    LAST_LLM_ERROR[purpose] = None

    record_llm_usage(
        provider="groq",
        model_id=selected_model_id,
        model=model_name,
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        total_tokens=usage["total_tokens"],
        success=True,
        fallback=False,
        purpose=purpose,
    )

    return content


def select_llm_model(
    model_id: str,
    provider_mode: str | None = None,
    purpose: str = DEFAULT_PURPOSE,
) -> dict[str, Any]:
    global CURRENT_LLM_MODEL_ID, LLM_PROVIDER_MODE
    global CONTENT_CREATION_MODEL_ID, CONTENT_CREATION_PROVIDER_MODE

    purpose = _normalize_purpose(purpose)

    if model_id not in LLM_MODELS:
        raise ValueError(f"Unknown model_id: {model_id}")

    if provider_mode is not None and provider_mode not in VALID_PROVIDER_MODES:
        raise ValueError(f"Invalid provider_mode: {provider_mode}")

    with _STATE_LOCK:
        if purpose == PURPOSE_CONTENT_CREATION:
            CONTENT_CREATION_MODEL_ID = model_id
            if provider_mode is not None:
                CONTENT_CREATION_PROVIDER_MODE = provider_mode
            if model_id == "local":
                CONTENT_CREATION_PROVIDER_MODE = "local"
            result_model_id = CONTENT_CREATION_MODEL_ID
            result_provider_mode = CONTENT_CREATION_PROVIDER_MODE
        else:
            CURRENT_LLM_MODEL_ID = model_id
            if provider_mode is not None:
                LLM_PROVIDER_MODE = provider_mode
            if model_id == "local":
                LLM_PROVIDER_MODE = "local"
            result_model_id = CURRENT_LLM_MODEL_ID
            result_provider_mode = LLM_PROVIDER_MODE

        LAST_FALLBACK_REASON[purpose] = None
        LAST_LLM_ERROR[purpose] = None

        _save_runtime_config()

        return {
            "purpose": purpose,
            "model_id": result_model_id,
            "provider_mode": result_provider_mode,
            "selected_model": _public_model_entry(result_model_id),
        }


def reset_usage_tracking() -> None:
    with _USAGE_LOCK:
        _save_usage_records([])


def _remaining_estimate(model_id: str) -> dict[str, int | None]:
    model_entry = LLM_MODELS.get(model_id, {})
    limits = model_entry.get("limits") or {}

    rpm = limits.get("rpm")
    rpd = limits.get("rpd")
    tpm = limits.get("tpm")
    tpd = limits.get("tpd")

    # Remaining quota reflects actual account usage across all purposes,
    # since Groq enforces limits per model regardless of caller purpose.
    usage = get_usage_summary(model_id)

    return {
        "requests_this_minute": None if rpm is None else max(0, rpm - usage["requests_last_minute"]),
        "tokens_this_minute": None if tpm is None else max(0, tpm - usage["tokens_last_minute"]),
        "requests_24h": None if rpd is None else max(0, rpd - usage["requests_last_24h"]),
        "tokens_24h": None if tpd is None else max(0, tpd - usage["tokens_last_24h"]),
    }


def _purpose_status(purpose: str) -> dict[str, Any]:
    selected_model_id = _get_purpose_model_id(purpose)
    if selected_model_id not in LLM_MODELS:
        selected_model_id = "local"

    provider_mode = _get_purpose_provider_mode(purpose)
    selected_model = _public_model_entry(selected_model_id)
    usage_model_id = selected_model_id if selected_model.get("provider") == "groq" else None

    if usage_model_id:
        # "usage" is purpose-specific (how much this purpose consumed);
        # "remaining_estimate" is the real shared-quota remaining for the model.
        usage = get_usage_summary(usage_model_id, purpose=purpose)
        remaining = _remaining_estimate(usage_model_id)
    else:
        usage = {
            "requests_last_minute": usage_requests_last_60s(provider="groq", purpose=purpose),
            "tokens_last_minute": usage_tokens_last_60s(provider="groq", purpose=purpose),
            "requests_last_24h": usage_requests_last_24h(provider="groq", purpose=purpose),
            "tokens_last_24h": usage_tokens_last_24h(provider="groq", purpose=purpose),
        }
        remaining = {
            "requests_this_minute": None,
            "tokens_this_minute": None,
            "requests_24h": None,
            "tokens_24h": None,
        }

    local_usage = {
        "requests_last_minute": len(
            _filter_usage_records(since=timedelta(seconds=60), provider="local", purpose=purpose)
        ),
        "tokens_last_minute": sum(
            int(r.get("total_tokens", 0))
            for r in _filter_usage_records(since=timedelta(seconds=60), provider="local", purpose=purpose)
        ),
        "requests_last_24h": len(
            _filter_usage_records(since=timedelta(hours=24), provider="local", purpose=purpose)
        ),
        "tokens_last_24h": sum(
            int(r.get("total_tokens", 0))
            for r in _filter_usage_records(since=timedelta(hours=24), provider="local", purpose=purpose)
        ),
    }

    return {
        "model_id": selected_model_id,
        "provider_mode": provider_mode,
        "selected_model": selected_model,
        "active_provider": LAST_ACTIVE_PROVIDER.get(purpose, "local"),
        "active_model_id": LAST_ACTIVE_MODEL_ID.get(purpose, "local"),
        "online_available": groq_is_available(),
        "online_configured": groq_is_configured(),
        "online_disabled_until": _utc_iso(ONLINE_DISABLED_UNTIL),
        "last_fallback_reason": LAST_FALLBACK_REASON.get(purpose),
        "last_error": LAST_LLM_ERROR.get(purpose),
        "usage": usage,
        "local_usage": local_usage,
        "remaining_estimate": remaining,
    }


def _usage_by_purpose_summary(purpose: str) -> dict[str, Any]:
    # Unlike usage_requests_last_60s/etc (which are online/Groq-only for rate
    # limiting), this reports ALL provider usage (local + groq) for a purpose.
    last_minute = _filter_usage_records(since=timedelta(seconds=60), purpose=purpose)
    last_24h = _filter_usage_records(since=timedelta(hours=24), purpose=purpose)
    return {
        "requests_last_minute": len(last_minute),
        "tokens_last_minute": sum(int(r.get("total_tokens", 0)) for r in last_minute),
        "requests_last_24h": len(last_24h),
        "tokens_last_24h": sum(int(r.get("total_tokens", 0)) for r in last_24h),
    }


def get_llm_status() -> dict[str, Any]:
    chat_status = _purpose_status(PURPOSE_CHAT)
    creation_status = _purpose_status(PURPOSE_CONTENT_CREATION)

    return {
        # --- existing top-level fields kept for backward compatibility (== chat purpose) ---
        "provider_mode": chat_status["provider_mode"],
        "selected_model_id": chat_status["model_id"],
        "selected_model": chat_status["selected_model"],
        "active_provider": chat_status["active_provider"],
        "active_model_id": chat_status["active_model_id"],
        "online_available": chat_status["online_available"],
        "online_configured": chat_status["online_configured"],
        "online_disabled_until": chat_status["online_disabled_until"],
        "last_fallback_reason": chat_status["last_fallback_reason"],
        "last_error": chat_status["last_error"],
        "usage": chat_status["usage"],
        "local_usage": chat_status["local_usage"],
        "remaining_estimate": chat_status["remaining_estimate"],
        "models_endpoint": "/llm/models",
        "select_endpoint": "/llm/select",
        # --- purpose-scoped detail ---
        "purposes": {
            PURPOSE_CHAT: chat_status,
            PURPOSE_CONTENT_CREATION: creation_status,
        },
        "usage_by_purpose": {
            PURPOSE_CHAT: _usage_by_purpose_summary(PURPOSE_CHAT),
            PURPOSE_CONTENT_CREATION: _usage_by_purpose_summary(PURPOSE_CONTENT_CREATION),
        },
    }


def get_llm_models_payload() -> dict[str, Any]:
    return {
        "current_model_id": CURRENT_LLM_MODEL_ID,
        "provider_mode": LLM_PROVIDER_MODE,
        "provider_modes": [
            {
                "id": "local",
                "label": "local — llama.cpp only (must be running on LOCAL_LLM_URL)",
            },
            {
                "id": "groq",
                "label": "groq — Groq only (fails if Groq is unavailable)",
            },
            {
                "id": "groq_with_local_fallback",
                "label": "groq_with_local_fallback — try Groq, fall back to local",
            },
        ],
        "models": _public_model_list(),
        "purposes": {
            PURPOSE_CHAT: {
                "current_model_id": CURRENT_LLM_MODEL_ID,
                "provider_mode": LLM_PROVIDER_MODE,
            },
            PURPOSE_CONTENT_CREATION: {
                "current_model_id": CONTENT_CREATION_MODEL_ID,
                "provider_mode": CONTENT_CREATION_PROVIDER_MODE,
            },
        },
    }


def init_llm_router() -> None:
    _load_secrets_file()
    _load_runtime_config()

    # Refresh local model metadata after env/secrets load.
    LLM_MODELS["local"]["model"] = LOCAL_LLM_MODEL
    LLM_MODELS["local"]["url"] = LOCAL_LLM_URL

    print(
        f"[LLM] router ready chat_model={CURRENT_LLM_MODEL_ID} chat_mode={LLM_PROVIDER_MODE} "
        f"content_creation_model={CONTENT_CREATION_MODEL_ID} content_creation_mode={CONTENT_CREATION_PROVIDER_MODE} "
        f"groq_configured={groq_is_configured()}",
        flush=True,
    )


init_llm_router()
