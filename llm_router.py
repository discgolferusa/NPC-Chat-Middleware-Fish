"""
Multi-provider LLM routing: local llama.cpp (default/fallback) and OpenRouter.

Requests are scoped by "purpose":
  - chat: spoken NPC dialogue turns (default, short completions).
  - content_creation: campaign document import / generation tooling
    (longer completions, its own selected model + provider mode).

When OpenRouter credit usage reaches OPENROUTER_CREDIT_FALLBACK_RATIO (default 80%)
of available credits, online modes automatically fall back to local llama.cpp.

Manual test (PowerShell):
  curl http://127.0.0.1:5000/llm/models
  curl http://127.0.0.1:5000/llm/status
  curl -X POST http://127.0.0.1:5000/llm/select -H "Content-Type: application/json" -d "{\"model_id\":\"local\",\"provider_mode\":\"local\"}"
  curl -X POST http://127.0.0.1:5000/llm/select -H "Content-Type: application/json" -d "{\"model_id\":\"openrouter-gpt-oss-20b\",\"provider_mode\":\"openrouter_with_local_fallback\"}"
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
# llama.cpp rejects requests when prompt_tokens + max_tokens exceeds n_ctx (HTTP 400).
LOCAL_LLM_CONTEXT_TOKENS = int(os.getenv("LOCAL_LLM_CONTEXT_TOKENS", "8192"))
LOCAL_LLM_TIMEOUT_SECONDS = int(os.getenv("LOCAL_LLM_TIMEOUT_SECONDS", "180"))
CONTENT_CREATION_LOCAL_TIMEOUT_SECONDS = int(
    os.getenv("CONTENT_CREATION_LOCAL_TIMEOUT_SECONDS", "600")
)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions"
)
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b")
OPENROUTER_TIMEOUT_SECONDS = int(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "30"))
# content_creation sends large prompts and asks for long JSON completions;
# the chat default is too short and causes cascading timeouts.
CONTENT_CREATION_OPENROUTER_TIMEOUT_SECONDS = int(
    os.getenv("CONTENT_CREATION_OPENROUTER_TIMEOUT_SECONDS", "120")
)
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "https://npcnarrator.com")
OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", "NPC Narrator")
# Auto-fall back to local llama when lifetime credit usage reaches this fraction
# of total available credits. Ratio comes from GET /key (per-key limit or
# OPENROUTER_CREDIT_BUDGET_USD + key usage). /credits is only used with
# OPENROUTER_MANAGEMENT_API_KEY.
OPENROUTER_CREDIT_FALLBACK_RATIO = float(os.getenv("OPENROUTER_CREDIT_FALLBACK_RATIO", "0.8"))
OPENROUTER_CREDITS_CACHE_SECONDS = int(os.getenv("OPENROUTER_CREDITS_CACHE_SECONDS", "60"))
# When /key reports limit: null (typical), divide key lifetime usage by this budget
# (USD credits purchased) to compute the 80% fallback ratio.
OPENROUTER_CREDIT_BUDGET_USD = float(os.getenv("OPENROUTER_CREDIT_BUDGET_USD", "0") or "0")
OPENROUTER_MANAGEMENT_API_KEY = os.getenv("OPENROUTER_MANAGEMENT_API_KEY", "")
# Comma-separated OpenRouter provider slugs. Default prefers Bedrock with Novita backup.
# Empty OPENROUTER_PROVIDER_ORDER disables preference routing (OpenRouter price load-balance).
_DEFAULT_OPENROUTER_PROVIDER_ORDER = "amazon-bedrock,novita"
_OPENROUTER_PROVIDER_ORDER_ENV_SET = "OPENROUTER_PROVIDER_ORDER" in os.environ
OPENROUTER_PROVIDER_ORDER_RAW = os.getenv(
    "OPENROUTER_PROVIDER_ORDER", _DEFAULT_OPENROUTER_PROVIDER_ORDER
)
# False = only try providers in order (no Parasail/DeepInfra/etc.). True = fall through.
_OPENROUTER_PROVIDER_ALLOW_FALLBACKS_ENV = os.getenv(
    "OPENROUTER_PROVIDER_ALLOW_FALLBACKS", "false"
)
OPENROUTER_PROVIDER_ALLOW_FALLBACKS = _OPENROUTER_PROVIDER_ALLOW_FALLBACKS_ENV.strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

# Spoken NPC replies are short, but reasoning models spend completion tokens on
# chain-of-thought first. max_tokens=180 with gpt-oss often yields empty
# message.content (finish_reason=length) -> "I have nothing more to say."
DEFAULT_MAX_TOKENS = 180
REASONING_MAX_TOKENS = int(os.getenv("LLM_REASONING_MAX_TOKENS", "768"))

# content_creation purpose (campaign document import, generation tooling) needs
# much larger completions than spoken NPC dialogue, so it gets its own default.
CONTENT_CREATION_DEFAULT_MAX_TOKENS = int(os.getenv("LLM_CONTENT_CREATION_MAX_TOKENS", "4096"))

LLM_USAGE_FILE = BASE_DIR / "llm_usage.json"
LLM_RUNTIME_CONFIG_FILE = BASE_DIR / "llm_runtime_config.json"
LLM_SECRETS_FILE = BASE_DIR / "llm_secrets.env"

VALID_PROVIDER_MODES = frozenset({"local", "openrouter", "openrouter_with_local_fallback"})
# Older llm_runtime_config.json / Admin UI selections used Groq mode names.
PROVIDER_MODE_ALIASES: dict[str, str] = {
    "groq": "openrouter",
    "groq_with_local_fallback": "openrouter_with_local_fallback",
}
ONLINE_PROVIDER_MODES = frozenset({"openrouter", "openrouter_with_local_fallback"})
ONLINE_WITH_FALLBACK_MODE = "openrouter_with_local_fallback"
ONLINE_STRICT_MODE = "openrouter"
REASONING_MODEL_MARKERS = ("gpt-oss", "qwen3", "nemotron", "deepseek-v4")

PURPOSE_CHAT = "chat"
PURPOSE_CONTENT_CREATION = "content_creation"
VALID_PURPOSES = frozenset({PURPOSE_CHAT, PURPOSE_CONTENT_CREATION})
DEFAULT_PURPOSE = PURPOSE_CHAT

# OpenRouter catalog. Credit budget (not RPM/TPM) gates online vs local fallback.
# `purposes` gates which Admin -> LLM dropdown offers the model.
# Optional `provider_order` pins OpenRouter routing for that catalog entry
# (overrides OPENROUTER_PROVIDER_ORDER for that model only).
# `provider_label` is shown in Admin dropdowns so each option names its host.
LLM_MODELS: dict[str, dict[str, Any]] = {
    "local": {
        "id": "local",
        "label": "Local llama.cpp — local",
        "provider": "local",
        "provider_label": "local",
        "model": LOCAL_LLM_MODEL,
        "url": LOCAL_LLM_URL,
        "online": False,
        "limits": None,
        "purposes": [PURPOSE_CHAT, PURPOSE_CONTENT_CREATION],
    },
    "openrouter-gpt-oss-20b": {
        "id": "openrouter-gpt-oss-20b",
        "label": "GPT-OSS 20B — Amazon Bedrock / Novita",
        "provider": "openrouter",
        "provider_label": "Amazon Bedrock / Novita",
        "model": "openai/gpt-oss-20b",
        "online": True,
        "limits": None,
        "purposes": [PURPOSE_CHAT, PURPOSE_CONTENT_CREATION],
    },
    "openrouter-nemotron-3.5-lightning-coreweave": {
        "id": "openrouter-nemotron-3.5-lightning-coreweave",
        "label": "Nemotron 3.5 Lightning — CoreWeave",
        "provider": "openrouter",
        "provider_label": "CoreWeave",
        "model": "nvidia/nemotron-3.5-lightning",
        "provider_order": ["coreweave"],
        "online": True,
        "limits": None,
        "purposes": [PURPOSE_CHAT, PURPOSE_CONTENT_CREATION],
    },
    "openrouter-nemotron-3.5-lightning-deepinfra": {
        "id": "openrouter-nemotron-3.5-lightning-deepinfra",
        "label": "Nemotron 3.5 Lightning — DeepInfra",
        "provider": "openrouter",
        "provider_label": "DeepInfra",
        "model": "nvidia/nemotron-3.5-lightning",
        "provider_order": ["deepinfra"],
        "online": True,
        "limits": None,
        "purposes": [PURPOSE_CHAT, PURPOSE_CONTENT_CREATION],
    },
    "openrouter-deepseek-v4-flash-0731-coreweave": {
        "id": "openrouter-deepseek-v4-flash-0731-coreweave",
        "label": "DeepSeek V4 Flash 0731 — CoreWeave",
        "provider": "openrouter",
        "provider_label": "CoreWeave",
        "model": "deepseek/deepseek-v4-flash-0731",
        "provider_order": ["coreweave"],
        "online": True,
        "limits": None,
        "purposes": [PURPOSE_CHAT, PURPOSE_CONTENT_CREATION],
    },
}

# Retired Groq catalog ids (and chained aliases) remap onto OpenRouter gpt-oss-20b.
DEPRECATED_MODEL_ALIASES: dict[str, str] = {
    "groq-llama-4-scout": "groq-llama-3.3-70b",
    "groq-qwen3-32b": "openrouter-gpt-oss-20b",
    "groq-llama-3.1-8b-instant": "openrouter-gpt-oss-20b",
    "groq-llama-3.3-70b": "openrouter-gpt-oss-20b",
    "groq-gpt-oss-20b": "openrouter-gpt-oss-20b",
    "groq-gpt-oss-120b": "openrouter-gpt-oss-20b",
    "groq-qwen3.6-27b": "openrouter-gpt-oss-20b",
    "groq-compound": "openrouter-gpt-oss-20b",
    "groq-compound-mini": "openrouter-gpt-oss-20b",
}

# --- chat purpose (default / backward-compatible globals) ---
CURRENT_LLM_MODEL_ID = os.getenv("CURRENT_LLM_MODEL_ID", "local")
LLM_PROVIDER_MODE = os.getenv("LLM_PROVIDER_MODE", "openrouter_with_local_fallback")
CHAT_FALLBACK_MODEL_IDS: list[str] = ["local"]

# --- content_creation purpose (campaign/document generation tooling) ---
CONTENT_CREATION_MODEL_ID = os.getenv("CONTENT_CREATION_LLM_MODEL_ID", CURRENT_LLM_MODEL_ID)
CONTENT_CREATION_PROVIDER_MODE = os.getenv("CONTENT_CREATION_LLM_PROVIDER_MODE", LLM_PROVIDER_MODE)
CONTENT_CREATION_FALLBACK_MODEL_IDS: list[str] = ["local"]

# Per-purpose runtime status. Mutated in place (never rebound) so callers never
# need `global` to update them.
LAST_ACTIVE_PROVIDER: dict[str, str] = {PURPOSE_CHAT: "local", PURPOSE_CONTENT_CREATION: "local"}
LAST_ACTIVE_MODEL_ID: dict[str, str] = {PURPOSE_CHAT: "local", PURPOSE_CONTENT_CREATION: "local"}
LAST_FALLBACK_REASON: dict[str, str | None] = {PURPOSE_CHAT: None, PURPOSE_CONTENT_CREATION: None}
LAST_LLM_ERROR: dict[str, str | None] = {PURPOSE_CHAT: None, PURPOSE_CONTENT_CREATION: None}

ONLINE_DISABLED_UNTIL: datetime | None = None

# Keep ledger longer than the 24h rate-limit window so Admin cost charts have history.
USAGE_RETENTION_HOURS = int(os.getenv("LLM_USAGE_RETENTION_HOURS", str(24 * 30)))

# Alpha OpenRouter list prices (USD per 1M tokens). Not billed by us — estimation only.
# Source: OpenRouter endpoint pricing (CoreWeave rows for pinned catalog entries).
ONLINE_ON_DEMAND_PRICES_USD: dict[str, dict[str, float]] = {
    "local": {"input": 0.0, "output": 0.0},
    "openrouter-gpt-oss-20b": {"input": 0.03, "output": 0.13},
    "openrouter-nemotron-3.5-lightning-coreweave": {"input": 0.10, "output": 0.25},
    "openrouter-nemotron-3.5-lightning-deepinfra": {"input": 0.08, "output": 0.20},
    "openrouter-deepseek-v4-flash-0731-coreweave": {"input": 0.13, "output": 0.28},
}
# Backward-compatible alias for older imports/tests.
GROQ_ON_DEMAND_PRICES_USD = ONLINE_ON_DEMAND_PRICES_USD

_STATE_LOCK = threading.Lock()
_USAGE_LOCK = threading.Lock()
_CREDITS_LOCK = threading.Lock()
_CREDITS_SNAPSHOT: dict[str, Any] | None = None
_CREDITS_LOOKUP_UNTIL: datetime | None = None


class LLMProviderError(Exception):
    """Raised when strict online-only mode cannot complete a request."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_provider_mode(mode: str | None, *, default: str = "local") -> str:
    candidate = PROVIDER_MODE_ALIASES.get(str(mode or "").strip(), str(mode or "").strip())
    if candidate in VALID_PROVIDER_MODES:
        return candidate
    return default


def _allows_local_fallback(provider_mode: str) -> bool:
    return provider_mode != ONLINE_STRICT_MODE


def _is_online_provider_mode(provider_mode: str) -> bool:
    return provider_mode in ONLINE_PROVIDER_MODES


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
    raw = CONTENT_CREATION_PROVIDER_MODE if purpose == PURPOSE_CONTENT_CREATION else LLM_PROVIDER_MODE
    return _normalize_provider_mode(raw, default="local")


def _get_purpose_fallback_model_ids(purpose: str) -> list[str]:
    if purpose == PURPOSE_CONTENT_CREATION:
        return list(CONTENT_CREATION_FALLBACK_MODEL_IDS)
    return list(CHAT_FALLBACK_MODEL_IDS)


def _normalize_fallback_model_ids(
    raw: Any,
    *,
    purpose: str,
    primary_model_id: str | None = None,
) -> list[str]:
    """Dedupe and validate an ordered fallback chain for a purpose."""
    if raw is None:
        items: list[Any] = []
    elif isinstance(raw, str):
        items = [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        items = []

    primary = (primary_model_id or "").strip()
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        candidate = _alias_model_id(str(item or "").strip())
        if not candidate or candidate == primary or candidate in seen:
            continue
        if candidate not in LLM_MODELS:
            continue
        if not model_supports_purpose(candidate, purpose):
            continue
        seen.add(candidate)
        out.append(candidate)
    return out


def _default_fallback_model_ids(provider_mode: str) -> list[str]:
    if provider_mode == ONLINE_WITH_FALLBACK_MODE:
        return ["local"]
    return []


def _parse_allow_fallbacks(raw: str | None, *, default: bool = False) -> bool:
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _load_secrets_file() -> None:
    """Load OpenRouter settings from llm_secrets.env when process env did not set them."""
    global OPENROUTER_API_KEY
    global OPENROUTER_PROVIDER_ORDER_RAW
    global OPENROUTER_PROVIDER_ALLOW_FALLBACKS
    global _OPENROUTER_PROVIDER_ORDER_ENV_SET

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
            if not value:
                continue
            if key == "OPENROUTER_API_KEY" and not OPENROUTER_API_KEY:
                OPENROUTER_API_KEY = value
            elif key == "OPENROUTER_PROVIDER_ORDER" and not _OPENROUTER_PROVIDER_ORDER_ENV_SET:
                OPENROUTER_PROVIDER_ORDER_RAW = value
                _OPENROUTER_PROVIDER_ORDER_ENV_SET = True
            elif (
                key == "OPENROUTER_PROVIDER_ALLOW_FALLBACKS"
                and "OPENROUTER_PROVIDER_ALLOW_FALLBACKS" not in os.environ
            ):
                OPENROUTER_PROVIDER_ALLOW_FALLBACKS = _parse_allow_fallbacks(value, default=False)
    except OSError as ex:
        print(f"[LLM] failed to read secrets file: {ex}", flush=True)


def model_supports_purpose(model_id: str, purpose: str) -> bool:
    """Whether the catalog offers ``model_id`` for ``purpose``.

    Entries with no explicit ``purposes`` list count as universal, so a trimmed
    catalog can never leave a dropdown with nothing to choose.
    """
    entry = LLM_MODELS.get(model_id)
    if not entry:
        return False
    allowed = entry.get("purposes")
    if not allowed:
        return True
    return _normalize_purpose(purpose) in allowed


def models_for_purpose(purpose: str) -> list[str]:
    normalized = _normalize_purpose(purpose)
    return [mid for mid in LLM_MODELS if model_supports_purpose(mid, normalized)]


def _alias_model_id(model_id: str | None) -> str:
    """Follow retired-model aliases only. Does not invent a fallback catalog id."""
    candidate = str(model_id or "").strip()
    if not candidate:
        return ""
    seen: set[str] = set()
    while candidate in DEPRECATED_MODEL_ALIASES and candidate not in seen:
        seen.add(candidate)
        candidate = DEPRECATED_MODEL_ALIASES[candidate]
    return candidate


def _resolve_model_id(
    model_id: str | None,
    *,
    default: str = "local",
    purpose: str | None = None,
) -> str:
    """Map retired catalog ids onto current models, else fall back to default.

    When ``purpose`` is given, a model the catalog no longer offers for that
    purpose is treated the same as a retired id.
    """
    candidate = _alias_model_id(model_id)
    if not candidate:
        candidate = default

    def _usable(mid: str) -> bool:
        if mid not in LLM_MODELS:
            return False
        return purpose is None or model_supports_purpose(mid, purpose)

    if _usable(candidate):
        return candidate
    if _usable(default):
        return default
    return "local"


def _load_runtime_config() -> None:
    global CURRENT_LLM_MODEL_ID, LLM_PROVIDER_MODE, CHAT_FALLBACK_MODEL_IDS
    global CONTENT_CREATION_MODEL_ID, CONTENT_CREATION_PROVIDER_MODE
    global CONTENT_CREATION_FALLBACK_MODEL_IDS

    if not LLM_RUNTIME_CONFIG_FILE.exists():
        CURRENT_LLM_MODEL_ID = "local"
        if os.getenv("LLM_PROVIDER_MODE"):
            LLM_PROVIDER_MODE = _normalize_provider_mode(LLM_PROVIDER_MODE, default="local")
        else:
            LLM_PROVIDER_MODE = "local"
        CHAT_FALLBACK_MODEL_IDS = _default_fallback_model_ids(LLM_PROVIDER_MODE)

        CONTENT_CREATION_MODEL_ID = _resolve_model_id(
            os.getenv("CONTENT_CREATION_LLM_MODEL_ID", CURRENT_LLM_MODEL_ID),
            default="local",
            purpose=PURPOSE_CONTENT_CREATION,
        )
        CONTENT_CREATION_PROVIDER_MODE = _normalize_provider_mode(
            os.getenv("CONTENT_CREATION_LLM_PROVIDER_MODE", LLM_PROVIDER_MODE),
            default="local",
        )
        CONTENT_CREATION_FALLBACK_MODEL_IDS = _default_fallback_model_ids(
            CONTENT_CREATION_PROVIDER_MODE
        )
        return

    try:
        raw = json.loads(LLM_RUNTIME_CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as ex:
        print(f"[LLM] invalid runtime config, using defaults: {ex}", flush=True)
        CURRENT_LLM_MODEL_ID = "local"
        LLM_PROVIDER_MODE = "local"
        CHAT_FALLBACK_MODEL_IDS = []
        CONTENT_CREATION_MODEL_ID = "local"
        CONTENT_CREATION_PROVIDER_MODE = "local"
        CONTENT_CREATION_FALLBACK_MODEL_IDS = []
        return

    data = raw if isinstance(raw, dict) else {}
    purposes = data.get("purposes") if isinstance(data.get("purposes"), dict) else {}

    # Backward compat: pre-purpose runtime config files only had top-level
    # model_id/provider_mode. Treat those as the "chat" purpose.
    chat_cfg = purposes.get(PURPOSE_CHAT) if isinstance(purposes.get(PURPOSE_CHAT), dict) else {}
    raw_chat_model_id = str(chat_cfg.get("model_id", data.get("model_id", CURRENT_LLM_MODEL_ID)) or "")
    chat_model_id = _resolve_model_id(raw_chat_model_id, default="local", purpose=PURPOSE_CHAT)
    raw_chat_provider_mode = chat_cfg.get("provider_mode", data.get("provider_mode", LLM_PROVIDER_MODE))

    CURRENT_LLM_MODEL_ID = chat_model_id
    LLM_PROVIDER_MODE = _normalize_provider_mode(raw_chat_provider_mode, default="local")
    raw_chat_fallbacks = chat_cfg.get(
        "fallback_model_ids",
        data.get("fallback_model_ids", _default_fallback_model_ids(LLM_PROVIDER_MODE)),
    )
    CHAT_FALLBACK_MODEL_IDS = _normalize_fallback_model_ids(
        raw_chat_fallbacks,
        purpose=PURPOSE_CHAT,
        primary_model_id=CURRENT_LLM_MODEL_ID,
    )

    creation_cfg = (
        purposes.get(PURPOSE_CONTENT_CREATION)
        if isinstance(purposes.get(PURPOSE_CONTENT_CREATION), dict)
        else {}
    )
    raw_creation_model_id = str(creation_cfg.get("model_id", CONTENT_CREATION_MODEL_ID) or "")
    creation_model_id = _resolve_model_id(
        raw_creation_model_id,
        default=CURRENT_LLM_MODEL_ID,
        purpose=PURPOSE_CONTENT_CREATION,
    )
    raw_creation_provider_mode = creation_cfg.get("provider_mode", CONTENT_CREATION_PROVIDER_MODE)

    CONTENT_CREATION_MODEL_ID = creation_model_id
    CONTENT_CREATION_PROVIDER_MODE = _normalize_provider_mode(
        raw_creation_provider_mode, default="local"
    )
    raw_creation_fallbacks = creation_cfg.get(
        "fallback_model_ids",
        _default_fallback_model_ids(CONTENT_CREATION_PROVIDER_MODE),
    )
    CONTENT_CREATION_FALLBACK_MODEL_IDS = _normalize_fallback_model_ids(
        raw_creation_fallbacks,
        purpose=PURPOSE_CONTENT_CREATION,
        primary_model_id=CONTENT_CREATION_MODEL_ID,
    )

    remapped_modes = (
        raw_chat_provider_mode != LLM_PROVIDER_MODE
        or raw_creation_provider_mode != CONTENT_CREATION_PROVIDER_MODE
    )
    if (
        raw_chat_model_id != CURRENT_LLM_MODEL_ID
        or raw_creation_model_id != CONTENT_CREATION_MODEL_ID
        or remapped_modes
    ):
        print(
            f"[LLM] remapped retired model/mode ids: chat {raw_chat_model_id!r}->{CURRENT_LLM_MODEL_ID!r} "
            f"mode {raw_chat_provider_mode!r}->{LLM_PROVIDER_MODE!r}, "
            f"content_creation {raw_creation_model_id!r}->{CONTENT_CREATION_MODEL_ID!r} "
            f"mode {raw_creation_provider_mode!r}->{CONTENT_CREATION_PROVIDER_MODE!r}",
            flush=True,
        )
        _save_runtime_config()


def _save_runtime_config() -> None:
    payload = {
        # Legacy top-level keys kept for backward compatibility; mirror "chat".
        "model_id": CURRENT_LLM_MODEL_ID,
        "provider_mode": LLM_PROVIDER_MODE,
        "fallback_model_ids": list(CHAT_FALLBACK_MODEL_IDS),
        "purposes": {
            PURPOSE_CHAT: {
                "model_id": CURRENT_LLM_MODEL_ID,
                "provider_mode": LLM_PROVIDER_MODE,
                "fallback_model_ids": list(CHAT_FALLBACK_MODEL_IDS),
            },
            PURPOSE_CONTENT_CREATION: {
                "model_id": CONTENT_CREATION_MODEL_ID,
                "provider_mode": CONTENT_CREATION_PROVIDER_MODE,
                "fallback_model_ids": list(CONTENT_CREATION_FALLBACK_MODEL_IDS),
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


def _public_model_list(purpose: str | None = None) -> list[dict[str, Any]]:
    model_ids = LLM_MODELS if purpose is None else models_for_purpose(purpose)
    return [_public_model_entry(model_id) for model_id in model_ids]


def openrouter_is_configured() -> bool:
    return bool(OPENROUTER_API_KEY)


# Backward-compatible aliases for older callers/tests.
groq_is_configured = openrouter_is_configured


def openrouter_is_available() -> bool:
    if not openrouter_is_configured():
        return False
    if ONLINE_DISABLED_UNTIL is not None and _utc_now() < ONLINE_DISABLED_UNTIL:
        return False
    return True


groq_is_available = openrouter_is_available


def _set_online_disabled(seconds: int, reason: str) -> None:
    global ONLINE_DISABLED_UNTIL

    ONLINE_DISABLED_UNTIL = _utc_now() + timedelta(seconds=max(1, seconds))
    print(
        f"[LLM] OpenRouter disabled until {_utc_iso(ONLINE_DISABLED_UNTIL)} ({reason})",
        flush=True,
    )


def _openrouter_api_root() -> str:
    base = (OPENROUTER_BASE_URL or "").rstrip("/")
    if base.endswith("/chat/completions"):
        return base[: -len("/chat/completions")]
    if base.endswith("/v1"):
        return base
    return "https://openrouter.ai/api/v1"


def _openrouter_auth_headers() -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    if OPENROUTER_HTTP_REFERER:
        headers["HTTP-Referer"] = OPENROUTER_HTTP_REFERER
    if OPENROUTER_APP_TITLE:
        headers["X-Title"] = OPENROUTER_APP_TITLE
    return headers


def _credit_snapshot_from_key_payload(data: dict[str, Any]) -> dict[str, Any] | None:
    """Build a credit snapshot from GET /api/v1/key ``data`` object."""
    limit = data.get("limit")
    limit_remaining = data.get("limit_remaining")
    usage = data.get("usage")
    try:
        usage_f = float(usage) if usage is not None else None
    except (TypeError, ValueError):
        usage_f = None

    total_credits: float | None = None
    total_usage: float | None = usage_f
    usage_ratio: float | None = None
    source = "key"

    try:
        limit_f = float(limit) if limit is not None else None
    except (TypeError, ValueError):
        limit_f = None
    try:
        remaining_f = float(limit_remaining) if limit_remaining is not None else None
    except (TypeError, ValueError):
        remaining_f = None

    if limit_f is not None and limit_f > 0:
        total_credits = limit_f
        if remaining_f is not None:
            total_usage = max(0.0, limit_f - remaining_f)
        elif usage_f is not None:
            total_usage = usage_f
        if total_usage is not None:
            usage_ratio = max(0.0, min(1.0, total_usage / limit_f))
    elif usage_f is not None and remaining_f is not None and remaining_f >= 0:
        # Unlimited key but remaining is reported — treat usage+remaining as total.
        total_usage = usage_f
        total_credits = usage_f + remaining_f
        if total_credits > 0:
            usage_ratio = max(0.0, min(1.0, total_usage / total_credits))
    elif usage_f is not None and OPENROUTER_CREDIT_BUDGET_USD > 0:
        # Typical setup: limit is null; compare key lifetime usage to configured budget.
        total_credits = OPENROUTER_CREDIT_BUDGET_USD
        total_usage = usage_f
        usage_ratio = max(0.0, min(1.0, usage_f / OPENROUTER_CREDIT_BUDGET_USD))
        source = "key+budget"

    if usage_ratio is None:
        return None

    return {
        "source": source,
        "total_credits": total_credits,
        "total_usage": total_usage,
        "usage_ratio": usage_ratio,
        "limit": limit_f,
        "limit_remaining": remaining_f,
        "budget_usd": OPENROUTER_CREDIT_BUDGET_USD if source == "key+budget" else None,
        "is_free_tier": bool(data.get("is_free_tier")),
        "fetched_at": _utc_iso(_utc_now()),
    }


def _credit_snapshot_from_credits_payload(data: dict[str, Any]) -> dict[str, Any] | None:
    try:
        total_credits = float(data.get("total_credits"))
        total_usage = float(data.get("total_usage"))
    except (TypeError, ValueError):
        return None
    if total_credits <= 0:
        return None
    usage_ratio = max(0.0, min(1.0, total_usage / total_credits))
    return {
        "source": "credits",
        "total_credits": total_credits,
        "total_usage": total_usage,
        "usage_ratio": usage_ratio,
        "limit": total_credits,
        "limit_remaining": max(0.0, total_credits - total_usage),
        "is_free_tier": None,
        "fetched_at": _utc_iso(_utc_now()),
    }


def _fetch_openrouter_credit_snapshot_live() -> dict[str, Any] | None:
    """Call OpenRouter once (no cache). Prefer /key; /credits only with management key."""
    root = _openrouter_api_root()
    headers = _openrouter_auth_headers()
    snapshot: dict[str, Any] | None = None

    try:
        key_resp = requests.get(f"{root}/key", headers=headers, timeout=10)
        if key_resp.status_code < 400:
            payload = key_resp.json() if key_resp.content else {}
            data = payload.get("data") if isinstance(payload, dict) else None
            if isinstance(data, dict):
                snapshot = _credit_snapshot_from_key_payload(data)
    except (requests.RequestException, ValueError, TypeError) as ex:
        print(f"[LLM] OpenRouter /key credit check failed: {ex}", flush=True)

    if snapshot is None and OPENROUTER_MANAGEMENT_API_KEY:
        mgmt_headers = {
            "Authorization": f"Bearer {OPENROUTER_MANAGEMENT_API_KEY}",
            "Content-Type": "application/json",
        }
        try:
            credits_resp = requests.get(f"{root}/credits", headers=mgmt_headers, timeout=10)
            if credits_resp.status_code < 400:
                payload = credits_resp.json() if credits_resp.content else {}
                data = payload.get("data") if isinstance(payload, dict) else None
                if isinstance(data, dict):
                    snapshot = _credit_snapshot_from_credits_payload(data)
        except (requests.RequestException, ValueError, TypeError) as ex:
            print(f"[LLM] OpenRouter /credits check failed: {ex}", flush=True)

    return snapshot


def _copy_credit_snapshot(snapshot: dict[str, Any] | None) -> dict[str, Any] | None:
    return dict(snapshot) if snapshot is not None else None


def fetch_openrouter_credit_snapshot(*, force: bool = False) -> dict[str, Any] | None:
    """Return cached OpenRouter credit usage, or refresh from /key (then /credits)."""
    global _CREDITS_SNAPSHOT, _CREDITS_LOOKUP_UNTIL

    if not openrouter_is_configured():
        return None

    now = _utc_now()
    with _CREDITS_LOCK:
        if (
            not force
            and _CREDITS_LOOKUP_UNTIL is not None
            and now < _CREDITS_LOOKUP_UNTIL
        ):
            return _copy_credit_snapshot(_CREDITS_SNAPSHOT)

    snapshot = _fetch_openrouter_credit_snapshot_live()

    with _CREDITS_LOCK:
        # Another thread may have refreshed while this one was in flight.
        refreshed_now = _utc_now()
        if (
            not force
            and _CREDITS_LOOKUP_UNTIL is not None
            and refreshed_now < _CREDITS_LOOKUP_UNTIL
        ):
            return _copy_credit_snapshot(_CREDITS_SNAPSHOT)

        ttl = timedelta(seconds=max(5, OPENROUTER_CREDITS_CACHE_SECONDS))
        if snapshot is not None:
            cached = dict(snapshot)
            _CREDITS_SNAPSHOT = cached
            _CREDITS_LOOKUP_UNTIL = refreshed_now + ttl
            return dict(cached)

        if _CREDITS_SNAPSHOT is not None:
            # Failed refresh — keep the last good snapshot; do not negative-cache over it.
            return dict(_CREDITS_SNAPSHOT)

        _CREDITS_SNAPSHOT = None
        _CREDITS_LOOKUP_UNTIL = refreshed_now + ttl
        return None


def openrouter_credit_threshold_reason() -> str | None:
    """If credit usage >= fallback ratio, return a human-readable fallback reason."""
    threshold = max(0.0, min(1.0, OPENROUTER_CREDIT_FALLBACK_RATIO))
    snapshot = fetch_openrouter_credit_snapshot()
    if not snapshot:
        return None
    ratio = snapshot.get("usage_ratio")
    if ratio is None:
        return None
    try:
        ratio_f = float(ratio)
    except (TypeError, ValueError):
        return None
    if ratio_f < threshold:
        return None
    total_credits = snapshot.get("total_credits")
    total_usage = snapshot.get("total_usage")
    pct = int(round(threshold * 100))
    return (
        f"OpenRouter credit usage is at {ratio_f:.0%} of available credits "
        f"(threshold {pct}%; used {total_usage} of {total_credits}). "
        "Falling back to local llama.cpp until credits are topped up."
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


def _apply_reasoning_options(payload: dict[str, Any], *, model_id: str, model_name: str) -> None:
    """Keep reasoning cheap so spoken dialogue still fits in max_tokens."""
    if not _is_reasoning_model(model_id, model_name):
        return
    name = (model_name or "").casefold()
    mid = (model_id or "").casefold()
    if "gpt-oss" in name or "gpt-oss" in mid:
        payload.setdefault("reasoning_effort", "low")
    elif "qwen3" in name or "qwen3" in mid:
        payload.setdefault("reasoning_effort", "none")
    elif "nemotron" in name or "nemotron" in mid:
        payload.setdefault("reasoning_effort", "low")
    elif "deepseek-v4" in name or "deepseek-v4" in mid:
        payload.setdefault("reasoning_effort", "low")


_apply_groq_reasoning_options = _apply_reasoning_options


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
    cutoff = _utc_now() - timedelta(hours=max(24, USAGE_RETENTION_HOURS))
    pruned = []
    for record in records:
        ts = _parse_utc_iso(record.get("timestamp"))
        if ts is None or ts >= cutoff:
            pruned.append(record)

    try:
        LLM_USAGE_FILE.write_text(json.dumps(pruned, indent=2), encoding="utf-8")
    except OSError as ex:
        print(f"[LLM] failed to save usage ledger: {ex}", flush=True)


def _request_owner_user_id() -> str:
    try:
        from session_context import get_request_owner

        return (get_request_owner() or "").strip()
    except Exception:
        return ""


def _pricing_for_model(model_id: str) -> dict[str, float]:
    entry = GROQ_ON_DEMAND_PRICES_USD.get(model_id)
    if entry:
        return {"input": float(entry["input"]), "output": float(entry["output"])}
    # Unknown online ids: treat as zero so we still show tokens without inventing rates.
    if model_id == "local" or not model_id:
        return {"input": 0.0, "output": 0.0}
    return {"input": 0.0, "output": 0.0}


def _estimate_cost_usd(model_id: str, prompt_tokens: int, completion_tokens: int) -> dict[str, float]:
    prices = _pricing_for_model(model_id)
    input_cost = (max(0, int(prompt_tokens)) / 1_000_000.0) * prices["input"]
    output_cost = (max(0, int(completion_tokens)) / 1_000_000.0) * prices["output"]
    return {
        "input_price_per_mtok": prices["input"],
        "output_price_per_mtok": prices["output"],
        "estimated_input_cost_usd": round(input_cost, 6),
        "estimated_output_cost_usd": round(output_cost, 6),
        "estimated_total_cost_usd": round(input_cost + output_cost, 6),
    }


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
    selected_model_id: str | None = None,
    attribute_cost_to_selected: bool = False,
) -> None:
    """Append one usage row.

    When ``attribute_cost_to_selected`` is true (local fallback while an online model
    was selected), ``cost_model_id`` points at the selected online model so Admin
    alpha cost charts treat the tokens as if that model had succeeded.
    """
    selected = (selected_model_id or model_id or "").strip() or model_id
    if attribute_cost_to_selected and selected and selected != "local":
        cost_model_id = selected
    else:
        cost_model_id = model_id

    record = {
        "timestamp": _utc_iso(_utc_now()),
        "provider": provider,
        "model_id": model_id,
        "model": model,
        "selected_model_id": selected,
        "cost_model_id": cost_model_id,
        "owner_user_id": _request_owner_user_id(),
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
        if online_only and record.get("provider") not in {"openrouter", "groq"}:
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


def _would_exceed_online_limits(
    model_id: str,
    estimated_total_tokens: int,
    *,
    purpose: str = DEFAULT_PURPOSE,
) -> str | None:
    """Preflight before an OpenRouter call.

    Primary gate is account credit usage (>= OPENROUTER_CREDIT_FALLBACK_RATIO).
    Optional catalog ``limits`` (rpm/rpd/tpm/tpd) remain supported if present.
    """
    credit_reason = openrouter_credit_threshold_reason()
    if credit_reason:
        return credit_reason

    model_entry = LLM_MODELS.get(model_id)
    if not model_entry or model_entry.get("provider") != "openrouter":
        return None

    limits = model_entry.get("limits") or {}
    if not limits:
        return None

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
            f"OpenRouter requests-per-minute limit reached for {label} "
            f"({rpm}/min). Wait about a minute, then retry."
        )

    rpd = limits.get("rpd")
    if rpd is not None and usage_requests_last_24h(model_id=model_id) + 1 > rpd:
        return (
            f"OpenRouter requests-per-day limit reached for {label} "
            f"({rpd}/day). Try again tomorrow or switch the {model_slot} in Admin → LLM."
        )

    tpm = limits.get("tpm")
    if tpm is not None:
        used = usage_tokens_last_60s(model_id=model_id)
        if used + estimated_total_tokens > tpm:
            return (
                f"OpenRouter tokens-per-minute budget too small for this request on {label} "
                f"(need ~{estimated_total_tokens} tokens, limit {tpm}/min, "
                f"~{used} used in the last minute)."
            )

    tpd = limits.get("tpd")
    if tpd is not None and usage_tokens_last_24h(model_id=model_id) + estimated_total_tokens > tpd:
        return (
            f"OpenRouter tokens-per-day limit reached for {label} "
            f"({tpd}/day). Try again tomorrow or switch the {model_slot} in Admin → LLM."
        )

    return None


_would_exceed_groq_limits = _would_exceed_online_limits



# Rough tokens reserved for system prompt + outline instructions outside the document body.
_CONTENT_CREATION_PROMPT_OVERHEAD_TOKENS = 1_000
# Import outline/detail passes may request up to 6144 completion tokens.
_CONTENT_CREATION_IMPORT_MAX_TOKENS = 6144


def _local_import_completion_reserve_tokens() -> int:
    """Completion share reserved when sizing local import prompts.

    Import callers often request up to 6144 max_tokens, but that much must not be
    locked aside before shrinking the document — otherwise the prompt is starved
    and end-biased NPC sections get dropped. Match the same share used by
    ``_local_content_creation_max_input_chars``.
    """
    completion = min(_CONTENT_CREATION_IMPORT_MAX_TOKENS, CONTENT_CREATION_DEFAULT_MAX_TOKENS)
    return min(completion, max(512, LOCAL_LLM_CONTEXT_TOKENS // 3))


def _local_content_creation_max_input_chars() -> int:
    """Document chars that fit local llama.cpp context for content_creation import."""
    completion = _local_import_completion_reserve_tokens()
    overhead = _CONTENT_CREATION_PROMPT_OVERHEAD_TOKENS + 128
    budget_tokens = LOCAL_LLM_CONTEXT_TOKENS - completion - overhead
    if budget_tokens <= 256:
        return 0
    return min(120_000, budget_tokens * 4)


def _trim_message_content_keep_tail(content: str, keep_ratio: float = 0.85) -> str:
    """Drop content from the start so end-biased import excerpts survive.

    Outline and detail prompts place NON-PLAYER CHARACTERS / name-focused windows
    toward the end of the largest message. Chopping the tail would undo
    ``_truncate_preserving_npc_section`` and forward-biased excerpts.
    """
    keep = max(1, int(len(content) * keep_ratio))
    if keep >= len(content):
        return content
    return "[... truncated to fit local model context ...]\n\n" + content[-keep:]


def _fit_messages_to_local_context(
    messages: list[dict[str, Any]],
    max_completion_tokens: int,
) -> tuple[list[dict[str, Any]], int]:
    """Shrink the largest message so prompt + completion fits LOCAL_LLM_CONTEXT_TOKENS.

    Preserves the caller's requested ``max_completion_tokens`` as a ceiling (so chat
    stays at 180). While shrinking an oversized prompt, reserves only the shared
    local import completion share (not the full 6144 request). Shrinks from the
    start of the largest message so NPC sections / forward-biased excerpts at the
    end remain. After the prompt fits, recomputes completion from remaining context.
    """
    slack = 128
    fitted = [dict(m) for m in messages]
    requested = max(1, int(max_completion_tokens))
    # Cap shrink-time reserve to the import share so a 6144 request does not
    # collapse the prompt; chat (180) still reserves only what it asked for.
    reserve = min(requested, _local_import_completion_reserve_tokens())
    reserve = min(reserve, max(1, LOCAL_LLM_CONTEXT_TOKENS - slack - 256))
    max_prompt_tokens = LOCAL_LLM_CONTEXT_TOKENS - reserve - slack

    longest_idx = max(
        range(len(fitted)),
        key=lambda i: len(str(fitted[i].get("content") or "")),
        default=0,
    )
    prompt_tokens = _estimate_prompt_tokens(fitted)
    while prompt_tokens > max_prompt_tokens and len(fitted) > 0:
        content = str(fitted[longest_idx].get("content") or "")
        if len(content) < 400:
            break
        fitted[longest_idx] = {
            **fitted[longest_idx],
            "content": _trim_message_content_keep_tail(content),
        }
        prompt_tokens = _estimate_prompt_tokens(fitted)

    available = LOCAL_LLM_CONTEXT_TOKENS - prompt_tokens - slack
    max_completion = max(1, min(requested, available))
    return fitted, max_completion


def content_creation_zero_budget_error_message() -> str:
    """User-facing reason when ``content_creation_max_input_chars`` returns 0."""
    model_id = CONTENT_CREATION_MODEL_ID
    provider_mode = CONTENT_CREATION_PROVIDER_MODE
    if provider_mode == "local" or model_id == "local":
        return (
            f"Local llama.cpp context is too small for document import "
            f"(LOCAL_LLM_CONTEXT_TOKENS={LOCAL_LLM_CONTEXT_TOKENS}). "
            "Raise llama-server -c / --ctx-size and set LOCAL_LLM_CONTEXT_TOKENS to match, "
            "or switch content_creation to OpenRouter in Admin → LLM."
        )
    return (
        "The content creation model cannot fit document import in the current budget. "
        "Switch content_creation to OpenRouter GPT-OSS 20B or local llama.cpp in Admin → LLM, "
        "or wait if recent usage exhausted a rate limit."
    )


def content_creation_max_input_chars() -> int:
    """Chars of document body that fit the content_creation model's current budget.

    Returns 0 when a strict-online setup cannot fit a minimal import request, or when
    local n_ctx / LOCAL_LLM_CONTEXT_TOKENS is too tight for import.
    For openrouter_with_local_fallback, do NOT pre-truncate using online TPM —
    otherwise online failures fall back to local with a head-chopped document
    and NPC/location sections later in the file are invisible to the local model.
    """
    model_id = CONTENT_CREATION_MODEL_ID
    provider_mode = CONTENT_CREATION_PROVIDER_MODE
    if provider_mode == "local" or model_id == "local":
        return _local_content_creation_max_input_chars()
    if provider_mode == ONLINE_WITH_FALLBACK_MODE:
        # OpenRouter may take the full doc; local fallback truncates in call_local_chat_completions.
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
    fitted_messages, effective_max = _fit_messages_to_local_context(messages, effective_max)
    payload: dict[str, Any] = {
        "model": LOCAL_LLM_MODEL,
        "messages": fitted_messages,
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
            "Start llama.cpp on that port, or set the content_creation / chat model to OpenRouter "
            "in Admin → LLM (provider mode 'openrouter' or 'openrouter_with_local_fallback')."
        ) from ex
    except requests.Timeout as ex:
        raise LLMProviderError(
            f"Local LLM timed out at {LOCAL_LLM_URL} after {timeout}s."
        ) from ex
    except requests.HTTPError as ex:
        detail = ""
        if ex.response is not None:
            detail = (ex.response.text or "").strip()[:500]
        prompt_tokens = _estimate_prompt_tokens(fitted_messages)
        hint = (
            f"Local LLM rejected the request (HTTP {ex.response.status_code if ex.response else '?'}). "
            f"Estimated prompt ~{prompt_tokens} tokens + max_tokens={effective_max} "
            f"exceeds LOCAL_LLM_CONTEXT_TOKENS={LOCAL_LLM_CONTEXT_TOKENS}. "
            "Raise n_ctx when starting llama.cpp, set LOCAL_LLM_CONTEXT_TOKENS to match, "
            "or use OpenRouter for content_creation in Admin → LLM."
        )
        if detail:
            hint = f"{hint} Server said: {detail}"
        raise LLMProviderError(hint) from ex

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



def _openrouter_provider_order() -> list[str]:
    # Prefer live process env so operators can set the key without restarting Python import order.
    raw = os.getenv("OPENROUTER_PROVIDER_ORDER")
    if raw is None:
        raw = OPENROUTER_PROVIDER_ORDER_RAW or ""
    raw = raw.strip()
    if not raw:
        return []
    parts = raw.replace(";", ",").split(",")
    return [part.strip() for part in parts if part.strip()]


def _catalog_provider_order(model_id: str | None) -> list[str] | None:
    """Return a catalog-pinned provider order, or None to use the env default."""
    if not model_id:
        return None
    entry = LLM_MODELS.get(model_id) or {}
    raw = entry.get("provider_order")
    if not isinstance(raw, (list, tuple)):
        return None
    order = [str(part).strip() for part in raw if str(part).strip()]
    return order or None


def _openrouter_allow_fallbacks() -> bool:
    if "OPENROUTER_PROVIDER_ALLOW_FALLBACKS" in os.environ:
        return _parse_allow_fallbacks(os.getenv("OPENROUTER_PROVIDER_ALLOW_FALLBACKS"), default=False)
    return OPENROUTER_PROVIDER_ALLOW_FALLBACKS


def _apply_openrouter_provider_preferences(
    payload: dict[str, Any],
    *,
    model_id: str | None = None,
) -> None:
    catalog_order = _catalog_provider_order(model_id)
    order = catalog_order if catalog_order is not None else _openrouter_provider_order()
    if not order:
        print(
            "[LLM] openrouter provider order empty — OpenRouter will price load-balance "
            "(set OPENROUTER_PROVIDER_ORDER=amazon-bedrock,novita)",
            flush=True,
        )
        return
    allow_fallbacks = _openrouter_allow_fallbacks()
    # Catalog pins (e.g. CoreWeave-only entries) must not spill to other providers.
    if catalog_order is not None:
        allow_fallbacks = False
    provider: dict[str, Any] = {
        "order": order,
        "allow_fallbacks": allow_fallbacks,
    }
    # When fallbacks are disabled, also pin `only` so other endpoints cannot be selected.
    if not allow_fallbacks:
        provider["only"] = list(order)
    payload["provider"] = provider
    print(
        f"[LLM] openrouter provider preferences order={order} "
        f"allow_fallbacks={allow_fallbacks} only={provider.get('only')} "
        f"model_id={model_id or '(none)'}",
        flush=True,
    )


def _openrouter_timeout_seconds(
    *,
    purpose: str = DEFAULT_PURPOSE,
    model_id: str | None = None,
    model_name: str | None = None,
    timeout: int | None = None,
) -> int:
    if timeout is not None:
        return max(1, int(timeout))
    if purpose == PURPOSE_CONTENT_CREATION:
        return max(CONTENT_CREATION_OPENROUTER_TIMEOUT_SECONDS, OPENROUTER_TIMEOUT_SECONDS)
    if _is_reasoning_model(model_id, model_name):
        return max(OPENROUTER_TIMEOUT_SECONDS, 30)
    return OPENROUTER_TIMEOUT_SECONDS


_groq_timeout_seconds = _openrouter_timeout_seconds


def call_openrouter_chat_completions(
    messages: list[dict[str, Any]],
    model_name: str,
    *,
    model_id: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: int | None = None,
    purpose: str = DEFAULT_PURPOSE,
) -> tuple[str, dict[str, int], requests.Response]:
    headers = _openrouter_auth_headers()
    effective_max = _effective_max_tokens(model_id, max_tokens, model_name=model_name, purpose=purpose)
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.80 if temperature is None else temperature,
        "max_tokens": effective_max,
    }
    _apply_reasoning_options(payload, model_id=model_id or "", model_name=model_name)
    _apply_openrouter_provider_preferences(payload, model_id=model_id)

    response = requests.post(
        OPENROUTER_BASE_URL,
        headers=headers,
        json=payload,
        timeout=_openrouter_timeout_seconds(
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
    routed_provider = data.get("provider") or choice.get("provider")
    if routed_provider:
        print(
            f"[LLM] openrouter routed provider={routed_provider!r} model={model_name}",
            flush=True,
        )
    usage = data.get("usage") or {}

    prompt_tokens = int(usage.get("prompt_tokens") or _estimate_prompt_tokens(messages))
    completion_tokens = int(usage.get("completion_tokens") or max(1, len(content) // 4))
    total_tokens = int(usage.get("total_tokens") or prompt_tokens + completion_tokens)

    if not content:
        print(
            f"[LLM] openrouter empty content model={model_name} finish={choice.get('finish_reason')!r} "
            f"completion_tokens={completion_tokens} max_tokens={effective_max} "
            f"reasoning_effort={payload.get('reasoning_effort')!r}",
            flush=True,
        )

    return content, {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }, response


call_groq_chat_completions = call_openrouter_chat_completions



def _effective_provider_mode(model_id: str, provider_mode: str) -> str:
    if model_id == "local" or LLM_MODELS.get(model_id, {}).get("provider") == "local":
        return "local"
    return provider_mode


def _set_active(provider: str, model_id: str, purpose: str = DEFAULT_PURPOSE) -> None:
    LAST_ACTIVE_PROVIDER[purpose] = provider
    LAST_ACTIVE_MODEL_ID[purpose] = model_id
    # Request-local timing attribution (ignore process-wide races / async memory).
    try:
        import request_timing

        selected = _get_purpose_model_id(purpose)
        request_timing.note_request_llm_model(
            active_model_id=model_id,
            selected_model_id=selected,
            purpose=purpose,
        )
    except Exception:
        pass


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
        selected_model_id=selected_model_id,
        attribute_cost_to_selected=True,
    )

    return content


def _fallback_after_failure(
    messages: list[dict[str, Any]],
    *,
    reason: str,
    selected_model_id: str,
    temperature: float | None,
    max_tokens: int | None,
    error: str | None = None,
    purpose: str = DEFAULT_PURPOSE,
    online_unavailable: bool = False,
    respect_online_disabled: bool = True,
    **kwargs: Any,
) -> str:
    """Walk Admin-configured fallback models, then local when allowed.

    ``online_unavailable`` skips other OpenRouter catalog entries (credit threshold,
    missing key, or OpenRouter globally disabled) and only tries local.
    ``respect_online_disabled`` when false still tries other OpenRouter models after a
    per-model outage (timeout/5xx), so CoreWeave → DeepInfra can recover.
    """
    purpose = _normalize_purpose(purpose)
    chain = _normalize_fallback_model_ids(
        _get_purpose_fallback_model_ids(purpose),
        purpose=purpose,
        primary_model_id=selected_model_id,
    )
    provider_mode = _get_purpose_provider_mode(purpose)
    if provider_mode == ONLINE_WITH_FALLBACK_MODE and "local" not in chain:
        chain = [*chain, "local"]

    last_error = error
    for fallback_id in chain:
        entry = LLM_MODELS.get(fallback_id) or {}
        if fallback_id == "local" or entry.get("provider") == "local":
            return _fallback_to_local(
                messages,
                reason=reason,
                selected_model_id=selected_model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                error=last_error,
                purpose=purpose,
                **kwargs,
            )

        if online_unavailable or not openrouter_is_configured():
            continue
        if (
            respect_online_disabled
            and ONLINE_DISABLED_UNTIL is not None
            and _utc_now() < ONLINE_DISABLED_UNTIL
        ):
            continue

        model_name = str(entry.get("model") or "")
        if not model_name:
            continue

        print(
            f"[LLM] trying fallback model={fallback_id} ({reason}) purpose={purpose}",
            flush=True,
        )
        try:
            content, usage, response = call_openrouter_chat_completions(
                messages,
                model_name,
                model_id=fallback_id,
                temperature=temperature,
                max_tokens=max_tokens,
                purpose=purpose,
            )
        except requests.RequestException as ex:
            last_error = str(ex)
            record_llm_usage(
                provider="openrouter",
                model_id=fallback_id,
                model=model_name,
                prompt_tokens=_estimate_prompt_tokens(messages),
                completion_tokens=0,
                total_tokens=_estimate_prompt_tokens(messages),
                success=False,
                fallback=True,
                error=last_error,
                purpose=purpose,
                selected_model_id=selected_model_id,
            )
            continue

        if response.status_code >= 400 or not content:
            last_error = (response.text or f"HTTP {response.status_code}")[:500] or "empty content"
            record_llm_usage(
                provider="openrouter",
                model_id=fallback_id,
                model=model_name,
                prompt_tokens=_estimate_prompt_tokens(messages),
                completion_tokens=0,
                total_tokens=_estimate_prompt_tokens(messages),
                success=False,
                fallback=True,
                error=last_error,
                purpose=purpose,
                selected_model_id=selected_model_id,
            )
            continue

        LAST_FALLBACK_REASON[purpose] = reason
        LAST_LLM_ERROR[purpose] = last_error
        _set_active("openrouter", fallback_id, purpose)
        record_llm_usage(
            provider="openrouter",
            model_id=fallback_id,
            model=model_name,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            success=True,
            fallback=True,
            error=last_error,
            purpose=purpose,
            selected_model_id=selected_model_id,
            # Bill the fallback catalog id that actually served tokens.
            attribute_cost_to_selected=False,
        )
        return content

    LAST_LLM_ERROR[purpose] = last_error or reason
    raise LLMProviderError(
        last_error or f"All LLM fallbacks failed after primary model error ({reason})"
    )


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
            selected_model_id=selected_model_id,
        )
        return content

    if not openrouter_is_configured():
        if not _allows_local_fallback(provider_mode):
            LAST_LLM_ERROR[purpose] = "OpenRouter API key is not configured"
            raise LLMProviderError(LAST_LLM_ERROR[purpose])
        return _fallback_after_failure(
            messages,
            reason="openrouter_missing_api_key",
            selected_model_id=selected_model_id,
            temperature=temperature,
            max_tokens=effective_max_tokens,
            error="OpenRouter API key is not configured",
            purpose=purpose,
            online_unavailable=True,
            **kwargs,
        )

    if ONLINE_DISABLED_UNTIL is not None and _utc_now() < ONLINE_DISABLED_UNTIL:
        disabled_msg = f"OpenRouter temporarily disabled until {_utc_iso(ONLINE_DISABLED_UNTIL)}"
        if not _allows_local_fallback(provider_mode):
            LAST_LLM_ERROR[purpose] = disabled_msg
            raise LLMProviderError(LAST_LLM_ERROR[purpose])
        return _fallback_after_failure(
            messages,
            reason="openrouter_disabled_until",
            selected_model_id=selected_model_id,
            temperature=temperature,
            max_tokens=effective_max_tokens,
            error=disabled_msg,
            purpose=purpose,
            online_unavailable=True,
            **kwargs,
        )

    limit_reason = _would_exceed_online_limits(
        selected_model_id,
        estimated_total_tokens,
        purpose=purpose,
    )
    if limit_reason:
        # Credit threshold is account-wide; skip other OpenRouter models and go to local.
        # Per-model rate preflight can still try another online fallback first.
        is_credit = "credit usage" in limit_reason.casefold()
        fallback_reason = "openrouter_credit_threshold" if is_credit else limit_reason
        if not _allows_local_fallback(provider_mode):
            LAST_LLM_ERROR[purpose] = limit_reason
            raise LLMProviderError(limit_reason)
        return _fallback_after_failure(
            messages,
            reason=fallback_reason,
            selected_model_id=selected_model_id,
            temperature=temperature,
            max_tokens=effective_max_tokens,
            error=limit_reason,
            purpose=purpose,
            online_unavailable=is_credit,
            **kwargs,
        )

    model_name = selected_model["model"]
    online_timeout = _openrouter_timeout_seconds(
        purpose=purpose,
        model_id=selected_model_id,
        model_name=model_name,
    )

    try:
        content, usage, response = call_openrouter_chat_completions(
            messages,
            model_name,
            model_id=selected_model_id,
            temperature=temperature,
            max_tokens=effective_max_tokens,
            timeout=online_timeout,
            purpose=purpose,
        )
    except requests.Timeout as ex:
        # Per-model timeout: try configured OpenRouter fallbacks. Do not circuit-break
        # the whole account — that would skip DeepInfra/etc. on later turns.
        if not _allows_local_fallback(provider_mode):
            LAST_LLM_ERROR[purpose] = str(ex)
            record_llm_usage(
                provider="openrouter",
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
            raise LLMProviderError(f"OpenRouter request timed out: {ex}") from ex
        return _fallback_after_failure(
            messages,
            reason="openrouter_timeout",
            selected_model_id=selected_model_id,
            temperature=temperature,
            max_tokens=effective_max_tokens,
            error=str(ex),
            purpose=purpose,
            respect_online_disabled=False,
            **kwargs,
        )
    except requests.RequestException as ex:
        if not _allows_local_fallback(provider_mode):
            LAST_LLM_ERROR[purpose] = str(ex)
            record_llm_usage(
                provider="openrouter",
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
            raise LLMProviderError(f"OpenRouter request failed: {ex}") from ex
        return _fallback_after_failure(
            messages,
            reason="openrouter_exception",
            selected_model_id=selected_model_id,
            temperature=temperature,
            max_tokens=effective_max_tokens,
            error=str(ex),
            purpose=purpose,
            respect_online_disabled=False,
            **kwargs,
        )

    # 402 = payment / credits exhausted on OpenRouter.
    if response.status_code in (402, 429):
        retry_seconds = _parse_retry_after_seconds(response) or 300
        reason = "openrouter_credits_exhausted" if response.status_code == 402 else "openrouter_rate_limited"
        _set_online_disabled(retry_seconds, reason)
        error_text = response.text[:500]
        record_llm_usage(
            provider="openrouter",
            model_id=selected_model_id,
            model=model_name,
            prompt_tokens=estimated_prompt_tokens,
            completion_tokens=0,
            total_tokens=estimated_prompt_tokens,
            success=False,
            fallback=_allows_local_fallback(provider_mode),
            error=error_text,
            purpose=purpose,
        )
        if not _allows_local_fallback(provider_mode):
            msg = "OpenRouter credits exhausted" if response.status_code == 402 else "OpenRouter rate limited"
            LAST_LLM_ERROR[purpose] = msg
            raise LLMProviderError(msg)
        return _fallback_after_failure(
            messages,
            reason=reason,
            selected_model_id=selected_model_id,
            temperature=temperature,
            max_tokens=effective_max_tokens,
            error=error_text,
            purpose=purpose,
            online_unavailable=True,
            **kwargs,
        )

    if response.status_code in (500, 502, 503, 504):
        # Transient per-endpoint errors: try other catalog fallbacks. Do not disable
        # OpenRouter account-wide or later turns skip DeepInfra/etc.
        error_text = response.text[:500]
        record_llm_usage(
            provider="openrouter",
            model_id=selected_model_id,
            model=model_name,
            prompt_tokens=estimated_prompt_tokens,
            completion_tokens=0,
            total_tokens=estimated_prompt_tokens,
            success=False,
            fallback=_allows_local_fallback(provider_mode),
            error=f"HTTP {response.status_code}: {error_text}",
            purpose=purpose,
        )
        if not _allows_local_fallback(provider_mode):
            LAST_LLM_ERROR[purpose] = f"OpenRouter HTTP {response.status_code}"
            raise LLMProviderError(LAST_LLM_ERROR[purpose])
        return _fallback_after_failure(
            messages,
            reason="openrouter_http_error",
            selected_model_id=selected_model_id,
            temperature=temperature,
            max_tokens=effective_max_tokens,
            error=error_text,
            purpose=purpose,
            respect_online_disabled=False,
            **kwargs,
        )

    if response.status_code >= 400:
        error_text = response.text[:500]
        model_missing = (
            "model_not_found" in error_text
            or "does not exist" in error_text
            or "do not have access" in error_text
            or "No endpoints found" in error_text
        )
        reason = "openrouter_model_not_found" if model_missing else "openrouter_http_error"
        friendly = (
            f"OpenRouter model '{model_name}' is unavailable on this API key/plan. "
            f"Pick openrouter-gpt-oss-20b (or local) in Admin → LLM."
            if model_missing
            else f"HTTP {response.status_code}: {error_text}"
        )
        record_llm_usage(
            provider="openrouter",
            model_id=selected_model_id,
            model=model_name,
            prompt_tokens=estimated_prompt_tokens,
            completion_tokens=0,
            total_tokens=estimated_prompt_tokens,
            success=False,
            fallback=_allows_local_fallback(provider_mode),
            error=friendly,
            purpose=purpose,
        )
        if not _allows_local_fallback(provider_mode):
            LAST_LLM_ERROR[purpose] = friendly
            raise LLMProviderError(friendly)
        return _fallback_after_failure(
            messages,
            reason=reason,
            selected_model_id=selected_model_id,
            temperature=temperature,
            max_tokens=effective_max_tokens,
            error=friendly,
            purpose=purpose,
            respect_online_disabled=False,
            **kwargs,
        )

    _set_active("openrouter", selected_model_id, purpose)
    LAST_FALLBACK_REASON[purpose] = None
    LAST_LLM_ERROR[purpose] = None

    record_llm_usage(
        provider="openrouter",
        model_id=selected_model_id,
        model=model_name,
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        total_tokens=usage["total_tokens"],
        success=True,
        fallback=False,
        purpose=purpose,
        selected_model_id=selected_model_id,
    )

    return content


def select_llm_model(
    model_id: str,
    provider_mode: str | None = None,
    purpose: str = DEFAULT_PURPOSE,
    fallback_model_ids: Any = None,
) -> dict[str, Any]:
    global CURRENT_LLM_MODEL_ID, LLM_PROVIDER_MODE, CHAT_FALLBACK_MODEL_IDS
    global CONTENT_CREATION_MODEL_ID, CONTENT_CREATION_PROVIDER_MODE
    global CONTENT_CREATION_FALLBACK_MODEL_IDS

    purpose = _normalize_purpose(purpose)

    if model_id not in LLM_MODELS:
        remapped = _alias_model_id(model_id)
        if remapped in LLM_MODELS:
            print(f"[LLM] select remapped retired model {model_id!r} -> {remapped!r}", flush=True)
            model_id = remapped
        else:
            raise ValueError(f"Unknown model_id: {model_id}")

    if not model_supports_purpose(model_id, purpose):
        raise ValueError(f"Model {model_id} is not available for purpose {purpose}")

    if provider_mode is not None:
        normalized_mode = _normalize_provider_mode(provider_mode, default="")
        if normalized_mode not in VALID_PROVIDER_MODES:
            raise ValueError(f"Invalid provider_mode: {provider_mode}")
        provider_mode = normalized_mode

    update_fallbacks = fallback_model_ids is not None
    normalized_fallbacks: list[str] | None = None
    if update_fallbacks:
        normalized_fallbacks = _normalize_fallback_model_ids(
            fallback_model_ids,
            purpose=purpose,
            primary_model_id=model_id,
        )

    with _STATE_LOCK:
        if purpose == PURPOSE_CONTENT_CREATION:
            CONTENT_CREATION_MODEL_ID = model_id
            if provider_mode is not None:
                CONTENT_CREATION_PROVIDER_MODE = provider_mode
            if model_id == "local":
                CONTENT_CREATION_PROVIDER_MODE = "local"
            if update_fallbacks:
                CONTENT_CREATION_FALLBACK_MODEL_IDS = list(normalized_fallbacks or [])
            else:
                CONTENT_CREATION_FALLBACK_MODEL_IDS = _normalize_fallback_model_ids(
                    CONTENT_CREATION_FALLBACK_MODEL_IDS,
                    purpose=purpose,
                    primary_model_id=CONTENT_CREATION_MODEL_ID,
                )
            result_model_id = CONTENT_CREATION_MODEL_ID
            result_provider_mode = CONTENT_CREATION_PROVIDER_MODE
            result_fallbacks = list(CONTENT_CREATION_FALLBACK_MODEL_IDS)
        else:
            CURRENT_LLM_MODEL_ID = model_id
            if provider_mode is not None:
                LLM_PROVIDER_MODE = provider_mode
            if model_id == "local":
                LLM_PROVIDER_MODE = "local"
            if update_fallbacks:
                CHAT_FALLBACK_MODEL_IDS = list(normalized_fallbacks or [])
            else:
                CHAT_FALLBACK_MODEL_IDS = _normalize_fallback_model_ids(
                    CHAT_FALLBACK_MODEL_IDS,
                    purpose=purpose,
                    primary_model_id=CURRENT_LLM_MODEL_ID,
                )
            result_model_id = CURRENT_LLM_MODEL_ID
            result_provider_mode = LLM_PROVIDER_MODE
            result_fallbacks = list(CHAT_FALLBACK_MODEL_IDS)

        LAST_FALLBACK_REASON[purpose] = None
        LAST_LLM_ERROR[purpose] = None

        _save_runtime_config()

        return {
            "purpose": purpose,
            "model_id": result_model_id,
            "provider_mode": result_provider_mode,
            "fallback_model_ids": result_fallbacks,
            "selected_model": _public_model_entry(result_model_id),
        }


def reset_usage_tracking() -> None:
    with _USAGE_LOCK:
        _save_usage_records([])


def _empty_token_bucket() -> dict[str, Any]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "requests": 0,
        "fallback_attributed_requests": 0,
        "estimated_input_cost_usd": 0.0,
        "estimated_output_cost_usd": 0.0,
        "estimated_total_cost_usd": 0.0,
    }


def _accumulate_usage_bucket(bucket: dict[str, Any], record: dict[str, Any], cost_model_id: str) -> None:
    prompt = int(record.get("prompt_tokens") or 0)
    completion = int(record.get("completion_tokens") or 0)
    total = int(record.get("total_tokens") or (prompt + completion))
    costs = _estimate_cost_usd(cost_model_id, prompt, completion)
    bucket["prompt_tokens"] += prompt
    bucket["completion_tokens"] += completion
    bucket["total_tokens"] += total
    bucket["requests"] += 1
    if record.get("fallback") and (record.get("cost_model_id") or "") not in {"", "local"}:
        if record.get("model_id") == "local" or record.get("provider") == "local":
            bucket["fallback_attributed_requests"] += 1
    bucket["estimated_input_cost_usd"] = round(
        bucket["estimated_input_cost_usd"] + costs["estimated_input_cost_usd"], 6
    )
    bucket["estimated_output_cost_usd"] = round(
        bucket["estimated_output_cost_usd"] + costs["estimated_output_cost_usd"], 6
    )
    bucket["estimated_total_cost_usd"] = round(
        bucket["estimated_total_cost_usd"] + costs["estimated_total_cost_usd"], 6
    )


def get_llm_usage_cost_report(*, hours: int = 24) -> dict[str, Any]:
    """Alpha cost/token report for Admin charts.

    Successful rows only. Local fallbacks with ``cost_model_id`` set to the selected
    online model are billed against that model for estimation.
    """
    hours = max(1, min(int(hours or 24), USAGE_RETENTION_HOURS))
    cutoff = _utc_now() - timedelta(hours=hours)

    by_model: dict[str, dict[str, Any]] = {}
    by_user: dict[str, dict[str, Any]] = {}
    series_map: dict[str, dict[str, dict[str, Any]]] = {}
    totals = _empty_token_bucket()

    with _USAGE_LOCK:
        records = list(_load_usage_records())

    for record in records:
        if not record.get("success"):
            continue
        ts = _parse_utc_iso(record.get("timestamp"))
        if ts is None or ts < cutoff:
            continue

        cost_model_id = str(
            record.get("cost_model_id") or record.get("selected_model_id") or record.get("model_id") or "local"
        ).strip() or "local"
        owner = str(record.get("owner_user_id") or "").strip() or "(unknown)"
        purpose = str(record.get("purpose") or DEFAULT_PURPOSE)

        model_bucket = by_model.get(cost_model_id)
        if model_bucket is None:
            prices = _pricing_for_model(cost_model_id)
            public = _public_model_entry(cost_model_id) if cost_model_id in LLM_MODELS else {
                "id": cost_model_id,
                "label": cost_model_id,
            }
            model_bucket = {
                **_empty_token_bucket(),
                "model_id": cost_model_id,
                "label": public.get("label") or cost_model_id,
                "input_price_per_mtok": prices["input"],
                "output_price_per_mtok": prices["output"],
                "by_purpose": {},
            }
            by_model[cost_model_id] = model_bucket
        _accumulate_usage_bucket(model_bucket, record, cost_model_id)
        purpose_bucket = model_bucket["by_purpose"].setdefault(purpose, _empty_token_bucket())
        _accumulate_usage_bucket(purpose_bucket, record, cost_model_id)

        user_bucket = by_user.get(owner)
        if user_bucket is None:
            user_bucket = {
                **_empty_token_bucket(),
                "owner_user_id": owner,
                "by_model": {},
            }
            by_user[owner] = user_bucket
        _accumulate_usage_bucket(user_bucket, record, cost_model_id)
        user_model = user_bucket["by_model"].setdefault(
            cost_model_id,
            {
                **_empty_token_bucket(),
                "model_id": cost_model_id,
                "label": model_bucket["label"],
            },
        )
        _accumulate_usage_bucket(user_model, record, cost_model_id)

        _accumulate_usage_bucket(totals, record, cost_model_id)

        hour_key = ts.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0).isoformat().replace(
            "+00:00", "Z"
        )
        hour_models = series_map.setdefault(hour_key, {})
        hour_bucket = hour_models.setdefault(
            cost_model_id,
            {
                **_empty_token_bucket(),
                "model_id": cost_model_id,
                "label": model_bucket["label"],
            },
        )
        _accumulate_usage_bucket(hour_bucket, record, cost_model_id)

    series = []
    for hour in sorted(series_map.keys()):
        models = sorted(series_map[hour].values(), key=lambda row: row["model_id"])
        series.append(
            {
                "hour": hour,
                "models": models,
                "prompt_tokens": sum(m["prompt_tokens"] for m in models),
                "completion_tokens": sum(m["completion_tokens"] for m in models),
                "estimated_total_cost_usd": round(sum(m["estimated_total_cost_usd"] for m in models), 6),
            }
        )

    return {
        "window_hours": hours,
        "retention_hours": USAGE_RETENTION_HOURS,
        "pricing_note": (
            "Alpha estimates using OpenRouter list prices for openai/gpt-oss-20b. "
            "When provider mode is openrouter_with_local_fallback, successful local fallbacks "
            "are attributed to the selected online model for cost estimation."
        ),
        "pricing": {
            model_id: {"input_per_mtok": prices["input"], "output_per_mtok": prices["output"]}
            for model_id, prices in ONLINE_ON_DEMAND_PRICES_USD.items()
        },
        "by_model": sorted(
            by_model.values(),
            key=lambda row: row["estimated_total_cost_usd"],
            reverse=True,
        ),
        "by_user": sorted(
            (
                {
                    **user,
                    "by_model": sorted(
                        user["by_model"].values(),
                        key=lambda row: row["estimated_total_cost_usd"],
                        reverse=True,
                    ),
                }
                for user in by_user.values()
            ),
            key=lambda row: row["estimated_total_cost_usd"],
            reverse=True,
        ),
        "series": series,
        "totals": totals,
    }


def _remaining_estimate(model_id: str) -> dict[str, int | None]:
    model_entry = LLM_MODELS.get(model_id, {})
    limits = model_entry.get("limits") or {}

    rpm = limits.get("rpm")
    rpd = limits.get("rpd")
    tpm = limits.get("tpm")
    tpd = limits.get("tpd")

    # Remaining quota reflects actual account usage across all purposes,
    # since online providers enforce limits per model regardless of caller purpose.
    usage = get_usage_summary(model_id)

    return {
        "requests_this_minute": None if rpm is None else max(0, rpm - usage["requests_last_minute"]),
        "tokens_this_minute": None if tpm is None else max(0, tpm - usage["tokens_last_minute"]),
        "requests_24h": None if rpd is None else max(0, rpd - usage["requests_last_24h"]),
        "tokens_24h": None if tpd is None else max(0, tpd - usage["tokens_last_24h"]),
    }



def _credits_status_payload(*, provider_mode: str) -> dict[str, Any]:
    snapshot = fetch_openrouter_credit_snapshot()
    threshold = max(0.0, min(1.0, OPENROUTER_CREDIT_FALLBACK_RATIO))
    fallback_applies = _allows_local_fallback(provider_mode)
    budget_configured = OPENROUTER_CREDIT_BUDGET_USD > 0

    if not snapshot:
        return {
            "source": None,
            "total_credits": None,
            "total_usage": None,
            "usage_ratio": None,
            "limit": None,
            "limit_remaining": None,
            "budget_usd": OPENROUTER_CREDIT_BUDGET_USD if budget_configured else None,
            "budget_configured": budget_configured,
            "monitoring_available": False,
            "fallback_threshold": threshold,
            "fallback_applies": fallback_applies,
            "above_threshold": None,
            "credit_fallback_active": False,
        }
    ratio = snapshot.get("usage_ratio")
    try:
        ratio_f = float(ratio) if ratio is not None else None
    except (TypeError, ValueError):
        ratio_f = None
    above = ratio_f is not None and ratio_f >= threshold
    return {
        **snapshot,
        "budget_configured": budget_configured,
        "monitoring_available": True,
        "fallback_threshold": threshold,
        "fallback_applies": fallback_applies,
        "above_threshold": above,
        "credit_fallback_active": above and fallback_applies,
    }


def _purpose_status(purpose: str) -> dict[str, Any]:
    selected_model_id = _get_purpose_model_id(purpose)
    if selected_model_id not in LLM_MODELS:
        selected_model_id = "local"

    provider_mode = _get_purpose_provider_mode(purpose)
    selected_model = _public_model_entry(selected_model_id)
    usage_model_id = selected_model_id if selected_model.get("provider") == "openrouter" else None

    if usage_model_id:
        # "usage" is purpose-specific (how much this purpose consumed);
        # "remaining_estimate" is optional catalog-limit remaining for the model.
        usage = get_usage_summary(usage_model_id, purpose=purpose)
        remaining = _remaining_estimate(usage_model_id)
    else:
        usage = {
            "requests_last_minute": usage_requests_last_60s(provider="openrouter", purpose=purpose),
            "tokens_last_minute": usage_tokens_last_60s(provider="openrouter", purpose=purpose),
            "requests_last_24h": usage_requests_last_24h(provider="openrouter", purpose=purpose),
            "tokens_last_24h": usage_tokens_last_24h(provider="openrouter", purpose=purpose),
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
        "fallback_model_ids": _get_purpose_fallback_model_ids(purpose),
        "selected_model": selected_model,
        "active_provider": LAST_ACTIVE_PROVIDER.get(purpose, "local"),
        "active_model_id": LAST_ACTIVE_MODEL_ID.get(purpose, "local"),
        "online_available": openrouter_is_available(),
        "online_configured": openrouter_is_configured(),
        "online_disabled_until": _utc_iso(ONLINE_DISABLED_UNTIL),
        "last_fallback_reason": LAST_FALLBACK_REASON.get(purpose),
        "last_error": LAST_LLM_ERROR.get(purpose),
        "usage": usage,
        "local_usage": local_usage,
        "remaining_estimate": remaining,
        "credits": _credits_status_payload(provider_mode=provider_mode),
    }


def _usage_by_purpose_summary(purpose: str) -> dict[str, Any]:
    # Unlike usage_requests_last_60s/etc (which are online-only for rate
    # limiting), this reports ALL provider usage (local + openrouter) for a purpose.
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
        "openrouter": {
            "provider_order": _openrouter_provider_order(),
            "provider_allow_fallbacks": _openrouter_allow_fallbacks(),
        },
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
                "id": "openrouter",
                "label": "openrouter — OpenRouter only (fails if OpenRouter is unavailable)",
            },
            {
                "id": "openrouter_with_local_fallback",
                "label": (
                    "openrouter_with_local_fallback — try primary + fallback models "
                    "(local last by default; also at 80% credit usage)"
                ),
            },
        ],
        # Full catalog for back-compat; each entry carries `purposes`, and every
        # purpose also gets its own pre-filtered list so clients need not re-derive it.
        "models": _public_model_list(),
        "purposes": {
            PURPOSE_CHAT: {
                "current_model_id": CURRENT_LLM_MODEL_ID,
                "provider_mode": LLM_PROVIDER_MODE,
                "fallback_model_ids": list(CHAT_FALLBACK_MODEL_IDS),
                "models": _public_model_list(PURPOSE_CHAT),
            },
            PURPOSE_CONTENT_CREATION: {
                "current_model_id": CONTENT_CREATION_MODEL_ID,
                "provider_mode": CONTENT_CREATION_PROVIDER_MODE,
                "fallback_model_ids": list(CONTENT_CREATION_FALLBACK_MODEL_IDS),
                "models": _public_model_list(PURPOSE_CONTENT_CREATION),
            },
        },
    }


def init_llm_router() -> None:
    _load_secrets_file()
    _load_runtime_config()

    # Refresh local model metadata after env/secrets load.
    LLM_MODELS["local"]["model"] = LOCAL_LLM_MODEL
    LLM_MODELS["local"]["url"] = LOCAL_LLM_URL

    provider_order = _openrouter_provider_order()
    print(
        f"[LLM] router ready chat_model={CURRENT_LLM_MODEL_ID} chat_mode={LLM_PROVIDER_MODE} "
        f"content_creation_model={CONTENT_CREATION_MODEL_ID} content_creation_mode={CONTENT_CREATION_PROVIDER_MODE} "
        f"openrouter_configured={openrouter_is_configured()} "
        f"provider_order={provider_order or '(none)'} "
        f"provider_allow_fallbacks={_openrouter_allow_fallbacks()}",
        flush=True,
    )


init_llm_router()
