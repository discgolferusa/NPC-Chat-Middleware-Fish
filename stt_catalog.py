"""Admin-editable STT model catalog and transcription attempt order.

Mirrors llm_catalog.py patterns: ``stt_catalog.json`` holds entries + order;
``stt_runtime_config.json`` holds the active primary / fallbacks / provider_mode.
"""

from __future__ import annotations

import json
import os
import re
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stt_capabilities import capabilities_public_dict

PURPOSE_TRANSCRIPTION = "transcription"
VALID_PURPOSES = frozenset({PURPOSE_TRANSCRIPTION})

PROVIDER_MODE_LOCAL = "local"
PROVIDER_MODE_OPENROUTER = "openrouter"
PROVIDER_MODE_OPENROUTER_WITH_LOCAL = "openrouter_with_local_fallback"
VALID_PROVIDER_MODES = frozenset(
    {
        PROVIDER_MODE_LOCAL,
        PROVIDER_MODE_OPENROUTER,
        PROVIDER_MODE_OPENROUTER_WITH_LOCAL,
    }
)

LOCAL_WHISPER_ID = "local-whisper-lite"

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_CATALOG_LOCK = threading.Lock()

STT_CATALOG_FILE: Path | None = None
STT_RUNTIME_FILE: Path | None = None

CATALOG_ENTRIES: dict[str, dict[str, Any]] = {}
MODEL_ORDER: dict[str, list[str]] = {PURPOSE_TRANSCRIPTION: []}

CURRENT_MODEL_ID = ""
FALLBACK_MODEL_IDS: list[str] = []
PROVIDER_MODE = PROVIDER_MODE_OPENROUTER_WITH_LOCAL


def _utc_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _slug(value: str) -> str:
    text = _SLUG_RE.sub("-", (value or "").strip().lower()).strip("-")
    return text or "model"


def make_openrouter_entry_id(openrouter_model: str) -> str:
    return f"openrouter-{_slug(openrouter_model.replace('/', '-'))}"


def _title_from_model(openrouter_model: str) -> str:
    leaf = (openrouter_model or "").split("/")[-1].replace("-", " ").replace("_", " ")
    return " ".join(part.capitalize() for part in leaf.split()) or openrouter_model


def seed_entries() -> dict[str, dict[str, Any]]:
    seeds = [
        {
            "id": "openrouter-assemblyai-universal-3-5-pro",
            "builtin": True,
            "enabled": True,
            "label": "AssemblyAI Universal-3.5 Pro",
            "provider": "openrouter",
            "provider_label": "OpenRouter / AssemblyAI",
            "model": "assemblyai/universal-3-5-pro",
            "online": True,
        },
        {
            "id": "openrouter-deepgram-nova-3",
            "builtin": True,
            "enabled": True,
            "label": "Deepgram Nova-3",
            "provider": "openrouter",
            "provider_label": "OpenRouter / Deepgram",
            "model": "deepgram/nova-3",
            "online": True,
        },
        {
            "id": "openrouter-openai-whisper-large-v3-turbo",
            "builtin": True,
            "enabled": True,
            "label": "Whisper Large V3 Turbo",
            "provider": "openrouter",
            "provider_label": "OpenRouter / OpenAI",
            "model": "openai/whisper-large-v3-turbo",
            "online": True,
        },
        {
            "id": "openrouter-openai-gpt-transcribe",
            "builtin": True,
            "enabled": True,
            "label": "GPT Transcribe",
            "provider": "openrouter",
            "provider_label": "OpenRouter / OpenAI",
            "model": "openai/gpt-transcribe",
            "online": True,
        },
        {
            "id": LOCAL_WHISPER_ID,
            "builtin": True,
            "enabled": True,
            "label": "Local Whisper Lite (base.en)",
            "provider": "local",
            "provider_label": "Local",
            "model": "base.en",
            "online": False,
            "device": "cpu",
            "compute_type": "int8",
        },
    ]
    return {entry["id"]: dict(entry) for entry in seeds}


def default_order(entries: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    preferred = [
        "openrouter-assemblyai-universal-3-5-pro",
        "openrouter-deepgram-nova-3",
        "openrouter-openai-whisper-large-v3-turbo",
        "openrouter-openai-gpt-transcribe",
        LOCAL_WHISPER_ID,
    ]
    chain = [mid for mid in preferred if mid in entries and entries[mid].get("enabled", True)]
    for mid, entry in entries.items():
        if mid not in chain and entry.get("enabled", True):
            chain.append(mid)
    return {PURPOSE_TRANSCRIPTION: chain}


def configure(*, catalog_file: Path, runtime_file: Path) -> None:
    global STT_CATALOG_FILE, STT_RUNTIME_FILE
    STT_CATALOG_FILE = catalog_file
    STT_RUNTIME_FILE = runtime_file


def _normalize_provider_mode(raw: Any) -> str:
    mode = str(raw or "").strip().lower()
    aliases = {
        "groq": PROVIDER_MODE_OPENROUTER,
        "cloud": PROVIDER_MODE_OPENROUTER,
        "online": PROVIDER_MODE_OPENROUTER,
        "openrouter_with_local": PROVIDER_MODE_OPENROUTER_WITH_LOCAL,
        "with_local_fallback": PROVIDER_MODE_OPENROUTER_WITH_LOCAL,
    }
    mode = aliases.get(mode, mode)
    if mode in VALID_PROVIDER_MODES:
        return mode
    return PROVIDER_MODE_OPENROUTER_WITH_LOCAL


def _normalize_entry(raw: dict[str, Any], *, entry_id: str | None = None) -> dict[str, Any]:
    mid = str(entry_id or raw.get("id") or "").strip()
    provider = str(raw.get("provider") or "").strip().lower()
    model = str(raw.get("model") or "").strip()

    if mid == LOCAL_WHISPER_ID or provider == "local":
        return {
            "id": LOCAL_WHISPER_ID,
            "builtin": True,
            "enabled": bool(raw.get("enabled", True)),
            "label": str(raw.get("label") or "Local Whisper Lite (base.en)"),
            "provider": "local",
            "provider_label": str(raw.get("provider_label") or "Local"),
            "model": model or "base.en",
            "online": False,
            "device": str(raw.get("device") or "cpu"),
            "compute_type": str(raw.get("compute_type") or "int8"),
        }

    if not model:
        raise ValueError("OpenRouter model is required")
    if not mid:
        mid = make_openrouter_entry_id(model)
    label = str(raw.get("label") or _title_from_model(model)).strip()
    return {
        "id": mid,
        "builtin": bool(raw.get("builtin", False)),
        "enabled": bool(raw.get("enabled", True)),
        "label": label,
        "provider": "openrouter",
        "provider_label": str(raw.get("provider_label") or "OpenRouter").strip(),
        "model": model,
        "online": True,
    }


def _normalize_order(
    raw: Any,
    entries: dict[str, dict[str, Any]],
    *,
    append_missing: bool = True,
) -> list[str]:
    if isinstance(raw, str):
        items = [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
    elif isinstance(raw, (list, tuple)):
        items = [str(x).strip() for x in raw if str(x).strip()]
    else:
        items = []

    seen: set[str] = set()
    out: list[str] = []
    for mid in items:
        entry = entries.get(mid)
        if not entry or mid in seen or entry.get("enabled") is False:
            continue
        seen.add(mid)
        out.append(mid)

    if append_missing:
        for mid, entry in entries.items():
            if mid in seen or entry.get("enabled") is False:
                continue
            seen.add(mid)
            out.append(mid)
    return out


def _serialize_catalog() -> dict[str, Any]:
    return {
        "version": 1,
        "updated_at": _utc_iso(),
        "entries": [deepcopy(CATALOG_ENTRIES[mid]) for mid in CATALOG_ENTRIES],
        "order": {
            PURPOSE_TRANSCRIPTION: list(MODEL_ORDER.get(PURPOSE_TRANSCRIPTION) or []),
        },
    }


def save_catalog() -> None:
    if STT_CATALOG_FILE is None:
        return
    try:
        STT_CATALOG_FILE.write_text(json.dumps(_serialize_catalog(), indent=2), encoding="utf-8")
    except OSError as ex:
        print(f"[STT] failed to save catalog: {ex}", flush=True)


def _serialize_runtime() -> dict[str, Any]:
    return {
        "model_id": CURRENT_MODEL_ID,
        "provider_mode": PROVIDER_MODE,
        "fallback_model_ids": list(FALLBACK_MODEL_IDS),
        "updated_at": _utc_iso(),
    }


def save_runtime() -> None:
    if STT_RUNTIME_FILE is None:
        return
    try:
        STT_RUNTIME_FILE.write_text(json.dumps(_serialize_runtime(), indent=2), encoding="utf-8")
    except OSError as ex:
        print(f"[STT] failed to save runtime config: {ex}", flush=True)


def _apply_order_to_runtime(order: list[str]) -> None:
    global CURRENT_MODEL_ID, FALLBACK_MODEL_IDS
    if not order:
        CURRENT_MODEL_ID = LOCAL_WHISPER_ID
        FALLBACK_MODEL_IDS = []
        return
    CURRENT_MODEL_ID = order[0]
    FALLBACK_MODEL_IDS = list(order[1:])


def _filter_chain_for_mode(chain: list[str], provider_mode: str) -> list[str]:
    out: list[str] = []
    for mid in chain:
        entry = CATALOG_ENTRIES.get(mid)
        if not entry or entry.get("enabled") is False:
            continue
        is_local = entry.get("provider") == "local"
        if provider_mode == PROVIDER_MODE_LOCAL and not is_local:
            continue
        if provider_mode == PROVIDER_MODE_OPENROUTER and is_local:
            continue
        out.append(mid)
    return out


def get_attempt_chain() -> list[str]:
    """Ordered catalog ids to try for the current provider_mode."""
    with _CATALOG_LOCK:
        order = list(MODEL_ORDER.get(PURPOSE_TRANSCRIPTION) or [])
        if not order and CURRENT_MODEL_ID:
            order = [CURRENT_MODEL_ID, *FALLBACK_MODEL_IDS]
        return _filter_chain_for_mode(order, PROVIDER_MODE)


def get_entry(entry_id: str) -> dict[str, Any] | None:
    with _CATALOG_LOCK:
        entry = CATALOG_ENTRIES.get(entry_id)
        return deepcopy(entry) if entry else None


def get_model_order() -> list[str]:
    with _CATALOG_LOCK:
        return list(MODEL_ORDER.get(PURPOSE_TRANSCRIPTION) or [])


def load_or_seed() -> None:
    global CATALOG_ENTRIES, MODEL_ORDER, CURRENT_MODEL_ID, FALLBACK_MODEL_IDS, PROVIDER_MODE

    seeded = seed_entries()
    order = default_order(seeded)
    provider_mode = _normalize_provider_mode(
        os.environ.get("STT_PROVIDER_MODE") or PROVIDER_MODE_OPENROUTER_WITH_LOCAL
    )
    bootstrap_model = (os.environ.get("STT_MODEL_ID") or "").strip()

    if STT_CATALOG_FILE is not None and STT_CATALOG_FILE.exists():
        try:
            raw = json.loads(STT_CATALOG_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as ex:
            print(f"[STT] invalid catalog file, reseeding: {ex}", flush=True)
            raw = None
        if isinstance(raw, dict):
            loaded: dict[str, dict[str, Any]] = {}
            entries_raw = raw.get("entries")
            if isinstance(entries_raw, list):
                for item in entries_raw:
                    if not isinstance(item, dict):
                        continue
                    try:
                        entry = _normalize_entry(item)
                    except ValueError:
                        continue
                    loaded[entry["id"]] = entry
            if loaded:
                if LOCAL_WHISPER_ID not in loaded:
                    loaded[LOCAL_WHISPER_ID] = seeded[LOCAL_WHISPER_ID]
                # Ensure builtin OpenRouter seeds exist so admins can re-enable them.
                for mid, entry in seeded.items():
                    if mid not in loaded:
                        loaded[mid] = entry
                seeded = loaded
            order_raw = raw.get("order") if isinstance(raw.get("order"), dict) else {}
            order = {
                PURPOSE_TRANSCRIPTION: _normalize_order(
                    order_raw.get(PURPOSE_TRANSCRIPTION) or order_raw.get("default"),
                    seeded,
                    append_missing=True,
                )
            }

    runtime_order = list(order.get(PURPOSE_TRANSCRIPTION) or [])
    if STT_RUNTIME_FILE is not None and STT_RUNTIME_FILE.exists():
        try:
            runtime = json.loads(STT_RUNTIME_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as ex:
            print(f"[STT] invalid runtime config, using catalog order: {ex}", flush=True)
            runtime = None
        if isinstance(runtime, dict):
            provider_mode = _normalize_provider_mode(
                runtime.get("provider_mode") or provider_mode
            )
            model_id = str(runtime.get("model_id") or "").strip()
            fallbacks = runtime.get("fallback_model_ids")
            if model_id:
                runtime_order = _normalize_order(
                    [model_id, *(fallbacks if isinstance(fallbacks, list) else [])],
                    seeded,
                    append_missing=False,
                )
                # Keep catalog order in sync with runtime when runtime is authoritative.
                if runtime_order:
                    order[PURPOSE_TRANSCRIPTION] = _normalize_order(
                        runtime_order + list(order.get(PURPOSE_TRANSCRIPTION) or []),
                        seeded,
                        append_missing=False,
                    )

    if bootstrap_model and bootstrap_model in seeded and not (
        STT_RUNTIME_FILE and STT_RUNTIME_FILE.exists()
    ):
        rest = [m for m in order[PURPOSE_TRANSCRIPTION] if m != bootstrap_model]
        order[PURPOSE_TRANSCRIPTION] = [bootstrap_model, *rest]

    with _CATALOG_LOCK:
        CATALOG_ENTRIES = seeded
        MODEL_ORDER = order
        PROVIDER_MODE = provider_mode
        _apply_order_to_runtime(list(MODEL_ORDER.get(PURPOSE_TRANSCRIPTION) or []))
        save_catalog()
        save_runtime()

    print(
        f"[STT] catalog loaded entries={len(CATALOG_ENTRIES)} "
        f"primary={CURRENT_MODEL_ID} mode={PROVIDER_MODE} "
        f"fallbacks={FALLBACK_MODEL_IDS}",
        flush=True,
    )


def set_model_order(model_ids: Any, *, provider_mode: str | None = None) -> list[str]:
    global PROVIDER_MODE
    with _CATALOG_LOCK:
        MODEL_ORDER[PURPOSE_TRANSCRIPTION] = _normalize_order(
            model_ids,
            CATALOG_ENTRIES,
            append_missing=False,
        )
        if not MODEL_ORDER[PURPOSE_TRANSCRIPTION]:
            raise ValueError("model_order must include at least one enabled model")
        if provider_mode is not None:
            PROVIDER_MODE = _normalize_provider_mode(provider_mode)
        _apply_order_to_runtime(MODEL_ORDER[PURPOSE_TRANSCRIPTION])
        save_catalog()
        save_runtime()
        return list(MODEL_ORDER[PURPOSE_TRANSCRIPTION])


def select_model(
    model_id: str,
    *,
    provider_mode: str | None = None,
    fallback_model_ids: Any = None,
) -> dict[str, Any]:
    global PROVIDER_MODE
    mid = (model_id or "").strip()
    with _CATALOG_LOCK:
        if mid not in CATALOG_ENTRIES or CATALOG_ENTRIES[mid].get("enabled") is False:
            raise ValueError(f"Unknown or disabled STT model_id: {mid}")
        if fallback_model_ids is not None:
            chain = _normalize_order(
                [mid, *list(fallback_model_ids or [])],
                CATALOG_ENTRIES,
                append_missing=False,
            )
        else:
            existing = list(MODEL_ORDER.get(PURPOSE_TRANSCRIPTION) or [])
            rest = [x for x in existing if x != mid]
            chain = [mid, *rest]
        if not chain:
            raise ValueError("model_order must include at least one enabled model")
        MODEL_ORDER[PURPOSE_TRANSCRIPTION] = chain
        if provider_mode is not None:
            PROVIDER_MODE = _normalize_provider_mode(provider_mode)
        _apply_order_to_runtime(chain)
        save_catalog()
        save_runtime()
        return {
            "model_id": mid,
            "model_order": list(chain),
            "provider_mode": PROVIDER_MODE,
            "fallback_model_ids": list(FALLBACK_MODEL_IDS),
        }


def upsert_entry(raw: dict[str, Any]) -> dict[str, Any]:
    with _CATALOG_LOCK:
        entry = _normalize_entry(raw)
        mid = entry["id"]
        previous = CATALOG_ENTRIES.get(mid)
        CATALOG_ENTRIES[mid] = entry
        order = list(MODEL_ORDER.get(PURPOSE_TRANSCRIPTION) or [])
        if entry.get("enabled") is False:
            order = [x for x in order if x != mid]
        elif mid not in order and (previous is None or previous.get("enabled") is False):
            order.append(mid)
        MODEL_ORDER[PURPOSE_TRANSCRIPTION] = _normalize_order(
            order, CATALOG_ENTRIES, append_missing=False
        )
        if CURRENT_MODEL_ID == mid and entry.get("enabled") is False:
            _apply_order_to_runtime(MODEL_ORDER[PURPOSE_TRANSCRIPTION])
        elif not CURRENT_MODEL_ID:
            _apply_order_to_runtime(MODEL_ORDER[PURPOSE_TRANSCRIPTION])
        save_catalog()
        save_runtime()
        return deepcopy(entry)


def delete_entry(entry_id: str) -> None:
    mid = (entry_id or "").strip()
    if mid == LOCAL_WHISPER_ID:
        raise ValueError("Cannot delete the local Whisper Lite entry")
    with _CATALOG_LOCK:
        if mid not in CATALOG_ENTRIES:
            raise ValueError(f"Unknown STT catalog entry: {mid}")
        if CATALOG_ENTRIES[mid].get("builtin"):
            # Soft-disable builtins instead of deleting seed rows.
            CATALOG_ENTRIES[mid]["enabled"] = False
        else:
            del CATALOG_ENTRIES[mid]
        MODEL_ORDER[PURPOSE_TRANSCRIPTION] = _normalize_order(
            MODEL_ORDER.get(PURPOSE_TRANSCRIPTION),
            CATALOG_ENTRIES,
            append_missing=False,
        )
        if CURRENT_MODEL_ID == mid or mid in FALLBACK_MODEL_IDS:
            _apply_order_to_runtime(MODEL_ORDER[PURPOSE_TRANSCRIPTION])
        save_catalog()
        save_runtime()


def get_models_payload() -> dict[str, Any]:
    with _CATALOG_LOCK:
        entries = []
        for mid, entry in CATALOG_ENTRIES.items():
            item = deepcopy(entry)
            item["capabilities"] = capabilities_public_dict(str(entry.get("model") or ""))
            entries.append(item)
        order = list(MODEL_ORDER.get(PURPOSE_TRANSCRIPTION) or [])
        return {
            "catalog": entries,
            "models": entries,
            "model_order": {PURPOSE_TRANSCRIPTION: order},
            "current_model_id": CURRENT_MODEL_ID,
            "provider_mode": PROVIDER_MODE,
            "fallback_model_ids": list(FALLBACK_MODEL_IDS),
            "purpose": PURPOSE_TRANSCRIPTION,
            "purposes": {
                PURPOSE_TRANSCRIPTION: {
                    "current_model_id": CURRENT_MODEL_ID,
                    "provider_mode": PROVIDER_MODE,
                    "fallback_model_ids": list(FALLBACK_MODEL_IDS),
                    "model_order": order,
                }
            },
        }


def get_status_payload(*, local_loaded: bool = False) -> dict[str, Any]:
    with _CATALOG_LOCK:
        primary = CATALOG_ENTRIES.get(CURRENT_MODEL_ID) or {}
        return {
            "ok": True,
            "model_id": CURRENT_MODEL_ID,
            "model": primary.get("model"),
            "provider": primary.get("provider"),
            "label": primary.get("label"),
            "provider_mode": PROVIDER_MODE,
            "fallback_model_ids": list(FALLBACK_MODEL_IDS),
            "model_order": list(MODEL_ORDER.get(PURPOSE_TRANSCRIPTION) or []),
            "attempt_chain": _filter_chain_for_mode(
                list(MODEL_ORDER.get(PURPOSE_TRANSCRIPTION) or []),
                PROVIDER_MODE,
            ),
            "local_whisper_loaded": bool(local_loaded),
            "updated_at": _serialize_runtime().get("updated_at"),
        }
