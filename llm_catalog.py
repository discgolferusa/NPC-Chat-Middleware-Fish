"""Admin-editable OpenRouter model+provider catalog and per-purpose precedence order.

Seed catalog matches the historic hardcoded LLM_MODELS rows. Persistence lives in
``llm_catalog.json`` beside the middleware process. Pricing is fetched from
OpenRouter ``/models/{author}/{slug}/endpoints`` when possible.
"""

from __future__ import annotations

import json
import re
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests

PURPOSE_CHAT = "chat"
PURPOSE_CONTENT_CREATION = "content_creation"
VALID_PURPOSES = frozenset({PURPOSE_CHAT, PURPOSE_CONTENT_CREATION})

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_CATALOG_LOCK = threading.Lock()

# Filled by llm_router.init after paths/env are known.
LLM_CATALOG_FILE: Path | None = None
_get_openrouter_api_root: Callable[[], str] | None = None
_get_openrouter_headers: Callable[[], dict[str, str]] | None = None
_get_local_model: Callable[[], str] | None = None
_get_local_url: Callable[[], str] | None = None

# Live mutable catalog mirrored into llm_router.LLM_MODELS.
CATALOG_ENTRIES: dict[str, dict[str, Any]] = {}
MODEL_ORDER: dict[str, list[str]] = {
    PURPOSE_CHAT: [],
    PURPOSE_CONTENT_CREATION: [],
}


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


def make_catalog_id(openrouter_model: str, host_provider: str) -> str:
    """Stable id for a model+provider pair (local is reserved)."""
    provider = _slug(host_provider)
    if provider == "local":
        return "local"
    model_slug = _slug(openrouter_model.replace("/", "-"))
    return f"openrouter-{model_slug}-{provider}"


def _title_from_model(openrouter_model: str) -> str:
    leaf = (openrouter_model or "").split("/")[-1].replace("-", " ").replace("_", " ")
    return " ".join(part.capitalize() for part in leaf.split()) or openrouter_model


def _provider_display(host_provider: str) -> str:
    known = {
        "local": "local",
        "amazon-bedrock": "Amazon Bedrock",
        "novita": "Novita",
        "coreweave": "CoreWeave",
        "deepinfra": "DeepInfra",
        "parasail": "Parasail",
    }
    key = (host_provider or "").strip().lower()
    if key in known:
        return known[key]
    return (host_provider or "").replace("-", " ").title() or "OpenRouter"


def seed_entries(
    *,
    local_model: str,
    local_url: str,
    default_prices: dict[str, dict[str, float]],
) -> dict[str, dict[str, Any]]:
    """Built-in catalog used when llm_catalog.json is missing."""
    seeds: list[dict[str, Any]] = [
        {
            "id": "local",
            "builtin": True,
            "enabled": True,
            "label": "Local llama.cpp — local",
            "provider": "local",
            "provider_label": "local",
            "host_provider": "local",
            "model": local_model,
            "url": local_url,
            "online": False,
            "limits": None,
            "purposes": [PURPOSE_CHAT, PURPOSE_CONTENT_CREATION],
            "provider_order": None,
        },
        {
            "id": "openrouter-gpt-oss-20b",
            "builtin": False,
            "enabled": True,
            "label": "GPT-OSS 20B — Amazon Bedrock / Novita",
            "provider": "openrouter",
            "provider_label": "Amazon Bedrock / Novita",
            "host_provider": "amazon-bedrock",
            "model": "openai/gpt-oss-20b",
            "online": True,
            "limits": None,
            "purposes": [PURPOSE_CHAT, PURPOSE_CONTENT_CREATION],
            # Legacy multi-host preference until an admin pins a single provider.
            "provider_order": ["amazon-bedrock", "novita"],
        },
        {
            "id": "openrouter-nemotron-3.5-lightning-coreweave",
            "builtin": False,
            "enabled": True,
            "label": "Nemotron 3.5 Lightning — CoreWeave",
            "provider": "openrouter",
            "provider_label": "CoreWeave",
            "host_provider": "coreweave",
            "model": "nvidia/nemotron-3.5-lightning",
            "online": True,
            "limits": None,
            "purposes": [PURPOSE_CHAT, PURPOSE_CONTENT_CREATION],
            "provider_order": ["coreweave"],
        },
        {
            "id": "openrouter-nemotron-3.5-lightning-deepinfra",
            "builtin": False,
            "enabled": True,
            "label": "Nemotron 3.5 Lightning — DeepInfra",
            "provider": "openrouter",
            "provider_label": "DeepInfra",
            "host_provider": "deepinfra",
            "model": "nvidia/nemotron-3.5-lightning",
            "online": True,
            "limits": None,
            "purposes": [PURPOSE_CHAT, PURPOSE_CONTENT_CREATION],
            "provider_order": ["deepinfra"],
        },
        {
            "id": "openrouter-deepseek-v4-flash-0731-coreweave",
            "builtin": False,
            "enabled": True,
            "label": "DeepSeek V4 Flash 0731 — CoreWeave",
            "provider": "openrouter",
            "provider_label": "CoreWeave",
            "host_provider": "coreweave",
            "model": "deepseek/deepseek-v4-flash-0731",
            "online": True,
            "limits": None,
            "purposes": [PURPOSE_CHAT, PURPOSE_CONTENT_CREATION],
            "provider_order": ["coreweave"],
        },
    ]
    out: dict[str, dict[str, Any]] = {}
    for entry in seeds:
        mid = entry["id"]
        prices = default_prices.get(mid) or {"input": 0.0, "output": 0.0}
        entry["pricing"] = {
            "input": float(prices.get("input", 0.0)),
            "output": float(prices.get("output", 0.0)),
            "source": "seed",
            "fetched_at": None,
        }
        out[mid] = entry
    return out


def default_order(entries: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """Default precedence: online seeds first, local last."""
    online = [mid for mid, e in entries.items() if e.get("online") and e.get("enabled", True)]
    local = ["local"] if "local" in entries else []
    chain = [*online, *[m for m in local if m not in online]]
    return {
        PURPOSE_CHAT: list(chain),
        PURPOSE_CONTENT_CREATION: list(chain),
    }


def configure(
    *,
    catalog_file: Path,
    get_openrouter_api_root: Callable[[], str],
    get_openrouter_headers: Callable[[], dict[str, str]],
    get_local_model: Callable[[], str],
    get_local_url: Callable[[], str],
) -> None:
    global LLM_CATALOG_FILE
    global _get_openrouter_api_root, _get_openrouter_headers
    global _get_local_model, _get_local_url
    LLM_CATALOG_FILE = catalog_file
    _get_openrouter_api_root = get_openrouter_api_root
    _get_openrouter_headers = get_openrouter_headers
    _get_local_model = get_local_model
    _get_local_url = get_local_url


def _normalize_purposes(raw: Any) -> list[str]:
    if raw is None:
        return [PURPOSE_CHAT, PURPOSE_CONTENT_CREATION]
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        items = []
    out: list[str] = []
    for item in items:
        purpose = str(item or "").strip()
        if purpose in VALID_PURPOSES and purpose not in out:
            out.append(purpose)
    return out or [PURPOSE_CHAT, PURPOSE_CONTENT_CREATION]


def _normalize_entry(raw: dict[str, Any], *, entry_id: str | None = None) -> dict[str, Any]:
    mid = str(entry_id or raw.get("id") or "").strip()
    if not mid:
        raise ValueError("Catalog entry id is required")

    host = str(raw.get("host_provider") or "").strip().lower()
    if mid == "local" or str(raw.get("provider") or "") == "local":
        local_model = _get_local_model() if _get_local_model else "npc"
        local_url = _get_local_url() if _get_local_url else ""
        return {
            "id": "local",
            "builtin": True,
            "enabled": bool(raw.get("enabled", True)),
            "label": str(raw.get("label") or "Local llama.cpp — local"),
            "provider": "local",
            "provider_label": "local",
            "host_provider": "local",
            "model": local_model,
            "url": local_url,
            "online": False,
            "limits": None,
            "purposes": _normalize_purposes(raw.get("purposes")),
            "provider_order": None,
            "pricing": {
                "input": 0.0,
                "output": 0.0,
                "source": "local",
                "fetched_at": None,
            },
        }

    model = str(raw.get("model") or "").strip()
    if not model:
        raise ValueError("OpenRouter model is required")
    if not host:
        # Derive from provider_order or id suffix.
        order = raw.get("provider_order")
        if isinstance(order, list) and order:
            host = str(order[0]).strip().lower()
        elif mid.startswith("openrouter-") and "-" in mid[len("openrouter-") :]:
            host = mid.rsplit("-", 1)[-1]
        else:
            raise ValueError("host_provider is required for online catalog entries")

    provider_label = str(raw.get("provider_label") or _provider_display(host)).strip()
    label = str(raw.get("label") or f"{_title_from_model(model)} — {provider_label}").strip()
    order_raw = raw.get("provider_order")
    if isinstance(order_raw, list) and order_raw:
        provider_order = [str(p).strip().lower() for p in order_raw if str(p).strip()]
    else:
        provider_order = [host]

    pricing_raw = raw.get("pricing") if isinstance(raw.get("pricing"), dict) else {}
    pricing = {
        "input": float(pricing_raw.get("input") or 0.0),
        "output": float(pricing_raw.get("output") or 0.0),
        "source": str(pricing_raw.get("source") or "unknown"),
        "fetched_at": pricing_raw.get("fetched_at"),
    }

    return {
        "id": mid,
        "builtin": bool(raw.get("builtin", False)),
        "enabled": bool(raw.get("enabled", True)),
        "label": label,
        "provider": "openrouter",
        "provider_label": provider_label,
        "host_provider": host,
        "model": model,
        "online": True,
        "limits": raw.get("limits"),
        "purposes": _normalize_purposes(raw.get("purposes")),
        "provider_order": provider_order,
        "pricing": pricing,
    }


def _normalize_order(
    raw: Any,
    entries: dict[str, dict[str, Any]],
    *,
    purpose: str,
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
        if not entry or mid in seen:
            continue
        if entry.get("enabled") is False:
            continue
        purposes = entry.get("purposes") or []
        if purposes and purpose not in purposes:
            continue
        seen.add(mid)
        out.append(mid)

    if append_missing:
        # Append any enabled purpose-capable entries not listed yet (stable catalog order).
        for mid, entry in entries.items():
            if mid in seen or entry.get("enabled") is False:
                continue
            purposes = entry.get("purposes") or []
            if purposes and purpose not in purposes:
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
            PURPOSE_CHAT: list(MODEL_ORDER.get(PURPOSE_CHAT) or []),
            PURPOSE_CONTENT_CREATION: list(MODEL_ORDER.get(PURPOSE_CONTENT_CREATION) or []),
        },
    }


def save_catalog() -> None:
    if LLM_CATALOG_FILE is None:
        return
    payload = _serialize_catalog()
    try:
        LLM_CATALOG_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as ex:
        print(f"[LLM] failed to save catalog: {ex}", flush=True)


def apply_to_router(
    llm_models: dict[str, dict[str, Any]],
    prices: dict[str, dict[str, float]],
) -> None:
    """Replace live router catalog/prices from CATALOG_ENTRIES."""
    llm_models.clear()
    prices.clear()
    for mid, entry in CATALOG_ENTRIES.items():
        public = deepcopy(entry)
        llm_models[mid] = public
        pricing = entry.get("pricing") or {}
        prices[mid] = {
            "input": float(pricing.get("input") or 0.0),
            "output": float(pricing.get("output") or 0.0),
        }


def load_or_seed(
    *,
    llm_models: dict[str, dict[str, Any]],
    prices: dict[str, dict[str, float]],
    default_prices: dict[str, dict[str, float]],
    local_model: str,
    local_url: str,
) -> None:
    """Load catalog file or seed defaults, then mirror into router dicts."""
    global CATALOG_ENTRIES, MODEL_ORDER

    seeded = seed_entries(
        local_model=local_model,
        local_url=local_url,
        default_prices=default_prices,
    )
    order = default_order(seeded)

    if LLM_CATALOG_FILE is not None and LLM_CATALOG_FILE.exists():
        try:
            raw = json.loads(LLM_CATALOG_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as ex:
            print(f"[LLM] invalid catalog file, reseeding: {ex}", flush=True)
            raw = None
        if isinstance(raw, dict):
            entries_raw = raw.get("entries")
            loaded: dict[str, dict[str, Any]] = {}
            if isinstance(entries_raw, list):
                for item in entries_raw:
                    if not isinstance(item, dict):
                        continue
                    try:
                        entry = _normalize_entry(item)
                    except ValueError:
                        continue
                    loaded[entry["id"]] = entry
            elif isinstance(entries_raw, dict):
                for mid, item in entries_raw.items():
                    if not isinstance(item, dict):
                        continue
                    try:
                        entry = _normalize_entry({**item, "id": mid}, entry_id=str(mid))
                    except ValueError:
                        continue
                    loaded[entry["id"]] = entry

            if loaded:
                # Always ensure local exists.
                if "local" not in loaded:
                    loaded["local"] = seeded["local"]
                else:
                    loaded["local"]["model"] = local_model
                    loaded["local"]["url"] = local_url
                seeded = loaded
            order_raw = raw.get("order") if isinstance(raw.get("order"), dict) else {}
            order = {
                PURPOSE_CHAT: _normalize_order(
                    order_raw.get(PURPOSE_CHAT), seeded, purpose=PURPOSE_CHAT
                ),
                PURPOSE_CONTENT_CREATION: _normalize_order(
                    order_raw.get(PURPOSE_CONTENT_CREATION),
                    seeded,
                    purpose=PURPOSE_CONTENT_CREATION,
                ),
            }

    with _CATALOG_LOCK:
        CATALOG_ENTRIES = seeded
        MODEL_ORDER = order
        apply_to_router(llm_models, prices)
        save_catalog()


def get_model_order(purpose: str) -> list[str]:
    purpose = purpose if purpose in VALID_PURPOSES else PURPOSE_CHAT
    return list(MODEL_ORDER.get(purpose) or [])


def purpose_chain_from_order(purpose: str) -> tuple[str, list[str]]:
    """Primary + fallbacks derived from the ordered precedence list."""
    chain = get_model_order(purpose)
    if not chain:
        return "local", []
    return chain[0], chain[1:]


def set_model_order(
    purpose: str,
    model_ids: Any,
    *,
    llm_models: dict[str, dict[str, Any]],
    prices: dict[str, dict[str, float]],
) -> list[str]:
    purpose = purpose if purpose in VALID_PURPOSES else PURPOSE_CHAT
    with _CATALOG_LOCK:
        MODEL_ORDER[purpose] = _normalize_order(
            model_ids,
            CATALOG_ENTRIES,
            purpose=purpose,
            append_missing=False,
        )
        apply_to_router(llm_models, prices)
        save_catalog()
        return list(MODEL_ORDER[purpose])


def _prune_orders(*, append_new_id: str | None = None) -> None:
    """Drop invalid ids from MODEL_ORDER without re-adding admin-excluded models.

    When ``append_new_id`` is set (a newly created catalog row), append it to the
    end of each purpose list it supports so it becomes selectable without
    reshuffling the admin's existing precedence.
    """
    for purpose in VALID_PURPOSES:
        order = _normalize_order(
            MODEL_ORDER.get(purpose),
            CATALOG_ENTRIES,
            purpose=purpose,
            append_missing=False,
        )
        if append_new_id:
            entry = CATALOG_ENTRIES.get(append_new_id)
            if (
                entry
                and entry.get("enabled") is not False
                and append_new_id not in order
            ):
                purposes = entry.get("purposes") or []
                if not purposes or purpose in purposes:
                    order.append(append_new_id)
        MODEL_ORDER[purpose] = order


def upsert_entry(
    payload: dict[str, Any],
    *,
    llm_models: dict[str, dict[str, Any]],
    prices: dict[str, dict[str, float]],
    fetch_pricing: bool = True,
) -> dict[str, Any]:
    model = str(payload.get("model") or "").strip()
    host = str(payload.get("host_provider") or payload.get("provider") or "").strip().lower()
    entry_id = str(payload.get("id") or "").strip()

    if not entry_id:
        if host == "local" or str(payload.get("provider") or "") == "local":
            entry_id = "local"
        elif model and host:
            entry_id = make_catalog_id(model, host)
        else:
            raise ValueError("model and host_provider are required")

    if entry_id == "local" or host == "local":
        with _CATALOG_LOCK:
            current = CATALOG_ENTRIES.get("local", {})
            merged = {
                **current,
                "id": "local",
                "enabled": bool(payload.get("enabled", current.get("enabled", True))),
                "purposes": _normalize_purposes(
                    payload.get("purposes", current.get("purposes"))
                ),
                "label": str(
                    payload.get("label") or current.get("label") or "Local llama.cpp — local"
                ),
            }
            entry = _normalize_entry(merged, entry_id="local")
            CATALOG_ENTRIES["local"] = entry
            _prune_orders()
            apply_to_router(llm_models, prices)
            save_catalog()
            return deepcopy(entry)

    if not model:
        existing = CATALOG_ENTRIES.get(entry_id)
        if existing:
            model = str(existing.get("model") or "")
    if not host:
        existing = CATALOG_ENTRIES.get(entry_id)
        if existing:
            host = str(existing.get("host_provider") or "")
    if not model or not host:
        raise ValueError("model and host_provider are required")

    is_new = entry_id not in CATALOG_ENTRIES
    with _CATALOG_LOCK:
        existing = CATALOG_ENTRIES.get(entry_id, {})
        merged = {
            **existing,
            **payload,
            "id": entry_id,
            "model": model,
            "host_provider": host,
            "provider": "openrouter",
            "online": True,
        }
        if "purposes" not in payload and existing.get("purposes"):
            merged["purposes"] = existing["purposes"]
        if "enabled" not in payload and "enabled" in existing:
            merged["enabled"] = existing["enabled"]
        # New/updated admin rows pin exactly one host.
        if "provider_order" not in payload:
            merged["provider_order"] = [host]
        entry = _normalize_entry(merged, entry_id=entry_id)

    if fetch_pricing and entry.get("online"):
        try:
            pricing = fetch_endpoint_pricing(entry["model"], entry["host_provider"])
            if pricing:
                entry["pricing"] = pricing
        except Exception as ex:  # noqa: BLE001 — keep catalog save even if pricing fails
            print(f"[LLM] pricing fetch failed for {entry_id}: {ex}", flush=True)

    with _CATALOG_LOCK:
        CATALOG_ENTRIES[entry_id] = entry
        _prune_orders(append_new_id=entry_id if is_new else None)
        apply_to_router(llm_models, prices)
        save_catalog()
        return deepcopy(entry)


def delete_entry(
    entry_id: str,
    *,
    llm_models: dict[str, dict[str, Any]],
    prices: dict[str, dict[str, float]],
) -> None:
    mid = str(entry_id or "").strip()
    if not mid:
        raise ValueError("id is required")
    if mid == "local":
        raise ValueError("Cannot delete the built-in local model")
    with _CATALOG_LOCK:
        if mid not in CATALOG_ENTRIES:
            raise ValueError(f"Unknown catalog id: {mid}")
        del CATALOG_ENTRIES[mid]
        prices.pop(mid, None)
        _prune_orders()
        apply_to_router(llm_models, prices)
        save_catalog()


def fetch_endpoint_pricing(openrouter_model: str, host_provider: str) -> dict[str, Any] | None:
    """Pull $/1M input+output for a model+provider from OpenRouter endpoints."""
    if not _get_openrouter_api_root or not _get_openrouter_headers:
        return None
    model = (openrouter_model or "").strip()
    host = (host_provider or "").strip().lower()
    if not model or "/" not in model or not host or host == "local":
        return None

    author, _, slug = model.partition("/")
    if not author or not slug:
        return None

    url = f"{_get_openrouter_api_root().rstrip('/')}/models/{author}/{slug}/endpoints"
    headers = _get_openrouter_headers()
    response = requests.get(url, headers=headers, timeout=20)
    if response.status_code >= 400:
        print(
            f"[LLM] OpenRouter endpoints HTTP {response.status_code} for {model}",
            flush=True,
        )
        return None

    data = response.json() if response.content else {}
    payload = data.get("data") if isinstance(data, dict) else None
    endpoints = []
    if isinstance(payload, dict):
        endpoints = payload.get("endpoints") or []
    elif isinstance(data, dict):
        endpoints = data.get("endpoints") or []
    if not isinstance(endpoints, list):
        return None

    host_compact = host.replace(" ", "").replace("_", "-")
    matched = None
    for ep in endpoints:
        if not isinstance(ep, dict):
            continue
        name = str(ep.get("provider_name") or ep.get("name") or "").strip().lower()
        tag = str(ep.get("tag") or "").strip().lower()
        name_compact = name.replace(" ", "").replace("_", "-")
        if host in {name, tag} or host_compact == name_compact or host in name_compact:
            matched = ep
            break
    if matched is None and endpoints:
        # Fall back to first endpoint pricing when provider spelling differs.
        matched = endpoints[0] if isinstance(endpoints[0], dict) else None
    if not matched:
        return None

    pricing = matched.get("pricing") if isinstance(matched.get("pricing"), dict) else {}
    try:
        prompt = float(pricing.get("prompt") or 0.0)
        completion = float(pricing.get("completion") or 0.0)
    except (TypeError, ValueError):
        return None

    return {
        "input": round(prompt * 1_000_000.0, 6),
        "output": round(completion * 1_000_000.0, 6),
        "source": "openrouter",
        "fetched_at": _utc_iso(),
        "provider_name": matched.get("provider_name") or matched.get("name"),
    }


def refresh_pricing(
    *,
    entry_id: str | None = None,
    llm_models: dict[str, dict[str, Any]],
    prices: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Refresh OpenRouter endpoint pricing without holding the catalog lock on HTTP I/O."""
    with _CATALOG_LOCK:
        if entry_id:
            targets = [entry_id]
        else:
            targets = [mid for mid, e in CATALOG_ENTRIES.items() if e.get("online")]
        # Snapshot model/host outside the lock so fetches cannot block upsert/order.
        snapshots: list[tuple[str, str, str]] = []
        for mid in targets:
            entry = CATALOG_ENTRIES.get(mid or "")
            if not entry or not entry.get("online"):
                continue
            snapshots.append(
                (
                    mid,
                    str(entry.get("model") or ""),
                    str(entry.get("host_provider") or ""),
                )
            )

    fetched: dict[str, dict[str, Any]] = {}
    failed: list[str] = []
    for mid, model, host in snapshots:
        try:
            pricing = fetch_endpoint_pricing(model, host)
        except Exception as ex:  # noqa: BLE001
            print(f"[LLM] pricing refresh failed for {mid}: {ex}", flush=True)
            pricing = None
        if not pricing:
            failed.append(mid)
            continue
        fetched[mid] = pricing

    # Mark requested ids that vanished / were offline.
    for mid in targets:
        if not mid:
            failed.append(str(mid or ""))
            continue
        if mid not in {s[0] for s in snapshots} and mid not in fetched:
            if mid not in failed:
                failed.append(mid)

    updated: list[str] = []
    with _CATALOG_LOCK:
        for mid, pricing in fetched.items():
            entry = CATALOG_ENTRIES.get(mid)
            if not entry:
                failed.append(mid)
                continue
            entry["pricing"] = pricing
            updated.append(mid)
        apply_to_router(llm_models, prices)
        save_catalog()
    return {"updated": updated, "failed": failed}


def public_catalog_entries(*, include_disabled: bool = True) -> list[dict[str, Any]]:
    rows = []
    for mid, entry in CATALOG_ENTRIES.items():
        if not include_disabled and entry.get("enabled") is False:
            continue
        rows.append(deepcopy(entry))
    return rows
