"""
In-memory tenant YAML content packs supplied by Yaml Editor.

Editor remains the source of truth on disk. Middleware keeps packs in memory
keyed by (owner_user_id, campaign_id) — never writes tenant YAML to local folders.
"""

from __future__ import annotations

import threading
import time
from copy import deepcopy
from typing import Any

import yaml

from session_context import (
    get_request_campaign,
    get_request_owner,
    session_key,
    set_request_campaign,
    set_request_owner,
)

_PACK_LOCK = threading.Lock()
# session_key(owner, campaign) -> pack dict
_PACKS: dict[str, dict[str, Any]] = {}
_MAX_PACKS = 2000


def clear_request_owner() -> None:
    from session_context import clear_request_context

    clear_request_context()


def _empty_pack(owner_user_id: str, campaign_id: str = "") -> dict[str, Any]:
    return {
        "owner_user_id": owner_user_id,
        "campaign_id": (campaign_id or "").strip(),
        "campaigns": {},
        "npcs": {},
        "locations": {},
        "parties": {},
        "tts": None,
        "llm_rules": None,
        "indexes": {
            "npc_ids_by_location": {},
            "npc_aliases": {},
        },
        "installed_at": time.time(),
        "touched_at": time.time(),
    }


def get_pack(
    owner_user_id: str | None = None,
    campaign_id: str | None = None,
) -> dict[str, Any] | None:
    key = session_key(owner_user_id, campaign_id)
    if not key:
        return None
    with _PACK_LOCK:
        pack = _PACKS.get(key)
        if pack:
            pack["touched_at"] = time.time()
        return deepcopy(pack) if pack else None


def _normalize_doc(data: Any, *, fallback_id: str = "") -> dict[str, Any]:
    if isinstance(data, str):
        parsed = yaml.safe_load(data) or {}
        data = parsed
    if not isinstance(data, dict):
        raise ValueError("Content pack document must be a YAML mapping or object")
    from yaml_list_text import repair_string_list_fields

    doc = repair_string_list_fields(dict(data))
    if fallback_id and not (
        str(doc.get("id") or "").strip()
        or str(doc.get("campaign_id") or "").strip()
        or str(doc.get("npc_id") or "").strip()
        or str(doc.get("party_id") or "").strip()
        or str(doc.get("location_id") or "").strip()
    ):
        doc.setdefault("id", fallback_id)
    return doc


def _doc_id(kind: str, doc: dict[str, Any], filename: str = "") -> str:
    stem = filename.rsplit(".", 1)[0] if filename else ""
    if kind == "campaign":
        return str(doc.get("campaign_id") or doc.get("id") or stem or "").strip()
    if kind == "npc":
        return str(doc.get("npc_id") or doc.get("id") or stem or "").strip()
    if kind == "location":
        return str(doc.get("location_id") or doc.get("id") or stem or "").strip()
    if kind == "party":
        return str(doc.get("party_id") or doc.get("id") or stem or "").strip()
    return stem.strip()


def _bucket_key(kind: str) -> str:
    mapping = {
        "campaign": "campaigns",
        "npc": "npcs",
        "location": "locations",
        "party": "parties",
    }
    if kind not in mapping:
        raise ValueError(f"Unsupported content pack kind: {kind}")
    return mapping[kind]


def _prune_locked() -> None:
    if len(_PACKS) <= _MAX_PACKS:
        return
    ordered = sorted(
        _PACKS.items(),
        key=lambda item: float(item[1].get("touched_at") or item[1].get("installed_at") or 0),
    )
    overflow = len(_PACKS) - _MAX_PACKS
    for key, _ in ordered[:overflow]:
        _PACKS.pop(key, None)


def install_content_pack(
    owner_user_id: str,
    *,
    campaign_id: str = "",
    content_pack: dict[str, Any] | None = None,
    files: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Replace the in-memory pack for this owner+campaign.

    Accepts either:
    - content_pack: { campaigns|npcs|locations|parties: { id: doc|yaml_string }, tts?: doc|yaml_string, llm_rules?: doc|yaml_string }
    - files: [{ kind, filename, content|yaml }]  (parsed into memory only)
    """
    owner = (owner_user_id or "").strip()
    campaign_id = (campaign_id or "").strip()
    if not owner:
        raise ValueError("Missing owner_user_id for content pack")
    if not campaign_id:
        raise ValueError("Missing campaign_id for content pack")

    set_request_owner(owner)
    set_request_campaign(campaign_id)
    key = session_key(owner, campaign_id)
    if not key:
        raise ValueError("Could not build content pack session key")

    pack = _empty_pack(owner, campaign_id)

    if isinstance(content_pack, dict):
        for kind, bucket in (
            ("campaign", "campaigns"),
            ("npc", "npcs"),
            ("location", "locations"),
            ("party", "parties"),
        ):
            raw_bucket = content_pack.get(bucket) or content_pack.get(kind) or {}
            if isinstance(raw_bucket, dict):
                for doc_id, raw_doc in raw_bucket.items():
                    doc = _normalize_doc(raw_doc, fallback_id=str(doc_id))
                    resolved = _doc_id(kind, doc, str(doc_id))
                    if not resolved:
                        continue
                    pack[bucket][resolved] = doc
            elif isinstance(raw_bucket, list):
                for raw_doc in raw_bucket:
                    doc = _normalize_doc(raw_doc)
                    resolved = _doc_id(kind, doc)
                    if not resolved:
                        continue
                    pack[bucket][resolved] = doc

        pack["tts"] = _normalize_tts(content_pack.get("tts"))
        pack["llm_rules"] = _normalize_llm_rules(content_pack.get("llm_rules"))

    for raw in files or []:
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("kind") or "").strip().lower()
        if kind not in {"campaign", "npc", "location", "party"}:
            raise ValueError(f"Unsupported content pack kind: {kind}")
        filename = str(raw.get("filename") or "").strip()
        content = raw.get("content")
        if content is None:
            content = raw.get("yaml") or raw.get("data")
        doc = _normalize_doc(content, fallback_id=filename.rsplit(".", 1)[0] if filename else "")
        resolved = _doc_id(kind, doc, filename)
        if not resolved:
            raise ValueError(f"Could not resolve id for {kind} document")
        pack[_bucket_key(kind)][resolved] = doc

    if campaign_id not in pack["campaigns"]:
        for cid, doc in list(pack["campaigns"].items()):
            embedded = str(doc.get("campaign_id") or doc.get("id") or cid).strip()
            if embedded.casefold() == campaign_id.casefold():
                pack["campaigns"][campaign_id] = doc
                break

    pack["indexes"] = _build_indexes(pack)

    with _PACK_LOCK:
        _PACKS[key] = pack
        _prune_locked()

    return {
        "owner_user_id": owner,
        "campaign_id": pack["campaign_id"],
        "counts": {
            "campaigns": len(pack["campaigns"]),
            "npcs": len(pack["npcs"]),
            "locations": len(pack["locations"]),
            "parties": len(pack["parties"]),
            "has_tts": pack.get("tts") is not None,
            "has_llm_rules": pack.get("llm_rules") is not None,
            "locations_indexed": len(pack["indexes"].get("npc_ids_by_location") or {}),
        },
    }


def _build_indexes(pack: dict[str, Any]) -> dict[str, Any]:
    """Derive location→npc and alias→npc maps once per pack install."""
    npc_ids_by_location: dict[str, list[str]] = {}
    npc_aliases: dict[str, str] = {}

    for key, doc in (pack.get("npcs") or {}).items():
        if not isinstance(doc, dict):
            continue
        npc_id = str(doc.get("npc_id") or doc.get("id") or key or "").strip()
        if not npc_id:
            continue
        location_id = str(doc.get("location_id") or "").strip()
        if location_id:
            loc_key = location_id.casefold()
            bucket = npc_ids_by_location.setdefault(loc_key, [])
            # Preserve original location casing via first seen id list; keys are casefolded.
            if npc_id not in bucket:
                bucket.append(npc_id)

        for alias in (npc_id, doc.get("name"), doc.get("role")):
            text = str(alias or "").strip()
            if len(text) < 3:
                continue
            npc_aliases.setdefault(text.casefold(), npc_id)

    # Stable ordering for deterministic prompts
    for loc_key, ids in npc_ids_by_location.items():
        npc_ids_by_location[loc_key] = sorted(ids, key=str.casefold)

    return {
        "npc_ids_by_location": npc_ids_by_location,
        "npc_aliases": npc_aliases,
    }


def _normalize_tts(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        parsed = yaml.safe_load(text) or {}
        return parsed if isinstance(parsed, dict) else None
    if isinstance(raw, dict):
        return dict(raw)
    return None


def _normalize_llm_rules(raw: Any) -> Any | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        return yaml.safe_load(text)
    if isinstance(raw, (dict, list)):
        return deepcopy(raw)
    return None


def get_pack_tts(
    owner_user_id: str | None = None,
    campaign_id: str | None = None,
) -> dict[str, Any] | None:
    """Return the owner's narrator TTS YAML from the installed content pack, if any."""
    pack = get_pack(owner_user_id, campaign_id)
    if not pack:
        return None
    tts = pack.get("tts")
    return deepcopy(tts) if isinstance(tts, dict) else None


def get_pack_llm_rules(
    owner_user_id: str | None = None,
    campaign_id: str | None = None,
) -> Any | None:
    """Return global llm_rules from the installed content pack, if any."""
    pack = get_pack(owner_user_id, campaign_id)
    if not pack:
        return None
    rules = pack.get("llm_rules")
    if rules is None:
        return None
    return deepcopy(rules)


def lookup_document(
    kind: str,
    doc_id: str,
    owner_user_id: str | None = None,
    campaign_id: str | None = None,
) -> dict[str, Any] | None:
    doc_id = (doc_id or "").strip()
    if not doc_id:
        return None

    key = session_key(owner_user_id, campaign_id)
    if not key:
        return None

    bucket = _bucket_key(kind)
    with _PACK_LOCK:
        pack = _PACKS.get(key)
        if not pack:
            return None
        pack["touched_at"] = time.time()
        docs = pack.get(bucket) or {}
        direct = docs.get(doc_id)
        if direct is not None:
            return deepcopy(direct)
        for item_key, doc in docs.items():
            if item_key.casefold() == doc_id.casefold():
                return deepcopy(doc)
            embedded = _doc_id(kind, doc, item_key)
            if embedded.casefold() == doc_id.casefold():
                return deepcopy(doc)
    return None


def list_document_ids(
    kind: str,
    owner_user_id: str | None = None,
    campaign_id: str | None = None,
) -> list[str]:
    key = session_key(owner_user_id, campaign_id)
    if not key:
        return []
    bucket = _bucket_key(kind)
    with _PACK_LOCK:
        pack = _PACKS.get(key)
        if not pack:
            return []
        pack["touched_at"] = time.time()
        return sorted((pack.get(bucket) or {}).keys(), key=str.casefold)


def has_pack(
    owner_user_id: str | None = None,
    campaign_id: str | None = None,
) -> bool:
    key = session_key(owner_user_id, campaign_id)
    if not key:
        return False
    with _PACK_LOCK:
        return key in _PACKS
