"""
In-memory tenant YAML content packs supplied by Yaml Editor.

Editor remains the source of truth on disk. Middleware keeps a session-scoped
copy in memory keyed by owner_user_id — never writes tenant YAML to local folders.
"""

from __future__ import annotations

import threading
import time
from contextvars import ContextVar
from copy import deepcopy
from typing import Any

import yaml

_REQUEST_OWNER: ContextVar[str] = ContextVar("content_pack_owner", default="")
_PACK_LOCK = threading.Lock()
# owner_user_id -> pack dict
_PACKS: dict[str, dict[str, Any]] = {}


def set_request_owner(owner_user_id: str | None) -> None:
    _REQUEST_OWNER.set((owner_user_id or "").strip())


def get_request_owner() -> str:
    return (_REQUEST_OWNER.get() or "").strip()


def clear_request_owner() -> None:
    _REQUEST_OWNER.set("")


def _empty_pack(owner_user_id: str, campaign_id: str = "") -> dict[str, Any]:
    return {
        "owner_user_id": owner_user_id,
        "campaign_id": (campaign_id or "").strip(),
        "campaigns": {},
        "npcs": {},
        "locations": {},
        "parties": {},
        "installed_at": time.time(),
    }


def get_pack(owner_user_id: str | None = None) -> dict[str, Any] | None:
    owner = (owner_user_id or get_request_owner() or "").strip()
    if not owner:
        return None
    with _PACK_LOCK:
        pack = _PACKS.get(owner)
        return deepcopy(pack) if pack else None


def _normalize_doc(data: Any, *, fallback_id: str = "") -> dict[str, Any]:
    if isinstance(data, str):
        parsed = yaml.safe_load(data) or {}
        data = parsed
    if not isinstance(data, dict):
        raise ValueError("Content pack document must be a YAML mapping or object")
    doc = dict(data)
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


def install_content_pack(
    owner_user_id: str,
    *,
    campaign_id: str = "",
    content_pack: dict[str, Any] | None = None,
    files: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Replace the in-memory pack for this owner.

    Accepts either:
    - content_pack: { campaigns|npcs|locations|parties: { id: doc|yaml_string } }
    - files: [{ kind, filename, content|yaml }]  (parsed into memory only)
    """
    owner = (owner_user_id or "").strip()
    if not owner:
        raise ValueError("Missing owner_user_id for content pack")

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

    if campaign_id and campaign_id not in pack["campaigns"]:
        # Allow select when campaign metadata arrived under a different key shape.
        for cid, doc in list(pack["campaigns"].items()):
            embedded = str(doc.get("campaign_id") or doc.get("id") or cid).strip()
            if embedded.casefold() == campaign_id.casefold():
                pack["campaigns"][campaign_id] = doc
                break

    if campaign_id:
        pack["campaign_id"] = campaign_id

    with _PACK_LOCK:
        _PACKS[owner] = pack

    return {
        "owner_user_id": owner,
        "campaign_id": pack["campaign_id"],
        "counts": {
            "campaigns": len(pack["campaigns"]),
            "npcs": len(pack["npcs"]),
            "locations": len(pack["locations"]),
            "parties": len(pack["parties"]),
        },
    }


def lookup_document(kind: str, doc_id: str, owner_user_id: str | None = None) -> dict[str, Any] | None:
    doc_id = (doc_id or "").strip()
    if not doc_id:
        return None

    owner = (owner_user_id or get_request_owner() or "").strip()
    if not owner:
        return None

    bucket = _bucket_key(kind)
    with _PACK_LOCK:
        pack = _PACKS.get(owner)
        if not pack:
            return None
        docs = pack.get(bucket) or {}
        direct = docs.get(doc_id)
        if direct is not None:
            return deepcopy(direct)
        for key, doc in docs.items():
            if key.casefold() == doc_id.casefold():
                return deepcopy(doc)
            embedded = _doc_id(kind, doc, key)
            if embedded.casefold() == doc_id.casefold():
                return deepcopy(doc)
    return None


def list_document_ids(kind: str, owner_user_id: str | None = None) -> list[str]:
    owner = (owner_user_id or get_request_owner() or "").strip()
    if not owner:
        return []
    bucket = _bucket_key(kind)
    with _PACK_LOCK:
        pack = _PACKS.get(owner)
        if not pack:
            return []
        return sorted((pack.get(bucket) or {}).keys(), key=str.casefold)


def has_pack(owner_user_id: str | None = None) -> bool:
    owner = (owner_user_id or get_request_owner() or "").strip()
    if not owner:
        return False
    with _PACK_LOCK:
        return owner in _PACKS
