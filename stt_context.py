"""Build concise RPG vocabulary hints for STT (keyterms + context prompt).

Uses already-loaded NPC / location / peer documents — no secret fields.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from stt_capabilities import STTModelCapabilities

_DEFAULT_CONTEXT_PROMPT = (
    "This is speech from a tabletop RPG session.\n\n"
    "Preserve character names, location names, fictional organizations, "
    "fantasy/science-fiction terminology, and other proper nouns exactly when "
    "they appear in the supplied context.\n\n"
    "Do not convert fictional terms into similar common English words."
)

_MIN_TERM_LEN = 3
_DEFAULT_MAX_TERMS = 80


@dataclass
class STTContext:
    keyterms: list[str] = field(default_factory=list)
    prompt: str = ""
    language: str = "en"


def _add_term(terms: list[str], seen: set[str], value: Any) -> None:
    text = str(value or "").strip()
    if len(text) < _MIN_TERM_LEN:
        return
    key = text.casefold()
    if key in seen:
        return
    seen.add(key)
    terms.append(text)


def _add_name_parts(terms: list[str], seen: set[str], name: str) -> None:
    _add_term(terms, seen, name)
    parts = [p for p in re.split(r"[\s,]+", name or "") if p]
    if len(parts) >= 2:
        # Common short form: last token (Komiskey from Theodora Komiskey).
        _add_term(terms, seen, parts[-1])
        if len(parts[0]) >= _MIN_TERM_LEN:
            _add_term(terms, seen, parts[0])


def _terms_from_list_field(doc: dict[str, Any], *keys: str) -> list[str]:
    out: list[str] = []
    for key in keys:
        raw = doc.get(key)
        if isinstance(raw, str) and raw.strip():
            out.append(raw.strip())
        elif isinstance(raw, (list, tuple)):
            for item in raw:
                if isinstance(item, str) and item.strip():
                    out.append(item.strip())
                elif isinstance(item, dict):
                    for nested_key in ("name", "label", "title", "id"):
                        val = item.get(nested_key)
                        if isinstance(val, str) and val.strip():
                            out.append(val.strip())
                            break
    return out


def build_stt_context(
    *,
    npc: dict[str, Any] | None = None,
    location: dict[str, Any] | None = None,
    colocated_npcs: list[tuple[str, dict[str, Any]]] | None = None,
    campaign_id: str = "",
    campaign: dict[str, Any] | None = None,
    language: str = "en",
    max_keyterms: int = _DEFAULT_MAX_TERMS,
    include_prompt: bool = True,
) -> STTContext:
    """Assemble a small, non-secret STT vocabulary set for the active scene."""
    terms: list[str] = []
    seen: set[str] = set()
    npc = npc or {}
    location = location or {}
    campaign = campaign or {}
    max_keyterms = max(1, min(int(max_keyterms or _DEFAULT_MAX_TERMS), 100))

    npc_name = str(npc.get("name") or "").strip()
    if npc_name:
        _add_name_parts(terms, seen, npc_name)
    for alias in _terms_from_list_field(npc, "aliases", "aka", "also_known_as"):
        _add_name_parts(terms, seen, alias)

    loc_name = str(location.get("name") or location.get("title") or "").strip()
    if loc_name:
        _add_term(terms, seen, loc_name)
    loc_id = str(location.get("location_id") or location.get("id") or "").strip()
    if loc_id and "_" not in loc_id:
        _add_term(terms, seen, loc_id.replace("-", " "))

    for sub in _terms_from_list_field(
        location,
        "sublocations",
        "areas",
        "rooms",
        "landmarks",
    ):
        _add_term(terms, seen, sub)

    camp_name = str(campaign.get("name") or campaign.get("title") or "").strip()
    if camp_name:
        _add_term(terms, seen, camp_name)
    elif campaign_id:
        pretty = campaign_id.replace("_", " ").replace("-", " ").strip()
        if pretty and pretty.casefold() != campaign_id.casefold():
            _add_term(terms, seen, pretty.title())

    for public_term in _terms_from_list_field(
        campaign,
        "factions",
        "organizations",
        "species",
        "terminology",
        "setting_terms",
    ):
        _add_term(terms, seen, public_term)
    for public_term in _terms_from_list_field(
        location,
        "factions",
        "organizations",
        "species",
        "terminology",
    ):
        _add_term(terms, seen, public_term)

    peer_cap = 12
    for npc_id, doc in (colocated_npcs or [])[:peer_cap]:
        peer_name = str((doc or {}).get("name") or npc_id or "").strip()
        if peer_name:
            _add_name_parts(terms, seen, peer_name)

    terms = terms[:max_keyterms]

    prompt = ""
    if include_prompt and terms:
        prompt = _DEFAULT_CONTEXT_PROMPT + "\n\nRelevant terms:\n" + "\n".join(
            f"- {term}" for term in terms
        )

    return STTContext(keyterms=terms, prompt=prompt, language=(language or "en").strip() or "en")


def apply_context_to_request(
    payload: dict[str, Any],
    context: STTContext | None,
    caps: STTModelCapabilities,
    *,
    enable_keyterms: bool = True,
    enable_context: bool = True,
) -> dict[str, Any]:
    """Mutate/return OpenRouter transcription payload with gated optional fields."""
    if context is None:
        return payload

    if caps.supports_language and context.language:
        payload["language"] = context.language

    keyterms = list(context.keyterms or [])
    if enable_keyterms and caps.supports_keyterms and keyterms:
        clipped = keyterms[: max(1, caps.max_keyterms)]
        if caps.use_top_level_keyterms:
            payload["keyterms"] = clipped
        elif caps.provider_options_slug and caps.keywords_option_key:
            provider = payload.setdefault("provider", {})
            options = provider.setdefault("options", {})
            slug_opts = options.setdefault(caps.provider_options_slug, {})
            slug_opts[caps.keywords_option_key] = clipped

    if (
        enable_context
        and caps.supports_context_prompt
        and context.prompt
        and caps.provider_options_slug
        and caps.context_option_key
    ):
        provider = payload.setdefault("provider", {})
        options = provider.setdefault("options", {})
        slug_opts = options.setdefault(caps.provider_options_slug, {})
        slug_opts[caps.context_option_key] = context.prompt

    return payload
