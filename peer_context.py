"""
Peer NPC prompt helpers: co-located directory + on-demand public_facts.
"""

from __future__ import annotations

import re
from typing import Any

from content_pack import get_pack, lookup_document
from npc_pronouns import get_npc_pronoun_context
from prompt_caps import get_prompt_caps
from session_context import get_request_campaign, get_request_owner


def _truncate(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 1:
        return text[:max_chars]
    return text[: max_chars - 1].rstrip() + "…"


def _npc_id_of(doc: dict[str, Any], fallback: str = "") -> str:
    return str(doc.get("npc_id") or doc.get("id") or fallback or "").strip()


def _location_id_of(doc: dict[str, Any]) -> str:
    return str(doc.get("location_id") or "").strip()


def _aliases_for(npc_id: str, doc: dict[str, Any]) -> list[str]:
    aliases: list[str] = []
    for value in (npc_id, doc.get("name"), doc.get("role")):
        text = str(value or "").strip()
        if text:
            aliases.append(text)
    # Deduplicate case-insensitively while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for alias in aliases:
        key = alias.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(alias)
    return out


def _pack_npc_docs() -> dict[str, dict[str, Any]]:
    pack = get_pack(get_request_owner(), get_request_campaign())
    if not pack:
        return {}
    npcs = pack.get("npcs") or {}
    return npcs if isinstance(npcs, dict) else {}


def list_colocated_npcs(
    location_id: str,
    *,
    exclude_npc_id: str = "",
) -> list[tuple[str, dict[str, Any]]]:
    location_id = (location_id or "").strip()
    if not location_id:
        return []

    exclude = (exclude_npc_id or "").strip().casefold()
    pack = get_pack(get_request_owner(), get_request_campaign())
    results: list[tuple[str, dict[str, Any]]] = []

    if pack and isinstance(pack.get("indexes"), dict):
        ids = (pack["indexes"].get("npc_ids_by_location") or {}).get(location_id.casefold()) or []
        npcs = pack.get("npcs") or {}
        for npc_id in ids:
            if str(npc_id).casefold() == exclude:
                continue
            doc = npcs.get(npc_id)
            if isinstance(doc, dict):
                results.append((str(npc_id), doc))
        if results:
            return results

    # Fallback scan when indexes are missing
    for key, doc in _pack_npc_docs().items():
        if not isinstance(doc, dict):
            continue
        npc_id = _npc_id_of(doc, key)
        if not npc_id or npc_id.casefold() == exclude:
            continue
        if _location_id_of(doc).casefold() != location_id.casefold():
            continue
        results.append((npc_id, doc))
    return results


def match_peer_npc_ids(
    player_text: str | None,
    location_id: str,
    *,
    exclude_npc_id: str = "",
    about_npc_id: str | None = None,
    max_targets: int = 1,
) -> list[str]:
    """Resolve peer NPC ids from an explicit about_npc_id and/or player text mentions."""
    max_targets = max(0, int(max_targets))
    if max_targets == 0:
        return []

    colocated = list_colocated_npcs(location_id, exclude_npc_id=exclude_npc_id)
    if not colocated:
        return []

    by_id = {npc_id.casefold(): npc_id for npc_id, _ in colocated}
    explicit = (about_npc_id or "").strip()
    matched: list[str] = []
    if explicit and explicit.casefold() in by_id:
        matched.append(by_id[explicit.casefold()])

    text = (player_text or "").strip()
    if text and len(matched) < max_targets:
        # Longest alias first so "Mayor Halden" beats "Halden"
        candidates: list[tuple[str, str]] = []
        for npc_id, doc in colocated:
            for alias in _aliases_for(npc_id, doc):
                if len(alias) < 3:
                    continue
                candidates.append((alias, npc_id))
        candidates.sort(key=lambda item: len(item[0]), reverse=True)

        lowered = text.casefold()
        for alias, npc_id in candidates:
            if npc_id in matched:
                continue
            pattern = r"(?<!\w)" + re.escape(alias.casefold()) + r"(?!\w)"
            if re.search(pattern, lowered, flags=re.IGNORECASE):
                matched.append(npc_id)
            if len(matched) >= max_targets:
                break

    return matched[:max_targets]


def format_directory_block(
    location_id: str,
    *,
    exclude_npc_id: str = "",
    caps: dict[str, int] | None = None,
) -> str:
    caps = caps or get_prompt_caps()
    max_npcs = caps.get("directory_max_npcs", 8)
    line_max = caps.get("directory_line_max_chars", 80)
    if max_npcs <= 0:
        return ""

    lines: list[str] = []
    for npc_id, doc in list_colocated_npcs(location_id, exclude_npc_id=exclude_npc_id):
        name = str(doc.get("name") or npc_id).strip()
        role = str(doc.get("role") or "").strip()
        line = f"{name} ({npc_id})" + (f" - {role}" if role else "")
        lines.append(f"- {_truncate(line, line_max)}")
        if len(lines) >= max_npcs:
            break

    if not lines:
        return ""
    return "People commonly known to be around here:\n" + "\n".join(lines)


def format_peer_public_facts_block(
    peer_ids: list[str],
    *,
    caps: dict[str, int] | None = None,
) -> str:
    caps = caps or get_prompt_caps()
    budget = caps.get("peer_public_facts_max_chars", 650)
    if budget <= 0 or not peer_ids:
        return ""

    sections: list[str] = []
    used = 0
    for peer_id in peer_ids:
        doc = lookup_document("npc", peer_id)
        if not isinstance(doc, dict):
            continue
        name = str(doc.get("name") or peer_id).strip()
        identity = get_npc_pronoun_context(doc)
        facts = doc.get("public_facts") or []
        if not isinstance(facts, list):
            facts = []
        fact_lines = [f"- {str(item).strip()}" for item in facts if str(item).strip()]
        if not fact_lines:
            fact_lines = ["- (no public facts listed)"]

        header = f"Public knowledge about {name} ({peer_id}):"
        identity_lines = (
            f"- Gender/identity: {identity['gender']}\n"
            f"- Pronouns: {identity['display']}\n"
            f"- Pronoun details: subject={identity['subject']}, object={identity['object']}, "
            f"possessive adjective={identity['possessive_adjective']}, "
            f"possessive pronoun={identity['possessive_pronoun']}, "
            f"reflexive={identity['reflexive']}"
        )
        body = "\n".join(fact_lines)
        block = f"{header}\n{identity_lines}\nPublic facts:\n{body}"
        remaining = budget - used
        if remaining <= 0:
            break
        if len(block) > remaining:
            block = _truncate(block, remaining)
        sections.append(block)
        used += len(block) + 2

    if not sections:
        return ""

    rules = (
        "Peer knowledge rules: You may share only these public facts and identity details "
        "about other NPCs. Do not invent private knowledge. Other NPCs' secrets and "
        "known_facts are unavailable unless listed here. "
        "Peer gender/pronoun lines describe THAT other NPC only when you refer to them "
        "in third person. They must NEVER replace, override, or blend into YOUR "
        "gender/identity and pronouns listed under 'NPC name' at the top of this prompt. "
        "Always speak as yourself using your own pronouns for first-person self-reference."
    )
    return "\n\n".join(sections) + "\n\n" + rules


def build_peer_location_prompt_sections(
    speaker_npc: dict[str, Any],
    *,
    player_text: str | None = None,
    about_npc_id: str | None = None,
    caps: dict[str, int] | None = None,
) -> tuple[str, str]:
    """
    Returns (directory_block, peer_public_facts_block) for the speaker's location.
    """
    caps = caps or get_prompt_caps()
    speaker_id = _npc_id_of(speaker_npc)
    location_id = _location_id_of(speaker_npc)
    directory = format_directory_block(location_id, exclude_npc_id=speaker_id, caps=caps)
    peer_ids = match_peer_npc_ids(
        player_text,
        location_id,
        exclude_npc_id=speaker_id,
        about_npc_id=about_npc_id,
        max_targets=caps.get("peer_max_targets", 1),
    )
    peer_facts = format_peer_public_facts_block(peer_ids, caps=caps)
    return directory, peer_facts
