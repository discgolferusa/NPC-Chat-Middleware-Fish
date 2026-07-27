"""
Campaign document import: turn a pasted campaign document (module notes,
session prep, wiki export, etc.) into structured campaign/location/NPC data
using multi-pass LLM calls against the "content_creation" purpose model.

This module never writes files or touches content packs — it only returns
structured dicts (shaped like the campaign/location/npc YAML schemas used
elsewhere in this repo) for the caller to review and persist.

Pipeline:
  1. Outline pass — one LLM call that reads the whole document and proposes
     the campaign, a location list, and an NPC list (name/role/locations).
  2. Location detail pass — small batches (<= MAX_LOCATIONS_PER_BATCH) using
     name-focused source excerpts; extract-over-invent; retry if sparse.
  3. NPC detail pass — one NPC per call with name-focused excerpts; maps
     NPCNarrator section headers and published RPG stat blocks (e.g. ALIEN RPG
     FULL NAME / PERSONAL AGENDA) into voice/personality/facts/secrets/llm_rules;
     retry if sparse.

When a NON-PLAYER CHARACTERS section is detected, stat-block names override
bad LLM outline guesses (see test_content_import.py / Chariot of the Gods).
"""

from __future__ import annotations

import json
import re
from typing import Any

from llm_router import (
    LLMProviderError,
    content_creation_max_input_chars,
    content_creation_zero_budget_error_message,
    generate_llm_response,
)

MAX_INPUT_CHARS = 120_000
# Small batches: local 8B models (and Groq fallbacks) often return empty/partial JSON
# when asked for many rich NPCs at once. One NPC per call preserves source detail.
MAX_LOCATIONS_PER_BATCH = 2
MAX_NPCS_PER_BATCH = 1
IMPORT_TEMPERATURE = 0.15
IMPORT_MAX_TOKENS = 6144
# NPCNarrator writeups put the name in a header, then many blocks after it.
# Bias the window forward so Knows/Hides/Memory/Finale are kept.
EXCERPT_WINDOW_BEFORE = 400
EXCERPT_WINDOW_AFTER = 7_000
EXCERPT_MAX_CHARS = 28_000
_EXCERPT_STOPWORDS = frozenset(
    {
        "a", "an", "and", "at", "by", "de", "for", "from", "in", "into", "la", "le",
        "npc", "of", "on", "or", "the", "to", "van", "von", "with",
    }
)

# Published module headers (PDFs often space letters: N O N - P L A Y E R).
_NPC_SECTION_PATTERN = re.compile(
    r"non\s*[- ]?\s*player[\s\n]+characters?",
    re.IGNORECASE,
)
_NPC_SECTION_END_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:auxiliary\s+vehicles|the\s+deck\s+plans|appendix\b|index\b|maps?\b)",
    re.IGNORECASE,
)
_STAT_BLOCK_HEADING = re.compile(
    r"^([A-Z][A-Z0-9][A-Z0-9 '\-]*?)\s*,\s*([A-Z][A-Z0-9][A-Z0-9 '\-/]*)$",
)
# Movie/sidebar names that are not scenario NPCs (Chariot of the Gods regression).
_KNOWN_IMPORT_FALSE_POSITIVE_NAMES = frozenset(
    {
        "captain miller",
        "miller",
        "david 8",
        "david",
        "miss vickers",
        "vickers",
    }
)

VALID_MODES = frozenset({"create_new", "merge"})

_FENCE_PATTERN = re.compile(r"```(?:json|yaml)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_SNAKE_INVALID = re.compile(r"[^a-z0-9]+")
_SNAKE_COLLAPSE = re.compile(r"_+")


class ContentImportError(Exception):
    """Raised when the document cannot be analyzed into structured content."""


# --------------------------------------------------------------------------
# JSON / text helpers
# --------------------------------------------------------------------------


def _strip_code_fences(text: str) -> str:
    match = _FENCE_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _find_balanced_json(text: str, open_ch: str, close_ch: str) -> Any | None:
    start = text.find(open_ch)
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    snippet = text[start : i + 1]
                    try:
                        return json.loads(snippet)
                    except json.JSONDecodeError:
                        break
        start = text.find(open_ch, start + 1)
    return None


def extract_json(text: str) -> Any:
    """Best-effort JSON extraction from an LLM completion.

    Handles markdown code fences, leading/trailing prose around the JSON, and
    falls back to scanning for the first balanced {...} or [...] block.
    """
    if not text or not text.strip():
        raise ContentImportError("Empty LLM response")

    candidates = [_strip_code_fences(text), text.strip()]
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    for candidate in candidates:
        for open_ch, close_ch in (("{", "}"), ("[", "]")):
            result = _find_balanced_json(candidate, open_ch, close_ch)
            if result is not None:
                return result

    raise ContentImportError(f"Could not extract JSON from LLM response: {text[:200]!r}")


def snake_case(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = _SNAKE_INVALID.sub("_", text)
    text = _SNAKE_COLLAPSE.sub("_", text).strip("_")
    return text or "unnamed"


def _unique_id(base: str, taken: set[str]) -> str:
    base = base or "unnamed"
    candidate = base
    suffix = 2
    while candidate in taken:
        candidate = f"{base}_{suffix}"
        suffix += 1
    taken.add(candidate)
    return candidate


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [value]


def _as_str_list(value: Any) -> list[str]:
    return [str(item).strip() for item in _as_list(value) if str(item).strip()]


def _batches(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)] if items else []


def _collapse_spaced_caps(text: str) -> str:
    """Collapse PDF letter-spacing (N O N - P L A Y E R -> NON-PLAYER)."""
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and re.fullmatch(r"[A-Z0-9](?:\s+[A-Z0-9-])+", stripped):
            lines.append(re.sub(r"\s+", "", stripped))
            continue
        lines.append(line)
    return "\n".join(lines)


def _find_npc_section_span(collapsed_text: str) -> tuple[int, int] | None:
    """Return [start, end) slice of the authoritative NPC section when present.

    ``collapsed_text`` should already be passed through ``_collapse_spaced_caps``.
    """
    if not collapsed_text:
        return None
    match = _NPC_SECTION_PATTERN.search(collapsed_text)
    if not match:
        return None
    start = match.start()
    remainder = collapsed_text[match.end() :]
    end_match = _NPC_SECTION_END_PATTERN.search(remainder)
    end = match.end() + end_match.start() if end_match else len(collapsed_text)
    return start, end


def _extract_alien_rpg_stat_block_npcs(section_text: str) -> list[dict[str, str]]:
    """Parse ALIEN RPG module stat blocks (JOHNS, OFFICER + FULL NAME lines)."""
    if not section_text.strip():
        return []

    lines = section_text.splitlines()
    results: list[dict[str, str]] = []
    seen_names: set[str] = set()

    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        segments = re.split(r"\t+", line) if "\t" in line else [line]
        for segment in segments:
            segment = segment.strip()
            if not _STAT_BLOCK_HEADING.match(segment):
                continue

            short_name = segment.split(",", 1)[0].strip()
            descriptor = segment.split(",", 1)[1].strip() if "," in segment else ""
            window = "\n".join(lines[index : index + 30])
            full_name_match = re.search(r"FULL NAME:\s*(.+)", window)
            if not full_name_match:
                # ALIEN RPG stat blocks always publish FULL NAME; skip generic headings.
                continue
            display_name = full_name_match.group(1).strip()
            name_key = display_name.casefold()
            if name_key in seen_names:
                continue

            role = descriptor.replace("/", " ").title()
            for look_ahead in lines[index + 1 : index + 4]:
                candidate = look_ahead.strip()
                if not candidate or candidate.startswith("FULL NAME:"):
                    continue
                if _STAT_BLOCK_HEADING.match(candidate):
                    break
                if not re.match(r"^(AGE|PERSONALITY|STRENGTH|HEALTH|SKILLS)\b", candidate):
                    role = candidate
                    break

            location_hint = ""
            if re.search(r"\bSotillo\b", window, re.IGNORECASE):
                location_hint = "sotillo"
            elif re.search(r"\bCronus\b", window, re.IGNORECASE):
                location_hint = "uscss_cronus"

            seen_names.add(name_key)
            results.append(
                {
                    "name": display_name,
                    "role": role,
                    "short_name": short_name,
                    "location_hint": location_hint,
                }
            )

    return results


def _prepare_outline_document_text(text: str, warnings: list[str]) -> str:
    """Bias the outline pass toward the module's NPC section when present."""
    collapsed = _collapse_spaced_caps(text)
    span = _find_npc_section_span(collapsed)
    if span is None:
        return text

    start, end = span
    intro = collapsed[: min(start, 2500)].strip()
    npc_section = collapsed[start:end].strip()
    warnings.append(
        "Detected a NON-PLAYER CHARACTERS section; outline will prioritize that "
        "section over names from intro text, quotes, or player-facing material."
    )
    if intro:
        return f"{intro}\n\n--- NON-PLAYER CHARACTERS (authoritative NPC list) ---\n\n{npc_section}"
    return npc_section


def _truncate_preserving_npc_section(text: str, max_chars: int) -> str:
    """Shrink long documents but keep the NON-PLAYER CHARACTERS section when present."""
    if len(text) <= max_chars:
        return text

    collapsed = _collapse_spaced_caps(text)
    span = _find_npc_section_span(collapsed)
    if span is None:
        return text[:max_chars]

    start, end = span
    section = collapsed[start:end].strip()
    separator = "\n\n--- NON-PLAYER CHARACTERS (priority excerpt) ---\n\n"
    if len(section) >= max_chars:
        return section[:max_chars]

    intro_budget = max_chars - len(section) - len(separator)
    if intro_budget > 0:
        intro_source = collapsed[:start].strip()
        intro = intro_source[:intro_budget]
        combined = f"{intro}{separator}{section}" if intro else section
        return combined[:max_chars]

    return section[:max_chars]


def _npc_stubs_from_stat_blocks(
    section_text: str,
    *,
    location_ids: set[str],
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Build outline NPC stubs from ALIEN RPG stat blocks."""
    parsed = _extract_alien_rpg_stat_block_npcs(section_text)
    if not parsed:
        return []

    def _resolve_hint(hint: str) -> str:
        if not location_ids:
            return ""
        if hint in location_ids:
            return hint
        for loc_id in location_ids:
            if hint in loc_id or loc_id in hint:
                return loc_id
        return next(iter(location_ids), "")

    stubs: list[dict[str, Any]] = []
    for row in parsed:
        loc = _resolve_hint(row.get("location_hint") or "")
        stubs.append(
            {
                "id": snake_case(row["name"]),
                "name": row["name"],
                "role": row.get("role") or "",
                "first_location_id": loc,
                "other_location_ids": [],
            }
        )

    warnings.append(
        f"Parsed {len(stubs)} NPC(s) from published stat blocks in NON-PLAYER CHARACTERS."
    )
    return stubs


def _npc_already_exists_in_campaign(
    name: str,
    *,
    mode: str,
    existing_npc_names: list[str],
) -> bool:
    if mode != "merge":
        return False
    key = name.casefold()
    existing_name_keys = {n.casefold() for n in existing_npc_names if n}
    existing_id_keys = {snake_case(n) for n in existing_npc_names if n}
    base_id = snake_case(name)
    return key in existing_name_keys or base_id in existing_id_keys


def _merge_outline_npcs_with_stat_blocks(
    llm_npcs: list[dict[str, Any]],
    document_text: str,
    *,
    mode: str,
    existing_npc_names: list[str],
    location_ids: set[str],
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Merge ALIEN RPG stat-block NPCs with outline stubs when confidently detected."""
    collapsed = _collapse_spaced_caps(document_text)
    span = _find_npc_section_span(collapsed)
    if span is None:
        return llm_npcs

    start, end = span
    section_text = collapsed[start:end]
    stat_stubs = _npc_stubs_from_stat_blocks(
        section_text,
        location_ids=location_ids,
        warnings=warnings,
    )
    if len(stat_stubs) < 3:
        return llm_npcs

    filtered_stat_stubs: list[dict[str, Any]] = []
    for stub in stat_stubs:
        name = str(stub.get("name") or "").strip()
        if not name:
            continue
        if _npc_already_exists_in_campaign(
            name, mode=mode, existing_npc_names=existing_npc_names
        ):
            warnings.append(
                f"NPC '{name}' already exists in the campaign; "
                "reusing existing character and skipping stat-block regeneration."
            )
            continue
        filtered_stat_stubs.append(stub)
    stat_stubs = filtered_stat_stubs
    if len(stat_stubs) < 3:
        return llm_npcs

    stat_names = {stub["name"].casefold() for stub in stat_stubs}
    dropped: list[str] = []
    kept_llm: list[dict[str, Any]] = []
    for npc in llm_npcs:
        name = str(npc.get("name") or "").strip()
        if not name:
            continue
        key = name.casefold()
        if key in stat_names:
            continue
        if key in _KNOWN_IMPORT_FALSE_POSITIVE_NAMES or any(
            bad in key for bad in ("miller", "vickers", "david 8")
        ):
            dropped.append(name)
            continue
        kept_llm.append(npc)

    if dropped:
        warnings.append(
            "Ignored outline NPC(s) outside the stat-block section: "
            + ", ".join(dropped)
        )

    return kept_llm + stat_stubs


_PRONOUN_OBJECTS: dict[str, dict[str, str]] = {
    "he/him": {
        "subject": "he",
        "object": "him",
        "possessive_adjective": "his",
        "possessive_pronoun": "his",
        "reflexive": "himself",
    },
    "she/her": {
        "subject": "she",
        "object": "her",
        "possessive_adjective": "her",
        "possessive_pronoun": "hers",
        "reflexive": "herself",
    },
    "they/them": {
        "subject": "they",
        "object": "them",
        "possessive_adjective": "their",
        "possessive_pronoun": "theirs",
        "reflexive": "themselves",
    },
}


def _normalize_pronouns(value: Any, gender: str) -> dict[str, str]:
    """Return structured pronouns the Yaml Editor can persist.

    Avoid substring checks like ``\"he\" in text`` — that matches inside ``the``.
    Prefer slash-forms and whole-word pronoun tokens only.
    """
    if isinstance(value, dict):
        subject = str(value.get("subject") or "").strip().lower()
        obj = str(value.get("object") or "").strip().lower()
        if subject and obj:
            return {
                "subject": subject,
                "object": obj,
                "possessive_adjective": str(value.get("possessive_adjective") or "").strip().lower()
                or ("her" if subject == "she" else "his" if subject == "he" else "their"),
                "possessive_pronoun": str(value.get("possessive_pronoun") or "").strip().lower()
                or ("hers" if subject == "she" else "his" if subject == "he" else "theirs"),
                "reflexive": str(value.get("reflexive") or "").strip().lower()
                or ("herself" if subject == "she" else "himself" if subject == "he" else "themselves"),
            }

    text = str(value or "").strip().lower()
    compact = re.sub(r"\s+", "", text)
    for key, obj in _PRONOUN_OBJECTS.items():
        head = key.split("/")[0]
        if compact == key or compact.startswith(head + "/"):
            return dict(obj)

    tokens = set(re.findall(r"[a-z]+", text))
    if tokens & {"she", "her", "hers", "herself"}:
        return dict(_PRONOUN_OBJECTS["she/her"])
    if tokens & {"they", "them", "their", "theirs", "themselves"}:
        return dict(_PRONOUN_OBJECTS["they/them"])
    if tokens & {"he", "him", "his", "himself"}:
        return dict(_PRONOUN_OBJECTS["he/him"])

    gender_l = (gender or "").strip().lower()
    if gender_l in {"female", "woman", "f"}:
        return dict(_PRONOUN_OBJECTS["she/her"])
    if gender_l in {"male", "man", "m"}:
        return dict(_PRONOUN_OBJECTS["he/him"])
    return dict(_PRONOUN_OBJECTS["they/them"])


def _distinctive_name_needles(name: str, entity_id: str = "") -> list[str]:
    """Needles safe for excerpt search — full name/id plus distinctive tokens only."""
    needles: list[str] = []
    for raw in (name, entity_id):
        value = str(raw or "").strip()
        if value:
            needles.append(value)

    for part in re.split(r"[\s_/]+", str(name or "").strip()):
        cleaned = part.strip("-").strip()
        if len(cleaned) < 4:
            continue
        if cleaned.casefold() in _EXCERPT_STOPWORDS:
            continue
        # Skip pure numeric fragments.
        if cleaned.isdigit():
            continue
        needles.append(cleaned)

    # Prefer longer / more specific needles first when budget is tight.
    unique: list[str] = []
    seen: set[str] = set()
    for needle in sorted(needles, key=lambda n: len(n), reverse=True):
        key = needle.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(needle)
    return unique


def _mention_indexes(haystack_cf: str, needle: str) -> list[int]:
    """Casefolded bounded substring matches (not bare 'he' inside 'the')."""
    v = str(needle or "").strip().casefold()
    if len(v) < 3:
        return []
    pattern = re.compile(rf"(?<![a-z0-9]){re.escape(v)}(?![a-z0-9])")
    return [match.start() for match in pattern.finditer(haystack_cf)]


def _relevant_excerpts(
    text: str,
    needles: list[str],
    *,
    window_before: int = EXCERPT_WINDOW_BEFORE,
    window_after: int = EXCERPT_WINDOW_AFTER,
) -> str:
    """Pull document windows around name/id mentions so detail passes stay focused."""
    if not text or not needles:
        return text

    lower = text.casefold()
    # Prefer long/specific needles so short tokens cannot flood the budget.
    ranked = sorted(
        {str(n).strip() for n in needles if str(n).strip()},
        key=lambda n: len(n),
        reverse=True,
    )
    primary = [n for n in ranked if len(n) >= 6 or any(ch in n for ch in (" ", "-", "_"))]
    search_needles = primary or [n for n in ranked if len(n) >= 4]

    spans: list[tuple[int, int, int]] = []  # start, end, needle_len
    for needle in search_needles:
        for idx in _mention_indexes(lower, needle):
            start = max(0, idx - window_before)
            end = min(len(text), idx + len(needle) + window_after)
            spans.append((start, end, len(needle)))

    if not spans:
        return text[: min(len(text), EXCERPT_MAX_CHARS)]

    # Keep matches from the longest needles when they exist.
    best_len = max(span[2] for span in spans)
    if best_len >= 6:
        spans = [span for span in spans if span[2] >= max(6, best_len - 4)]

    spans.sort(key=lambda s: s[0])
    merged: list[list[int]] = []
    for start, end, _needle_len in spans:
        if not merged or start > merged[-1][1] + 80:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    chunks: list[str] = []
    total = 0
    for start, end in merged:
        piece = text[start:end].strip()
        if not piece:
            continue
        if total + len(piece) > EXCERPT_MAX_CHARS:
            remain = EXCERPT_MAX_CHARS - total
            if remain > 400:
                chunks.append(piece[:remain])
            break
        chunks.append(piece)
        total += len(piece)

    return "\n\n---\n\n".join(chunks) if chunks else text[: min(len(text), EXCERPT_MAX_CHARS)]


def _npc_detail_is_sparse(detail: dict[str, Any] | None) -> bool:
    if not detail:
        return True
    filled = 0
    for key in ("personality", "public_facts", "known_facts", "secrets", "llm_rules", "voice"):
        if _as_str_list(detail.get(key)):
            filled += 1
    if str(detail.get("race") or "").strip().lower() not in {"", "unknown"}:
        filled += 1
    return filled < 2


def _location_detail_is_sparse(detail: dict[str, Any] | None) -> bool:
    if not detail:
        return True
    if str(detail.get("summary") or "").strip():
        return False
    for key in ("current_events", "rumors", "local_tone", "important_locations", "knowledge_rules"):
        if _as_list(detail.get(key)):
            return False
    return True


def _coerce_container(data: Any, key: str) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        # One-item batch often collapses to {"locations": {...}} / {"npcs": {...}}.
        if isinstance(value, dict):
            return [value]
        # Bare object at the top level (single entity without the wrapper key).
        if "id" in data or "name" in data:
            return [data]
    return []


def _index_details_by_stub_id(
    detail_items: list[dict[str, Any]],
    stubs: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Map detail objects onto stub ids, including one-item bare-object batches.

    Detail passes often echo ids with different casing/spacing than the outline
    stubs (which are always snake_case). Normalize before indexing so lookups
    do not fall through to empty defaults.
    """
    details_by_id: dict[str, dict[str, Any]] = {}
    for item in detail_items:
        raw_id = str(item.get("id") or "").strip()
        if raw_id:
            details_by_id[snake_case(raw_id)] = item
        # NPCs frequently return the display name as id; index by name too.
        raw_name = str(item.get("name") or "").strip()
        if raw_name:
            details_by_id.setdefault(snake_case(raw_name), item)

    if len(stubs) == 1 and len(detail_items) == 1 and stubs[0]["id"] not in details_by_id:
        details_by_id[stubs[0]["id"]] = detail_items[0]
        return details_by_id

    # Positional fallback when counts match and some stubs still unresolved
    # (model returned fields but mismatched/missing ids).
    if len(detail_items) == len(stubs) and len(stubs) > 1:
        for stub, item in zip(stubs, detail_items):
            stub_id = str(stub.get("id") or "")
            if stub_id and stub_id not in details_by_id:
                details_by_id[stub_id] = item

    return details_by_id


def _call_llm_json(
    system_prompt: str,
    user_prompt: str,
    *,
    warnings: list[str],
    step: str,
) -> Any:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    raw = generate_llm_response(
        messages,
        purpose="content_creation",
        max_tokens=IMPORT_MAX_TOKENS,
        temperature=IMPORT_TEMPERATURE,
    )
    try:
        return extract_json(raw)
    except ContentImportError as ex:
        warnings.append(f"{step}: {ex}")
        raise


_SYSTEM_PROMPT = (
    "You are a tabletop RPG campaign content structuring assistant for NPCNarrator. "
    "You convert campaign documents into strict JSON matching the requested schema. "
    "Prefer extracting and preserving concrete facts, wording, goals, secrets, and "
    "voice notes from the source text over inventing new content. "
    "Respond with ONLY the JSON value — no markdown code fences, no commentary."
)


# --------------------------------------------------------------------------
# Pass 1: outline
# --------------------------------------------------------------------------


def _build_outline_prompt(
    text: str,
    mode: str,
    existing_campaign_id: str | None,
    existing_location_ids: list[str],
    existing_npc_names: list[str],
) -> str:
    if mode == "merge" and existing_campaign_id:
        mode_guidance = (
            f"This document is being merged into the EXISTING campaign \"{existing_campaign_id}\". "
            f"Set campaign.campaign_id to exactly \"{existing_campaign_id}\".\n"
        )
        if existing_location_ids:
            mode_guidance += (
                "These location ids already exist in the campaign: "
                f"{', '.join(existing_location_ids)}. If the document clearly describes one of "
                "these same locations, reuse that exact id; otherwise invent a new distinct id.\n"
            )
        if existing_npc_names:
            mode_guidance += (
                "These NPC names already exist in the campaign: "
                f"{', '.join(existing_npc_names)}. If the document clearly describes one of these "
                "same NPCs, reuse that exact name; otherwise invent a new distinct name.\n"
            )
    else:
        mode_guidance = (
            "This is a brand new campaign. Invent a short, memorable campaign_id and name from "
            "the document's title or setting if one is not explicitly given.\n"
        )

    return (
        "Read the campaign document text below and produce an outline as JSON with exactly "
        "this shape:\n\n"
        "{\n"
        '  "campaign": {"campaign_id": "snake_case_id", "name": "Display Name", '
        '"description": "1-3 sentence summary"},\n'
        '  "locations": [{"id": "snake_case_id", "name": "Display Name", '
        '"type": "city|town|dungeon|forest|building|region|other"}],\n'
        '  "npcs": [{"name": "Display Name", "role": "short role or title", '
        '"first_location_id": "snake_case id of the location where this NPC is first '
        'encountered", "other_location_ids": ["snake_case_id", ...]}]\n'
        "}\n\n"
        "Rules:\n"
        "- Use snake_case for every id (lowercase letters, digits, underscores only).\n"
        "- Every id used as an NPC's first_location_id or in other_location_ids MUST also "
        "appear in the locations list.\n"
        "- If the document has a NON-PLAYER CHARACTERS (or NPCs / Supporting Cast) section, "
        "that section is the authoritative NPC list. List every stat-block NPC there and do "
        "NOT list names only mentioned in intro text, movie quotes, sidebars, or player "
        "(PC) material.\n"
        "- Do NOT list player characters, the PC party, or the PC ship crew as NPCs.\n"
        "- Published RPG stat blocks (e.g. lines like 'JOHNS, OFFICER' with FULL NAME, "
        "PERSONALITY, PERSONAL AGENDA) are always NPCs when inside the NPC section.\n"
        "- Include every named location and every named NPC from the authoritative NPC section; "
        "minor NPCs in that section still count.\n"
        "- other_location_ids should only be non-empty for NPCs who travel or appear in more "
        "than one place in the text.\n"
        f"{mode_guidance}"
        "- Output ONLY the JSON object described above. No prose, no markdown fences.\n\n"
        "Campaign document text:\n"
        '"""\n'
        f"{text}\n"
        '"""'
    )


def _normalize_outline(
    outline: dict[str, Any],
    *,
    mode: str,
    existing_campaign_id: str | None,
    existing_location_ids: list[str],
    existing_npc_names: list[str],
    warnings: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    campaign_raw = outline.get("campaign") if isinstance(outline.get("campaign"), dict) else {}

    if mode == "merge" and existing_campaign_id:
        campaign_id = existing_campaign_id.strip()
    else:
        campaign_id = snake_case(campaign_raw.get("campaign_id") or campaign_raw.get("name") or "campaign")

    campaign = {
        "campaign_id": campaign_id,
        "name": str(campaign_raw.get("name") or campaign_id.replace("_", " ").title()).strip(),
        "description": str(campaign_raw.get("description") or "").strip(),
    }

    existing_location_set: set[str] = {
        loc_id.strip() for loc_id in existing_location_ids if loc_id and loc_id.strip()
    }
    # Tracks ids already claimed by locations we will emit (write). Existing
    # campaign ids are intentionally NOT pre-seeded here so merge mode can reuse
    # them for NPC location refs without inventing greyhaven_2.
    taken_location_ids: set[str] = set()
    id_remap: dict[str, str] = {}
    locations: list[dict[str, Any]] = []

    for raw_loc in _as_list(outline.get("locations")):
        if not isinstance(raw_loc, dict):
            continue
        name = str(raw_loc.get("name") or "").strip()
        raw_id = str(raw_loc.get("id") or name or "location").strip()
        base_id = snake_case(raw_id)
        if not base_id:
            continue

        # Merge: model reused an existing campaign location — keep the id for
        # NPC refs, but do not re-emit it for writing (editor won't overwrite).
        if mode == "merge" and base_id in existing_location_set:
            id_remap[raw_id] = base_id
            if base_id not in id_remap:
                id_remap[base_id] = base_id
            warnings.append(
                f"Location id '{base_id}' already exists in the campaign; "
                "reusing it for references and skipping regeneration."
            )
            continue

        final_id = base_id
        if final_id in taken_location_ids:
            # Within-document collision (two outline rows share a snake_case id).
            # Keep id_remap[base_id] pointing at the FIRST location; only the
            # later raw_id maps to the renamed final_id.
            final_id = _unique_id(base_id, taken_location_ids)
            warnings.append(
                f"Location id '{base_id}' collided with another outline location; "
                f"renamed later entry to '{final_id}'."
            )
        else:
            taken_location_ids.add(final_id)

        id_remap[raw_id] = final_id
        # Prefer the first claimant of the snake_case key so NPC refs that use
        # the base id keep pointing at the original location.
        if base_id not in id_remap:
            id_remap[base_id] = final_id

        locations.append(
            {
                "id": final_id,
                "name": name or final_id.replace("_", " ").title(),
                "type": str(raw_loc.get("type") or "other").strip() or "other",
            }
        )

    # Existing locations remain valid NPC targets even when not re-emitted.
    known_location_ids = {loc["id"] for loc in locations} | existing_location_set

    def _resolve_location_ref(ref: str) -> str | None:
        ref = str(ref or "").strip()
        if not ref:
            return None
        if ref in id_remap:
            return id_remap[ref]
        candidate = snake_case(ref)
        if candidate in id_remap:
            return id_remap[candidate]
        if candidate in known_location_ids:
            return candidate
        return None

    existing_npc_name_keys: set[str] = {
        n.strip().casefold() for n in existing_npc_names if n and n.strip()
    }
    existing_npc_id_set: set[str] = {
        snake_case(n) for n in existing_npc_names if n and n.strip()
    }
    taken_npc_ids: set[str] = set()
    npcs: list[dict[str, Any]] = []

    for raw_npc in _as_list(outline.get("npcs")):
        if not isinstance(raw_npc, dict):
            continue
        name = str(raw_npc.get("name") or "").strip()
        if not name:
            continue
        base_id = snake_case(name)
        if not base_id:
            continue

        # Merge: reuse existing NPC identity — skip regeneration/write.
        if mode == "merge" and (
            name.casefold() in existing_npc_name_keys or base_id in existing_npc_id_set
        ):
            warnings.append(
                f"NPC '{name}' already exists in the campaign; "
                "reusing existing character and skipping regeneration."
            )
            continue

        final_id = base_id
        if final_id in taken_npc_ids:
            final_id = _unique_id(base_id, taken_npc_ids)
            warnings.append(
                f"NPC id '{base_id}' collided with another outline NPC; "
                f"renamed later entry to '{final_id}'."
            )
        else:
            taken_npc_ids.add(final_id)

        first_location_id = _resolve_location_ref(raw_npc.get("first_location_id"))
        other_location_ids: list[str] = []
        for ref in _as_list(raw_npc.get("other_location_ids")):
            resolved = _resolve_location_ref(ref)
            if resolved and resolved != first_location_id and resolved not in other_location_ids:
                other_location_ids.append(resolved)

        if raw_npc.get("first_location_id") and not first_location_id:
            warnings.append(
                f"NPC '{name}': could not resolve first_location_id "
                f"'{raw_npc.get('first_location_id')}' to a known location."
            )

        npcs.append(
            {
                "id": final_id,
                "name": name,
                "role": str(raw_npc.get("role") or "").strip(),
                "first_location_id": first_location_id or (locations[0]["id"] if locations else ""),
                "other_location_ids": other_location_ids,
            }
        )

    return campaign, locations, npcs


# --------------------------------------------------------------------------
# Pass 2: location detail
# --------------------------------------------------------------------------


def _build_location_detail_prompt(batch: list[dict[str, Any]], text: str, campaign: dict[str, Any]) -> str:
    location_list = "\n".join(f"- id: {loc['id']} | name: {loc['name']} | type: {loc['type']}" for loc in batch)
    return (
        f'Campaign: "{campaign["name"]}" ({campaign["campaign_id"]}).\n\n'
        "Fill in details for EACH location below from the SOURCE EXCERPTS. "
        'Respond with exactly {"locations": [ ... ]} where each item has this shape:\n\n'
        "{\n"
        '  "id": "<exact id from the list below>",\n'
        '  "summary": "faithful overview; keep concrete sensory and plot details from the source; '
        'use as many sentences as needed (typically 2-8)",\n'
        '  "current_events": ["ongoing situations from the source; include all that are stated"],\n'
        '  "important_locations": [{"name": "sub-location name", "description": "what the source says", '
        '"public_knowledge": "what ordinary people believe or know about it"}],\n'
        '  "rumors": ["rumors or uncertain talk from the source"],\n'
        '  "local_tone": ["mood/atmosphere descriptors from the source"],\n'
        '  "knowledge_rules": ["rules about what NPCs here generally do or do not know"]\n'
        "}\n\n"
        "CRITICAL EXTRACTION RULES:\n"
        "- Prefer copying concrete names, objects, factions, and plot facts from the source.\n"
        "- Do NOT invent major plot facts. If a field has no source support, use [] or \"\".\n"
        "- If the source uses section headers (Appearance, Green Line, etc.), mine them for this location.\n"
        "- Keep fidelity high: longer accurate lists beat short generic ones.\n\n"
        "Locations to fill in:\n"
        f"{location_list}\n\n"
        "SOURCE EXCERPTS (may be partial windows around these locations):\n"
        '"""\n'
        f"{text}\n"
        '"""\n\n'
        'Output ONLY {"locations": [...]}. No prose, no markdown fences.'
    )


def _merge_location_detail(stub: dict[str, Any], detail: dict[str, Any], campaign_id: str) -> dict[str, Any]:
    important_locations = []
    for item in _as_list(detail.get("important_locations")):
        if isinstance(item, dict):
            important_locations.append(
                {
                    "name": str(item.get("name") or "").strip(),
                    "description": str(item.get("description") or "").strip(),
                    "public_knowledge": str(item.get("public_knowledge") or "").strip(),
                }
            )
        elif isinstance(item, str) and item.strip():
            important_locations.append({"name": item.strip(), "description": "", "public_knowledge": ""})

    return {
        "id": stub["id"],
        "campaign_id": campaign_id,
        "name": stub["name"],
        "type": stub["type"],
        "summary": str(detail.get("summary") or "").strip(),
        "current_events": _as_str_list(detail.get("current_events")),
        "important_locations": important_locations,
        "rumors": _as_str_list(detail.get("rumors")),
        "local_tone": _as_str_list(detail.get("local_tone")),
        "knowledge_rules": _as_str_list(detail.get("knowledge_rules")),
    }


# --------------------------------------------------------------------------
# Pass 3: NPC detail
# --------------------------------------------------------------------------


def _build_npc_detail_prompt(batch: list[dict[str, Any]], text: str, campaign: dict[str, Any]) -> str:
    npc_lines = []
    for npc in batch:
        locations_note = npc["first_location_id"]
        if npc["other_location_ids"]:
            locations_note += f" (also appears at: {', '.join(npc['other_location_ids'])})"
        npc_lines.append(
            f"- id: {npc['id']} | name: {npc['name']} | role: {npc['role'] or 'unspecified'} | "
            f"location(s): {locations_note}"
        )
    npc_list = "\n".join(npc_lines)

    return (
        f'Campaign: "{campaign["name"]}" ({campaign["campaign_id"]}).\n\n'
        "Fill in roleplay details for EACH NPC below from the SOURCE EXCERPTS. "
        "These excerpts often come from NPCNarrator-ready writeups. Map section headers "
        "into fields when present:\n"
        "- Role → keep/refine role (already provided on the stub; do not blank it)\n"
        "- Appearance → public_facts and/or personality cues\n"
        "- Voice Direction → voice (short delivery traits) AND llm_rules about speech style\n"
        "- Personality and Motivation / Goals → personality + known_facts + llm_rules\n"
        "- PERSONALITY / PERSONAL AGENDA (ALIEN RPG modules) → personality + known_facts + llm_rules\n"
        "- BUDDY / RIVAL (ALIEN RPG) → llm_rules about relationships\n"
        "- What the NPC Knows → known_facts (one bullet per distinct fact)\n"
        "- What the NPC Hides → secrets\n"
        "- Initial Attitude and Development / Breaking Point → llm_rules\n"
        "- NPCNarrator Memory Triggers → llm_rules (remember X when players do Y)\n"
        "- Sample Dialogue → optional llm_rules quoting tone; do not dump every line\n\n"
        'Respond with exactly {"npcs": [ ... ]} where each array item has this shape:\n\n'
        "{\n"
        '  "id": "<exact id from the list below>",\n'
        '  "pronouns": {"subject":"she","object":"her","possessive_adjective":"her",'
        '"possessive_pronoun":"hers","reflexive":"herself"},\n'
        '  "race": "species/ancestry if stated, else \\"unknown\\"",\n'
        '  "gender": "male|female|nonbinary|unspecified",\n'
        '  "voice": ["short spoken-delivery traits from Voice Direction, e.g. quiet, compassionate, controlled"],\n'
        '  "personality": ["traits and motives from Personality/Motivation; include all distinct ones"],\n'
        '  "public_facts": ["appearance and reputation others can observe or commonly know"],\n'
        '  "known_facts": ["EVERY distinct item from What the NPC Knows / Goals the NPC is pursuing"],\n'
        '  "secrets": ["EVERY distinct item from What the NPC Hides"],\n'
        '  "llm_rules": ["behavioral rules from Attitude, Development, Memory Triggers, Breaking Point, '
        'and Voice Direction; include all actionable guidance"]\n'
        "}\n\n"
        "CRITICAL EXTRACTION RULES:\n"
        "- EXTRACT over invent. Copy concrete facts from the source; do not replace them with generics.\n"
        "- Empty lists only when the source truly has nothing for that field.\n"
        "- If the source says she/her (or uses she/her language), pronouns and gender must match.\n"
        "- public_facts = what OTHER characters already know or can see (appearance, public role).\n"
        "- known_facts = what THIS NPC knows and may share.\n"
        "- secrets = what THIS NPC will not willingly reveal.\n"
        "- If secrets is non-empty, include at least one llm_rule of this exact form: "
        '"If asked directly about <topic>, do not reveal it. Respond in character, then '
        'end your reply with [DM CHECK: <short description of what the player is trying to '
        'learn>] so the DM can decide if it comes out."\n'
        "- If an NPC appears at more than one location, prefix location-specific lines with "
        '"Only when scene location is <location_id>: ...".\n'
        "- Prefer longer accurate lists over short summaries when the source is explicit.\n\n"
        "NPCs to fill in:\n"
        f"{npc_list}\n\n"
        "SOURCE EXCERPTS (windows around these NPCs — use them as the authority):\n"
        '"""\n'
        f"{text}\n"
        '"""\n\n'
        'Output ONLY {"npcs": [...]}. No prose, no markdown fences.'
    )


def _merge_npc_detail(stub: dict[str, Any], detail: dict[str, Any], campaign_id: str, warnings: list[str]) -> dict[str, Any]:
    secrets = _as_str_list(detail.get("secrets"))
    llm_rules = _as_str_list(detail.get("llm_rules"))
    gender = str(detail.get("gender") or "unspecified").strip().lower() or "unspecified"
    if gender not in {"male", "female", "nonbinary", "unspecified"}:
        gender = "unspecified"

    if secrets and not any("[DM CHECK:" in rule for rule in llm_rules):
        llm_rules.append(
            "If asked directly about information you are keeping secret, do not reveal it. "
            "Respond in character, then end your reply with [DM CHECK: the player is probing "
            "one of this NPC's secrets] so the DM can decide if it comes out."
        )
        warnings.append(f"NPC '{stub['name']}': added a default [DM CHECK] rule for its secrets.")

    role = stub["role"] or str(detail.get("role") or "").strip()

    return {
        "id": stub["id"],
        "campaign_id": campaign_id,
        "name": stub["name"],
        "role": role,
        "location_id": stub["first_location_id"],
        "other_location_ids": stub["other_location_ids"],
        "gender": gender,
        "race": str(detail.get("race") or "unknown").strip() or "unknown",
        "pronouns": _normalize_pronouns(detail.get("pronouns"), gender),
        "voice": _as_str_list(detail.get("voice")),
        "personality": _as_str_list(detail.get("personality")),
        "public_facts": _as_str_list(detail.get("public_facts")),
        "known_facts": _as_str_list(detail.get("known_facts")),
        "secrets": secrets,
        "llm_rules": llm_rules,
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def _truncate_input(text: str, warnings: list[str]) -> str:
    text = text or ""
    # Fit the selected content_creation model's Groq TPM when configured.
    # 0 means this model cannot host an import request at all right now.
    max_chars = content_creation_max_input_chars()
    if max_chars <= 0:
        raise ContentImportError(content_creation_zero_budget_error_message())
    max_chars = min(MAX_INPUT_CHARS, max_chars)
    if len(text) > max_chars:
        warnings.append(
            f"Input document was {len(text)} characters; truncated to fit the current "
            f"content_creation context budget ({max_chars} characters). "
            "When a NON-PLAYER CHARACTERS section exists it is kept in priority. "
            "For long PDFs use Groq for content_creation or raise LOCAL_LLM_CONTEXT_TOKENS "
            "to match llama.cpp n_ctx."
        )
        return _truncate_preserving_npc_section(text, max_chars)
    return text


def _fetch_location_details(
    batch: list[dict[str, Any]],
    document_text: str,
    campaign: dict[str, Any],
    warnings: list[str],
) -> dict[str, dict[str, Any]]:
    needles: list[str] = []
    for loc in batch:
        needles.extend(_distinctive_name_needles(loc["name"], loc["id"]))
    excerpt = _relevant_excerpts(document_text, needles)
    prompt = _build_location_detail_prompt(batch, excerpt, campaign)
    try:
        detail_raw = _call_llm_json(_SYSTEM_PROMPT, prompt, warnings=warnings, step="location detail")
        detail_items = _coerce_container(detail_raw, "locations")
        return _index_details_by_stub_id(detail_items, batch)
    except Exception as ex:
        warnings.append(f"Location detail batch failed ({[loc['id'] for loc in batch]}): {ex}")
        return {}


def _fetch_npc_details(
    batch: list[dict[str, Any]],
    document_text: str,
    campaign: dict[str, Any],
    warnings: list[str],
) -> dict[str, dict[str, Any]]:
    needles: list[str] = []
    for npc in batch:
        needles.extend(_distinctive_name_needles(npc["name"], npc["id"]))
    excerpt = _relevant_excerpts(document_text, needles)
    prompt = _build_npc_detail_prompt(batch, excerpt, campaign)
    try:
        detail_raw = _call_llm_json(_SYSTEM_PROMPT, prompt, warnings=warnings, step="npc detail")
        detail_items = _coerce_container(detail_raw, "npcs")
        return _index_details_by_stub_id(detail_items, batch)
    except Exception as ex:
        warnings.append(f"NPC detail batch failed ({[npc['id'] for npc in batch]}): {ex}")
        return {}


def analyze_campaign_document(
    text: str,
    mode: str,
    existing_campaign_id: str | None = None,
    existing_location_ids: list[str] | None = None,
    existing_npc_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Analyze a campaign document (module notes, session prep, wiki export) and
    return structured campaign/location/NPC data via multi-pass LLM calls.

    Returns: {"campaign": {...}, "locations": [...], "npcs": [...], "warnings": [...]}
    Raises ContentImportError if the document cannot be analyzed at all.
    """
    warnings: list[str] = []

    mode = (mode or "create_new").strip().lower()
    if mode not in VALID_MODES:
        warnings.append(f"Unknown mode '{mode}'; defaulting to 'create_new'.")
        mode = "create_new"

    existing_location_ids = [str(x).strip() for x in (existing_location_ids or []) if str(x).strip()]
    existing_npc_names = [str(x).strip() for x in (existing_npc_names or []) if str(x).strip()]
    existing_campaign_id = (existing_campaign_id or "").strip() or None

    if mode == "merge" and not existing_campaign_id:
        warnings.append("mode='merge' but no existing_campaign_id was provided; treating as create_new.")
        mode = "create_new"

    if not text or not text.strip():
        raise ContentImportError("No document text provided")

    document_text = _truncate_input(text, warnings)
    outline_source = _prepare_outline_document_text(document_text, warnings)

    outline_prompt = _build_outline_prompt(
        outline_source, mode, existing_campaign_id, existing_location_ids, existing_npc_names
    )

    try:
        outline_raw = _call_llm_json(_SYSTEM_PROMPT, outline_prompt, warnings=warnings, step="outline")
    except LLMProviderError as ex:
        raise ContentImportError(str(ex)) from ex
    except Exception as ex:
        raise ContentImportError(f"Outline pass failed: {ex}") from ex

    if not isinstance(outline_raw, dict):
        raise ContentImportError("Outline pass did not return a JSON object")

    campaign, location_stubs, npc_stubs = _normalize_outline(
        outline_raw,
        mode=mode,
        existing_campaign_id=existing_campaign_id,
        existing_location_ids=existing_location_ids,
        existing_npc_names=existing_npc_names,
        warnings=warnings,
    )

    existing_location_set = {snake_case(loc_id) for loc_id in existing_location_ids}
    location_ids = {loc["id"] for loc in location_stubs} | existing_location_set
    npc_stubs = _merge_outline_npcs_with_stat_blocks(
        npc_stubs,
        document_text,
        mode=mode,
        existing_npc_names=existing_npc_names,
        location_ids=location_ids,
        warnings=warnings,
    )

    locations_by_id: dict[str, dict[str, Any]] = {}
    for batch in _batches(location_stubs, MAX_LOCATIONS_PER_BATCH):
        details_by_id = _fetch_location_details(batch, document_text, campaign, warnings)

        # Retry sparse singles once with a tighter excerpt focus.
        for stub in batch:
            detail = details_by_id.get(stub["id"], {})
            if _location_detail_is_sparse(detail):
                retry = _fetch_location_details([stub], document_text, campaign, warnings)
                retry_detail = retry.get(stub["id"], {})
                if not _location_detail_is_sparse(retry_detail):
                    details_by_id[stub["id"]] = retry_detail
                    warnings.append(f"Location '{stub['id']}': recovered details on retry.")
                elif not detail:
                    warnings.append(f"Location '{stub['id']}': no detail returned; using minimal defaults.")

            locations_by_id[stub["id"]] = _merge_location_detail(
                stub, details_by_id.get(stub["id"], {}), campaign["campaign_id"]
            )

    npcs_by_id: dict[str, dict[str, Any]] = {}
    for batch in _batches(npc_stubs, MAX_NPCS_PER_BATCH):
        details_by_id = _fetch_npc_details(batch, document_text, campaign, warnings)

        for stub in batch:
            detail = details_by_id.get(stub["id"], {})
            if _npc_detail_is_sparse(detail):
                retry = _fetch_npc_details([stub], document_text, campaign, warnings)
                retry_detail = retry.get(stub["id"], {})
                if not _npc_detail_is_sparse(retry_detail):
                    details_by_id[stub["id"]] = retry_detail
                    warnings.append(f"NPC '{stub['id']}': recovered details on retry.")
                elif not detail:
                    warnings.append(
                        f"NPC '{stub['id']}': no detail returned; using minimal defaults. "
                        "Local small models often need content_creation set to Groq or a "
                        "stronger local model for dense NPC writeups."
                    )

            npcs_by_id[stub["id"]] = _merge_npc_detail(
                stub, details_by_id.get(stub["id"], {}), campaign["campaign_id"], warnings
            )

    return {
        "campaign": campaign,
        "locations": [locations_by_id[stub["id"]] for stub in location_stubs],
        "npcs": [npcs_by_id[stub["id"]] for stub in npc_stubs],
        "warnings": warnings,
    }
