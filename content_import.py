"""
Campaign document import: turn a pasted campaign document (module notes,
session prep, wiki export, etc.) into structured campaign/location/NPC data
using multi-pass LLM calls against the "content_creation" purpose model.

This module never writes files or touches content packs — it only returns
structured dicts (shaped like the campaign/location/npc YAML schemas used
elsewhere in this repo) for the caller to review and persist.

Pipeline:
  1. Outline pass — one LLM call, or deterministic stub seeding when recoverable
     published callouts are present (ALIEN FULL NAME blocks, or Cubicle 7-style
     name + Awareness/Presence + TRAITS character boxes). Exposed as
     ``outline_campaign_document``.
  2. Location detail — format-agnostic LLM extraction from name-focused excerpts.
     One call per Funnel hop; editor retries sparse results as a separate hop, then
     may request ``allow_seed_fallback``.
  3. NPC detail — same pattern. Layout examples (labeled writeups, FULL NAME blocks,
     trait grids + bios / callout boxes) are described generically so new RPG lines
     do not need dedicated parsers. Deterministic excerpt seeding is last-resort only.

``analyze_campaign_document`` orchestrates outline + detail (+ local retry/seed)
for local/dev. Production Yaml Editor drives chunked Funnel hops.
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
# One entity per Funnel hop. Location batching + sparse retries stacked past the
# editor's per-hop HTTP timeout (CONTENT_CREATION_LOCAL_TIMEOUT_SECONDS each).
MAX_LOCATIONS_PER_BATCH = 1
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
    r"(?:^|\n)\s*(?:auxiliary\s+vehicles|(?:the\s+)?deck\s+plans)\b",
    re.IGNORECASE,
)
# Whole-line headers only. Group the alternation so ^ and $ apply to every option;
# mid-prose "Appendix I on page 38." must not match.
_NPC_SECTION_HEADER_END_PATTERN = re.compile(
    r"(?im)^\s*(?:APPENDIX(?:\s+[A-Z0-9]+)?|INDEX|MAPS?)\s*$",
)
_STAT_BLOCK_HEADING = re.compile(
    r"^([A-Z][A-Z0-9][A-Z0-9 '\-]*?)\s*,\s*([A-Z][A-Z0-9][A-Z0-9 '\-/]*)$",
)
# Short name before a comma in a heading (AVA 6 allowed; not multi-word titles).
_STAT_BLOCK_SHORT_NAME = re.compile(
    r"([A-Z][A-Z0-9'\-]*(?:\s+\d+)?)\s*,\s*"
)
_STAT_BLOCK_FALSE_HEADING_NAMES = frozenset(
    {
        "STRENGTH",
        "AGILITY",
        "WITS",
        "EMPATHY",
        "HEALTH",
        "SKILLS",
        "GEAR",
        "TALENT",
        "BUDDY",
        "RIVAL",
        "AGE",
        "SCENARIO",
    }
)
_FULL_NAME_LINE = re.compile(r"^FULL NAME:\s*(.+)$", re.IGNORECASE)
# PdfPig two-column lines often contain multiple FULL NAME markers on one line.
# Restrict to Title-Case name tokens; stop before the next FULL NAME or lowercase prose.
_FULL_NAME_ANY = re.compile(
    r"(?:FULL|Full|full)\s+(?:NAME|Name|name):\s*"
    r"([A-Z][a-zA-Z0-9'\-]+(?:\s+[A-Z][a-zA-Z0-9'\-]+){0,4})"
    r"(?=\s*(?:FULL|Full|full)\s+(?:NAME|Name|name)\s*:|\s+[a-z]|\s*$|[,;\n])",
)
_STAT_BLOCK_FIELD_STOP = re.compile(
    r"^(?:AGE|PERSONALITY|PERSONAL\s+AGENDA|STRENGTH|HEALTH|SKILLS|GEAR|TALENT|"
    r"BUDDY|RIVAL|FULL\s+NAME)\b",
    re.IGNORECASE,
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
    end = len(collapsed_text)
    hard_end: int | None = None

    end_match = _NPC_SECTION_END_PATTERN.search(remainder)
    if end_match:
        hard_end = match.end() + end_match.start()
        end = hard_end

    # Only treat APPENDIX/INDEX/MAPS as section ends when the entire line is that
    # header — not mid-sentence references like "Appendix I on page 38".
    for header_match in _NPC_SECTION_HEADER_END_PATTERN.finditer(remainder):
        abs_pos = match.end() + header_match.start()
        if abs_pos >= end:
            break
        line_start = collapsed_text.rfind("\n", start, abs_pos) + 1
        line_end = collapsed_text.find("\n", abs_pos)
        if line_end < 0:
            line_end = len(collapsed_text)
        line = collapsed_text[line_start:line_end].strip()
        # Regex already requires a whole-line header; keep the length guard as a
        # belt-and-suspenders check against odd whitespace collapses.
        if line and len(line) <= 40 and _NPC_SECTION_HEADER_END_PATTERN.match(line):
            end = min(end, line_start)
            break

    section = collapsed_text[start:end]
    # PdfPig two-column glue can trip an early false end after the first NPC.
    # If we see too few FULL NAME markers, extend only to the next *hard* end
    # (deck plans / auxiliary vehicles). Never blow past a known hard boundary
    # with a blind start+20_000 window — that swallows following chapters when
    # a legitimate module simply has fewer than three FULL NAME rows.
    if len(_FULL_NAME_ANY.findall(section)) < 3:
        hard = _NPC_SECTION_END_PATTERN.search(collapsed_text, start)
        if hard is not None:
            hard_pos = hard.start()
            if hard_pos > end:
                end = hard_pos
            # else: already at/before the hard end — keep it
        elif hard_end is None:
            # No hard boundary in the document; allow a bounded look-ahead only
            # when we have not already locked onto deck-plans / auxiliary vehicles.
            end = min(len(collapsed_text), max(end, start + 20_000))

    return start, end


def _iter_stat_block_headings(line: str) -> list[tuple[str, str]]:
    """Return (short_name, role) pairs from a line; supports PDF two-column glue.

    Split on short-name commas rather than a role lookahead that can swallow the
    next heading (``REID, COLONIAL MARINE FLYNN, MEDIC`` → Reid/Colonial Marine +
    Flynn/Medic, not Marine Flynn as a short name).
    """
    stripped = line.strip()
    if not stripped:
        return []

    starts = list(_STAT_BLOCK_SHORT_NAME.finditer(stripped))
    if not starts:
        return []

    results: list[tuple[str, str]] = []
    for index, match in enumerate(starts):
        short_name = match.group(1).strip()
        short_key = short_name.upper()
        if short_key in _STAT_BLOCK_FALSE_HEADING_NAMES:
            continue
        if short_key.split()[0] in _STAT_BLOCK_FALSE_HEADING_NAMES:
            continue
        # Allow ``AVA 6`` but reject codes like ``LV-1113``.
        if re.search(r"\d", short_name) and not re.fullmatch(
            r"[A-Z][A-Z'\-]*\s+\d+", short_name
        ):
            continue

        role_start = match.end()
        role_end = starts[index + 1].start() if index + 1 < len(starts) else len(stripped)
        role = stripped[role_start:role_end].strip(" ,\t")
        if not role:
            continue
        # Prose after a comma (``LV-1113, and suffers...``) is not a role.
        if not role[0].isupper():
            continue
        role_key = role.upper()
        if role_key in _STAT_BLOCK_FALSE_HEADING_NAMES:
            continue
        # Attribute rows like ``STRENGTH 4, AGILITY 3``.
        if re.fullmatch(r"\d+(?:\s*,\s*[A-Z]+ \d+)*", role):
            continue
        # Roles are short labels (COLONIAL MARINE), not sentences.
        if len(role.split()) > 6 or "," in role:
            continue
        results.append((short_name, role))
    return results


def _heading_matches_display_name(short_name: str, display_name: str) -> bool:
    short_key = short_name.casefold().strip()
    display_key = display_name.casefold().strip()
    if not short_key or not display_key:
        return False
    if short_key in display_key:
        return True
    # PIN -> Pinion, AVA 6 -> Ava 6
    short_token = re.sub(r"[^a-z0-9]+", "", short_key)
    display_compact = re.sub(r"[^a-z0-9]+", "", display_key)
    return bool(short_token) and short_token in display_compact


def _block_has_own_full_name(lines: list[str], heading_index: int) -> bool:
    """True when a FULL NAME appears before this heading's PERSONAL AGENDA.

    Used to distinguish humans (FULL NAME then agenda) from androids like Ava 6
    whose agenda window may later include a column partner's FULL NAME.
    """
    for raw in lines[heading_index + 1 : heading_index + 35]:
        candidate = raw.strip()
        if not candidate:
            continue
        if _FULL_NAME_LINE.match(candidate):
            return True
        if re.match(r"^PERSONAL AGENDA:", candidate, re.IGNORECASE):
            return False
        # A later heading line usually starts the next block / column partner body.
        if _iter_stat_block_headings(candidate) and not candidate.upper().startswith("FULL NAME:"):
            # Same-line partners are on heading_index itself; a new heading line ends the block.
            return False
    return False


def _labeled_field_from_window(window: str, label: str) -> str:
    """Read a ``LABEL:`` value; PdfPig often places labels mid-line between columns."""
    match = re.search(rf"(?i)\b{re.escape(label)}\s*:\s*(.+)", window)
    if not match:
        return ""
    rest = match.group(1)
    first_line = rest.split("\n", 1)[0].strip(" -\t")
    if label.casefold() == "personality":
        # Published trait is short (Submissive, On Edge); drop glued column prose.
        cleaned = re.split(
            r"\b(?:PERSONALITY|PERSONAL\s+AGENDA|STRENGTH|HEALTH|SKILLS|AGE|FULL\s+NAME)\b",
            first_line,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip(" -\t")
        trait = re.match(
            r"([A-Za-z][A-Za-z\-]*(?:\s+[A-Za-z][A-Za-z\-]*){0,3})",
            cleaned,
        )
        return trait.group(1).strip() if trait else (cleaned.split()[0] if cleaned else "")

    # PERSONAL AGENDA (and similar) may wrap; stop at next known field marker.
    chunk = first_line
    more = rest.split("\n", 1)
    if len(more) > 1:
        for line in more[1].splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if _STAT_BLOCK_FIELD_STOP.match(stripped) or re.search(
                r"\b(?:STRENGTH|HEALTH|SKILLS|GEAR|TALENT|BUDDY|RIVAL|FULL\s+NAME|PERSONALITY|AGE)\b",
                stripped,
                re.IGNORECASE,
            ):
                break
            if re.search(r"\bPERSONAL\s+AGENDA\s*:", stripped, re.IGNORECASE):
                break
            chunk += " " + stripped
            if len(chunk) > 240:
                break
    value = re.sub(r"\s+", " ", chunk).strip(" -\t")
    value = re.split(
        r"\b(?:STRENGTH|HEALTH|SKILLS|GEAR|TALENT|BUDDY|RIVAL|PERSONAL\s+AGENDA|PERSONALITY)\b",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" -\t")
    return value


def _iter_full_names_on_line(line: str) -> list[str]:
    """Return every FULL NAME value on a line (PdfPig often glues two columns)."""
    return [match.group(1).strip() for match in _FULL_NAME_ANY.finditer(line.strip())]


def _enrich_row_from_window(row: dict[str, str], window: str) -> dict[str, str]:
    personality = _labeled_field_from_window(window, "PERSONALITY")
    agenda = _labeled_field_from_window(window, "PERSONAL AGENDA")
    if personality:
        row["personality"] = personality
    if agenda:
        row["personal_agenda"] = agenda
    return row


def _extract_alien_rpg_stat_block_npcs(section_text: str) -> list[dict[str, str]]:
    """Parse ALIEN RPG module stat blocks.

    Prefer ``FULL NAME:`` rows (authoritative). PDF two-column extraction often glues
    headings like ``REID, COLONIAL MARINE FLYNN, MEDIC`` onto one line, so whole-line
    heading matches alone miss most NPCs.
    """
    if not section_text.strip():
        return []

    lines = section_text.splitlines()
    results: list[dict[str, str]] = []
    seen_names: set[str] = set()
    claimed_heading_keys: set[str] = set()

    def _location_hint_from(window: str) -> str:
        if re.search(r"\bSotillo\b", window, re.IGNORECASE):
            return "sotillo"
        if re.search(r"\bCronus\b", window, re.IGNORECASE):
            return "uscss_cronus"
        return ""

    def _role_near_full_name(index: int, display_name: str) -> tuple[str, str]:
        """Return (short_name, role) from nearby headings matching this full name."""
        before = lines[max(0, index - 40) : index]
        # Prefer a heading whose short name appears in the full name.
        for raw in reversed(before):
            for short_name, role in _iter_stat_block_headings(raw):
                key = short_name.casefold()
                if key in claimed_heading_keys:
                    continue
                if _heading_matches_display_name(short_name, display_name):
                    return short_name, role.replace("/", " ").title()
        # Fallback: nearest unclaimed heading, then subtitle line under it.
        for raw_index in range(len(before) - 1, -1, -1):
            headings = _iter_stat_block_headings(before[raw_index])
            if not headings:
                continue
            for short_name, role in reversed(headings):
                key = short_name.casefold()
                if key in claimed_heading_keys:
                    continue
                role_text = role.replace("/", " ").title()
                # Prefer a short subtitle under the heading; skip prose sentences.
                if raw_index + 1 < len(before):
                    candidate = before[raw_index + 1].strip()
                    if (
                        candidate
                        and not candidate.upper().startswith("FULL NAME:")
                        and not _iter_stat_block_headings(candidate)
                        and len(candidate.split()) <= 8
                        and not candidate.endswith(("-", ","))
                        and not re.match(
                            r"^(AGE|PERSONALITY|STRENGTH|HEALTH|SKILLS|GEAR|TALENT)\b",
                            candidate,
                            re.IGNORECASE,
                        )
                    ):
                        role_text = candidate
                return short_name, role_text
        return "", ""

    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line or "FULL NAME:" not in line.upper():
            continue

        for display_name in _iter_full_names_on_line(line):
            name_key = display_name.casefold()
            if not display_name or name_key in seen_names:
                continue

            short_name, role = _role_near_full_name(index, display_name)
            if short_name:
                claimed_heading_keys.add(short_name.casefold())

            window = "\n".join(lines[max(0, index - 8) : index + 25])
            if not role:
                for look_ahead in lines[index + 1 : index + 4]:
                    candidate = look_ahead.strip()
                    if not candidate or "FULL NAME:" in candidate.upper():
                        continue
                    if _iter_stat_block_headings(candidate):
                        break
                    if not _STAT_BLOCK_FIELD_STOP.match(candidate):
                        role = candidate
                        break

            seen_names.add(name_key)
            row = {
                "name": display_name,
                "role": role or (short_name.title() if short_name else ""),
                "short_name": short_name or display_name.split()[-1].upper(),
                "location_hint": _location_hint_from(window),
            }
            results.append(_enrich_row_from_window(row, window))

    # Androids / entries with PERSONAL AGENDA but no FULL NAME (e.g. AVA 6).
    for index, raw_line in enumerate(lines):
        for short_name, role in _iter_stat_block_headings(raw_line):
            key = short_name.casefold()
            if key in claimed_heading_keys:
                continue
            # Already imported under a FULL NAME that matches this heading.
            if any(_heading_matches_display_name(short_name, row["name"]) for row in results):
                claimed_heading_keys.add(key)
                continue
            window = "\n".join(lines[index : index + 30])
            if not re.search(r"PERSONAL AGENDA:", window, re.IGNORECASE):
                continue
            # Only skip when THIS block has its own FULL NAME before its agenda.
            # A column partner's FULL NAME after Ava's agenda must not suppress Ava.
            if _block_has_own_full_name(lines, index):
                continue
            display_name = short_name.title()
            name_key = display_name.casefold()
            if name_key in seen_names:
                continue
            claimed_heading_keys.add(key)
            seen_names.add(name_key)
            role_text = role.replace("/", " ").title()
            for look_ahead in lines[index + 1 : index + 4]:
                candidate = look_ahead.strip()
                if not candidate or "FULL NAME:" in candidate.upper():
                    continue
                if _iter_stat_block_headings(candidate):
                    break
                if not _STAT_BLOCK_FIELD_STOP.match(candidate):
                    role_text = candidate
                    break
            row = {
                "name": display_name,
                "role": role_text,
                "short_name": short_name,
                "location_hint": _location_hint_from(window),
            }
            results.append(_enrich_row_from_window(row, window))

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
    """Shrink long documents but keep the NON-PLAYER CHARACTERS section when present.

    Must not exceed ``max_chars`` — that budget is the Groq/local outline limit.
    Deterministic stat-block merge uses the full document separately.
    """
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


def _document_has_recoverable_stat_blocks(text: str) -> bool:
    collapsed = _collapse_spaced_caps(text or "")
    span = _find_npc_section_span(collapsed)
    if span is None:
        return False
    parsed = _extract_alien_rpg_stat_block_npcs(collapsed[span[0] : span[1]])
    return len(parsed) >= 3


def _guess_campaign_title(text: str) -> str | None:
    """Best-effort title from the document head when the LLM returns an empty campaign."""
    head = (text or "")[:2500]
    collapsed_head = _collapse_spaced_caps(head)
    # Prefer an explicit scenario title near the top of the extract.
    for pattern in (
        r"(?is)\b(CHARIOT\s+OF\s+THE\s+GODS)\b",
        r"(?m)^\s*([A-Z0-9][A-Z0-9' \-]{2,48})\s*$",
    ):
        match = re.search(pattern, collapsed_head)
        if not match:
            continue
        title = re.sub(r"\s+", " ", match.group(1)).strip(" :-")
        if not title:
            continue
        words = title.split()
        if len(words) > 8:
            continue
        if title.casefold() in {"scenario", "contents", "introduction", "non-player characters"}:
            continue
        if title.isupper() and len(words) >= 2:
            return title.title()
        if title.isupper() and len(words) == 1 and len(title) >= 4:
            return title.title()
    return None


def _ensure_locations_for_npc_hints(
    location_stubs: list[dict[str, Any]],
    npc_stubs: list[dict[str, Any]],
    *,
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Add minimal location stubs referenced by NPC first_location_id hints."""
    known = {loc["id"] for loc in location_stubs}
    defaults = {
        "uscss_cronus": ("USCSS Cronus", "building"),
        "sotillo": ("Sotillo", "building"),
    }
    added: list[str] = []
    for npc in npc_stubs:
        loc_id = str(npc.get("first_location_id") or "").strip()
        if not loc_id or loc_id in known:
            continue
        name, loc_type = defaults.get(loc_id, (loc_id.replace("_", " ").title(), "other"))
        location_stubs.append(
            {
                "id": loc_id,
                "name": name,
                "type": loc_type,
                "seeded_from_stat_blocks": True,
            }
        )
        known.add(loc_id)
        added.append(loc_id)
    if added:
        warnings.append(
            "Added location stub(s) referenced by stat-block NPCs: " + ", ".join(added)
        )
    return location_stubs


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
        hint = (hint or "").strip()
        if not hint:
            return ""
        if hint in location_ids:
            return hint
        for loc_id in location_ids:
            if hint in loc_id or loc_id in hint:
                return loc_id
        if location_ids:
            return next(iter(location_ids), "")
        # Keep the hint so callers can seed a location stub later.
        return hint

    stubs: list[dict[str, Any]] = []
    for row in parsed:
        loc = _resolve_hint(row.get("location_hint") or "")
        personality = str(row.get("personality") or "").strip()
        agenda = str(row.get("personal_agenda") or "").strip()
        stubs.append(
            {
                "id": snake_case(row["name"]),
                "name": row["name"],
                "role": row.get("role") or "",
                "first_location_id": loc,
                "other_location_ids": [],
                "seeded_from_stat_blocks": True,
                "seed_personality": personality,
                "seed_personal_agenda": agenda,
            }
        )

    warnings.append(
        f"Parsed {len(stubs)} NPC(s) from published stat blocks in NON-PLAYER CHARACTERS."
    )
    return stubs


def _npc_stub_has_stat_seed(stub: dict[str, Any]) -> bool:
    return bool(stub.get("seeded_from_stat_blocks")) and bool(
        stub.get("seed_personality") or stub.get("seed_personal_agenda")
    )


def _npc_detail_from_stat_seed(stub: dict[str, Any]) -> dict[str, Any]:
    """Build usable NPC detail fields from published ALIEN block text (no LLM)."""
    personality = str(stub.get("seed_personality") or "").strip()
    agenda = str(stub.get("seed_personal_agenda") or "").strip()
    role = str(stub.get("role") or "").strip()
    detail: dict[str, Any] = {
        "gender": "unspecified",
        "race": "unknown",
        "voice": [],
        "personality": [personality] if personality else [],
        "public_facts": [f"Role: {role}"] if role else [],
        "known_facts": [f"Personal agenda: {agenda}"] if agenda else [],
        "secrets": [agenda] if agenda else [],
        "llm_rules": [],
    }
    if agenda:
        detail["llm_rules"] = [
            f"Pursue this personal agenda when it fits the scene: {agenda}",
            "If asked directly about information you are keeping secret, do not reveal it. "
            "Respond in character, then end your reply with [DM CHECK: the player is probing "
            "one of this NPC's secrets] so the DM can decide if it comes out.",
        ]
    return detail


_TRAIT_SECTION_HEADER = re.compile(r"(?im)^\s*traits?\s*$")
_TRAIT_ITEM_START = re.compile(r"^([A-Za-z][A-Za-z][A-Za-z \-/]{0,32}):\s*(.*)$")
_TRAIT_SECTION_END = re.compile(
    r"(?i)^\s*(?:tech\s*level|story\s*points|skills|awareness|presence|coordination|"
    r"resolve|ingenuity|strength|exploring|water\s+vs)\b"
)
_STAT_OR_SECTION_LINE = re.compile(
    r"(?i)^\s*(?:traits?|skills?|tech\s*level|story\s*points|awareness|presence|"
    r"coordination|resolve|ingenuity|strength)\b"
)
# Seed windows stay tighter than LLM excerpts so packed modules do not blend NPCs.
_SEED_EXCERPT_WINDOW_BEFORE = 120
_SEED_EXCERPT_WINDOW_AFTER = 2_200
_SEED_WRITEUP_MAX_CHARS = 1_800
_BIO_CHUNK_MAX_CHARS = 280


def _parse_trait_blocks(
    excerpt: str,
    *,
    first_section_only: bool = False,
) -> list[tuple[str, str]]:
    """Parse Cubicle 7-style TRAITS blocks (Trait: description, possibly wrapped)."""
    lines = excerpt.splitlines()
    traits: list[tuple[str, str]] = []
    in_traits = False
    section_count = 0
    current_name = ""
    current_parts: list[str] = []

    def flush() -> None:
        nonlocal current_name, current_parts
        if current_name and current_parts:
            text = " ".join(p.strip() for p in current_parts if p.strip()).strip()
            if text:
                traits.append((current_name, text))
        current_name = ""
        current_parts = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _TRAIT_SECTION_HEADER.match(stripped):
            if first_section_only and section_count >= 1:
                break
            flush()
            in_traits = True
            section_count += 1
            continue
        if not in_traits:
            continue
        if _TRAIT_SECTION_END.match(stripped):
            flush()
            in_traits = False
            if first_section_only and section_count >= 1:
                break
            continue
        item = _TRAIT_ITEM_START.match(stripped)
        if item and not stripped.lower().startswith("tech level"):
            flush()
            current_name = item.group(1).strip()
            rest = item.group(2).strip()
            current_parts = [rest] if rest else []
            continue
        if current_name:
            current_parts.append(stripped)
    flush()
    return traits


def _looks_like_npc_name_header(line: str, *, exclude_name: str = "") -> bool:
    """True for short title-like lines that often introduce the next NPC writeup."""
    s = str(line or "").strip().strip("\"“”")
    if not s or len(s) < 3 or len(s) > 42:
        return False
    if s.endswith((".", ":", ";", ",")):
        return False
    if re.search(r"\d", s):
        return False
    if _STAT_OR_SECTION_LINE.match(s) or _TRAIT_SECTION_HEADER.match(s):
        return False
    words = s.split()
    if not (1 <= len(words) <= 4):
        return False
    if exclude_name and _mention_indexes(s.casefold(), exclude_name):
        return False
    # Avoid cutting on ordinary prose openers.
    if words[0].casefold() in {
        "the",
        "a",
        "an",
        "for",
        "and",
        "but",
        "as",
        "in",
        "on",
        "at",
        "to",
        "of",
        "with",
        "from",
        "this",
        "that",
        "he",
        "she",
        "they",
        "his",
        "her",
        "their",
    }:
        return False
    return True


def _clip_npc_writeup(excerpt: str, name: str) -> str:
    """Keep only the local writeup for ``name`` so neighbor TRAITS/bios are excluded."""
    name = str(name or "").strip()
    if not excerpt:
        return ""
    if not name:
        return excerpt[:_SEED_WRITEUP_MAX_CHARS]

    cf = excerpt.casefold()
    hits = _mention_indexes(cf, name)
    if not hits and " " in name:
        hits = _mention_indexes(cf, name.split()[-1])
    if not hits:
        return excerpt[:_SEED_WRITEUP_MAX_CHARS]

    # Prefer a short header-line hit (PDF-split / quoted names) over an earlier
    # prose mention ("including the formidable Chad").
    start = None
    for idx in hits:
        line_start = excerpt.rfind("\n", 0, idx) + 1
        line_end = excerpt.find("\n", idx)
        line = excerpt[line_start : line_end if line_end >= 0 else len(excerpt)].strip()
        label = line.strip("\"“”").strip()
        if not label or not _mention_indexes(label.casefold(), name):
            continue
        if _looks_like_npc_name_header(line) or len(label.split()) <= max(1, len(name.split())):
            start = line_start
            break
    if start is None:
        start = excerpt.rfind("\n", 0, hits[0]) + 1

    hard_end = min(len(excerpt), start + _SEED_WRITEUP_MAX_CHARS)
    region = excerpt[start:hard_end]
    offset = 0
    for i, line in enumerate(region.splitlines(keepends=True)):
        stripped = line.strip()
        if i > 0 and stripped and _looks_like_npc_name_header(stripped, exclude_name=name):
            return excerpt[start : start + offset]
        offset += len(line)
    return excerpt[start:hard_end]


def _split_writeup_blocks(excerpt: str) -> list[str]:
    """Split PdfPig-style single-newline text on blank lines and structural headers."""
    blocks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            blocks.append("\n".join(current))
            current.clear()

    for line in excerpt.splitlines():
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if (
            _TRAIT_SECTION_HEADER.match(stripped)
            or _STAT_OR_SECTION_LINE.match(stripped)
            or _TRAIT_SECTION_END.match(stripped)
            or _looks_like_npc_name_header(stripped)
        ):
            flush()
            # Keep name headers so flexible matching can see "Peter Hutc HiNs".
            if _looks_like_npc_name_header(stripped) and not _STAT_OR_SECTION_LINE.match(stripped):
                current.append(stripped)
            continue
        current.append(stripped)
    flush()
    return blocks


def _bio_paragraphs_for_name(excerpt: str, name: str) -> list[str]:
    """Grab prose paragraphs that clearly describe this NPC by name."""
    name = str(name or "").strip()
    if not name or not excerpt:
        return []
    surname = name.split()[-1] if name.split() else ""
    paragraphs: list[str] = []
    for block in _split_writeup_blocks(excerpt):
        text = " ".join(line.strip() for line in block.splitlines() if line.strip()).strip()
        if len(text) < 40:
            continue
        head = text[:120]
        name_in_head = bool(_mention_indexes(head.casefold(), name))
        surname_in_open = bool(
            len(surname) >= 4 and _mention_indexes(text[:160].casefold(), surname)
        )
        if name_in_head or (
            surname_in_open
            and any(
                marker in text.casefold()
                for marker in (" is ", " suffers", " has ", " would ", " can ")
            )
        ):
            if len(text) > _BIO_CHUNK_MAX_CHARS:
                text = text[: _BIO_CHUNK_MAX_CHARS - 3] + "..."
            paragraphs.append(text)
    return paragraphs[:3]


def _seed_npc_detail_from_document(
    name: str,
    role: str,
    document_text: str,
) -> dict[str, Any] | None:
    """Deterministic detail seed from Cubicle 7 / Doctor Who-style stat writeups."""
    needles = _distinctive_name_needles(name, snake_case(name))
    excerpt = _relevant_excerpts(
        document_text,
        needles,
        window_before=_SEED_EXCERPT_WINDOW_BEFORE,
        window_after=_SEED_EXCERPT_WINDOW_AFTER,
    )
    excerpt = _clip_npc_writeup(excerpt, name)
    if not excerpt or len(excerpt.strip()) < 60:
        return None

    # Only the first TRAITS section in this clipped writeup belongs to the target NPC.
    traits = _parse_trait_blocks(excerpt, first_section_only=True)
    bios = _bio_paragraphs_for_name(excerpt, name)
    personality: list[str] = []
    llm_rules: list[str] = []
    public_facts: list[str] = []
    known_facts: list[str] = []

    for trait_name, trait_text in traits:
        personality.append(f"{trait_name}: {trait_text}")
        llm_rules.append(f"Play the '{trait_name}' trait: {trait_text}")

    if role:
        public_facts.append(f"Role: {role}")
    for bio in bios:
        # First bio sentence-ish chunks as public/known facts.
        if len(public_facts) < 4:
            public_facts.append(bio)
        else:
            known_facts.append(bio)

    detail: dict[str, Any] = {
        "gender": "unspecified",
        "race": "unknown",
        "voice": [],
        "personality": personality,
        "public_facts": public_facts,
        "known_facts": known_facts,
        "secrets": [],
        "llm_rules": llm_rules,
    }
    if _npc_detail_is_sparse(detail):
        return None
    return detail


_CALLOUT_AWARENESS_PRESENCE = re.compile(
    r"(?i)^\s*awareness\s+\d+\s+presence\s+\d+\s*$"
)
_CALLOUT_STORY_OR_TECH = re.compile(r"(?i)\b(?:story\s*points|tech\s*level)\b")
_CALLOUT_MONSTER_CONTEXT = re.compile(
    r"(?i)\b(?:creature\s*statistics?|monster(?:s)?|enemy\s*stat|adversary\s*stat|"
    r"alien\s*creature|beast(?:s)?\s*stat)\b"
)
_CREDIT_ROLE_LINE = re.compile(
    r"(?i)^\s*(?:writing|writer|line\s*developer|editing|editors?|cover|"
    r"graphic\s*design(?:\s*&\s*layout)?|layout|creative\s*director|"
    r"art\s*director|publisher|proofreading|playtesters?|thanks)\s*$"
)


def _normalize_pdf_person_name(raw: str) -> str:
    """Repair PdfPig mid-word spaces: 'Peter Hutc HiNs' → 'Peter Hutchins'."""
    s = str(raw or "").strip().strip("\"“”'‘’")
    if not s:
        return ""
    tokens = re.sub(r"\s+", " ", s).split()
    # Merge trailing syllable splits (Peterse N, Hutc HiNs).
    if len(tokens) >= 3 and (
        len(tokens[-1]) <= 3 or (len(tokens[-2]) <= 4 and len(tokens[-1]) <= 4)
    ):
        if tokens[-1].casefold() not in {"vs", "the", "of", "and", "or"}:
            tokens = tokens[:-2] + [tokens[-2] + tokens[-1]]
    return " ".join((t[0].upper() + t[1:].lower()) if t else t for t in tokens)


def _guess_callout_role(preceding: str) -> str:
    window = (preceding or "")[-900:].casefold()
    if "squatter" in window:
        return "squatter"
    if "villain" in window or "antagonist" in window:
        return "antagonist"
    if "guest" in window or "reception" in window:
        return "hotel guest"
    return ""


def _extract_cubicle7_callout_npcs(document_text: str) -> list[dict[str, str]]:
    """Find Cubicle 7-style character callout boxes (name + attributes + TRAITS).

    These appear mid-scene (like ALIEN sidebar/stat callouts), not only under an
    NPC appendix. Credits/TOC names lack Awareness/Presence + TRAITS + Story Points
    and are ignored. Creature-only blocks under monster headings are skipped when
    that context is adjacent.
    """
    text = document_text or ""
    if not text.strip():
        return []

    lines = text.splitlines()
    found: list[dict[str, str]] = []
    seen: set[str] = set()

    for i, line in enumerate(lines):
        if not _CALLOUT_AWARENESS_PRESENCE.match(line.strip()):
            continue

        # Name header is the nearest prior non-empty line.
        header_idx = -1
        for j in range(i - 1, max(-1, i - 6), -1):
            candidate = lines[j].strip()
            if not candidate:
                continue
            if _CREDIT_ROLE_LINE.match(candidate) or _STAT_OR_SECTION_LINE.match(candidate):
                header_idx = -1
                break
            header_idx = j
            break
        if header_idx < 0:
            continue

        header_raw = lines[header_idx].strip()
        if not _looks_like_npc_name_header(header_raw):
            continue

        # Look ahead for character-sheet signals (TRAITS + Story Points / Tech Level).
        window = "\n".join(lines[i : min(len(lines), i + 22)])
        if not _TRAIT_SECTION_HEADER.search(window):
            continue
        if not _CALLOUT_STORY_OR_TECH.search(window):
            continue

        preceding = "\n".join(lines[max(0, header_idx - 12) : header_idx])
        if _CALLOUT_MONSTER_CONTEXT.search(preceding):
            continue

        name = _normalize_pdf_person_name(header_raw)
        if not name or len(name) < 3:
            continue
        key = name.casefold()
        if key in seen:
            continue
        if key in _KNOWN_IMPORT_FALSE_POSITIVE_NAMES:
            continue

        seen.add(key)
        role = _guess_callout_role(preceding + "\n" + window)
        found.append({"name": name, "role": role, "location_hint": ""})

    return found


def _npc_stubs_from_cubicle7_callouts(
    document_text: str,
    *,
    location_ids: set[str],
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Build outline NPC stubs from Cubicle 7 character callout boxes."""
    parsed = _extract_cubicle7_callout_npcs(document_text)
    if not parsed:
        return []

    loc = ""
    for candidate in ("majestic_hotel", "the_majestic", "majestic"):
        if candidate in location_ids:
            loc = candidate
            break
    if not loc and location_ids:
        # Prefer a hotel/building-looking id when present.
        for loc_id in location_ids:
            if "majestic" in loc_id or "hotel" in loc_id:
                loc = loc_id
                break

    stubs: list[dict[str, Any]] = []
    for row in parsed:
        stubs.append(
            {
                "id": snake_case(row["name"]),
                "name": row["name"],
                "role": row.get("role") or "",
                "first_location_id": loc,
                "other_location_ids": [],
                "seeded_from_stat_blocks": True,
                "seed_personality": "",
                "seed_personal_agenda": "",
            }
        )

    warnings.append(
        f"Parsed {len(stubs)} NPC(s) from Cubicle 7-style character callout boxes "
        "(name + Awareness/Presence + TRAITS)."
    )
    return stubs


def _document_has_recoverable_callouts(text: str) -> bool:
    return len(_extract_cubicle7_callout_npcs(text or "")) >= 2


def _location_detail_from_seed(stub: dict[str, Any]) -> dict[str, Any]:
    name = str(stub.get("name") or stub.get("id") or "Location").strip()
    return {
        "summary": f"{name} appears in the imported module.",
        "current_events": [],
        "important_locations": [],
        "rumors": [],
        "local_tone": [],
        "knowledge_rules": [],
    }


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


def _merge_outline_npcs_with_callouts(
    llm_npcs: list[dict[str, Any]],
    document_text: str,
    *,
    mode: str,
    existing_npc_names: list[str],
    location_ids: set[str],
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Prefer Cubicle 7 character callouts over LLM/credits when confidently detected.

    When enough callout boxes are present, they are authoritative — credits-page
    and TOC names from the outline LLM are dropped.
    """
    # ALIEN FULL NAME stubs already carry PERSONALITY seeds; leave them alone.
    alien_seeded = sum(
        1
        for n in llm_npcs
        if n.get("seeded_from_stat_blocks")
        and str(n.get("seed_personality") or "").strip()
    )
    if alien_seeded >= 3:
        return llm_npcs

    callout_stubs = _npc_stubs_from_cubicle7_callouts(
        document_text,
        location_ids=location_ids,
        warnings=warnings,
    )
    if len(callout_stubs) < 2:
        return llm_npcs

    filtered: list[dict[str, Any]] = []
    for stub in callout_stubs:
        name = str(stub.get("name") or "").strip()
        if not name:
            continue
        if _npc_already_exists_in_campaign(
            name, mode=mode, existing_npc_names=existing_npc_names
        ):
            warnings.append(
                f"NPC '{name}' already exists in the campaign; "
                "reusing existing character and skipping callout regeneration."
            )
            continue
        filtered.append(stub)
    if len(filtered) < 2:
        return llm_npcs

    callout_keys = {stub["name"].casefold() for stub in filtered}
    dropped = [
        str(npc.get("name") or "").strip()
        for npc in llm_npcs
        if str(npc.get("name") or "").strip()
        and str(npc.get("name") or "").strip().casefold() not in callout_keys
    ]
    if dropped:
        warnings.append(
            "Ignored outline NPC(s) not found in character callout boxes "
            "(often credits/TOC): " + ", ".join(dropped)
        )

    return filtered


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
    """Casefolded matches; allow PDF mid-word spaces (Hutc HiNs ≈ Hutchins)."""
    v = str(needle or "").strip().casefold()
    if len(v) < 3:
        return []
    # Exact bounded match.
    exact = re.compile(rf"(?<![a-z0-9]){re.escape(v)}(?![a-z0-9])")
    hits = {match.start() for match in exact.finditer(haystack_cf)}
    # Also collect flexible matches so PDF-split headers are not skipped when a
    # later contiguous prose mention exists (e.g. "Peter Hutc HiNs" vs bio name).
    flex_words: list[str] = []
    for word in v.split():
        chars = [re.escape(ch) for ch in word if ch.isalnum()]
        if len(chars) < 3:
            continue
        flex_words.append(r"\s*".join(chars))
    if flex_words:
        flexible = re.compile(rf"(?<![a-z0-9]){r'\s+'.join(flex_words)}(?![a-z0-9])")
        hits.update(match.start() for match in flexible.finditer(haystack_cf))
    return sorted(hits)


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
        "- Character callout boxes are also authoritative NPCs even mid-scene: a short name "
        "header followed by attributes (Awareness/Presence/...), Skills, TRAITS "
        "(Trait: description), Tech Level / Story Points, and a prose biography that is more "
        "than physical appearance. List each such callout NPC.\n"
        "- Do NOT list credits / staff / acknowledgements (Writing, Editing, Cover, Art "
        "Director, etc.), table-of-contents page references, publisher names, or trademark "
        "boilerplate as NPCs.\n"
        "- Do NOT list player characters, the PC party, or the PC ship crew as NPCs.\n"
        "- Prefer named supporting characters with dialogue, motives, or callout boxes over "
        "generic monsters/enemies unless the text clearly presents them as individual "
        "named characters with personality.\n"
        "- Published RPG stat blocks (e.g. lines like 'JOHNS, OFFICER' with FULL NAME, "
        "PERSONALITY, PERSONAL AGENDA) are always NPCs when inside the NPC section.\n"
        "- Include every named location and every named NPC from the authoritative NPC section "
        "or character callout boxes; minor NPCs in those places still count.\n"
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


def _build_location_detail_prompt(
    batch: list[dict[str, Any]],
    text: str,
    campaign: dict[str, Any],
    *,
    is_retry: bool = False,
) -> str:
    location_list = "\n".join(f"- id: {loc['id']} | name: {loc['name']} | type: {loc['type']}" for loc in batch)
    retry_note = ""
    if is_retry:
        retry_note = (
            "RETRY: A previous pass returned little or nothing useful for this location. "
            "Read the excerpts again carefully and fill every field you can support from the text.\n\n"
        )
    return (
        f'Campaign: "{campaign["name"]}" ({campaign["campaign_id"]}).\n\n'
        f"{retry_note}"
        "Fill in details for EACH location below from the SOURCE EXCERPTS only.\n"
        "Modules use many layouts — room-by-room keys, ship deck notes, scene boxed text, "
        "faction writeups, maps with captions, or free prose. Do not assume one game's headings.\n\n"
        "How to read unfamiliar layouts:\n"
        "- Treat nearby headings, bullets, boxed text, and captions as belonging to this place "
        "when they clearly describe it.\n"
        "- Sensory description, hazards, who is present, and what is going on → summary / "
        "current_events / local_tone.\n"
        "- Named sub-areas, rooms, decks, or landmarks → important_locations.\n"
        "- Gossip, hearsay, or uncertain talk → rumors.\n"
        "- What locals or visitors typically know or must not know → knowledge_rules.\n\n"
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


def _build_npc_detail_prompt(
    batch: list[dict[str, Any]],
    text: str,
    campaign: dict[str, Any],
    *,
    is_retry: bool = False,
) -> str:
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

    retry_note = ""
    if is_retry:
        retry_note = (
            "RETRY: A previous pass returned little or nothing useful for this NPC. "
            "The excerpts below almost certainly contain usable personality, traits, or biography. "
            "Extract every concrete beat you can; do not return empty lists if the text has substance.\n\n"
        )

    return (
        f'Campaign: "{campaign["name"]}" ({campaign["campaign_id"]}).\n\n'
        f"{retry_note}"
        "Fill in roleplay details for EACH NPC below from the SOURCE EXCERPTS only.\n"
        "Published modules and GM notes use many layouts. Do not require a particular game's "
        "section titles. Recognize these common shapes when they appear (examples, not a checklist):\n"
        "- Labeled writeups (Appearance, Voice, Personality, Goals, What they know/hide, "
        "Attitude, Memory triggers, sample dialogue).\n"
        "- Compact stat blocks with FULL NAME / role lines and PERSONALITY or agenda lines.\n"
        "- Character callout boxes: short name header, attribute grid (Awareness/Presence/...), "
        "Skills, TRAITS (Trait: description), Tech Level / Story Points, then a biography that "
        "covers motives, relationships, and behavior — not just appearance.\n"
        "- Attribute/skill grids followed by TRAITS (Name: description) and a prose biography.\n"
        "- Free prose paragraphs that open with the character's name.\n\n"
        "Field mapping (layout-agnostic):\n"
        "- Keep/refine role from the stub; do not blank it.\n"
        "- Motives, temperament, labeled traits, agendas → personality and llm_rules.\n"
        "- Appearance, public reputation, social role → public_facts.\n"
        "- What this NPC knows, plans, or can share → known_facts.\n"
        "- What they conceal → secrets.\n"
        "- How they speak or behave in play → voice and llm_rules.\n"
        "- Relationship hooks (allies, rivals, who they protect) → llm_rules.\n"
        "- Attribute/skill numbers are optional; prefer roleplay substance over statting.\n\n"
        'Respond with exactly {"npcs": [ ... ]} where each array item has this shape:\n\n'
        "{\n"
        '  "id": "<exact id from the list below>",\n'
        '  "pronouns": {"subject":"she","object":"her","possessive_adjective":"her",'
        '"possessive_pronoun":"hers","reflexive":"herself"},\n'
        '  "race": "species/ancestry if stated, else \\"unknown\\"",\n'
        '  "gender": "male|female|nonbinary|unspecified",\n'
        '  "voice": ["short spoken-delivery traits"],\n'
        '  "personality": ["distinct traits and motives from the source"],\n'
        '  "public_facts": ["appearance and reputation others can observe or commonly know"],\n'
        '  "known_facts": ["what THIS NPC knows / is pursuing"],\n'
        '  "secrets": ["what THIS NPC will not willingly reveal"],\n'
        '  "llm_rules": ["actionable play guidance from the source"]\n'
        "}\n\n"
        "CRITICAL EXTRACTION RULES:\n"
        "- EXTRACT over invent. Copy concrete facts from the source; do not replace them with generics.\n"
        "- Empty lists only when the source truly has nothing for that field.\n"
        "- If the source says she/her (or uses she/her language), pronouns and gender must match.\n"
        "- public_facts = what OTHER characters already know or can see.\n"
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
    *,
    is_retry: bool = False,
) -> dict[str, dict[str, Any]]:
    needles: list[str] = []
    for loc in batch:
        needles.extend(_distinctive_name_needles(loc["name"], loc["id"]))
    excerpt = _relevant_excerpts(document_text, needles)
    prompt = _build_location_detail_prompt(batch, excerpt, campaign, is_retry=is_retry)
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
    *,
    is_retry: bool = False,
) -> dict[str, dict[str, Any]]:
    needles: list[str] = []
    for npc in batch:
        needles.extend(_distinctive_name_needles(npc["name"], npc["id"]))
    excerpt = _relevant_excerpts(document_text, needles)
    prompt = _build_npc_detail_prompt(batch, excerpt, campaign, is_retry=is_retry)
    try:
        detail_raw = _call_llm_json(_SYSTEM_PROMPT, prompt, warnings=warnings, step="npc detail")
        detail_items = _coerce_container(detail_raw, "npcs")
        return _index_details_by_stub_id(detail_items, batch)
    except Exception as ex:
        warnings.append(f"NPC detail batch failed ({[npc['id'] for npc in batch]}): {ex}")
        return {}


def _resolve_import_mode(
    mode: str | None,
    *,
    existing_campaign_id: str | None,
    warnings: list[str],
) -> tuple[str, str | None]:
    mode = (mode or "create_new").strip().lower()
    if mode not in VALID_MODES:
        warnings.append(f"Unknown mode '{mode}'; defaulting to 'create_new'.")
        mode = "create_new"
    existing_campaign_id = (existing_campaign_id or "").strip() or None
    if mode == "merge" and not existing_campaign_id:
        warnings.append("mode='merge' but no existing_campaign_id was provided; treating as create_new.")
        mode = "create_new"
    return mode, existing_campaign_id


def outline_campaign_document(
    text: str,
    mode: str,
    existing_campaign_id: str | None = None,
    existing_location_ids: list[str] | None = None,
    existing_npc_names: list[str] | None = None,
) -> dict[str, Any]:
    """Pass 1: campaign + location/NPC stubs (no detail LLM sweeps).

    Used by the editor as a short Funnel hop before per-batch detail calls.
    """
    warnings: list[str] = []
    mode, existing_campaign_id = _resolve_import_mode(
        mode, existing_campaign_id=existing_campaign_id, warnings=warnings
    )
    existing_location_ids = [str(x).strip() for x in (existing_location_ids or []) if str(x).strip()]
    existing_npc_names = [str(x).strip() for x in (existing_npc_names or []) if str(x).strip()]

    if not text or not text.strip():
        raise ContentImportError("No document text provided")

    full_document_text = text
    use_stat_block_fast_path = _document_has_recoverable_stat_blocks(full_document_text)

    if use_stat_block_fast_path:
        warnings.append(
            "Published NON-PLAYER CHARACTERS stat blocks detected; skipping LLM outline "
            "and seeding NPC stubs from FULL NAME blocks (detail sweep still runs)."
        )
        outline_raw = {"campaign": {}, "locations": [], "npcs": []}
    else:
        outline_document_text = _truncate_input(text, warnings)
        outline_source = _prepare_outline_document_text(outline_document_text, warnings)
        outline_prompt = _build_outline_prompt(
            outline_source, mode, existing_campaign_id, existing_location_ids, existing_npc_names
        )
        try:
            outline_raw = _call_llm_json(
                _SYSTEM_PROMPT, outline_prompt, warnings=warnings, step="outline"
            )
        except Exception as ex:
            if isinstance(ex, LLMProviderError):
                raise ContentImportError(str(ex)) from ex
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

    if (
        mode == "create_new"
        and campaign.get("campaign_id") in {"campaign", ""}
        and (not campaign.get("name") or str(campaign.get("name")).strip().casefold() in {"campaign", ""})
    ):
        guessed = _guess_campaign_title(full_document_text)
        if guessed:
            campaign["name"] = guessed
            campaign["campaign_id"] = snake_case(guessed)
            warnings.append(f"Outline omitted campaign title; inferred '{guessed}' from the document head.")

    existing_location_set = {snake_case(loc_id) for loc_id in existing_location_ids}
    location_ids = {loc["id"] for loc in location_stubs} | existing_location_set
    npc_stubs = _merge_outline_npcs_with_stat_blocks(
        npc_stubs,
        full_document_text,
        mode=mode,
        existing_npc_names=existing_npc_names,
        location_ids=location_ids,
        warnings=warnings,
    )
    npc_stubs = _merge_outline_npcs_with_callouts(
        npc_stubs,
        full_document_text,
        mode=mode,
        existing_npc_names=existing_npc_names,
        location_ids=location_ids,
        warnings=warnings,
    )
    location_stubs = _ensure_locations_for_npc_hints(
        location_stubs, npc_stubs, warnings=warnings
    )
    for stub in npc_stubs:
        hint = str(stub.get("first_location_id") or "").strip()
        if not hint and location_stubs:
            stub["first_location_id"] = location_stubs[0]["id"]

    return {
        "campaign": campaign,
        "locations": location_stubs,
        "npcs": npc_stubs,
        "warnings": warnings,
    }


def _normalize_campaign_meta(campaign: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize campaign fields for detail hops without rewriting assigned ids.

    Merge mode and the editor pass a concrete ``campaign_id`` that must be stamped
    onto locations/NPCs unchanged (e.g. ``MultiTest``). Only invent via snake_case
    when no id was provided.
    """
    raw = campaign if isinstance(campaign, dict) else {}
    raw_id = str(raw.get("campaign_id") or "").strip()
    if raw_id:
        campaign_id = raw_id
    else:
        campaign_id = snake_case(raw.get("name") or "campaign")
    return {
        "campaign_id": campaign_id,
        "name": str(raw.get("name") or campaign_id.replace("_", " ").title()).strip(),
        "description": str(raw.get("description") or "").strip(),
    }


def _optional_location_id(value: Any) -> str:
    """Snake-case a location id, or return empty when the source had none.

    ``snake_case("")`` yields ``\"unnamed\"``, which must not become a fake location.
    """
    raw = str(value or "").strip()
    if not raw:
        return ""
    return snake_case(raw)


def _normalize_location_stubs(raw_locations: Any) -> list[dict[str, Any]]:
    stubs: list[dict[str, Any]] = []
    for raw in _as_list(raw_locations):
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip()
        loc_id = snake_case(raw.get("id") or name or "location")
        if not loc_id:
            continue
        stubs.append(
            {
                "id": loc_id,
                "name": name or loc_id.replace("_", " ").title(),
                "type": str(raw.get("type") or "other").strip() or "other",
                "seeded_from_stat_blocks": bool(raw.get("seeded_from_stat_blocks")),
            }
        )
    return stubs


def _normalize_npc_stubs(raw_npcs: Any) -> list[dict[str, Any]]:
    stubs: list[dict[str, Any]] = []
    for raw in _as_list(raw_npcs):
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip()
        npc_id = snake_case(raw.get("id") or name or "npc")
        if not npc_id or not name:
            continue
        other = [
            _optional_location_id(x)
            for x in _as_list(raw.get("other_location_ids"))
            if str(x).strip()
        ]
        stubs.append(
            {
                "id": npc_id,
                "name": name,
                "role": str(raw.get("role") or "").strip(),
                "first_location_id": _optional_location_id(raw.get("first_location_id")),
                "other_location_ids": [x for x in other if x],
                "seeded_from_stat_blocks": bool(raw.get("seeded_from_stat_blocks")),
                "seed_personality": str(raw.get("seed_personality") or "").strip(),
                "seed_personal_agenda": str(raw.get("seed_personal_agenda") or "").strip(),
            }
        )
    return stubs


def detail_location_batch(
    text: str,
    campaign: dict[str, Any],
    location_stubs: list[dict[str, Any]],
    *,
    is_retry: bool = False,
    allow_seed_fallback: bool = False,
) -> dict[str, Any]:
    """Pass 2: fill one location batch from document excerpts.

    One LLM call per Funnel hop. Editor should retry with ``is_retry=True`` when
    sparse; only pass ``allow_seed_fallback=True`` after LLM retries are exhausted.
    """
    warnings: list[str] = []
    if not text or not text.strip():
        raise ContentImportError("No document text provided")
    campaign = _normalize_campaign_meta(campaign)
    batch = _normalize_location_stubs(location_stubs)
    if not batch:
        raise ContentImportError("location-details requires at least one location stub")
    if len(batch) > MAX_LOCATIONS_PER_BATCH:
        raise ContentImportError(
            f"location-details accepts at most {MAX_LOCATIONS_PER_BATCH} locations per call"
        )

    details_by_id: dict[str, dict[str, Any]] = {}
    details_by_id.update(
        _fetch_location_details(batch, text, campaign, warnings, is_retry=is_retry)
    )
    for stub in batch:
        detail = details_by_id.get(stub["id"], {})
        if _location_detail_is_sparse(detail) and allow_seed_fallback:
            if stub.get("seeded_from_stat_blocks"):
                details_by_id[stub["id"]] = _location_detail_from_seed(stub)
                warnings.append(
                    f"Location '{stub['id']}': detail sparse after LLM; fell back to "
                    "minimal location stub defaults."
                )
            elif not detail:
                warnings.append(
                    f"Location '{stub['id']}': no detail returned; using minimal defaults."
                )
        elif _location_detail_is_sparse(detail) and not detail:
            warnings.append(
                f"Location '{stub['id']}': LLM detail sparse (editor may retry this hop)."
            )

    locations = [
        _merge_location_detail(stub, details_by_id.get(stub["id"], {}), campaign["campaign_id"])
        for stub in batch
    ]
    return {"locations": locations, "warnings": warnings}


def detail_npc_batch(
    text: str,
    campaign: dict[str, Any],
    npc_stubs: list[dict[str, Any]],
    *,
    is_retry: bool = False,
    allow_seed_fallback: bool = False,
) -> dict[str, Any]:
    """Pass 3: fill one NPC batch from document excerpts.

    One LLM call per Funnel hop. Prefer editor-driven retries over in-hop seeds.
    ``allow_seed_fallback`` enables last-resort ALIEN stub seeds or generic
    excerpt seeding only after LLM hops have failed.
    """
    warnings: list[str] = []
    if not text or not text.strip():
        raise ContentImportError("No document text provided")
    campaign = _normalize_campaign_meta(campaign)
    batch = _normalize_npc_stubs(npc_stubs)
    if not batch:
        raise ContentImportError("npc-details requires at least one NPC stub")
    if len(batch) > MAX_NPCS_PER_BATCH:
        raise ContentImportError(
            f"npc-details accepts at most {MAX_NPCS_PER_BATCH} NPC(s) per call"
        )

    details_by_id: dict[str, dict[str, Any]] = {}
    details_by_id.update(
        _fetch_npc_details(batch, text, campaign, warnings, is_retry=is_retry)
    )
    for stub in batch:
        detail = details_by_id.get(stub["id"], {})
        if not _npc_detail_is_sparse(detail):
            continue
        if allow_seed_fallback:
            if _npc_stub_has_stat_seed(stub):
                details_by_id[stub["id"]] = _npc_detail_from_stat_seed(stub)
                warnings.append(
                    f"NPC '{stub['id']}': detail sparse after LLM; fell back to "
                    "PERSONALITY / PERSONAL AGENDA from the stat block."
                )
                continue
            seeded = _seed_npc_detail_from_document(
                str(stub.get("name") or ""),
                str(stub.get("role") or ""),
                text,
            )
            if seeded is not None:
                details_by_id[stub["id"]] = seeded
                warnings.append(
                    f"NPC '{stub['id']}': detail sparse after LLM; seeded from "
                    "document excerpts as a last resort."
                )
                continue
            if not detail:
                warnings.append(
                    f"NPC '{stub['id']}': no detail returned; using minimal defaults. "
                    "Local small models often need content_creation set to Groq or a "
                    "stronger local model for dense NPC writeups."
                )
        elif not detail:
            warnings.append(
                f"NPC '{stub['id']}': LLM detail sparse (editor may retry this hop)."
            )

    npcs = [
        _merge_npc_detail(
            stub, details_by_id.get(stub["id"], {}), campaign["campaign_id"], warnings
        )
        for stub in batch
    ]
    return {"npcs": npcs, "warnings": warnings}


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

    Orchestrates outline + detail batches for local/dev callers. Production
    Yaml Editor should call the chunked endpoints instead so Tailscale Funnel
    is not holding one HTTP request across every LLM pass.

    Returns: {"campaign": {...}, "locations": [...], "npcs": [...], "warnings": [...]}
    Raises ContentImportError if the document cannot be analyzed at all.
    """
    outline = outline_campaign_document(
        text,
        mode,
        existing_campaign_id=existing_campaign_id,
        existing_location_ids=existing_location_ids,
        existing_npc_names=existing_npc_names,
    )
    warnings = list(outline.get("warnings") or [])
    campaign = outline["campaign"]
    location_stubs = list(outline.get("locations") or [])
    npc_stubs = list(outline.get("npcs") or [])

    locations: list[dict[str, Any]] = []
    for batch in _batches(location_stubs, MAX_LOCATIONS_PER_BATCH):
        result = detail_location_batch(text, campaign, batch)
        if any(_location_detail_is_sparse(loc) for loc in (result.get("locations") or [])):
            result = detail_location_batch(text, campaign, batch, is_retry=True)
        if any(_location_detail_is_sparse(loc) for loc in (result.get("locations") or [])):
            result = detail_location_batch(
                text, campaign, batch, is_retry=True, allow_seed_fallback=True
            )
        warnings.extend(result.get("warnings") or [])
        locations.extend(result.get("locations") or [])

    npcs: list[dict[str, Any]] = []
    for batch in _batches(npc_stubs, MAX_NPCS_PER_BATCH):
        result = detail_npc_batch(text, campaign, batch)
        if any(_npc_detail_is_sparse(npc) for npc in (result.get("npcs") or [])):
            result = detail_npc_batch(text, campaign, batch, is_retry=True)
        if any(_npc_detail_is_sparse(npc) for npc in (result.get("npcs") or [])):
            result = detail_npc_batch(
                text, campaign, batch, is_retry=True, allow_seed_fallback=True
            )
        warnings.extend(result.get("warnings") or [])
        npcs.extend(result.get("npcs") or [])

    return {
        "campaign": campaign,
        "locations": locations,
        "npcs": npcs,
        "warnings": warnings,
    }
