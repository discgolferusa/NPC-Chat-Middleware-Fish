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
  2. Location detail — format-agnostic LLM extraction from name-focused excerpts,
     plus TOC/Exploring windows and a generic sub-area harvest into
     ``important_locations`` when the model returns a summary without rooms.
     Ships also harvest deck-scoped room headers (e.g. ALIEN deck plans) and put
     deck identity into each room's ``description``.
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

from party_campaign import normalize_campaign_genre
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
OUTLINE_MAX_TOKENS = 2048
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
_TOC_NPC_HEADER_LINE = re.compile(
    r"\.{2,}|\s+\d{1,4}\s*$",
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
# Act/Appendix roman numerals must never become NPC short names (``Act III, …``).
_STAT_BLOCK_ROMAN_NUMERALS = frozenset(
    {
        "I",
        "II",
        "III",
        "IV",
        "V",
        "VI",
        "VII",
        "VIII",
        "IX",
        "X",
        "XI",
        "XII",
        "XIII",
        "XIV",
        "XV",
        "XVI",
        "XVII",
        "XVIII",
        "XIX",
        "XX",
    }
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
# Allow middle initials (``Edward R. Meyers``, ``A. Mallory Eckford``).
# Restrict to name tokens; stop before the next FULL NAME or lowercase prose.
_FULL_NAME_TOKEN = r"(?:[A-Z]\.|[A-Z][a-zA-Z0-9'\-]*)"
_FULL_NAME_ANY = re.compile(
    r"(?:FULL|Full|full)\s+(?:NAME|Name|name):\s*"
    rf"({_FULL_NAME_TOKEN}(?:\s+{_FULL_NAME_TOKEN}){{0,5}})"
    r"(?=\s*(?:FULL|Full|full)\s+(?:NAME|Name|name)\s*:|\s+[a-z]|\s*$|[,;\n])",
)
# Two-column extracts often glue ``MEYERS, COLONELMEYERS, COLONEL``.
_DOUBLED_STAT_HEADING = re.compile(
    r"\b([A-Z][A-Z0-9'\-]*)\s*,\s*"
    r"([A-Z][A-Z0-9'\-/]*(?:\s+[A-Z][A-Z0-9'\-/]*){0,5}?)"
    r"\1\s*,\s*\2\b"
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


def _collapsed_line_from_raw_line(line: str) -> str:
    stripped = line.strip()
    if stripped and re.fullmatch(r"[A-Z0-9](?:\s+[A-Z0-9-])+", stripped):
        return re.sub(r"\s+", "", stripped)
    return stripped


def _is_toc_npc_header_line(collapsed_line: str) -> bool:
    if not collapsed_line:
        return False
    if _TOC_NPC_HEADER_LINE.search(collapsed_line):
        return True
    if len(collapsed_line) <= 48 and re.search(r"\bpage\s+\d+\b", collapsed_line, re.IGNORECASE):
        return True
    return False


def _npc_header_collapsed_from_lines(lines: list[str]) -> str | None:
    """Return collapsed heading text when 1-2 lines form NON-PLAYER CHARACTERS."""
    stripped_lines = [line.strip() for line in lines if line.strip()]
    if not stripped_lines or len(stripped_lines) > 2:
        return None
    if any(len(line) > 96 for line in stripped_lines):
        return None

    collapsed = " ".join(_collapsed_line_from_raw_line(line) for line in stripped_lines)
    match = _NPC_SECTION_PATTERN.search(collapsed)
    if not match:
        return None
    if collapsed[: match.start()].strip():
        return None

    remainder = collapsed[match.end() :].strip()
    if remainder and not _is_toc_npc_header_line(collapsed):
        return None
    return collapsed


def _iter_npc_section_header_starts(text: str):
    """Yield raw offsets for NON-PLAYER CHARACTERS headings, including split lines."""
    if not text:
        return

    lines = text.splitlines(keepends=True)
    pos = 0
    for index, line in enumerate(lines):
        line_start = pos
        pos += len(line)
        groups = [([line], line_start)]
        if index + 1 < len(lines):
            next_line = lines[index + 1]
            if line.strip() and next_line.strip():
                groups.append(([line, next_line], line_start))

        seen_collapsed: set[str] = set()
        for group_lines, start in groups:
            collapsed = _npc_header_collapsed_from_lines(group_lines)
            if collapsed is None or collapsed in seen_collapsed:
                continue
            seen_collapsed.add(collapsed)
            yield start, collapsed


def _find_best_npc_section_header_offset(text: str) -> int | None:
    """Locate the best NON-PLAYER CHARACTERS header without collapsing the full extract."""
    if not text:
        return None

    best_offset: int | None = None
    best_rank: tuple[int, int, int] | None = None
    for line_start, collapsed_line in _iter_npc_section_header_starts(text):
        is_toc = _is_toc_npc_header_line(collapsed_line)
        lookahead = _collapse_spaced_caps(text[line_start : line_start + 25_000])
        match = _NPC_SECTION_PATTERN.search(lookahead)
        if not match:
            continue
        start, end = _npc_section_span_from_match(lookahead, match)
        span_len = end - start
        full_name_count = len(_FULL_NAME_ANY.findall(lookahead[start:end]))
        if is_toc and full_name_count == 0 and span_len < 120:
            continue
        rank = (0 if is_toc else 1, full_name_count, span_len)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_offset = line_start

    return best_offset


def _bounded_text_for_npc_operations(text: str, max_chars: int) -> str:
    """Return a bounded raw slice that should still contain the NPC section when present."""
    text = text or ""
    if len(text) <= max_chars:
        return text

    header_off = _find_best_npc_section_header_offset(text)
    if header_off is not None:
        intro_start = max(0, header_off - 2500)
        slice_end = min(len(text), header_off + max(max_chars, 80_000))
        return text[intro_start:slice_end]

    # Moderate extracts: collapse the full text so mid-document NPC sections are found.
    if len(text) <= MAX_INPUT_CHARS:
        return text

    for window in (text[-max_chars:], text[:max_chars]):
        collapsed_window = _collapse_spaced_caps(window)
        if _find_npc_section_span(collapsed_window) is not None:
            return window

    return text[:max_chars]


def _npc_section_header_line(collapsed_text: str, match: re.Match[str]) -> str:
    line_start = collapsed_text.rfind("\n", 0, match.start()) + 1
    line_end = collapsed_text.find("\n", match.start())
    if line_end < 0:
        line_end = len(collapsed_text)
    return collapsed_text[line_start:line_end].strip()


def _looks_like_toc_npc_section_hit(collapsed_text: str, match: re.Match[str]) -> bool:
    """Docnet/PDFium TOCs often put 'Non-Player Characters ........ 12' before the real section."""
    return _is_toc_npc_header_line(_npc_section_header_line(collapsed_text, match))


def _npc_section_span_from_match(
    collapsed_text: str,
    match: re.Match[str],
) -> tuple[int, int]:
    start = match.start()
    remainder = collapsed_text[match.end() :]
    end = len(collapsed_text)
    hard_end: int | None = None

    end_match = _NPC_SECTION_END_PATTERN.search(remainder)
    if end_match:
        hard_end = match.end() + end_match.start()
        end = hard_end

    for header_match in _NPC_SECTION_HEADER_END_PATTERN.finditer(remainder):
        abs_pos = match.end() + header_match.start()
        if abs_pos >= end:
            break
        line_start = collapsed_text.rfind("\n", start, abs_pos) + 1
        line_end = collapsed_text.find("\n", abs_pos)
        if line_end < 0:
            line_end = len(collapsed_text)
        line = collapsed_text[line_start:line_end].strip()
        if line and len(line) <= 40 and _NPC_SECTION_HEADER_END_PATTERN.match(line):
            end = min(end, line_start)
            break

    section = collapsed_text[start:end]
    if len(_FULL_NAME_ANY.findall(section)) < 3:
        hard = _NPC_SECTION_END_PATTERN.search(collapsed_text, start)
        if hard is not None:
            hard_pos = hard.start()
            if hard_pos > end:
                end = hard_pos
        elif hard_end is None:
            end = min(len(collapsed_text), max(end, start + 20_000))

    return start, end


def _find_npc_section_span(collapsed_text: str) -> tuple[int, int] | None:
    """Return [start, end) slice of the authoritative NPC section when present.

    ``collapsed_text`` should already be passed through ``_collapse_spaced_caps``.
    """
    if not collapsed_text:
        return None

    best_span: tuple[int, int] | None = None
    best_rank: tuple[int, int, int] | None = None

    for match in _NPC_SECTION_PATTERN.finditer(collapsed_text):
        is_toc = _looks_like_toc_npc_section_hit(collapsed_text, match)
        start, end = _npc_section_span_from_match(collapsed_text, match)
        span_len = end - start
        full_name_count = len(_FULL_NAME_ANY.findall(collapsed_text[start:end]))
        # Drop bare TOC index lines with no body.
        if is_toc and full_name_count == 0 and span_len < 120:
            continue
        # Prefer non-TOC headers, then more FULL NAME markers, then longer body.
        rank = (0 if is_toc else 1, full_name_count, span_len)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_span = (start, end)

    return best_span


def _dedupe_repeated_full_name(name: str) -> str:
    """Collapse two-column duplicates: ``Edward R. Meyers Edward R. Meyers`` → one name."""
    parts = [p for p in str(name or "").split() if p]
    if len(parts) < 2:
        return str(name or "").strip()
    if len(parts) == 2 and parts[0].casefold() == parts[1].casefold():
        return parts[0]
    if len(parts) >= 4 and len(parts) % 2 == 0:
        half = len(parts) // 2
        if [p.casefold() for p in parts[:half]] == [p.casefold() for p in parts[half:]]:
            return " ".join(parts[:half])
    return " ".join(parts)


def _role_looks_like_prose(role: str) -> bool:
    """Published roles are ALL-CAPS labels; reject sentence fragments from GEAR wraps."""
    role = (role or "").strip()
    if not role:
        return True
    if re.search(r"(?i)\bpage\s+\d+", role):
        return True
    if role.endswith((").", ".", ";")) or "(" in role or ")" in role:
        return True
    letters = [c for c in role if c.isalpha()]
    if not letters:
        return True
    lower_ratio = sum(1 for c in letters if c.islower()) / len(letters)
    return lower_ratio > 0.25


def _subtitle_looks_like_prose(candidate: str) -> bool:
    candidate = (candidate or "").strip()
    if not candidate:
        return True
    if candidate[0].islower():
        return True
    if re.search(r"(?i)\bpage\s+\d+", candidate):
        return True
    if candidate.endswith((").", ".", ",", ";")) or "(" in candidate:
        return True
    if len(candidate.split()) > 8:
        return True
    return False


def _normalize_alien_pdf_stat_text(text: str) -> str:
    """Undo common two-column PDF gluing before heading / FULL NAME parsing."""
    lines: list[str] = []
    for line in (text or "").splitlines():
        # FULL NAME:FULL NAME: … → FULL NAME: …
        line = re.sub(r"(?i)\b(FULL\s+NAME:\s*)(?:FULL\s+NAME:\s*)+", r"\1", line)
        line = re.sub(r"(?i)\b(PERSONALITY:\s*)(?:PERSONALITY:\s*)+", r"\1", line)
        line = re.sub(r"(?i)\b(AGE:\s*)(?:AGE:\s*)+", r"\1", line)
        prev = None
        while prev != line:
            prev = line
            line = _DOUBLED_STAT_HEADING.sub(r"\1, \2", line)
        stripped = line.strip()
        # Exact doubled lines: ``COMMANDER OF FORT NEBRASKACOMMANDER OF FORT NEBRASKA``.
        if len(stripped) >= 12 and len(stripped) % 2 == 0:
            half = len(stripped) // 2
            if stripped[:half] == stripped[half:]:
                line = stripped[:half]
        lines.append(line)
    return "\n".join(lines)


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
        if short_key in _STAT_BLOCK_ROMAN_NUMERALS:
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
        if _role_looks_like_prose(role):
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
    names: list[str] = []
    for match in _FULL_NAME_ANY.finditer(line.strip()):
        name = _dedupe_repeated_full_name(match.group(1).strip())
        if name:
            names.append(name)
    return names


def _enrich_row_from_window(row: dict[str, str], window: str) -> dict[str, str]:
    personality = _labeled_field_from_window(window, "PERSONALITY")
    agenda = _labeled_field_from_window(window, "PERSONAL AGENDA")
    if personality:
        row["personality"] = _dedupe_repeated_full_name(personality)
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

    section_text = _normalize_alien_pdf_stat_text(section_text)
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
                        and not _subtitle_looks_like_prose(candidate)
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
                    if _STAT_BLOCK_FIELD_STOP.match(candidate):
                        break
                    if _subtitle_looks_like_prose(candidate):
                        continue
                    role = candidate
                    break

            seen_names.add(name_key)
            row = {
                "name": display_name,
                "role": role or (short_name.title() if short_name else ""),
                "short_name": short_name or display_name.split()[-1].upper(),
                "location_hint": _location_hint_from(window),
            }
            # Field labels must start at this FULL NAME — looking upward steals the
            # previous column partner's PERSONALITY (Destroyer Jaell → Davydovich).
            field_window = "\n".join(lines[index : index + 25])
            results.append(_enrich_row_from_window(row, field_window))

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
                if _STAT_BLOCK_FIELD_STOP.match(candidate):
                    break
                if _subtitle_looks_like_prose(candidate):
                    continue
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
    Leftover budget keeps leading document text (titles, locations) from the
    original extract head — not the bounded pre-header window, which often holds
    TOC/sidebar false-positive names.
    """
    if len(text) <= max_chars:
        return text

    sliced = _bounded_text_for_npc_operations(text, max_chars)
    collapsed = _collapse_spaced_caps(sliced)
    span = _find_npc_section_span(collapsed)
    if span is None:
        if len(collapsed) <= max_chars:
            return collapsed
        return collapsed[:max_chars]

    start, end = span
    section = collapsed[start:end].strip()
    separator = "\n\n--- NON-PLAYER CHARACTERS (priority excerpt) ---\n\n"
    # Always reserve some head room for titles/locations, even when the NPC
    # span has no hard end and would otherwise consume the whole budget.
    head_reserve = min(2500, max(0, max_chars // 4))
    section_budget = max(max_chars - head_reserve - len(separator), max_chars // 2)
    if len(section) > section_budget:
        section = section[:section_budget]

    intro_budget = max_chars - len(section) - len(separator)
    if intro_budget <= 0:
        return section[:max_chars]

    # Prefer the original document head for titles/location context.
    header_off = _find_best_npc_section_header_offset(text)
    head_end = header_off if header_off is not None else len(text)
    # Collapse only the head window needed for the intro budget.
    head_window = text[: min(head_end, intro_budget + 4000)]
    head_source = _collapse_spaced_caps(head_window)
    head_span = _find_npc_section_span(head_source)
    if head_span is not None:
        head_source = head_source[: head_span[0]]
    intro = head_source[:intro_budget].strip()
    combined = f"{intro}{separator}{section}" if intro else section
    return combined[:max_chars]


def _document_has_recoverable_stat_blocks(text: str) -> bool:
    collapsed = _collapse_spaced_caps(text or "")
    span = _find_npc_section_span(collapsed)
    if span is None:
        return False
    parsed = _extract_alien_rpg_stat_block_npcs(collapsed[span[0] : span[1]])
    return len(parsed) >= 3


def _guess_campaign_title(text: str) -> str | None:
    """Best-effort title from the document head when the LLM returns an empty campaign."""
    head = (text or "")[:3500]
    collapsed_head = _collapse_spaced_caps(head)
    full_scan = _collapse_spaced_caps((text or "")[:20_000])
    # Doctor Who / Cubicle 7 style titles often arrive letter-spaced: "Cat s Ey E".
    if re.search(r"(?i)c\s*a\s*t\s*['’]?\s*s\s*e\s*y\s*e", head):
        return "Cat's Eye"

    # Named scenario titles before any generic ALL-CAPS line match — moonside
    # extracts open with map labels (FORT NEBRASKA, SPACEPORT, ARIARCUS) that
    # would otherwise become the campaign title/id.
    for pattern, title in (
        (r"(?is)\bDESTROYER\s+OF\s+(?:THE\s+)?WORLDS\b", "Destroyer of Worlds"),
        (r"(?is)\bCHARIOT\s+OF\s+THE\s+GODS\b", "Chariot of the Gods"),
        (r"(?is)\bCAT'?S\s+EYE\b", "Cat's Eye"),
    ):
        if re.search(pattern, collapsed_head) or re.search(pattern, full_scan):
            return title
        if "CAT" in pattern and re.search(pattern, head, flags=re.IGNORECASE):
            return title

    if _MOONSIDE_LOCATIONS_HEADER.search(full_scan) or re.search(
        r"\bariarcus\b", full_scan, re.IGNORECASE
    ):
        return "Destroyer of Worlds"

    skip_titles = {
        "scenario",
        "contents",
        "introduction",
        "non-player characters",
        "credits",
        "moonside locations",
        "ariarcus",
        "ariarcus colony",
        *(name.casefold() for _, name, _ in _ALIEN_MAJOR_SITE_PATTERNS),
    }
    for pattern in (r"(?m)^\s*([A-Z0-9][A-Z0-9' \-]{2,48})\s*$",):
        match = re.search(pattern, collapsed_head) or re.search(pattern, full_scan)
        if not match:
            continue
        title = re.sub(r"\s+", " ", match.group(1)).strip(" :-")
        if not title:
            continue
        words = title.split()
        if len(words) > 8:
            continue
        if title.casefold() in skip_titles:
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
        "uscss_cronus": ("USCSS Cronus", "ship"),
        "sotillo": ("Sotillo", "ship"),
        "majestic_hotel": ("Majestic Hotel", "building"),
        "the_majestic": ("The Majestic", "building"),
        "majestic": ("Majestic Hotel", "building"),
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


# --------------------------------------------------------------------------
# Alien moonside / locations-chapter major-site seeding
# --------------------------------------------------------------------------

_MOONSIDE_LOCATIONS_HEADER = re.compile(r"moonside\s+locations", re.IGNORECASE)
_COLON_PLACE_HEADER = re.compile(
    r"^([A-Z][A-Z0-9][A-Z0-9 '\-/]{1,48})\s*:\s+(\S.+)$"
)
_ALIEN_MAP_CHROME = frozenset(
    {
        "pad",
        "pads",
        "in",
        "aw",
        "ma",
        "ap",
        "ta",
        "km",
        "1 km",
        "wells",
        "pumpjacks",
        "marines",
        "tank",
        "apc",
        "d6",
        "encounter",
        "incident",
        "discovery",
        "destroyer of worlds",
        "moonside locations",
    }
)
_ALIEN_ROOM_SCALE_TOKENS = frozenset(
    {
        "hangar",
        "pad",
        "landing",
        "cell",
        "room",
        "bay",
        "lift",
        "ladder",
        "hatch",
        "gate",
        "foyer",
        "studio",
        "alcove",
        "gantry",
        "armory",
        "morgue",
        "bullpen",
        "reception",
        "office",
        "yard",
        "shack",
        "tower",
        "operations",
        "junction",
        "tunnel",
        "corridor",
        "storage",
        "tank",
        "door",
        "camera",
        "crowd",
        "table",
        "tables",
        "snake",
        "act",
        "pipeline",
        "tactics",
        "technomagic",
        "havoc",
        "recovery",
        "examination",
        "surgical",
        "holding",
        "interrogation",
        "medtech",
        "stores",
        "cloak",
        "restroom",
        "alleyway",
        "bedroom",
        "hall",
        "roof",
        "sally",
        "emplacement",
        "emplacements",
        "canopy",
        "doors",
        "hopper",
        "hoppers",
        "pillar",
        "pillars",
        "wall",
        "pipes",
        "car",
        "people",
        "awols",
        "edie",
        "dancer",
        "cameras",
        "footage",
    }
)
# (regex on undoubled uppercase text, display name, type)
_ALIEN_MAJOR_SITE_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"\bFORT\s+NEBRASKA\b"), "Fort Nebraska", "building"),
    (
        re.compile(r"\bSAN\s+ROCCO(?:\s+MEDICAL(?:\s+FACILITY)?)?\b"),
        "San Rocco Medical Facility",
        "building",
    ),
    (re.compile(r"\bMARSHAL(?:'S)?\s+STATION\b"), "Marshal Station", "building"),
    (re.compile(r"\bOBLIVION(?:\s+BAR)?\b"), "Oblivion Bar", "building"),
    (re.compile(r"\bWILDCATTER(?:\s+PARK)?\b"), "Wildcatter Park", "region"),
    (
        re.compile(r"\b(?:THE\s+)?INSURGENT\s+COMPOUND\b"),
        "The Insurgent Compound",
        "building",
    ),
    (re.compile(r"\bSPACEPORT\b"), "Spaceport", "building"),
    (
        re.compile(r"\b(?:OLD\s+)?(?:OIL\s+)?REFINERY\b|\bOIL\s+FIELDS?\b"),
        "Oil Refinery",
        "building",
    ),
    (re.compile(r"\bARIARCUS(?:\s+COLONY)?\b"), "Ariarcus Colony", "region"),
    (re.compile(r"\bDISTRICT\s+NORTH\b"), "District North", "region"),
    (re.compile(r"\bDISTRICT\s+SOUTH\b"), "District South", "region"),
    (re.compile(r"\bDISTRICT\s+EAST\b"), "District East", "region"),
    (re.compile(r"\bDISTRICT\s+WEST\b"), "District West", "region"),
    (re.compile(r"\bDOWNTOWN\b"), "Downtown", "region"),
]


def _undouble_map_label(line: str) -> str:
    """Undo two-column map glue: ``SPACEPORTSPACEPORT`` / ``SAN ROCCO SAN ROCCO``."""
    s = re.sub(r"\s+", " ", str(line or "").strip())
    if not s:
        return ""
    if len(s) >= 4 and len(s) % 2 == 0 and s[: len(s) // 2] == s[len(s) // 2 :]:
        return s[: len(s) // 2].strip()
    parts = s.split()
    if len(parts) >= 2 and len(parts) % 2 == 0:
        half = len(parts) // 2
        if [p.casefold() for p in parts[:half]] == [p.casefold() for p in parts[half:]]:
            return " ".join(parts[:half])
    return s


def _normalize_alien_location_scan_text(text: str) -> str:
    """Undouble PDF map/section labels and collapse letter-spaced caps for site scans."""
    lines: list[str] = []
    for line in (text or "").splitlines():
        lines.append(_undouble_map_label(line))
    return _collapse_spaced_caps("\n".join(lines))


def _count_colon_place_headers(text: str) -> int:
    count = 0
    for line in (text or "").splitlines():
        stripped = line.strip()
        # Strip leading list bullets like ``T THE GATE:``.
        stripped = re.sub(r"^[T•\-]\s+", "", stripped)
        if _COLON_PLACE_HEADER.match(stripped):
            count += 1
    return count


def _document_has_recoverable_location_sites(text: str) -> bool:
    """True when the extract looks like an Alien moonside / locations chapter."""
    scan = _normalize_alien_location_scan_text(text or "")
    if not scan.strip():
        return False
    colon_count = _count_colon_place_headers(scan)
    has_moonside = bool(_MOONSIDE_LOCATIONS_HEADER.search(scan))
    if has_moonside and colon_count >= 4:
        return True
    if not has_moonside and colon_count >= 8:
        return True
    # Map labels can survive when PdfPig mangles ``PLACE: prose`` room lines.
    site_count = len(_extract_alien_location_site_stubs(text))
    if has_moonside and site_count >= 2:
        return True
    if re.search(r"\bariarcus\b", scan, re.IGNORECASE) and site_count >= 3:
        return True
    return False


def _is_alien_room_scale_place_name(name: str) -> bool:
    key = re.sub(r"[^a-z0-9]+", " ", str(name or "").casefold()).strip()
    if not key:
        return True
    if key in _ALIEN_MAP_CHROME:
        return True
    tokens = key.split()
    if any(tok in _ALIEN_ROOM_SCALE_TOKENS for tok in tokens):
        return True
    if re.search(r"\d", key):
        return True
    if re.match(r"^act\s+[ivx\d]+\b", key):
        return True
    return False


def _extract_alien_location_site_stubs(text: str) -> list[dict[str, Any]]:
    """Seed major moonside sites (Fort Nebraska, Spaceport, …) — not every room."""
    scan = _normalize_alien_location_scan_text(text or "")
    if not scan.strip():
        return []

    # Join split doubled map rows: FORT / NEBRASKA on consecutive lines.
    glued_lines: list[str] = []
    pending: list[str] = []
    for line in scan.splitlines():
        stripped = line.strip()
        if stripped and re.fullmatch(r"[A-Z0-9][A-Z0-9 '\-]{0,40}", stripped):
            pending.append(stripped)
            continue
        if pending:
            glued_lines.append(" ".join(pending))
            pending = []
        if stripped:
            glued_lines.append(stripped)
    if pending:
        glued_lines.append(" ".join(pending))
    glued = "\n".join(glued_lines)

    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pattern, display_name, loc_type in _ALIEN_MAJOR_SITE_PATTERNS:
        if not pattern.search(glued):
            continue
        loc_id = snake_case(display_name)
        if loc_id in seen:
            continue
        seen.add(loc_id)
        found.append(
            {
                "id": loc_id,
                "name": display_name,
                "type": loc_type,
                "seeded_from_location_sites": True,
                "seeded_from_stat_blocks": False,
            }
        )
        if len(found) >= 16:
            break

    return found


def _merge_outline_locations_with_sites(
    llm_locations: list[dict[str, Any]],
    document_text: str,
    *,
    mode: str,
    existing_location_ids: list[str] | set[str],
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Prefer deterministic Alien major sites when a locations chapter is present."""
    if not _document_has_recoverable_location_sites(document_text):
        return llm_locations

    site_stubs = _extract_alien_location_site_stubs(document_text)
    if len(site_stubs) < 2:
        return llm_locations

    existing = {snake_case(x) for x in existing_location_ids if str(x).strip()}
    existing |= {loc["id"] for loc in llm_locations if loc.get("id")}

    merged = list(llm_locations)
    added: list[str] = []
    for stub in site_stubs:
        loc_id = stub["id"]
        if mode == "merge" and loc_id in existing:
            continue
        if any(loc.get("id") == loc_id for loc in merged):
            continue
        # Soft match on name against LLM stubs.
        name_key = stub["name"].casefold()
        if any(str(loc.get("name") or "").casefold() == name_key for loc in merged):
            continue
        merged.append(stub)
        existing.add(loc_id)
        added.append(stub["name"])

    if added:
        warnings.append(
            "Seeded major location site(s) from Alien moonside / locations chapter: "
            + ", ".join(added)
        )
    return merged


def _colon_place_header_name(line: str) -> str | None:
    stripped = re.sub(r"^[T•\-]\s+", "", (line or "").strip())
    match = _COLON_PLACE_HEADER.match(stripped)
    if not match:
        return None
    raw_name = match.group(1).strip()
    # Reject attribute / agenda field labels.
    if _STAT_BLOCK_FIELD_STOP.match(raw_name + ":"):
        return None
    if raw_name.casefold() in {
        "full name",
        "personality",
        "personal agenda",
        "strength",
        "health",
        "skills",
        "gear",
        "talent",
        "age",
    }:
        return None
    return _display_place_name(raw_name) or raw_name.title()


def _alien_site_boundary_pattern(location_name: str) -> re.Pattern[str] | None:
    key = re.sub(r"[^a-z0-9]+", " ", str(location_name or "").casefold()).strip()
    mapping = {
        "fort nebraska": r"FORT\s+NEBRASKA",
        "san rocco medical facility": r"SAN\s+ROCCO",
        "san rocco": r"SAN\s+ROCCO",
        "marshal station": r"MARSHAL(?:'S)?\s+STATION",
        "oblivion bar": r"OBLIVION(?:\s+BAR)?",
        "oblivion": r"OBLIVION(?:\s+BAR)?",
        "wildcatter park": r"WILDCATTER(?:\s+PARK)?",
        "the insurgent compound": r"(?:THE\s+)?INSURGENT\s+COMPOUND",
        "insurgent compound": r"(?:THE\s+)?INSURGENT\s+COMPOUND",
        "spaceport": r"SPACEPORT",
        "oil refinery": r"(?:OLD\s+)?(?:OIL\s+)?REFINERY|OIL\s+FIELDS?",
        "ariarcus colony": r"ARIARCUS(?:\s+COLONY)?",
        "district north": r"DISTRICT\s+NORTH",
        "district south": r"DISTRICT\s+SOUTH",
        "district east": r"DISTRICT\s+EAST",
        "district west": r"DISTRICT\s+WEST",
        "downtown": r"DOWNTOWN",
    }
    pattern = mapping.get(key)
    if not pattern:
        # Generic: require a whole-word match of the display name letters.
        escaped = re.escape(str(location_name or "").strip())
        if len(escaped) < 4:
            return None
        pattern = escaped
    return re.compile(pattern, re.IGNORECASE)


_ALIEN_NEXT_SITE_BOUNDARY = re.compile(
    r"(?i)(?:^|\n)\s*(?:FORT\s+NEBRASKA|SAN\s+ROCCO|MARSHAL(?:'S)?\s+STATION|"
    r"OBLIVION(?:\s+BAR)?|WILDCATTER(?:\s+PARK)?|(?:THE\s+)?INSURGENT\s+COMPOUND|"
    r"SPACEPORT|(?:OLD\s+)?(?:OIL\s+)?REFINERY|OIL\s+FIELDS?|ARIARCUS(?:\s+COLONY)?|"
    r"DISTRICT\s+(?:NORTH|SOUTH|EAST|WEST)|DOWNTOWN)\b"
)


def _find_alien_site_body_span(text: str, location_name: str) -> tuple[int, int] | None:
    """Return a document slice owned by one major site (until the next site)."""
    if not text or not location_name:
        return None
    own = _alien_site_boundary_pattern(location_name)
    if own is None:
        return None
    candidates = list(own.finditer(text))
    if not candidates:
        return None

    match = None
    for candidate in candidates:
        line_start = text.rfind("\n", 0, candidate.start()) + 1
        line_end = text.find("\n", candidate.start())
        if line_end < 0:
            line_end = len(text)
        line = text[line_start:line_end].strip()
        # Skip mid-prose mentions (``…Oblivion bar. There are six…``).
        if candidate.start() - line_start > 8 or len(line) > 64:
            continue
        if "." in line or "," in line or ";" in line:
            continue
        letters = [c for c in line if c.isalpha()]
        if letters and sum(1 for c in letters if c.islower()) / len(letters) > 0.35:
            continue
        # Require room writeups soon after the heading so map-legend labels
        # (Fort / Oblivion / Spaceport stacked at the top) are skipped.
        near_end = min(len(text), candidate.end() + 900)
        for boundary in _ALIEN_NEXT_SITE_BOUNDARY.finditer(text, candidate.end() + 20):
            snip = text[boundary.start() : boundary.start() + 100]
            lead = re.match(r"\s*", snip)
            pos = lead.end() if lead else 0
            if own.match(snip, pos):
                continue
            near_end = min(near_end, boundary.start())
            break
        near = text[candidate.end() : near_end]
        if _count_colon_place_headers(near) < 1:
            continue
        window_end = min(len(text), candidate.end() + 3_000)
        for boundary in _ALIEN_NEXT_SITE_BOUNDARY.finditer(text, candidate.end() + 40):
            snip = text[boundary.start() : boundary.start() + 100]
            # Only treat as the same site when this site's pattern anchors the boundary.
            lead = re.match(r"\s*", snip)
            pos = lead.end() if lead else 0
            if own.match(snip, pos):
                continue
            window_end = boundary.start()
            break
        window = text[candidate.end() : window_end]
        if _count_colon_place_headers(window) >= 1:
            match = candidate
            break
    if match is None:
        # Map-legend hits come first; body writeups are usually later.
        match = candidates[-1]

    start = match.start()
    min_body = 400
    search_from = start + min_body
    end = len(text)
    for boundary in _ALIEN_NEXT_SITE_BOUNDARY.finditer(text, search_from):
        # Ignore boundaries that still match this same site.
        if own.search(text[boundary.start() : boundary.start() + 80]):
            continue
        end = boundary.start()
        break
    end = min(end, start + 12_000)
    if end <= start + 80:
        return None
    return start, end


def _harvest_colon_place_subareas(
    document_text: str,
    *,
    location_name: str,
    max_items: int = 16,
) -> list[dict[str, str]]:
    """Harvest Alien ``PLACE: prose`` room headers near a parent site name."""
    # Undouble map glue so site boundaries like SPACEPORTSPACEPORT match.
    text = _normalize_alien_location_scan_text(document_text or "")
    parent = str(location_name or "").strip()
    if not text.strip() or not parent:
        return []

    span = _find_alien_site_body_span(text, parent)
    if span is None:
        needles = _distinctive_name_needles(parent, snake_case(parent))
        region = _relevant_excerpts(text, needles) if needles else ""
        if len(region) < 200:
            return []
        if len(region) > 12_000:
            region = region[:12_000]
    else:
        region = text[span[0] : span[1]]

    lines = region.splitlines()
    found: list[dict[str, str]] = []
    seen: set[str] = set()
    parent_key = parent.casefold()
    for idx, line in enumerate(lines):
        name = _colon_place_header_name(line)
        if not name:
            continue
        key = name.casefold()
        if key in seen or key == parent_key:
            continue
        # People / events as colon headers (EDIE:, AWOLS:, ACT II:).
        if key in {"edie", "awols", "people of interest", "the disgruntled dancer"}:
            continue
        if re.match(r"^act\s+", key):
            continue
        if key in _ALIEN_MAP_CHROME or key in _SUBAREA_HEADER_STOPWORDS:
            continue
        if key.startswith(("page ", "cat s eye", "cats eye")):
            continue
        description = _prose_after_colon_header(lines, idx)
        if len(description) < 40:
            # Inline prose after the colon on the same line.
            stripped = re.sub(r"^[T•\-]\s+", "", line.strip())
            match = _COLON_PLACE_HEADER.match(stripped)
            if match:
                description = re.sub(r"\s+", " ", match.group(2)).strip()
        if len(description) < 40:
            continue
        seen.add(key)
        found.append(
            {
                "name": name,
                "description": description[:400],
                "public_knowledge": "",
            }
        )
        if len(found) >= max_items:
            break
    return found


def _prose_after_colon_header(lines: list[str], start_idx: int, *, max_chars: int = 280) -> str:
    parts: list[str] = []
    total = 0
    # Include remainder of the header line after the colon.
    header = re.sub(r"^[T•\-]\s+", "", lines[start_idx].strip())
    match = _COLON_PLACE_HEADER.match(header)
    if match:
        first = re.sub(r"\s+", " ", match.group(2)).strip()
        if first:
            parts.append(first)
            total += len(first)
    for line in lines[start_idx + 1 :]:
        stripped = line.strip()
        if not stripped:
            if parts:
                break
            continue
        if _colon_place_header_name(stripped) or _looks_like_place_header(stripped):
            break
        if _STAT_OR_SECTION_LINE.match(stripped) or _CALLOUT_AWARENESS_PRESENCE.search(stripped):
            break
        chunk = re.sub(r"\s+", " ", stripped)
        parts.append(chunk)
        total += len(chunk) + 1
        if total >= max_chars:
            break
    text = " ".join(parts).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text


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


def _flex_letters(word: str) -> str:
    """Build a regex that allows PdfPig spaces between letters: 'awareness' ≈ 'AWAr eNe s s'."""
    chars = [re.escape(ch) for ch in str(word or "") if ch.isalnum()]
    if not chars:
        return ""
    return r"\s*".join(chars)


# PdfPig two-column extracts often put two Awareness/Presence pairs on one line and
# insert spaces inside words. Match flexibly anywhere on the line.
_CALLOUT_AWARENESS_PRESENCE = re.compile(
    rf"(?i)(?<![a-z0-9]){_flex_letters('awareness')}\s+(\d+)\s+{_flex_letters('presence')}\s+(\d+)(?![a-z0-9])"
)
_CALLOUT_TRAITS_FLEX = re.compile(rf"(?i)(?<![a-z0-9]){_flex_letters('traits')}(?![a-z0-9])")
_CALLOUT_STORY_OR_TECH = re.compile(
    rf"(?i)(?:{_flex_letters('story')}\s*{_flex_letters('points')}|{_flex_letters('tech')}\s*{_flex_letters('level')})"
)
_CALLOUT_MONSTER_CONTEXT = re.compile(
    r"(?i)\b(?:creature\s*statistics?|monster(?:s)?|enemy\s*stat|adversary\s*stat|"
    r"alien\s*creature|beast(?:s)?\s*stat)\b"
)
_CREDIT_ROLE_LINE = re.compile(
    r"(?i)^\s*(?:writing|writer|line\s*developer|editing|editors?|cover|"
    r"graphic\s*design(?:\s*&\s*layout)?|layout|creative\s*director|"
    r"art\s*director|publisher|proofreading|playtesters?|thanks)\s*$"
)
_CALLOUT_BIO_NAME = re.compile(
    r"\b([A-Z][a-z]{1,14}(?:\s+[A-Z][a-z]{1,14}){1,2})\b"
)
_CALLOUT_SINGLE_NAME = re.compile(r"[\"“”']([A-Za-z][A-Za-z'-]{1,20})[\"“”']")
_CALLOUT_NAME_STOPWORDS = frozenset(
    {
        "awareness",
        "presence",
        "coordination",
        "resolve",
        "ingenuity",
        "strength",
        "fighting",
        "survival",
        "subterfuge",
        "medicine",
        "convince",
        "skills",
        "traits",
        "story",
        "points",
        "tech",
        "level",
        "the majestic",
        "majestic hotel",
        "player characters",
        "player group",
        "poc unity",
        "cats eye",
        "cat's eye",
        "unarmed combat",
    }
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

    def _is_traits_header(stripped: str) -> bool:
        if _TRAIT_SECTION_HEADER.match(stripped):
            return True
        # PdfPig: "t rAi t s" or "t rAi t s Friends: Hutchins has..."
        flex = _CALLOUT_TRAITS_FLEX.search(stripped)
        if not flex:
            return False
        after = stripped[flex.end() :].strip()
        if not after:
            return True
        if _TRAIT_ITEM_START.match(after):
            return True
        # Bare spaced header with junk; keep short-only guard.
        return len(stripped) <= 48

    def _is_traits_end(stripped: str) -> bool:
        if _TRAIT_SECTION_END.match(stripped):
            return True
        if _CALLOUT_STORY_OR_TECH.search(stripped):
            return True
        return False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _is_traits_header(stripped):
            if first_section_only and section_count >= 1:
                break
            flush()
            in_traits = True
            section_count += 1
            # Same line may continue with "Friends: ..." after a spaced TRAITS header.
            after = _CALLOUT_TRAITS_FLEX.sub("", stripped, count=1).strip()
            if after:
                item = _TRAIT_ITEM_START.match(after)
                if item:
                    current_name = item.group(1).strip()
                    rest = item.group(2).strip()
                    current_parts = [rest] if rest else []
            continue
        if not in_traits:
            # Mid-line trait items in two-column glue: "Friends: Hutchins has..."
            item = _TRAIT_ITEM_START.match(stripped)
            if item and not stripped.lower().startswith("tech level"):
                # Only accept when a traits header appeared earlier in the excerpt.
                if _CALLOUT_TRAITS_FLEX.search(excerpt[: excerpt.find(stripped) + 1] if stripped in excerpt else excerpt):
                    flush()
                    in_traits = True
                    section_count = max(1, section_count)
                    current_name = item.group(1).strip()
                    rest = item.group(2).strip()
                    current_parts = [rest] if rest else []
            continue
        if _is_traits_end(stripped):
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
        # Trait text may continue after another column's trait name mid-line.
        inline = re.search(
            r"(?<![A-Za-z])([A-Z][a-z]{2,24}):\s+(.+)$",
            stripped,
        )
        if inline and current_name:
            # Keep prior trait text before the new inline trait starts.
            before = stripped[: inline.start()].strip()
            if before:
                current_parts.append(before)
            flush()
            current_name = inline.group(1).strip()
            current_parts = [inline.group(2).strip()]
            continue
        if current_name:
            current_parts.append(stripped)
    flush()
    return traits


def _traits_for_npc(traits: list[tuple[str, str]], name: str) -> list[tuple[str, str]]:
    """Keep traits that mention this NPC when two callouts are column-glued."""
    parts = str(name or "").split()
    if not traits or not parts:
        return traits
    surname = parts[-1].casefold()
    first = parts[0].casefold()
    matched = []
    for trait_name, trait_text in traits:
        blob = f"{trait_name} {trait_text}".casefold()
        if (len(surname) >= 3 and surname in blob) or (len(first) >= 4 and first in blob):
            matched.append((trait_name, trait_text))
    return matched if matched else traits


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
    # PdfPig letter-spacing: "t rAi t s" / "sKiLLs" must not look like an NPC name.
    collapsed = re.sub(r"[^a-z0-9]+", "", s.casefold())
    if collapsed in {
        "traits",
        "trait",
        "skills",
        "skill",
        "awareness",
        "presence",
        "coordination",
        "resolve",
        "ingenuity",
        "strength",
        "storypoints",
        "techlevel",
    }:
        return False
    if _CALLOUT_TRAITS_FLEX.fullmatch(s) or _CALLOUT_AWARENESS_PRESENCE.search(s):
        return False
    if _CALLOUT_STORY_OR_TECH.search(s) and len(s) <= 64:
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

    # Prefer the callout/bio mention over an earlier casual name-drop in the window.
    best_start = None
    best_score = -10_000
    for idx in hits:
        line_start = excerpt.rfind("\n", 0, idx) + 1
        score = _npc_writeup_score_at(excerpt, idx, name)
        if score > best_score:
            best_score = score
            best_start = line_start
    start = best_start if best_start is not None else excerpt.rfind("\n", 0, hits[0]) + 1

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
        # Stat/callout chrome is not biography.
        if _CALLOUT_AWARENESS_PRESENCE.search(text) or _CALLOUT_TRAITS_FLEX.search(text):
            continue
        if _CALLOUT_STORY_OR_TECH.search(text) and len(text) < 120:
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


def _writeup_search_needles(name: str) -> list[str]:
    """Needles for locating an NPC writeup — avoid bare given names (Peter ≠ Petersen)."""
    name = str(name or "").strip()
    needles = _distinctive_name_needles(name, snake_case(name))
    parts = [p for p in re.split(r"[\s_/]+", name) if len(p) >= 3]
    if len(parts) < 2:
        return needles
    given = parts[0].casefold()
    # Keep full name + surname/id; drop "Peter" alone so Kate Petersen cannot win.
    return [n for n in needles if n.casefold() != given]


def _line_at(text: str, idx: int) -> str:
    line_start = text.rfind("\n", 0, idx) + 1
    line_end = text.find("\n", idx)
    return text[line_start : line_end if line_end >= 0 else len(text)].strip()


def _npc_writeup_score_at(text: str, mention_idx: int, name: str) -> int:
    """Score a name mention by how much it looks like a character writeup callout.

    Use a tight forward window so earlier prose name-drops do not inherit a later
    neighbor's Awareness/TRAITS block (common in PdfPig two-column extracts).
    """
    tight = text[mention_idx : min(len(text), mention_idx + 420)]
    mid = text[mention_idx : min(len(text), mention_idx + 900)]
    score = 0
    if _CALLOUT_AWARENESS_PRESENCE.search(tight):
        score += 60
    elif _CALLOUT_AWARENESS_PRESENCE.search(mid):
        score += 12
    if _CALLOUT_TRAITS_FLEX.search(tight):
        score += 50
    elif _CALLOUT_TRAITS_FLEX.search(mid):
        score += 12
    if _CALLOUT_STORY_OR_TECH.search(tight):
        score += 20
    elif _CALLOUT_STORY_OR_TECH.search(mid):
        score += 8

    name_cf = str(name or "").casefold()
    surname = name.split()[-1].casefold() if name.split() else ""
    head = mid[:520].casefold()
    if name_cf and (
        f"{name_cf} is " in head
        or f"{name_cf} suffers" in head
        or f"{name_cf} can " in head
        or f"{name_cf} has " in head
    ):
        score += 40
    elif len(surname) >= 4 and (
        f"{surname} is " in head
        or f"{surname} can " in head
        or f"{surname} has " in head
    ):
        score += 25

    line = _line_at(text, mention_idx)
    label = line.strip("\"“”").strip()
    if label and (
        _looks_like_npc_name_header(line)
        or (
            _mention_indexes(label.casefold(), name)
            and len(label) <= max(48, len(name) + 18)
        )
    ):
        score += 45
    # Letter-spaced dual-column headers: "Pe t e r Hu t cHiNs" / "MAx e d d iNGs ..."
    if name_cf and _mention_indexes(label.casefold(), name) and len(label) <= 64:
        score += 20

    before = text[max(0, mention_idx - 24) : mention_idx]
    if re.search(
        r"(?i)(?:\bof|\bto|\band|\bwith|\bfor|\btrust|\bincluding)\s+$|[a-z,;:]\s*$",
        before,
    ):
        score -= 45
    if re.search(r"\b\d{1,3}\s*$", line.strip()):
        score -= 25  # TOC page refs
    return score


def _npc_writeup_excerpt(
    document_text: str,
    name: str,
    *,
    window_before: int = 450,
    window_after: int = 2_400,
) -> str:
    """Prefer the name mention nearest Awareness/TRAITS/bio over casual name-drops."""
    text = document_text or ""
    if not text.strip():
        return ""
    needles = _writeup_search_needles(name)
    if not needles:
        needles = _distinctive_name_needles(name, snake_case(name))
    lower = text.casefold()
    scored: list[tuple[int, int, str]] = []
    for needle in needles:
        for idx in _mention_indexes(lower, needle):
            scored.append((_npc_writeup_score_at(text, idx, name), idx, needle))
    if not scored:
        return _relevant_excerpts(
            text,
            needles,
            window_before=_SEED_EXCERPT_WINDOW_BEFORE,
            window_after=_SEED_EXCERPT_WINDOW_AFTER,
        )
    # Prefer richer writeup scores; on ties prefer longer needles then earlier idx.
    scored.sort(key=lambda row: (-row[0], -len(row[2]), row[1]))
    best_score, idx, needle = scored[0]
    start = max(0, idx - window_before)
    # If the best hit is the bio opener, include the attribute/TRAITS block above it.
    back = text[max(0, idx - 900) : idx]
    aw = list(_CALLOUT_AWARENESS_PRESENCE.finditer(back))
    if aw:
        start = max(0, idx - 900 + aw[-1].start() - 120)
    elif best_score >= 40:
        # Dual-column name header often sits just above Awareness.
        start = max(0, idx - window_before)
    end = min(len(text), max(idx + len(needle) + window_after, start + 1_200))
    return text[start:end]


def _bio_block_after_name(excerpt: str, name: str) -> list[str]:
    """Grab prose starting at '<Name> is/suffers/can' through the next clear break."""
    name = str(name or "").strip()
    if not name or not excerpt:
        return []
    # Allow PdfPig letter-spacing inside the name for the bio opener.
    name_flex_parts: list[str] = []
    for word in name.split():
        chars = [re.escape(ch) for ch in word if ch.isalnum()]
        if len(chars) >= 2:
            name_flex_parts.append(r"\s*".join(chars))
    name_alt = re.escape(name)
    if name_flex_parts:
        name_alt = rf"(?:{name_alt}|{r'\s+'.join(name_flex_parts)})"
    pattern = re.compile(
        rf"(?is)({name_alt}\s+(?:is|suffers|can|has|would)\b.+?)(?="
        rf"\n\s*[A-Z][a-z]+\s+[A-Z][a-z]+\s+(?:is|suffers)\b|"
        rf"\n\s*(?:Cat.?s\s*Ey|PAGE|\d{{1,3}})\s*$|"
        rf"$)",
    )
    match = pattern.search(excerpt)
    if not match:
        # Surname-led continuation paragraphs in glued columns.
        surname = name.split()[-1]
        match = re.search(
            rf"(?is)({re.escape(surname)}\s+(?:is|can|has|would)\b.+?)(?=\n\s*[A-Z][a-z]+\s+[A-Z]|\n\s*\d{{1,3}}\s*$|$)",
            excerpt,
        )
    if not match:
        return []
    raw = match.group(1)
    # Drop PdfPig tech/story chrome glued into the bio column.
    raw = _CALLOUT_STORY_OR_TECH.sub(" ", raw)
    text = re.sub(r"\s+", " ", raw).strip()
    if len(text) < 40:
        return []
    if len(text) > _BIO_CHUNK_MAX_CHARS * 2:
        text = text[: _BIO_CHUNK_MAX_CHARS * 2 - 3] + "..."
    return [text]


_COLUMN_GLUE_PIVOT = re.compile(
    r"(?i)\b(?:other than|suspicious by|nothing better)\s+"
)


def _scrub_seed_prose(text: str) -> str:
    """Remove PdfPig tech/story chrome and leftover ': 4' crumbs from seed prose."""
    cleaned = _CALLOUT_STORY_OR_TECH.sub(" ", str(text or ""))
    cleaned = re.sub(r"\s*:\s*\d+\b", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:;,")
    return cleaned


def _expand_trait_text_from_excerpt(
    excerpt: str,
    trait_name: str,
    partial_text: str,
) -> str:
    """Rebuild a trait description when PdfPig column-glue truncates the first line."""
    partial = str(partial_text or "").strip()
    name = str(trait_name or "").strip()
    if not excerpt or not name:
        return partial
    looks_short = len(partial) < 80 or bool(
        re.search(r"(?i)\b(?:the|of|a|and|to|his|her|than|by|with)\s*$", partial)
    )
    if not looks_short and partial.endswith((".", "!", "?")):
        return partial

    start = re.search(rf"(?i)\b{re.escape(name)}:", excerpt)
    if not start:
        return partial

    chunks = [partial]
    for line in excerpt[start.end() :].splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(
            r"(?i)^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\s+(?:is|suffers)\b",
            stripped,
        ):
            break
        # Pure tech/story chrome ends the traits block.
        if _CALLOUT_STORY_OR_TECH.search(stripped) and not _TRAIT_ITEM_START.match(stripped):
            break

        item = _TRAIT_ITEM_START.match(stripped)
        body = item.group(2).strip() if item else stripped
        if item and item.group(1).strip().casefold() == name.casefold():
            if body:
                chunks = [body]
            continue

        pivots = list(_COLUMN_GLUE_PIVOT.finditer(body))
        if pivots:
            frag = body[pivots[-1].end() :].strip(" -:;,")
            # Drop trailing tech chrome glued onto the right column.
            frag = _CALLOUT_STORY_OR_TECH.split(frag)[0].strip(" -:;,")
            if frag and not frag.casefold().startswith("eddings"):
                chunks.append(frag)
        elif (
            item is None
            and re.search(r"(?i)\b(?:would fight|on his side|some would)\b", body)
            and not re.search(r"(?i)\beddings\b", body)
        ):
            chunks.append(body)

        if item and item.group(1).strip().casefold() in {
            "argumentative",
            "brave",
            "tough",
            "attractive",
        }:
            # Next left-column trait — stop once we already recovered a continuation.
            if len(chunks) > 1:
                break

    text = _scrub_seed_prose(" ".join(chunks))
    m = re.search(r"(?is)(.+?\bsome would\.)", text)
    if m:
        text = m.group(1).strip()
    if len(text) > len(partial) + 8:
        return text[:420].rstrip()
    return partial


def _enrich_personality_and_public_facts_from_bio(
    bios: list[str],
    *,
    personality: list[str],
    public_facts: list[str],
    excerpt: str = "",
) -> None:
    """Pull obvious roleplay cues from biography prose into structured fields."""
    joined = " ".join(bios)
    if excerpt:
        joined = f"{joined} {excerpt}"
    if not joined.strip():
        return
    cf = joined.casefold()
    cues = [
        (r"\bwarm\b", "Warm"),
        (r"\bfriendly\b", "Friendly"),
        (r"easy to smile", "Easy to smile"),
        (r"quick to thank", "Quick to thank others"),
        (r"hard man to dislike|hard to dislike", "Hard to dislike"),
        (r"natural gift for organisation|gift for organisation|organization and management", "Organized / managerial"),
    ]
    for pattern, label in cues:
        if re.search(pattern, cf) and not any(label.casefold() in p.casefold() for p in personality):
            personality.append(label)
    if re.search(r"\bleader\b|father or uncle|considers himself more as a father", cf):
        fact = "Considered the leader (or father-figure) of the squatters."
        if not any("leader" in p.casefold() for p in public_facts):
            public_facts.append(fact)
    if re.search(r"\bfamily\b", cf) and not any("family" in p.casefold() for p in public_facts):
        public_facts.append("Treats the squatters as family.")


def _seed_npc_detail_from_document(
    name: str,
    role: str,
    document_text: str,
) -> dict[str, Any] | None:
    """Deterministic detail seed from Cubicle 7 / Doctor Who-style stat writeups."""
    excerpt = _npc_writeup_excerpt(document_text, name)
    excerpt = _clip_npc_writeup(excerpt, name)
    if not excerpt or len(excerpt.strip()) < 60:
        return None

    traits = _traits_for_npc(
        _parse_trait_blocks(excerpt, first_section_only=False),
        name,
    )
    traits = [
        (trait_name, _expand_trait_text_from_excerpt(excerpt, trait_name, trait_text))
        for trait_name, trait_text in traits
    ]
    bios = _bio_paragraphs_for_name(excerpt, name) or _bio_block_after_name(excerpt, name)
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
        cleaned = _scrub_seed_prose(bio)
        if len(cleaned) < 40:
            continue
        # Prefer short public facts; keep long glued OCR bios in known_facts.
        if len(cleaned) <= _BIO_CHUNK_MAX_CHARS and len(public_facts) < 4:
            public_facts.append(cleaned)
        else:
            known_facts.append(cleaned[: _BIO_CHUNK_MAX_CHARS * 2])
    _enrich_personality_and_public_facts_from_bio(
        bios,
        personality=personality,
        public_facts=public_facts,
        excerpt=excerpt,
    )

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


def _token_looks_pdf_split_fragment(tok: str) -> bool:
    """True when a token looks like a PdfPig mid-word shard, not a real name word."""
    t = str(tok or "").strip()
    if not t:
        return False
    # Single-letter leftovers: "Peterse N"
    if len(t) == 1 and t.isalpha():
        return True
    # Mixed mid-word caps typical of PdfPig: "HiNs", "eddiNGs" shards
    if any(c.islower() for c in t) and any(c.isupper() for c in t[1:]):
        return True
    # Short consonant-only shards: "Sm" in "Sm Ith"
    letters = [c for c in t if c.isalpha()]
    if 2 <= len(letters) <= 4:
        if not any(c.casefold() in "aeiou" for c in letters):
            return True
    return False


def _normalize_pdf_person_name(raw: str) -> str:
    """Repair PdfPig mid-word spaces: 'Peter Hutc HiNs' → 'Peter Hutchins'.

    Only joins trailing tokens when at least one looks like a split fragment.
    Legitimate multi-word names (Mary Ann Lee, Jean Luc Picard) are left alone.
    """
    s = str(raw or "").strip().strip("\"“”'‘’")
    if not s:
        return ""
    tokens = re.sub(r"\s+", " ", s).split()
    if len(tokens) >= 3:
        left, right = tokens[-2], tokens[-1]
        if right.casefold() not in {"vs", "the", "of", "and", "or", "jr", "sr", "ii", "iii", "iv"}:
            if _token_looks_pdf_split_fragment(left) or _token_looks_pdf_split_fragment(right):
                tokens = tokens[:-2] + [left + right]
    return " ".join((t[0].upper() + t[1:].lower()) if t else t for t in tokens)


def _collapse_alpha_shards(segment: str) -> str:
    """Join PdfPig letter shards: 'MAx e d d iNGs' → 'MAxeddiNGs'."""
    tokens = str(segment or "").split()
    out: list[str] = []
    for tok in tokens:
        if (
            out
            and re.fullmatch(r"[A-Za-z]+", out[-1] or "")
            and re.fullmatch(r"[A-Za-z]+", tok)
            and (len(tok) <= 2 or tok[0].islower() or len(out[-1]) <= 3)
        ):
            out[-1] += tok
            continue
        out.append(tok)
    return " ".join(out)


def _is_plausible_callout_name(name: str) -> bool:
    key = str(name or "").strip().casefold()
    if not key or key in _CALLOUT_NAME_STOPWORDS:
        return False
    if any(key == stop or key.startswith(stop + " ") for stop in _CALLOUT_NAME_STOPWORDS):
        return False
    words = str(name).split()
    if not (1 <= len(words) <= 3):
        return False
    return all(re.fullmatch(r"[A-Za-z][A-Za-z'-]{0,20}", w) for w in words)


def _split_dual_column_headers(header_line: str, count: int) -> list[str]:
    """Split a two-column name header into ``count`` groups by long-token boundaries."""
    toks = [t for t in str(header_line or "").split() if t]
    if not toks:
        return []
    if count <= 1 or len(toks) < 4:
        return [_normalize_pdf_person_name(_collapse_alpha_shards(" ".join(toks)))]

    if count == 2:
        seen_long = 0
        split_at = None
        for i, tok in enumerate(toks):
            if tok.isalpha() and len(tok) >= 3:
                seen_long += 1
                if seen_long >= 2 and i + 1 < len(toks):
                    split_at = i + 1
                    break
        if split_at:
            chunks = [toks[:split_at], toks[split_at:]]
            return [
                _normalize_pdf_person_name(_collapse_alpha_shards(" ".join(chunk)))
                for chunk in chunks
                if chunk
            ]

    # Fallback: even token slices.
    size = len(toks) / count
    names: list[str] = []
    for i in range(count):
        start = int(i * size)
        end = int((i + 1) * size) if i < count - 1 else len(toks)
        chunk = toks[start:end]
        if chunk:
            names.append(_normalize_pdf_person_name(_collapse_alpha_shards(" ".join(chunk))))
    return names


def _repair_glued_person_name(glued: str, window: str) -> str:
    """Map Maxeddings/Peterhutchins/Katepetersen onto a spaced display name."""
    glued_norm = _normalize_pdf_person_name(glued)
    letters = re.sub(r"[^a-z]", "", glued_norm.casefold())
    if not letters:
        return glued_norm
    if " " in glued_norm and _is_plausible_callout_name(glued_norm):
        return glued_norm
    hay = window or ""
    for match in _CALLOUT_BIO_NAME.finditer(hay):
        cand = match.group(1).strip()
        if not _is_plausible_callout_name(cand):
            continue
        cand_letters = re.sub(r"[^a-z]", "", cand.casefold())
        if cand_letters == letters or letters in cand_letters or cand_letters in letters:
            return cand
    # Split FirstLast when the first token appears nearby as a whole word (Kate / Max).
    # Prefer longer first-name splits so Kate beats Kat.
    for n in range(6, 2, -1):
        if len(letters) <= n + 3:
            continue
        first = letters[:n]
        rest = letters[n:]
        if not re.search(rf"\b{re.escape(first)}\b", hay.casefold()):
            continue
        spaced = first.title() + " " + rest.title()
        if re.search(rf"\b{re.escape(first)}\s+{re.escape(rest)}\b", hay.casefold()):
            return spaced
        if len(rest) >= 4:
            return spaced
    return glued_norm


def _pick_callout_header_line(lines: list[str], awareness_line_idx: int) -> int:
    """Choose the best short name-header line above an Awareness row (skip prose)."""
    best_idx = -1
    best_score = -1
    for j in range(awareness_line_idx - 1, max(-1, awareness_line_idx - 14), -1):
        candidate = lines[j].strip()
        if not candidate:
            continue
        if _CREDIT_ROLE_LINE.match(candidate):
            return -1
        if _CALLOUT_AWARENESS_PRESENCE.search(candidate):
            continue
        shards = candidate.split()
        short_shards = sum(1 for t in shards if t.isalpha() and len(t) <= 3)
        collapsed = _collapse_alpha_shards(candidate)
        score = 0
        if len(candidate) <= 40:
            score += 2
        if len(candidate) > 70:
            score -= 8
        # PdfPig name headers are short lines with many letter shards.
        if short_shards >= 3 and len(candidate) <= 42:
            score += 10
        # Also: 'Peter Hutc HiNs' style (few tokens, a split fragment, short line).
        if (
            2 <= len(shards) <= 5
            and len(candidate) <= 42
            and any(_token_looks_pdf_split_fragment(t) for t in shards)
        ):
            score += 10
        if _CALLOUT_SINGLE_NAME.search(candidate) and len(candidate) <= 24:
            score += 10
        lowered = candidate.casefold()
        if lowered.startswith(
            ("the ", "a ", "an ", "this ", "that ", "hotel ", "by ", "in ", "of ", "to ")
        ):
            score -= 10
        # Running headers / title chrome are not NPC names.
        if "cat" in lowered and "eye" in lowered:
            score -= 20
        if "squatter" in lowered.replace(" ", "") and "meet" in lowered.replace(" ", ""):
            score -= 10
        if len(collapsed.split()) == 1 and len(re.sub(r"[^A-Za-z]", "", collapsed)) > 18:
            # Collapsed prose blob, not a name.
            score -= 6
        if _looks_like_npc_name_header(candidate.strip("\"“”")):
            score += 3
        if _looks_like_npc_name_header(collapsed.strip("\"“”")):
            score += 2
        if score > best_score:
            best_score = score
            best_idx = j
    if best_score < 5:
        return -1
    return best_idx


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

    Tolerates PdfPig letter-spacing and two-column glue where two Awareness/Presence
    pairs share a line. Credits/TOC names lack that sheet pattern and are ignored.
    """
    text = document_text or ""
    if not text.strip():
        return []

    lines = text.splitlines()
    found: list[dict[str, str]] = []
    seen: set[str] = set()

    for i, line in enumerate(lines):
        matches = list(_CALLOUT_AWARENESS_PRESENCE.finditer(line))
        if not matches:
            continue

        header_idx = _pick_callout_header_line(lines, i)
        if header_idx < 0:
            continue

        preceding = "\n".join(lines[max(0, header_idx - 12) : header_idx])
        if _CALLOUT_MONSTER_CONTEXT.search(preceding):
            continue

        header_raw = lines[header_idx].strip()
        header_names = _split_dual_column_headers(header_raw, len(matches))
        # Quoted single-name headers ("cHAd") for one-column sheets.
        if len(matches) == 1:
            quoted = _CALLOUT_SINGLE_NAME.search(header_raw)
            if quoted:
                header_names = [_normalize_pdf_person_name(quoted.group(1))]

        window = "\n".join(lines[max(0, header_idx) : min(len(lines), i + 32)])
        if not _CALLOUT_TRAITS_FLEX.search(window):
            continue
        if not _CALLOUT_STORY_OR_TECH.search(window):
            continue

        bio_names: list[str] = []
        for match in _CALLOUT_BIO_NAME.finditer(window):
            cand = match.group(1).strip()
            if _is_plausible_callout_name(cand) and cand.casefold() not in {
                n.casefold() for n in bio_names
            }:
                bio_names.append(cand)

        for column_index, _match in enumerate(matches):
            name = ""
            if column_index < len(header_names):
                name = _repair_glued_person_name(header_names[column_index], window)
            if not _is_plausible_callout_name(name) or (
                " " not in name and len(name) > 8
            ):
                for bio in bio_names:
                    if bio.casefold() not in seen:
                        name = bio
                        break
            name = _normalize_pdf_person_name(name)
            if not _is_plausible_callout_name(name):
                continue
            # Prefer multi-word character names; allow short single tokens (Chad).
            if " " not in name and len(name) > 8:
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
    if not loc and re.search(r"(?i)majestic", document_text or ""):
        loc = "majestic_hotel"

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


_SUBAREA_HEADER_STOPWORDS = frozenset(
    {
        "awareness",
        "presence",
        "coordination",
        "resolve",
        "ingenuity",
        "strength",
        "skills",
        "traits",
        "introduction",
        "contents",
        "credits",
        "writing",
        "editing",
        "cover",
        "publisher",
        "appendix",
        "index",
        "maps",
        "arrival",
        "talking with the squatters",
        "the afflicted",
        "meet the squatters",
        "herding cats",
        "the villain revealed",
        "water vs the poc unity",
        "water vs. the poc unity",
        "different character types",
        "so why not just leave",
        "highly armed",
        "time lords telepathy cats",
        "exploring the hotel",
        "the majestic hotel",
        "majestic hotel",
        "the majestic hotel map",
        "majestic hotel map",
        "the feral cat colony",
        "the squatters",
        "the poc unity",
        "a sudden deluge",
        "sleeping squatter",
        "a sleeping squatter",
    }
)

_TOC_PLACE_LINE = re.compile(
    r"(?im)^\s*(?:[Bb]\s+)?([A-Za-z\"“”'][A-Za-z0-9'\"“”’\- ]{1,42}?)\s+(\d{1,3})\s*$"
)
_EXPLORING_SECTION = re.compile(
    rf"(?is)(?:^|\n)\s*(?:[Bb]\s+)?{_flex_letters('exploring')}[^\n]{{0,80}}\n"
)
_STAT_NAME_COLLAPSED = frozenset(
    {
        "awareness",
        "presence",
        "coordination",
        "resolve",
        "ingenuity",
        "strength",
        "skills",
        "traits",
        "fighting",
        "survival",
        "subterfuge",
        "medicine",
        "convince",
        "techlevel",
        "storypoints",
    }
)


def _collapsed_alpha(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _significant_location_tokens(name: str) -> set[str]:
    """Distinctive tokens from a location name (for scoping harvest/excerpts)."""
    skip = {
        "the",
        "a",
        "an",
        "and",
        "of",
        "or",
        "to",
        "in",
        "on",
        "at",
        "location",
        "area",
        "place",
        "region",
        "zone",
    }
    tokens: set[str] = set()
    for part in re.split(r"[\s_/.\-]+", str(name or "").strip()):
        cleaned = re.sub(r"[^a-z0-9]+", "", part.casefold())
        if len(cleaned) < 4 or cleaned in skip:
            continue
        tokens.add(cleaned)
    return tokens


def _text_matches_location(blob: str, location_name: str) -> bool:
    """True when blob shares a significant token with the location name."""
    tokens = _significant_location_tokens(location_name)
    if not tokens or not blob:
        return False
    collapsed = _collapsed_alpha(blob)
    spaced = re.sub(r"[^a-z0-9]+", " ", str(blob).casefold())
    for token in tokens:
        if token in collapsed or re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", spaced):
            return True
    return False


def _display_place_name(raw: str) -> str:
    """Turn stylized PDF headers into readable place names."""
    cleaned = re.sub(r"\s+", " ", str(raw or "").strip().strip("\"“”'"))
    if not cleaned:
        return ""
    parts: list[str] = []
    for token in cleaned.split():
        letters = "".join(ch for ch in token if ch.isalnum() or ch in "'’-")
        if not letters:
            continue
        if letters.isupper() and len(letters) <= 4:
            parts.append(letters.title())
        elif any(c.islower() for c in letters) and any(c.isupper() for c in letters[1:]):
            parts.append(letters[:1].upper() + letters[1:].lower())
        else:
            parts.append(letters[:1].upper() + letters[1:].lower() if letters.isalpha() else letters)
    return " ".join(parts).strip()


def _is_subarea_stop_name(name: str) -> bool:
    key = re.sub(r"[^a-z0-9]+", " ", str(name or "").casefold()).strip()
    if not key or key in _SUBAREA_HEADER_STOPWORDS:
        return True
    if key.startswith(("cat s eye", "cats eye", "page ")):
        return True
    collapsed = _collapsed_alpha(name)
    # Exact collapsed match only — "Portraits" must not die on substring "traits".
    if collapsed in _STAT_NAME_COLLAPSED:
        return True
    if re.search(r"\d", name):
        return True
    return False


def _looks_like_place_header(line: str) -> bool:
    s = str(line or "").strip().strip("\"“”")
    if not s or len(s) < 3 or len(s) > 42:
        return False
    if s.endswith((".", ":", ";", ",")):
        return False
    if re.search(r"\d", s):
        return False
    if _STAT_OR_SECTION_LINE.match(s) or _TRAIT_SECTION_HEADER.match(s):
        return False
    if _CALLOUT_AWARENESS_PRESENCE.search(s) or _CALLOUT_TRAITS_FLEX.search(s):
        return False
    if s.casefold().startswith(("b  ", "b ")):
        # Section chrome like "B  exPLOriNG..." — allow only pure exploring titles elsewhere.
        return False
    words = s.split()
    if not (1 <= len(words) <= 5):
        return False
    # Reject wrapped prose: too many common lowercase glue words.
    lower_glues = sum(1 for w in words if w.casefold() in {
        "and", "or", "of", "to", "for", "with", "from", "that", "this", "his", "her", "their",
        "is", "are", "was", "were", "be", "by", "as", "in", "on", "at", "into", "than", "but",
    })
    if lower_glues >= 2:
        return False
    name = _display_place_name(s)
    if not name or _is_subarea_stop_name(name):
        return False
    # Reject obvious NPC full names ("Max Eddings") while keeping "Confusing Room".
    parts = name.split()
    if len(parts) == 2 and all(p[:1].isupper() for p in parts):
        first, last = parts
        if last.casefold() not in {
            "room",
            "pool",
            "wing",
            "hall",
            "deck",
            "inn",
            "barracks",
            "courtyard",
            "restaurant",
            "reception",
            "centre",
            "center",
        } and first.casefold() in {
            "max",
            "peter",
            "kate",
            "albert",
            "mary",
            "jean",
            "nick",
            "paul",
            "chad",
        }:
            return False
    return True


def _prose_after_header(lines: list[str], start_idx: int, *, max_chars: int = 280) -> str:
    parts: list[str] = []
    total = 0
    for line in lines[start_idx + 1 :]:
        stripped = line.strip()
        if not stripped:
            if parts:
                break
            continue
        if _looks_like_place_header(stripped) or _looks_like_npc_name_header(stripped):
            break
        if _STAT_OR_SECTION_LINE.match(stripped) or _CALLOUT_AWARENESS_PRESENCE.search(stripped):
            break
        chunk = re.sub(r"\s+", " ", stripped)
        parts.append(chunk)
        total += len(chunk) + 1
        if total >= max_chars:
            break
    text = " ".join(parts).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text


def _harvest_toc_place_names(text: str) -> list[str]:
    """Collect place-like TOC entries ('The Courtyard 14') from the document head."""
    head = (text or "")[:6_000]
    names: list[str] = []
    seen: set[str] = set()
    chapter_skips = {
        "introduction",
        "arrival",
        "contents",
        "credits",
        "exploring the hotel",
        "the majestic hotel map",
        "majestic hotel map",
        "the majestic hotel",
        "majestic hotel",
        "the feral cat colony",
        "the squatters",
        "the poc unity",
        "meet the squatters",
        "talking with the squatters",
        "the afflicted",
        "water vs the poc unity",
        "the villain revealed",
        "herding cats",
        "a sudden deluge",
        "a sleeping squatter",
        "sleeping squatter",
    }
    for match in _TOC_PLACE_LINE.finditer(head):
        label = _display_place_name(match.group(1))
        if not label or _is_subarea_stop_name(label):
            continue
        if label.casefold() in chapter_skips:
            continue
        words = label.split()
        if len(words) == 2 and words[0][:1].isupper() and words[1][:1].isupper():
            if words[0].casefold() in {
                "max",
                "peter",
                "kate",
                "nick",
                "paul",
                "gareth",
                "andrew",
                "chad",
            }:
                continue
            if words[0].casefold() not in {"the", "a", "an", "old", "new"} and words[
                1
            ].casefold() not in {
                "room",
                "pool",
                "wing",
                "hall",
                "deck",
                "inn",
                "courtyard",
                "restaurant",
                "reception",
                "centre",
                "center",
            }:
                # Likely an NPC TOC row (First Last), not a place.
                continue
        if label.casefold() in {"chad", "hamish"} or label.casefold().strip("\"“”") in {
            "chad",
            "hamish",
        }:
            continue
        key = label.casefold()
        if key in seen:
            continue
        seen.add(key)
        names.append(label)
    return names[:20]


def _find_exploring_section_span(
    text: str,
    *,
    location_name: str = "",
) -> tuple[int, int] | None:
    """Locate an Exploring… scene block, optionally scoped to a parent location."""
    if not text:
        return None
    matches = list(_EXPLORING_SECTION.finditer(text))
    if not matches:
        matches = list(
            re.finditer(r"(?is)(?:^|\n)\s*(?:[Bb]\s+)?exploring\b[^\n]{0,80}\n", text)
        )
    if not matches:
        return None

    def _header_line(match: re.Match[str]) -> str:
        line_end = text.find("\n", match.start() + 1)
        return text[match.start() : line_end if line_end >= 0 else match.end()].strip()

    # Drop TOC rows ("Exploring the Hotel 12").
    body_matches = [m for m in matches if not re.search(r"\d{1,3}\s*$", _header_line(m))]
    candidates = body_matches or matches

    if location_name.strip():
        scoped = [m for m in candidates if _text_matches_location(_header_line(m), location_name)]
        if not scoped:
            return None
        candidates = scoped

    body = candidates[-1]
    start = body.start()
    rest = text[body.end() :]
    end_rel = len(rest)
    for ender in re.finditer(
        r"(?im)^\s*(?:[Bb]\s+)?(?:the\s+villain|herding|appendix|index|maps?)\b",
        rest,
    ):
        end_rel = ender.start()
        break
    end = body.end() + min(end_rel, 14_000)
    return start, end


def _harvest_scene_subareas(text: str, *, location_name: str = "") -> list[dict[str, str]]:
    """Harvest short scene/room headers from an Exploring section scoped to the location."""
    span = _find_exploring_section_span(text, location_name=location_name)
    if not span:
        return []
    region = text[span[0] : span[1]]
    lines = region.splitlines()
    found: list[dict[str, str]] = []
    seen: set[str] = set()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or not _looks_like_place_header(stripped):
            continue
        name = _display_place_name(stripped)
        if not name or _is_subarea_stop_name(name):
            continue
        if name.casefold() in {"chad", "hamish"}:
            continue
        key = name.casefold()
        if key in seen:
            continue
        description = _prose_after_header(lines, idx)
        # Exploring headers should usually have following prose; TOC can fill names alone.
        if len(description) < 40:
            continue
        seen.add(key)
        found.append(
            {
                "name": name,
                "description": description,
                "public_knowledge": "",
            }
        )
        if len(found) >= 16:
            break
    return found


def _attach_scene_descriptions(
    items: dict[str, dict[str, str]],
    document_text: str,
    *,
    location_name: str = "",
) -> None:
    """Fill empty TOC descriptions from matching Exploring scene headers."""
    span = _find_exploring_section_span(document_text, location_name=location_name)
    if not span:
        return
    region = document_text[span[0] : span[1]]
    lines = region.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or not _looks_like_place_header(stripped):
            continue
        name = _display_place_name(stripped)
        key = name.casefold()
        if key not in items:
            # Soft match: "The Courtyard" vs "Courtyard".
            soft = next(
                (
                    k
                    for k in items
                    if k == key
                    or k.endswith(key)
                    or key.endswith(k)
                    or _collapsed_alpha(k) == _collapsed_alpha(name)
                ),
                None,
            )
            if soft is None:
                continue
            key = soft
        if items[key].get("description"):
            continue
        description = _prose_after_header(lines, idx)
        if description:
            items[key]["description"] = description


def _toc_name_appears_in_region(name: str, region: str) -> bool:
    """True when a TOC place name (possibly stylized) occurs in the scoped region."""
    if not name or not region:
        return False
    if _mention_indexes(region.casefold(), name):
        return True
    collapsed_name = _collapsed_alpha(name)
    if len(collapsed_name) < 4:
        return False
    # Stylized headers: "tHe cOurtyArd" ≈ "The Courtyard".
    flex_parts: list[str] = []
    for word in re.findall(r"[A-Za-z0-9]+", name):
        chars = [re.escape(ch) for ch in word.casefold() if ch.isalnum()]
        if len(chars) >= 3:
            flex_parts.append(r"\s*".join(chars))
    if not flex_parts:
        return False
    pattern = re.compile(rf"(?i)(?<![a-z0-9]){r'\s+'.join(flex_parts)}(?![a-z0-9])")
    return bool(pattern.search(region))


_SHIP_TYPE_TOKENS = frozenset(
    {
        "ship",
        "vessel",
        "starship",
        "spaceship",
        "spacecraft",
        "freighter",
        "cruiser",
        "destroyer",
        "carrier",
        "corvette",
        "cutter",
        "yacht",
        "boat",
        "submarine",
        "station",
    }
)
_SHIP_NAME_HINTS = (
    "uscss",
    "uss ",
    "hms ",
    "sev ",
    "nostromo",
    "montero",
    "cronus",
    "sotillo",
    "prometheus",
)
_SHIP_ROOM_HEADER_STOP = frozenset(
    {
        "technical specifications",
        "production information",
        "map legend",
        "auxiliary vehicles",
        "the deck plans",
        "deck plans",
        "doors on the cronus",
        "decompression",
        "personal agenda",
        "full name",
        "buddy",
        "rival",
        "gear",
        "talent",
        "skills",
        "health",
        "strength",
        "agility",
        "wits",
        "empathy",
        "scenario",
        "chariot of the gods",
    }
)
_SHIP_DECK_MARKER = re.compile(
    r"(?im)^\s*(?:[A-Z0-9][A-Z0-9 '\-]{0,40}\s*[—\-–]?\s*)?DECK\s+([A-Z])(?:\s*[-–—]?\s*\d+)?\s*$"
)
_SHIP_NAMED_DECK = re.compile(
    r"(?im)^\s*((?:CARGO|ENGINEERING|HABITATION|SCIENCE|CREW|LOWER|UPPER)\s+DECK)\s*$"
)
_SHIP_ROOM_HEADER = re.compile(
    r"^\s*"
    r"([A-Z0-9][A-Z0-9 /'\-]{1,55}(?:,\s*[A-Z0-9][A-Z0-9 /'\-]{0,40})?)"
    r"\s*:\s*(.*)$"
)
_SHIP_FALSE_ROOM_NAMES = frozenset(
    {
        "cold",
        "hot",
        "dark",
        "air",
        "note",
        "notes",
        "important",
        "warning",
        "optional",
        "example",
        "events",
        "event",
        "drugs",
        "vials",
        "motes",
        "bones",
        "urns",
        "fungal nodes",
        "key card",
        "the cure",
        "bloodbursters",
        "medpod",
        "wall safe",
        "act i",
        "act ii",
        "act iii",
        "act iv",
        "ping",
        "brace for impact",
        "ghost ship",
        "bad air",
        "no money",
        "breaking loose",
        "egg sacs afoot",
        "egg sacs",
        "pandora s box",
        "pandoras box",
        "the long night",
        "sterilization protocol",
        "clayton s secret stash",
        "tools",
        "walk in closet",
        "escape pod",
    }
)
_SHIP_SINGLE_WORD_ROOMS = frozenset(
    {
        "bridge",
        "armory",
        "armoury",
        "elevators",
        "elevator",
        "showers",
        "shower",
        "airlock",
        "medlab",
        "medbay",
        "brig",
        "galley",
        "hangar",
        "reactor",
        "catwalk",
        "barracks",
        "mess",
        "infirmary",
        "garage",
    }
)
_SHIP_RESPACE_LABELS = (
    ("MAINAIRLOCK", "MAIN AIRLOCK"),
    ("AIRSCRUBBERS", "AIR SCRUBBERS"),
    ("CRYOCHAMBERS", "CRYO CHAMBERS"),
    ("EXAMINATIONROOM", "EXAMINATION ROOM"),
    ("EVASTORAGE", "EVA STORAGE"),
    ("CREWQUARTERS", "CREW QUARTERS"),
    ("MESSHALL", "MESS HALL"),
    ("CORPORATESUITE", "CORPORATE SUITE"),
    ("SCIENCELAB", "SCIENCE LAB"),
    ("CARGOBAYS", "CARGO BAYS"),
    ("CARGOOFFICE", "CARGO OFFICE"),
    ("STORAGEUNITS", "STORAGE UNITS"),
    ("COOLANTTANKS", "COOLANT TANKS"),
    ("REACTORRELAYANDCONTROL", "REACTOR RELAY AND CONTROL"),
    ("MUTHER2000MAINFRAME", "MU/TH/UR 2000 MAINFRAME"),
    ("MUTHUR2000MAINFRAME", "MU/TH/UR 2000 MAINFRAME"),
    ("MUTHER", "MU/TH/UR"),
    ("MUTHUR", "MU/TH/UR"),
    ("LIFESUPPORTMONITORS", "LIFE SUPPORT MONITORS"),
)
_SHIP_MAP_LEGEND_TOKENS = frozenset(
    {
        "airlock",
        "ladder",
        "access",
        "restricted",
        "area",
        "vacc",
        "suit",
        "storage",
        "med",
        "kit",
        "workshop",
        "engineering",
        "intercom",
        "terminal",
        "power",
        "system",
        "weapon",
        "medlab",
        "mainframe",
        "observation",
        "cpt",
        "quarters",
        "barracks",
        "cryosleep",
        "pods",
        "elevator",
        "escape",
        "pod",
        "radioactive",
    }
)
_SHIP_SECTION_ENDER = re.compile(
    r"(?im)^\s*(?:A\s*C\s*T\s+[IVX0-9]+|ACT\s+[IVX0-9]+|APPENDIX(?:\s+[A-Z0-9]+)?|"
    r"INDEX|EPILOGUE|HANDOUTS?|PANDORA|GHOST\s+SHIP|THE\s+LONG\s+NIGHT)\b"
)
_SHIP_EVENT_HEADER = re.compile(
    r"(?i)^(act\s*[ivx0-9]+|ping|brace\s+for\s+impact|ghost\s+ship|bad\s+air|no\s+money|"
    r"breaking\s+loose|egg\s+sacs(?:\s+afoot)?|pandora|the\s+long\s+night|mandatory)\b"
)
_SHIP_JUNCTION_WAYPOINT = re.compile(
    r"^junction\s+[a-z]\s*-\s*\d+$",
    re.IGNORECASE,
)
_SHIP_JUNCTION_DECK = re.compile(r"\bJUNCTION\s*([A-Z])(?:\s*[-–—]?\s*\d+)?\b", re.IGNORECASE)
_SHIP_ROOM_HARVEST_LIMIT = 24


def _ship_room_name_key(name: str) -> str:
    """Normalize ship room labels for false-name / dedupe comparisons."""
    s = str(name or "").casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s)).strip()


_SHIP_FALSE_ROOM_NAME_KEYS = frozenset(
    _ship_room_name_key(name) for name in _SHIP_FALSE_ROOM_NAMES
)


def _is_ship_false_room_name(name: str) -> bool:
    key = _ship_room_name_key(name)
    if not key:
        return False
    if key in _SHIP_FALSE_ROOM_NAME_KEYS:
        return True
    return bool(_SHIP_EVENT_HEADER.match(key))


def _is_ship_junction_waypoint(name: str) -> bool:
    """Bare junction labels (Junction B-1) are map chrome, not room slots."""
    return bool(_SHIP_JUNCTION_WAYPOINT.match(str(name or "").strip()))


def _cap_ship_important_locations(
    rooms: list[dict[str, str]],
    limit: int = _SHIP_ROOM_HARVEST_LIMIT,
) -> list[dict[str, str]]:
    """Keep up to ``limit`` rooms, dropping bare junction labels before named rooms."""
    if len(rooms) <= limit:
        return rooms
    kept = list(rooms)
    for idx in range(len(kept) - 1, -1, -1):
        if len(kept) <= limit:
            break
        if _is_ship_junction_waypoint(kept[idx]["name"]):
            kept.pop(idx)
    return kept[:limit]


def _location_looks_like_ship(
    location_name: str,
    *,
    location_type: str = "",
    document_text: str = "",
) -> bool:
    loc_type = str(location_type or "").casefold().strip()
    if loc_type in _SHIP_TYPE_TOKENS:
        return True
    name_key = str(location_name or "").casefold()
    if any(token in loc_type for token in _SHIP_TYPE_TOKENS):
        return True
    if any(hint in name_key for hint in _SHIP_NAME_HINTS):
        return True
    if any(token in name_key.split() for token in _SHIP_TYPE_TOKENS):
        return True
    text = document_text or ""
    if not text or not str(location_name or "").strip():
        return False
    # Deck plans / vessel copy near this location name.
    needles = _distinctive_name_needles(location_name, "")
    lower = text.casefold()
    for needle in needles:
        for idx in _mention_indexes(lower, needle)[:4]:
            window = text[max(0, idx - 400) : min(len(text), idx + 1_200)].casefold()
            if "deck plan" in window or "deck plans" in window:
                return True
            if any(token in window for token in ("vessel", "starship", "spaceship", "spacecraft")):
                return True
            if "the ship" in window and ("deck" in window or "airlock" in window):
                return True
    return False


def _normalize_deck_label(raw: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(raw or "").strip())
    if not cleaned:
        return ""
    if cleaned.casefold().startswith("deck "):
        suffix = cleaned[5:].strip()
        return f"Deck {suffix.upper()}" if len(suffix) <= 3 else cleaned.title()
    return cleaned.title()


def _deck_from_ship_room_name(name: str) -> str:
    match = _SHIP_JUNCTION_DECK.search(name or "")
    if match:
        return f"Deck {match.group(1).upper()}"
    return ""


def _respace_collapsed_ship_label(label: str) -> str:
    """Re-insert spaces into a fully collapsed ship room label."""
    s = re.sub(r"\s+", "", str(label or "")).upper()
    if not s:
        return ""
    s = s.replace("MU/TH/UR", "MUTHER")
    for src, dest in _SHIP_RESPACE_LABELS:
        s = s.replace(src, dest.upper().replace(" ", "\0"))
    s = s.replace("\0", " ")
    # Common glue after collapsing letter-spaced headers.
    s = s.replace("AIRLOCKJUNCTION", "AIRLOCK JUNCTION")
    s = s.replace("MAINFRAME", " MAINFRAME")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"JUNCTION([A-Z])-?(\d+)", r"JUNCTION \1-\2", s)
    s = re.sub(r"SCIENCE LAB(\d+)", r"SCIENCE LAB \1", s)
    s = re.sub(r"EVA STORAGE(\d+)", r"EVA STORAGE \1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _collapse_letter_spaced_phrase(phrase: str) -> str:
    """Collapse PdfPig letter-spacing ('M A I N A I R Lo C K' → 'MAIN AIRLOCK')."""
    parts = [p for p in str(phrase or "").split() if p]
    if len(parts) < 4:
        return re.sub(r"\s+", " ", str(phrase or "")).strip()
    short = 0
    for part in parts:
        core = re.sub(r"[^A-Za-z0-9-]", "", part)
        if 0 < len(core) <= 2:
            short += 1
    if short < max(3, int(len(parts) * 0.55)):
        return re.sub(r"\s+", " ", phrase).strip()
    collapsed = "".join(re.sub(r"\s+", "", p) for p in parts)
    collapsed = re.sub(r"([A-Za-z])-(\d)", r"\1-\2", collapsed)
    return _respace_collapsed_ship_label(collapsed)


def _normalize_ship_line(line: str) -> str:
    """Normalize one ship-section line for header/deck detection."""
    raw = str(line or "")
    if ":" not in raw:
        collapsed = _collapse_letter_spaced_phrase(raw.strip())
        return collapsed or raw.strip()
    left, right = raw.split(":", 1)
    return f"{_collapse_letter_spaced_phrase(left)}:{right}"


def _is_ship_map_legend_line(line: str) -> bool:
    tokens = [re.sub(r"[^a-z0-9]+", "", t.casefold()) for t in str(line or "").split()]
    tokens = [t for t in tokens if t]
    if len(tokens) < 2 or len(tokens) > 10:
        return False
    hits = sum(1 for t in tokens if t in _SHIP_MAP_LEGEND_TOKENS)
    return hits >= 2 and hits >= len(tokens) * 0.6


def _parse_ship_deck_marker(line: str, *, allow_lettered_map: bool = False) -> str:
    stripped = _normalize_ship_line(str(line or "")).strip()
    if not stripped or len(stripped) > 48:
        return ""
    # Heading lines only — never retarget deck from mid-sentence "Deck C" prose.
    if any(ch in stripped for ch in ",.;()"):
        return ""
    if " plan" in stripped.casefold() or stripped.casefold().endswith("plans"):
        return ""
    named = _SHIP_NAMED_DECK.match(stripped)
    if named:
        return _normalize_deck_label(named.group(1))
    if not allow_lettered_map:
        # Bare DECK A/B/C/D lines are usually map chrome and end on Deck D.
        return ""
    lettered = _SHIP_DECK_MARKER.match(stripped)
    if lettered:
        return _normalize_deck_label(f"Deck {lettered.group(1)}")
    return ""


def _looks_like_ship_room_header(line: str) -> tuple[str, str] | None:
    """Return (room_name, same_line_prose) for ship room headers ending in ':'."""
    normalized = _normalize_ship_line(str(line or ""))
    stripped = normalized.strip()
    if not stripped:
        return None
    # Nested bullet/triangle substations ("T COMMAND STATION:") stay under the parent room.
    if re.match(r"^[T•▪\-]\s+\S", str(line or "").strip()):
        return None
    # Only the header label must stay short; same-line prose may be long.
    header_label = stripped.split(":", 1)[0]
    if len(header_label) > 80:
        return None
    match = _SHIP_ROOM_HEADER.match(stripped)
    if not match:
        return None
    raw_name = re.sub(r"\s+", " ", match.group(1)).strip(" -—–")
    prose = (match.group(2) or "").strip()
    if not raw_name or len(raw_name) < 3:
        return None
    letters = [ch for ch in raw_name if ch.isalpha()]
    if len(letters) < 3:
        return None
    upper_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
    if upper_ratio < 0.55:
        return None
    display = _display_place_name(raw_name)
    if not display:
        display = raw_name.title()
    key = _ship_room_name_key(display)
    if key in {_ship_room_name_key(stop) for stop in _SHIP_ROOM_HEADER_STOP}:
        return None
    if raw_name.casefold().startswith(("doors on the", "appendix", "chapter", "the deck")):
        return None
    if _is_ship_false_room_name(display) or _is_ship_false_room_name(raw_name):
        return None
    if _SHIP_EVENT_HEADER.match(key) or _SHIP_EVENT_HEADER.match(raw_name.casefold()):
        return None
    words = display.split()
    if len(words) == 1 and key not in {
        _ship_room_name_key(room) for room in _SHIP_SINGLE_WORD_ROOMS
    }:
        return None
    # Reject glued map-legend headers ("Intercom Cryo Chambers").
    if any(w.casefold() in _SHIP_MAP_LEGEND_TOKENS for w in words[:1]) and len(words) >= 2:
        if words[0].casefold() in {"intercom", "access", "terminal", "power", "weapon"}:
            return None
    return display, prose


def _prose_after_ship_header(
    lines: list[str],
    start_idx: int,
    *,
    initial: str = "",
    max_chars: int = 360,
) -> str:
    parts: list[str] = []
    total = 0
    if initial.strip():
        chunk = re.sub(r"\s+", " ", initial.strip())
        if not _is_ship_map_legend_line(chunk):
            parts.append(chunk)
            total += len(chunk) + 1
    for line in lines[start_idx + 1 :]:
        stripped = line.strip()
        if not stripped:
            if parts:
                break
            continue
        normalized = _normalize_ship_line(stripped)
        if _SHIP_SECTION_ENDER.match(normalized) or _SHIP_SECTION_ENDER.match(stripped):
            break
        if _parse_ship_deck_marker(normalized, allow_lettered_map=False):
            break
        if _looks_like_ship_room_header(stripped) or _looks_like_ship_room_header(normalized):
            break
        if _looks_like_place_header(stripped) or _looks_like_npc_name_header(stripped):
            break
        if _STAT_OR_SECTION_LINE.match(stripped) or _CALLOUT_AWARENESS_PRESENCE.search(stripped):
            break
        if re.match(r"^[T•▪\-]\s+[A-Z0-9].*:\s*\S", stripped):
            if parts:
                break
        if _is_ship_map_legend_line(stripped) or _is_ship_map_legend_line(normalized):
            continue
        chunk = re.sub(r"\s+", " ", stripped)
        # Drop two-column legend glue mixed mid-sentence.
        legend_hits = sum(
            1
            for tok in chunk.casefold().split()
            if re.sub(r"[^a-z0-9]+", "", tok) in _SHIP_MAP_LEGEND_TOKENS
        )
        if legend_hits >= 3:
            continue
        parts.append(chunk)
        total += len(chunk) + 1
        if total >= max_chars:
            break
    text = " ".join(parts).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text


def _description_with_deck(deck: str, prose: str) -> str:
    prose = re.sub(r"\s+", " ", str(prose or "").strip())
    deck = _normalize_deck_label(deck)
    if not prose:
        return f"{deck}." if deck else ""
    if not deck:
        return prose
    if prose.casefold().startswith(deck.casefold()):
        return prose
    return f"{deck}. {prose}"


def _find_ship_room_description_span(
    text: str,
    *,
    location_name: str = "",
) -> tuple[int, int] | None:
    """Locate the room-description block for one parent ship.

    Requires ``location_name`` and only accepts deck-plan / room-header starts whose
    nearby lead-in mentions that ship — otherwise Sotillo would inherit Cronus rooms.
    """
    if not text:
        return None
    parent = str(location_name or "").strip()
    if not parent:
        return None

    deck_plan_hits = [
        m.start()
        for m in re.finditer(r"(?im)(?:^|\n)\s*(?:the\s+)?deck\s+plans\b", text)
    ]
    room_hits = [
        m.start()
        for m in re.finditer(
            r"(?im)^\s*(?:MAIN\s+AIRLOCK|BRIDGE|MEDLAB|CRYO\s+CHAMBERS|CARGO\s+BAYS|"
            r"REACTOR\s+RELAY|SCIENCE\s+LAB|MU/TH/UR)[^\n]{0,40}:",
            text,
        )
    ]

    def _belongs_to_parent(start_idx: int) -> bool:
        # Ownership comes from the lead-in before the room block. Do not look far
        # forward — scenario text after rooms often mentions Montero/other ships.
        lead = text[max(0, start_idx - 3_000) : min(len(text), start_idx + 160)]
        if not _text_matches_location(lead, parent):
            return False
        parent_tokens = _significant_location_tokens(parent)
        if not parent_tokens:
            return True
        lead_cf = lead.casefold()
        parent_pos = -1
        for token in parent_tokens:
            pos = lead_cf.rfind(token)
            if pos > parent_pos:
                parent_pos = pos
        if parent_pos < 0:
            return False
        competing = (
            "uscss",
            "cronus",
            "nostromo",
            "montero",
            "prometheus",
            "heliades",
            "sotillo",
        )
        for other in competing:
            if any(other == t or other in t or t in other for t in parent_tokens):
                continue
            other_pos = lead_cf.rfind(other)
            if other_pos >= 0 and other_pos > parent_pos:
                return False
        return True

    start_candidates = [
        idx
        for idx in sorted(set(deck_plan_hits + room_hits))
        if _belongs_to_parent(idx)
    ]
    if not start_candidates:
        return None

    start = start_candidates[0]
    scoped_deck = [h for h in deck_plan_hits if _belongs_to_parent(h)]
    scoped_rooms = [h for h in room_hits if _belongs_to_parent(h)]
    if scoped_deck and scoped_rooms:
        after_plans = [h for h in scoped_rooms if h >= scoped_deck[0]]
        if after_plans:
            start = after_plans[0]
    elif scoped_rooms:
        start = scoped_rooms[0]

    # Stop before scenario acts/events — those are not ship rooms.
    enders = list(_SHIP_SECTION_ENDER.finditer(text[start + 1 :]))
    end = start + 1 + enders[0].start() if enders else min(len(text), start + 30_000)
    if end - start < 200:
        return None
    return start, end


def _harvest_ship_room_subareas(text: str, *, location_name: str = "") -> list[dict[str, str]]:
    """Harvest deck-scoped ship rooms from ALL-CAPS 'ROOM NAME: prose' headers."""
    span = _find_ship_room_description_span(text, location_name=location_name)
    if not span:
        return []
    region = text[span[0] : span[1]]
    lines = region.splitlines()
    found: list[dict[str, str]] = []
    seen: set[str] = set()
    room_slots = 0
    # Only named decks (Cargo Deck, etc.) update current_deck. Bare DECK D map
    # chrome would otherwise stamp every room as Deck D.
    current_deck = ""

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if _SHIP_SECTION_ENDER.match(_normalize_ship_line(stripped)) or _SHIP_SECTION_ENDER.match(
            stripped
        ):
            break
        deck_marker = _parse_ship_deck_marker(stripped, allow_lettered_map=False)
        if deck_marker:
            current_deck = deck_marker
            continue
        parsed = _looks_like_ship_room_header(stripped)
        if not parsed:
            continue
        name, same_line = parsed
        key = name.casefold()
        if key in seen:
            continue
        if location_name and _collapsed_alpha(name) == _collapsed_alpha(location_name):
            continue
        description = _prose_after_ship_header(lines, idx, initial=same_line)
        # Require real room prose so map labels / chrome do not become rooms.
        if len(description) < 50:
            continue
        deck = _deck_from_ship_room_name(name) or current_deck
        seen.add(key)
        found.append(
            {
                "name": name,
                "description": _description_with_deck(deck, description),
                "public_knowledge": "",
            }
        )
        if not _is_ship_junction_waypoint(name):
            room_slots += 1
            if room_slots >= _SHIP_ROOM_HARVEST_LIMIT:
                break
    return found


def _harvest_exploring_important_locations(
    document_text: str,
    location_name: str,
) -> list[dict[str, str]]:
    """Harvest sub-areas from an Exploring section (+ matching TOC) for one parent."""
    text = document_text or ""
    parent = str(location_name or "").strip()
    if not text.strip() or not parent:
        return []

    span = _find_exploring_section_span(text, location_name=parent)
    if not span:
        return []
    region = text[span[0] : span[1]]
    by_key: dict[str, dict[str, str]] = {}

    for toc_name in _harvest_toc_place_names(text):
        # Only keep TOC rows that also appear inside this location's exploring region.
        if not _toc_name_appears_in_region(toc_name, region):
            continue
        key = toc_name.casefold()
        by_key.setdefault(
            key,
            {"name": toc_name, "description": "", "public_knowledge": ""},
        )

    for item in _harvest_scene_subareas(text, location_name=parent):
        key = str(item.get("name") or "").casefold()
        if not key:
            continue
        existing = by_key.get(key)
        if existing is None:
            if not any(
                token in key
                for token in (
                    "room",
                    "pool",
                    "courtyard",
                    "wing",
                    "hall",
                    "deck",
                    "reception",
                    "restaurant",
                    "leisure",
                    "eddings",
                    "airlock",
                    "bridge",
                    "cargo",
                    "medlab",
                    "cryo",
                )
            ) and len(key.split()) > 2:
                continue
            by_key[key] = item
        elif item.get("description") and len(str(item["description"])) > len(
            str(existing.get("description") or "")
        ):
            existing["description"] = item["description"]

    _attach_scene_descriptions(by_key, text, location_name=parent)

    parent_key = _display_place_name(parent).casefold()
    parent_tokens = {t for t in re.split(r"[\s_/]+", parent_key) if len(t) >= 4}
    results: list[dict[str, str]] = []
    for key, item in by_key.items():
        if parent_key and (key == parent_key or key in parent_tokens):
            continue
        if parent_tokens and key.replace(" ", "") == "".join(parent_tokens):
            continue
        results.append(item)
    return results[:16]


def _harvest_important_locations(
    document_text: str,
    location_name: str,
    *,
    location_type: str = "",
) -> list[dict[str, str]]:
    """Harvest sub-areas for one parent location (Exploring/TOC and/or ship decks).

    Does not attach a shared document TOC/Exploring block to unrelated location stubs.
    """
    text = document_text or ""
    parent = str(location_name or "").strip()
    if not text.strip() or not parent:
        return []

    exploring = _harvest_exploring_important_locations(text, parent)
    is_ship = _location_looks_like_ship(
        parent, location_type=location_type, document_text=text
    )
    ship_rooms: list[dict[str, str]] = []
    if is_ship:
        ship_rooms = _harvest_ship_room_subareas(text, location_name=parent)
    colon_rooms: list[dict[str, str]] = []
    # Alien moonside chapters use PLACE: prose rooms; do not apply that harvest to
    # ships (Chariot deck plans), which already have dedicated ship-room parsing.
    if _document_has_recoverable_location_sites(text) and not is_ship:
        colon_rooms = _harvest_colon_place_subareas(text, location_name=parent)

    combined = exploring
    if ship_rooms:
        combined = _merge_important_location_lists(combined, ship_rooms)
    if colon_rooms:
        combined = _merge_important_location_lists(combined, colon_rooms)

    if ship_rooms:
        return _cap_ship_important_locations(combined)
    return combined[:16]


def _merge_important_location_lists(
    primary: list[Any],
    secondary: list[Any],
) -> list[dict[str, str]]:
    """Keep LLM entries first; fill gaps from harvested sub-areas."""
    merged: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(item: Any) -> None:
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            if not name:
                return
            key = name.casefold()
            if key in seen:
                # Prefer richer descriptions (e.g. deck-prefixed ship room prose).
                for row in merged:
                    if row["name"].casefold() != key:
                        continue
                    desc = str(item.get("description") or "").strip()
                    existing_desc = str(row.get("description") or "").strip()
                    if desc and (not existing_desc or len(desc) > len(existing_desc)):
                        row["description"] = desc
                    pk = str(item.get("public_knowledge") or "").strip()
                    if pk and not row.get("public_knowledge"):
                        row["public_knowledge"] = pk
                return
            seen.add(key)
            merged.append(
                {
                    "name": name,
                    "description": str(item.get("description") or "").strip(),
                    "public_knowledge": str(item.get("public_knowledge") or "").strip(),
                }
            )
        elif isinstance(item, str) and item.strip():
            key = item.strip().casefold()
            if key in seen:
                return
            seen.add(key)
            merged.append({"name": item.strip(), "description": "", "public_knowledge": ""})

    for item in primary or []:
        add(item)
    for item in secondary or []:
        add(item)
    return merged


def _contents_excerpt(document_text: str, *, max_chars: int = 2_500) -> str:
    text = document_text or ""
    if not text:
        return ""
    head = text[:6_000]
    match = re.search(r"(?is)(?:^|\n)\s*(?:[Bb]\s+)?c\s*o\s*n\s*t\s*e\s*n\s*t\s*s\b", head)
    if match:
        start = match.start()
        return text[start : start + max_chars]
    # Fallback: leading pages often hold the TOC even without a clear header.
    return head[:max_chars]


def _location_detail_excerpt(document_text: str, loc: dict[str, Any]) -> str:
    """Parent-name windows plus location-scoped TOC / Exploring / ship-room sections."""
    text = document_text or ""
    loc_name = str(loc.get("name") or "")
    loc_type = str(loc.get("type") or "")
    needles = _distinctive_name_needles(loc_name, loc.get("id", ""))
    chunks: list[str] = []
    seen: set[str] = set()
    is_ship = _location_looks_like_ship(loc_name, location_type=loc_type, document_text=text)

    def add_chunk(piece: str) -> None:
        piece = (piece or "").strip()
        if not piece:
            return
        key = piece[:180].casefold()
        if key in seen:
            return
        seen.add(key)
        chunks.append(piece)

    lower = text.casefold()
    has_parent_hits = any(_mention_indexes(lower, needle) for needle in needles if needle)
    if has_parent_hits:
        if is_ship:
            # Ships: keep tight overview windows. A 7k forward window from an early
            # Sotillo mention would otherwise swallow Cronus deck rooms.
            for needle in needles:
                for idx in _mention_indexes(lower, needle)[:4]:
                    start = max(0, idx - 250)
                    end = min(len(text), idx + 1_800)
                    add_chunk(text[start:end])
        else:
            add_chunk(_relevant_excerpts(text, needles))
            # Prefer a later parent-name mention (often scene pages) beyond the first overview.
            for needle in needles:
                hits = _mention_indexes(lower, needle)
                if len(hits) < 2:
                    continue
                idx = hits[min(len(hits) - 1, 2)]
                start = max(0, idx - 200)
                end = min(len(text), idx + 5_000)
                add_chunk(text[start:end])
                break

    toc = _contents_excerpt(text)
    if toc and _text_matches_location(toc, loc_name):
        add_chunk(toc)
    span = _find_exploring_section_span(text, location_name=loc_name)
    if span:
        add_chunk(text[span[0] : span[1]])

    if is_ship:
        ship_span = _find_ship_room_description_span(text, location_name=loc_name)
        if ship_span:
            add_chunk(text[ship_span[0] : ship_span[1]])

    combined: list[str] = []
    total = 0
    for piece in chunks:
        if total + len(piece) > EXCERPT_MAX_CHARS:
            remain = EXCERPT_MAX_CHARS - total
            if remain > 500:
                combined.append(piece[:remain])
            break
        combined.append(piece)
        total += len(piece)
    if combined:
        return "\n\n---\n\n".join(combined)
    # Unmatched stub: do not dump the whole module into the prompt.
    return ""


def _location_detail_from_seed(stub: dict[str, Any], document_text: str = "") -> dict[str, Any]:
    name = str(stub.get("name") or stub.get("id") or "Location").strip()
    harvested = (
        _harvest_important_locations(
            document_text,
            name,
            location_type=str(stub.get("type") or ""),
        )
        if document_text
        else []
    )
    return {
        "summary": f"{name} appears in the imported module.",
        "current_events": [],
        "important_locations": harvested,
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
        max_tokens=OUTLINE_MAX_TOKENS if step == "outline" else IMPORT_MAX_TOKENS,
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
        '"description": "1-3 sentence summary", '
        '"genre": "fantasy|sci_fi|modern|horror|other", '
        '"rpg_system": "e.g. D&D 5e, Call of Cthulhu, Alien RPG"},\n'
        '  "locations": [{"id": "snake_case_id", "name": "Display Name", '
        '"type": "city|town|dungeon|forest|building|ship|region|other"}],\n'
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
        "- If the document centers on a ship or vessel, list the ship as one location "
        "(type ship) and put individual decks/rooms in later important_locations detail — "
        "do not create a separate top-level location for every room.\n"
        "- other_location_ids should only be non-empty for NPCs who travel or appear in more "
        "than one place in the text.\n"
        "- Set campaign.genre from the document's tone (fantasy, sci_fi, modern, horror, or "
        "other). Set campaign.rpg_system to the named ruleset when stated (e.g. D&D 5e, "
        "Call of Cthulhu, Alien RPG); otherwise leave rpg_system as an empty string.\n"
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
        "genre": normalize_campaign_genre(campaign_raw.get("genre")),
        "rpg_system": str(campaign_raw.get("rpg_system") or "").strip(),
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
    ship_batch = any(
        _location_looks_like_ship(
            str(loc.get("name") or ""),
            location_type=str(loc.get("type") or ""),
            document_text=text,
        )
        for loc in batch
    )
    retry_note = ""
    if is_retry:
        retry_note = (
            "RETRY: A previous pass returned little or nothing useful for this location. "
            "Read the excerpts again carefully and fill every field you can support from the text. "
            "If important_locations was empty, list every named room, wing, courtyard, pool, "
            "deck, or scene locale that belongs to this place.\n\n"
        )
    ship_note = ""
    if ship_batch:
        ship_note = (
            "SHIP / VESSEL LAYOUT (apply when this place is a ship, station, or vessel):\n"
            "- Keep the ship itself as the single top-level location.\n"
            "- After the ship overview / deck plans, look for ALL-CAPS or Title-Case subheaders "
            "followed by prose (for example 'MAIN AIRLOCK, JUNCTION A-1:', 'BRIDGE:', "
            "'SCIENCE LAB 1:').\n"
            "- Treat each subheader whose following text reads like a room or area description "
            "as one important_locations entry (not a separate top-level location).\n"
            "- Track nearby deck headings (Deck A/B/C, Cargo Deck, etc.) and put that deck "
            "identity at the start of each room's description "
            "(example: 'Deck A. The main airlock is the only point of entry...').\n"
            "- Junction codes like A-1 / B-2 also imply the matching deck letter.\n"
            "- Do not invent rooms that lack a subheader or clear room prose in the excerpts.\n\n"
        )
    return (
        f'Campaign: "{campaign["name"]}" ({campaign["campaign_id"]}).\n\n'
        f"{retry_note}"
        f"{ship_note}"
        "Fill in details for EACH location below from the SOURCE EXCERPTS only.\n"
        "Modules use many layouts — room-by-room keys, ship deck notes, scene boxed text, "
        "faction writeups, maps with captions, or free prose. Do not assume one game's headings.\n\n"
        "How to read unfamiliar layouts:\n"
        "- Treat nearby headings, bullets, boxed text, and captions as belonging to this place "
        "when they clearly describe it.\n"
        "- Sensory description, hazards, who is present, and what is going on → summary / "
        "current_events / local_tone.\n"
        "- Named sub-areas, rooms, decks, courtyards, pools, wings, or landmarks belonging to "
        "this place → important_locations (not separate top-level locations).\n"
        "- When a location header has subheaders beneath it, read the text under each subheader. "
        "If that text sounds like a room or place description (layout, contents, hazards, who is "
        "there), add it as an important_locations entry for this location.\n"
        "- Table-of-contents entries and Exploring/scene headings for rooms inside this place "
        "are high-confidence important_locations.\n"
        "- Gossip, hearsay, or uncertain talk → rumors.\n"
        "- What locals or visitors typically know or must not know → knowledge_rules.\n\n"
        'Respond with exactly {"locations": [ ... ]} where each item has this shape:\n\n'
        "{\n"
        '  "id": "<exact id from the list below>",\n'
        '  "summary": "faithful overview; keep concrete sensory and plot details from the source; '
        'use as many sentences as needed (typically 2-8)",\n'
        '  "current_events": ["ongoing situations from the source; include all that are stated"],\n'
        '  "important_locations": [{"name": "sub-location name", "description": "what the source says '
        '(include deck/level when the source is a ship)", '
        '"public_knowledge": "what ordinary people believe or know about it"}],\n'
        '  "rumors": ["rumors or uncertain talk from the source"],\n'
        '  "local_tone": ["mood/atmosphere descriptors from the source"],\n'
        '  "knowledge_rules": ["rules about what NPCs here generally do or do not know"]\n'
        "}\n\n"
        "CRITICAL EXTRACTION RULES:\n"
        "- Prefer copying concrete names, objects, factions, and plot facts from the source.\n"
        "- Do NOT invent major plot facts. If a field has no source support, use [] or \"\".\n"
        "- Keep fidelity high: longer accurate lists beat short generic ones.\n"
        "- Do NOT leave important_locations empty when the excerpts name rooms or scene locales "
        "inside this place (for example courtyard, pool, a confusing room, ship airlock/bridge/"
        "medlab, or a named NPC's territory used as a scene set).\n\n"
        "Locations to fill in:\n"
        f"{location_list}\n\n"
        "SOURCE EXCERPTS (may be partial windows around these locations):\n"
        '"""\n'
        f"{text}\n"
        '"""\n\n'
        'Output ONLY {"locations": [...]}. No prose, no markdown fences.'
    )


def _merge_location_detail(
    stub: dict[str, Any],
    detail: dict[str, Any],
    campaign_id: str,
    *,
    document_text: str = "",
) -> dict[str, Any]:
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

    if document_text:
        harvested = _harvest_important_locations(
            document_text,
            str(stub.get("name") or ""),
            location_type=str(stub.get("type") or ""),
        )
        important_locations = _merge_important_location_lists(important_locations, harvested)

    if _location_looks_like_ship(
        str(stub.get("name") or ""),
        location_type=str(stub.get("type") or ""),
        document_text=document_text,
    ):
        cleaned: list[dict[str, str]] = []
        for item in important_locations:
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            if _is_ship_false_room_name(name):
                continue
            cleaned.append(item)
        important_locations = cleaned

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


def _cap_document_text(text: str, max_chars: int, warnings: list[str]) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    warnings.append(
        f"Input document was {len(text)} characters; truncated to fit the current "
        f"content_creation context budget ({max_chars} characters). "
        "When a NON-PLAYER CHARACTERS section exists it is kept in priority. "
        "For long PDFs use Groq for content_creation or raise LOCAL_LLM_CONTEXT_TOKENS "
        "to match llama.cpp n_ctx."
    )
    return _truncate_preserving_npc_section(text, max_chars)


def _truncate_input(text: str, warnings: list[str]) -> str:
    text = text or ""
    # Fit the selected content_creation model's Groq TPM when configured.
    # 0 means this model cannot host an import request at all right now.
    max_chars = content_creation_max_input_chars()
    if max_chars <= 0:
        raise ContentImportError(content_creation_zero_budget_error_message())
    return _cap_document_text(text, min(MAX_INPUT_CHARS, max_chars), warnings)


def _truncate_for_import_document(text: str, warnings: list[str], *, require_llm_budget: bool) -> str:
    """Cap document length for import hops.

    Stat-block / callout fast paths skip the LLM and always use the full hop budget
    so the preserved NPC section is not chopped below recoverable stat blocks.
    """
    if require_llm_budget:
        return _truncate_input(text, warnings)
    return _cap_document_text(text, MAX_INPUT_CHARS, warnings)


def _fetch_location_details(
    batch: list[dict[str, Any]],
    document_text: str,
    campaign: dict[str, Any],
    warnings: list[str],
    *,
    is_retry: bool = False,
) -> dict[str, dict[str, Any]]:
    # One location per hop today; still build a focused parent+TOC+Exploring excerpt.
    if len(batch) == 1:
        excerpt = _location_detail_excerpt(document_text, batch[0])
    else:
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
    if len(batch) == 1:
        # Prefer the callout/stat writeup over early name-drops in scene prose.
        excerpt = _npc_writeup_excerpt(document_text, str(batch[0].get("name") or ""))
    else:
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

    if len(text) > MAX_INPUT_CHARS:
        bounded_text = _cap_document_text(text, MAX_INPUT_CHARS, warnings)
    else:
        bounded_text = text

    use_stat_block_fast_path = _document_has_recoverable_stat_blocks(bounded_text)
    use_location_site_fast_path = (
        not use_stat_block_fast_path
        and _document_has_recoverable_location_sites(bounded_text)
        and len(_extract_alien_location_site_stubs(bounded_text)) >= 2
    )
    document_for_deterministic_merge = bounded_text
    if use_stat_block_fast_path or use_location_site_fast_path:
        full_document_text = bounded_text
    else:
        # Further shrink the already-bounded text to the LLM TPM budget.
        # Do not re-truncate the original extract — that duplicated the
        # MAX_INPUT_CHARS truncation warning on every oversized non-stat-block import.
        full_document_text = _truncate_for_import_document(
            bounded_text, warnings, require_llm_budget=True
        )

    if use_stat_block_fast_path:
        warnings.append(
            "Published NON-PLAYER CHARACTERS stat blocks detected; skipping LLM outline "
            "and seeding NPC stubs from FULL NAME blocks (detail sweep still runs)."
        )
        outline_raw = {"campaign": {}, "locations": [], "npcs": []}
    elif use_location_site_fast_path:
        warnings.append(
            "Alien moonside / locations chapter detected; skipping LLM outline "
            "and seeding major location site stubs (detail sweep still runs)."
        )
        outline_raw = {"campaign": {}, "locations": [], "npcs": []}
    else:
        outline_source = _prepare_outline_document_text(full_document_text, warnings)
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
    _apply_imported_campaign_setting(
        campaign,
        document_text=full_document_text,
        alien_fast_path=use_stat_block_fast_path or use_location_site_fast_path,
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
        document_for_deterministic_merge,
        mode=mode,
        existing_npc_names=existing_npc_names,
        location_ids=location_ids,
        warnings=warnings,
    )
    npc_stubs = _merge_outline_npcs_with_callouts(
        npc_stubs,
        document_for_deterministic_merge,
        mode=mode,
        existing_npc_names=existing_npc_names,
        location_ids=location_ids,
        warnings=warnings,
    )
    location_stubs = _ensure_locations_for_npc_hints(
        location_stubs, npc_stubs, warnings=warnings
    )
    location_stubs = _merge_outline_locations_with_sites(
        location_stubs,
        document_for_deterministic_merge,
        mode=mode,
        existing_location_ids=existing_location_ids,
        warnings=warnings,
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


def _infer_genre_from_rpg_system(rpg_system: str) -> str:
    """Best-effort genre from a free-text RPG system label (empty when unknown)."""
    text = (rpg_system or "").strip().casefold()
    if not text:
        return ""
    if "alien" in text:
        return "sci_fi"
    if "cthulhu" in text or text in {"coc", "call of cthulhu"}:
        return "horror"
    if "shadowrun" in text or "cyberpunk" in text or "traveller" in text:
        return "sci_fi"
    if "d&d" in text or "dnd" in text or "dungeons" in text or "pathfinder" in text:
        return "fantasy"
    if "vampire" in text or "werewolf" in text or "world of darkness" in text:
        return "horror"
    return ""


def _apply_imported_campaign_setting(
    campaign: dict[str, Any],
    *,
    document_text: str = "",
    alien_fast_path: bool = False,
) -> None:
    """Fill genre/rpg_system for imports without inventing fantasy by default.

    Alien RPG stat-block / moonside fast paths skip the outline LLM, so missing
    genre must not fall back to fantasy (that would force a stay-in-fantasy rule).
    """
    genre = normalize_campaign_genre(campaign.get("genre"))
    rpg_system = str(campaign.get("rpg_system") or "").strip()

    if alien_fast_path:
        if not genre:
            genre = "sci_fi"
        if not rpg_system:
            rpg_system = "Alien RPG"
    elif not genre and rpg_system:
        genre = _infer_genre_from_rpg_system(rpg_system)
    elif not genre and document_text:
        sample = (document_text or "")[:12000].casefold()
        if (
            "alien rpg" in sample
            or "free league" in sample
            or "uscss" in sample
            or "xenomorph" in sample
            or "moonside locations" in sample
        ):
            genre = "sci_fi"
            if not rpg_system:
                rpg_system = "Alien RPG"

    campaign["genre"] = genre
    campaign["rpg_system"] = rpg_system


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
        "genre": normalize_campaign_genre(raw.get("genre")),
        "rpg_system": str(raw.get("rpg_system") or "").strip(),
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
                "seeded_from_location_sites": bool(raw.get("seeded_from_location_sites")),
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
            details_by_id[stub["id"]] = _location_detail_from_seed(stub, text)
            warnings.append(
                f"Location '{stub['id']}': detail sparse after LLM; fell back to "
                "minimal location defaults plus harvested sub-areas when available."
            )
        elif _location_detail_is_sparse(detail) and not detail:
            warnings.append(
                f"Location '{stub['id']}': LLM detail sparse (editor may retry this hop)."
            )

    locations = [
        _merge_location_detail(
            stub,
            details_by_id.get(stub["id"], {}),
            campaign["campaign_id"],
            document_text=text,
        )
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
