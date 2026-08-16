"""
Hard enforcement for NPC language-barrier llm_rules.

Prompt-only rules lose to global "speak dialogue" instructions. When an NPC only
understands a named language and the DM has not confirmed the speaker knows it,
force a stage-direction-only reply and a [DM CHECK] so TTS stays narrator-only.
"""

from __future__ import annotations

import re
from typing import Any, Callable

# Require an explicit Language <Name> form so ordinary phrasing like
# "only understands that…" / "only understands the situation…" never matches.
ONLY_UNDERSTANDS_LANGUAGE_PATTERN = re.compile(
    r"only\s+understands?\s+language\s+(?P<lang>[A-Za-z][\w'-]*(?:\s+[A-Za-z][\w'-]*){0,2})",
    re.IGNORECASE,
)

# Secondary explicit forms used in authored rules / DM CHECK templates.
EXPLICIT_LANGUAGE_NAME_PATTERNS = (
    re.compile(
        r"\bknow(?:s)?\s+language\s+(?P<lang>[A-Za-z][\w'-]*(?:\s+[A-Za-z][\w'-]*){0,2})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\[DM CHECK:\s*does this speaker know language\s+(?P<lang>[A-Za-z][\w'-]*(?:\s+[A-Za-z][\w'-]*){0,2})\s*\]",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bspeaks?\s+only\s+language\s+(?P<lang>[A-Za-z][\w'-]*(?:\s+[A-Za-z][\w'-]*){0,2})",
        re.IGNORECASE,
    ),
)

# Stopwords that are never valid fantasy language names when captured alone.
_LANGUAGE_NAME_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "that",
        "this",
        "these",
        "those",
        "them",
        "they",
        "their",
        "and",
        "or",
        "but",
        "not",
        "only",
        "common",
        "english",
        "words",
        "speech",
        "dialogue",
        "requests",
        "meaning",
        "speaker",
        "speakers",
        "player",
        "party",
    }
)

STAGE_DIRECTION_PATTERN = re.compile(r"\*([^*]{1,800})\*")

# Require knowledge/speech framing around the language name in DM rulings.
_LANGUAGE_KNOWLEDGE_VERBS = (
    r"know|knows|knew|understand|understands|understood|speak|speaks|spoke|spoken|"
    r"fluent|fluency|comprehend|comprehends"
)


def _npc_rules_blob(npc: dict[str, Any] | None) -> str:
    npc = npc or {}
    raw = npc.get("llm_rules")
    if raw is None:
        raw = npc.get("npc_rules")

    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        return "\n".join(str(item) for item in raw if item is not None)
    if isinstance(raw, dict):
        nested = raw.get("llm_rules") or raw.get("rules") or []
        if isinstance(nested, list):
            return "\n".join(str(item) for item in nested if item is not None)
        return str(nested)
    return str(raw)


def _normalize_language_name(raw: str | None) -> str | None:
    if not raw:
        return None
    cleaned = re.sub(r"[^\w\s'-]+", " ", str(raw), flags=re.UNICODE)
    parts = [part for part in cleaned.split() if part]
    if not parts:
        return None

    filtered: list[str] = []
    for part in parts:
        if part.lower() in _LANGUAGE_NAME_STOPWORDS:
            # Stop at trailing filler ("Slaadi They do…") rather than inventing a name.
            break
        filtered.append(part)
    if not filtered:
        return None

    display = " ".join(part.capitalize() for part in filtered)
    if display.lower() in _LANGUAGE_NAME_STOPWORDS:
        return None
    return display


def _extract_explicit_language(blob: str) -> str | None:
    for pattern in (ONLY_UNDERSTANDS_LANGUAGE_PATTERN, *EXPLICIT_LANGUAGE_NAME_PATTERNS):
        match = pattern.search(blob)
        if not match:
            continue
        language = _normalize_language_name(match.group("lang"))
        if language:
            return language
    return None


def detect_language_barrier(npc: dict[str, Any] | None) -> dict[str, str] | None:
    """
    Return {language, claim} when NPC rules explicitly name the only understood
    language (``Language <Name>``), otherwise None.

    Does not default to any fantasy language. ``does not understand Common`` alone
    is not enough without a parseable language name elsewhere in the rules.
    """
    blob = _npc_rules_blob(npc)
    if not blob.strip():
        return None

    language = _extract_explicit_language(blob)
    if not language:
        return None

    return {
        "language": language,
        "claim": f"does this speaker know Language {language}",
    }


def _ruling_mentions_language(text: str, language: str) -> bool:
    """
    True only when the ruling both names the language and frames it as
    knowledge/speech of that language (not a coincidental race/creature word).
    """
    blob = (text or "").strip()
    lang = (language or "").strip()
    if not blob or not lang:
        return False

    lang_pattern = re.escape(lang)
    # Allow optional "Language " prefix before the name.
    name_group = rf"(?:language\s+)?{lang_pattern}"

    knowledge_patterns = (
        rf"\b(?:{_LANGUAGE_KNOWLEDGE_VERBS})\b.{{0,40}}\b{name_group}\b",
        rf"\b{name_group}\b.{{0,40}}\b(?:{_LANGUAGE_KNOWLEDGE_VERBS})\b",
        rf"\bdoes this speaker know\s+{name_group}\b",
        rf"\bspeaker know(?:s)?\s+{name_group}\b",
    )
    return any(
        re.search(pattern, blob, flags=re.IGNORECASE | re.DOTALL)
        for pattern in knowledge_patterns
    )


def language_barrier_status(
    npc: dict[str, Any] | None,
    dm_context: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """
    Resolve barrier + DM verdict.

    status:
      - unknown: no ruling yet (force confused stage + DM CHECK)
      - denied: DM said they do not know it (force confused stage, no new check)
      - confirmed: speak normally
    """
    barrier = detect_language_barrier(npc)
    if not barrier:
        return None

    language = barrier["language"]
    status = "unknown"
    matched_ruling = None

    for ruling in reversed((dm_context or {}).get("rulings") or []):
        claim = str(ruling.get("claim_summary") or "")
        notes = str(ruling.get("ruling_text") or "")
        if not (
            _ruling_mentions_language(claim, language)
            or _ruling_mentions_language(notes, language)
        ):
            continue
        verdict = str(ruling.get("verdict") or "").strip().lower()
        if verdict in ("confirmed", "denied"):
            status = verdict
            matched_ruling = ruling
            break

    return {
        **barrier,
        "status": status,
        "matched_ruling": matched_ruling,
    }


def build_language_barrier_response(
    npc: dict[str, Any] | None,
    *,
    language: str,
    include_dm_check: bool,
) -> str:
    name = str((npc or {}).get("name") or "The NPC").strip() or "The NPC"
    stage = f"*{name} looks confused and clearly does not understand.*"
    if include_dm_check:
        return f"{stage}\n[DM CHECK: does this speaker know Language {language}]"
    return stage


def enforce_language_barrier_text(
    npc: dict[str, Any] | None,
    dm_context: dict[str, Any] | None,
    raw_text: str,
) -> str:
    """
    Post-LLM safety net: strip spoken dialogue and ensure DM CHECK when required.
    """
    resolved = language_barrier_status(npc, dm_context)
    if not resolved or resolved["status"] == "confirmed":
        return raw_text

    include_check = resolved["status"] == "unknown"
    # Prefer a model-authored stage direction if present; otherwise use template.
    match = STAGE_DIRECTION_PATTERN.search(raw_text or "")
    if match and match.group(1).strip():
        stage = f"*{match.group(1).strip()}*"
        if include_check:
            return f"{stage}\n[DM CHECK: does this speaker know Language {resolved['language']}]"
        return stage

    return build_language_barrier_response(
        npc,
        language=resolved["language"],
        include_dm_check=include_check,
    )


def generate_npc_reply_text(
    npc: dict[str, Any] | None,
    dm_context: dict[str, Any] | None,
    messages: list[dict[str, str]],
    ask_llm_fn: Callable[[list[dict[str, str]]], str],
) -> str:
    """
    Skip the LLM when a language barrier blocks comprehension; otherwise call LLM
    and still enforce stage-only + DM CHECK on the output.
    """
    resolved = language_barrier_status(npc, dm_context)
    if resolved and resolved["status"] != "confirmed":
        return build_language_barrier_response(
            npc,
            language=resolved["language"],
            include_dm_check=resolved["status"] == "unknown",
        )

    raw = ask_llm_fn(messages)
    return enforce_language_barrier_text(npc, dm_context, raw)
