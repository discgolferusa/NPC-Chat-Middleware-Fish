"""
Hard enforcement for NPC language-barrier llm_rules.

Prompt-only rules lose to global "speak dialogue" instructions. When an NPC only
understands a named language and the DM has not confirmed the speaker knows it,
force a stage-direction-only reply and a [DM CHECK] so TTS stays narrator-only.
"""

from __future__ import annotations

import re
from typing import Any, Callable

ONLY_UNDERSTANDS_PATTERN = re.compile(
    r"only\s+understands?\s+(?:the\s+)?(?:language\s+)?(?P<lang>[A-Za-z][\w'-]*(?:\s+[A-Za-z][\w'-]*){0,3})",
    re.IGNORECASE,
)

DOES_NOT_UNDERSTAND_COMMON_PATTERN = re.compile(
    r"do(?:es)?\s+not\s+understand\s+(?:common(?:/english)?|english)",
    re.IGNORECASE,
)

STAGE_DIRECTION_PATTERN = re.compile(r"\*([^*]{1,800})\*")

# Loose match for DM rulings about the speaker knowing/speaking a language.
LANGUAGE_RULING_HINT = re.compile(
    r"\b(know|knows|knew|understand|understands|speak|speaks|language)\b",
    re.IGNORECASE,
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


def detect_language_barrier(npc: dict[str, Any] | None) -> dict[str, str] | None:
    """
    Return {language, claim} when NPC rules declare a single understood language,
    otherwise None.
    """
    blob = _npc_rules_blob(npc)
    if not blob.strip():
        return None

    match = ONLY_UNDERSTANDS_PATTERN.search(blob)
    if not match and not DOES_NOT_UNDERSTAND_COMMON_PATTERN.search(blob):
        return None

    language = (match.group("lang") if match else "Slaadi").strip()
    # Drop trailing filler words that sometimes appear in freeform rules.
    language = re.sub(
        r"\b(they|and|do|does|not|never|only|common|english)\b.*$",
        "",
        language,
        flags=re.IGNORECASE,
    ).strip(" .,-")
    if not language:
        language = "Slaadi"

    # Title-case multi-word fantasy language names without shouting acronyms.
    display = " ".join(part.capitalize() for part in language.split())
    return {
        "language": display,
        "claim": f"does this speaker know Language {display}",
    }


def _ruling_mentions_language(text: str, language: str) -> bool:
    blob = (text or "").lower()
    lang = (language or "").lower().strip()
    if not blob or not lang:
        return False
    if lang in blob:
        return True
    # "Language Slaadi" / "know Slaadi" style claims without repeating "Language".
    return LANGUAGE_RULING_HINT.search(blob) is not None and any(
        token and token in blob for token in lang.split()
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
