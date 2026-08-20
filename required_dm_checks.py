"""
DM CHECK handling: LLM judges; platform parses tags.

Enforcement is intentionally narrow:
1. Parse ``[DM CHECK: …]`` from model output (see dm_channel.extract_dm_check_tags).
2. If NPC llm_rules author a literal ``[DM CHECK: …]`` template and no matching
   DM ruling exists yet, ensure that exact tag is present on the reply.
3. Do not regex-guess player intent to invent server-side checks — global llm_rules
   tell the model when to emit tags.
"""

from __future__ import annotations

import re
import threading
from typing import Any, Callable

DM_CHECK_IN_RULES_PATTERN = re.compile(
    r"\[DM CHECK:\s*([^\]]+?)\]",
    re.IGNORECASE,
)

# YAML colon-split can leave fragments like: [DM CHECK': 'claim text]
DM_CHECK_MANGLED_IN_RULES_PATTERN = re.compile(
    r"\[DM\s*CHECK['\"\s]*:\s*['\"]?\s*([^\]]+?)\s*['\"]?\s*\]",
    re.IGNORECASE,
)

DM_RULING_FAILURE_PATTERN = re.compile(
    r"\[DM\s*Ruling:\s*Failure\]\s*(.+)",
    re.IGNORECASE,
)

_STAGE_DIRECTION_PATTERN = re.compile(r"^\*[^*]+\*\s*")

# Claim must be about speaking/comprehending a language — not generic "understand X".
_LANGUAGE_BARRIER_CLAIM_PHRASES = (
    "necessary language",
    "know the language",
    "knows the language",
    "know language",
    "speak the language",
    "speaks the language",
    "speak language",
    "language to talk",
    "language to speak",
)

_LANGUAGE_BARRIER_CLAIM_PATTERN = re.compile(
    r"language\s+[a-z]{4,}",
    re.IGNORECASE,
)

_ENGLISH_COMPREHENSION_WORDS = frozenset(
    {
        "i", "you", "we", "they", "he", "she", "it", "the", "a", "an", "is", "are", "was", "were",
        "can", "could", "will", "would", "have", "has", "had", "do", "does", "did",
        "yes", "no", "hello", "hi", "help", "please", "thank", "thanks", "what", "how", "why",
        "who", "where", "when", "understand", "understood", "speak", "english", "sure", "of", "course",
        "may", "might", "must", "shall", "should", "want", "need", "know", "think", "believe",
        "your", "my", "our", "their", "this", "that", "these", "those", "here", "there",
        "well", "very", "good", "great", "fine", "okay", "ok", "right", "indeed", "certainly",
        "welcome", "sorry", "excuse", "pardon", "tell", "say", "said", "ask", "answer",
        "perfectly", "clearly", "absolutely", "however", "perhaps", "maybe", "come", "go",
    }
)


def _npc_rules_blob(npc: dict[str, Any] | None) -> str:
    from yaml_list_text import yaml_list_item_as_text

    npc = npc or {}
    raw = npc.get("llm_rules")
    if raw is None:
        raw = npc.get("npc_rules")

    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        return "\n".join(
            text for text in (yaml_list_item_as_text(item) for item in raw if item is not None) if text.strip()
        )
    if isinstance(raw, dict):
        nested = raw.get("llm_rules") or raw.get("rules") or []
        if isinstance(nested, list):
            return "\n".join(
                text
                for text in (yaml_list_item_as_text(item) for item in nested if item is not None)
                if text.strip()
            )
        return str(nested)
    return str(raw)


def _normalize_claim(text: str) -> str:
    cleaned = re.sub(r"[^\w\s]+", " ", (text or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def extract_dm_check_templates_from_rules(npc: dict[str, Any] | None) -> list[str]:
    """Unique DM CHECK claim strings authored inside NPC llm_rules."""
    blob = _npc_rules_blob(npc)
    claims: list[str] = []
    seen: set[str] = set()

    def _add_claim(claim: str) -> None:
        claim = (claim or "").strip()
        if not claim:
            return
        key = _normalize_claim(claim)
        if not key or key in seen:
            return
        seen.add(key)
        claims.append(claim)

    for match in DM_CHECK_IN_RULES_PATTERN.finditer(blob):
        # Preserve authored punctuation (possessives, trailing quotes in the claim).
        _add_claim(match.group(1))

    for match in DM_CHECK_MANGLED_IN_RULES_PATTERN.finditer(blob):
        # Only mangled YAML fragments may wrap the claim in leftover ' / " chars.
        _add_claim(match.group(1).strip().strip("'\""))

    return claims


_CLAIM_STOPWORDS = frozenset(
    {
        "a", "an", "the", "to", "of", "and", "or", "for", "is", "are", "was", "were",
        "be", "been", "being", "do", "does", "did", "this", "that", "these", "those",
        "it", "its", "in", "on", "at", "as", "by", "with", "from", "into", "about",
        "than", "then", "so", "if", "not", "no", "yes", "any", "all", "some",
    }
)


def _content_claim_tokens(text: str) -> set[str]:
    return {
        w for w in _normalize_claim(text).split()
        if len(w) >= 4 and w not in _CLAIM_STOPWORDS
    }


def _claim_matches(rule_claim: str, other: str) -> bool:
    """
    Match an authored template to a DM claim_summary / emitted tag.

    Equality and long substring only for exact-ish text. Near-paraphrase uses
    content tokens (stopwords excluded). Overlap thresholds scale with claim
    length so short templates (e.g. secrets probing — three content tokens)
    can still match slight rewordings.
    """
    a = _normalize_claim(rule_claim)
    b = _normalize_claim(other)
    if not a or not b:
        return False
    if a == b:
        return True

    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) >= 20 and shorter in longer:
        return True

    ca = _content_claim_tokens(a)
    cb = _content_claim_tokens(b)
    if not ca or not cb:
        return False

    overlap = ca & cb
    shorter_n = min(len(ca), len(cb))
    # Scale: short claims need (almost) all tokens; long ones need ~90% with a floor.
    if shorter_n <= 3:
        needed = shorter_n
        distinctive_needed = 1
    elif shorter_n <= 5:
        needed = max(3, int(0.9 * shorter_n + 0.999))
        distinctive_needed = 2
    else:
        needed = max(4, int(0.9 * shorter_n + 0.999))
        distinctive_needed = 2

    if len(overlap) < needed:
        return False
    if sum(1 for w in overlap if len(w) >= 6) < distinctive_needed:
        return False
    return True


def _ruling_covers_claim(dm_context: dict[str, Any] | None, claim: str) -> bool:
    """A ruling covers a template only via claim_summary identity, not ruling_text."""
    for ruling in (dm_context or {}).get("rulings") or []:
        verdict = str(ruling.get("verdict") or "").strip().lower()
        if verdict not in ("confirmed", "denied"):
            continue
        summary = str(ruling.get("claim_summary") or "")
        if _claim_matches(claim, summary):
            return True
    return False


def required_dm_checks(
    npc: dict[str, Any] | None,
    dm_context: dict[str, Any] | None,
) -> list[str]:
    """NPC-authored DM CHECK claims that still lack a matching DM ruling."""
    return [
        claim
        for claim in extract_dm_check_templates_from_rules(npc)
        if not _ruling_covers_claim(dm_context, claim)
    ]


def _is_language_comprehension_claim(claim: str) -> bool:
    """True only for language-barrier checks, not generic 'does the player understand X'."""
    normalized = _normalize_claim(claim)
    if not normalized:
        return False
    if _LANGUAGE_BARRIER_CLAIM_PATTERN.search(normalized):
        return True
    if "language" not in normalized:
        return False
    if any(phrase in normalized for phrase in _LANGUAGE_BARRIER_CLAIM_PHRASES):
        return True
    # Authored template: "Does the speaker know the necessary language to talk to the NPC"
    if "speaker" in normalized and "know" in normalized:
        return True
    return False


def _npc_has_language_barrier_rules(npc: dict[str, Any] | None) -> bool:
    blob = _npc_rules_blob(npc).lower()
    if "do not treat the speaker as understood" in blob:
        return True
    if "fantasy-language" in blob:
        return True
    failure_hint = _extract_failure_ruling_hint(npc).lower()
    return bool(
        failure_hint
        and (
            "fantasy-language" in failure_hint
            or "do not understand them" in failure_hint
            or "cannot understand them" in failure_hint
            or "show you cannot understand" in failure_hint
        )
    )


def _extract_failure_ruling_hint(npc: dict[str, Any] | None) -> str:
    for line in _npc_rules_blob(npc).splitlines():
        match = DM_RULING_FAILURE_PATTERN.search(line)
        if match:
            return match.group(1).strip().strip("'\"")
    return ""


def _split_stage_direction(text: str) -> tuple[str, str]:
    text = (text or "").strip()
    match = _STAGE_DIRECTION_PATTERN.match(text)
    if not match:
        return "", text
    return match.group(0).strip(), text[match.end():].strip()


def _looks_like_english_comprehension(spoken: str) -> bool:
    """Heuristic: did the NPC reply as if they understood English?"""
    _stage, dialogue = _split_stage_direction(spoken)
    check_text = (dialogue or spoken).strip()
    if not check_text:
        return False
    words = re.findall(r"[a-zA-Z']+", check_text.lower())
    if len(words) < 2:
        return False
    english_hits = sum(1 for word in words if word in _ENGLISH_COMPREHENSION_WORDS)
    if english_hits >= 2:
        return True
    return len(words) >= 3 and english_hits >= 1 and len(words) <= 8


def _default_fantasy_language_line(_npc: dict[str, Any] | None) -> str:
    return "Skraa-vek?"


def _rewrite_language_barrier_spoken(spoken: str, npc: dict[str, Any] | None) -> str:
    stage, _dialogue = _split_stage_direction(spoken)
    fantasy = _default_fantasy_language_line(npc)
    if stage:
        return f"{stage} {fantasy}".strip()
    return fantasy


def enforce_unruled_check_spoken_behavior(
    npc: dict[str, Any] | None,
    dm_context: dict[str, Any] | None,
    raw_text: str,
) -> str:
    """
    When a language-barrier DM CHECK is still unruled, strip English comprehension
    dialogue the model may have added before/after the tag.
    """
    pending = required_dm_checks(npc, dm_context)
    language_pending = [claim for claim in pending if _is_language_comprehension_claim(claim)]
    if not language_pending or not _npc_has_language_barrier_rules(npc):
        return raw_text

    from dm_channel import extract_dm_check_tags

    spoken, claims = extract_dm_check_tags(raw_text)
    if not _looks_like_english_comprehension(spoken):
        return raw_text

    rewritten = _rewrite_language_barrier_spoken(spoken, npc)
    result = rewritten.rstrip()
    tag_claims = list(claims)
    for claim in pending:
        if not any(_claim_matches(claim, existing) for existing in tag_claims):
            tag_claims.append(claim)
    for claim in tag_claims:
        result += f"\n[DM CHECK: {claim}]"
    return result


def format_required_dm_checks_prompt_block(
    npc: dict[str, Any] | None,
    dm_context: dict[str, Any] | None,
) -> str:
    """Remind the model of NPC-authored tags that are still waiting on a ruling."""
    pending = required_dm_checks(npc, dm_context)
    if not pending:
        return ""

    lines = "\n".join(f"- [DM CHECK: {claim}]" for claim in pending)
    language_waiting = any(_is_language_comprehension_claim(claim) for claim in pending)
    language_block = ""
    if language_waiting and _npc_has_language_barrier_rules(npc):
        failure_hint = _extract_failure_ruling_hint(npc)
        failure_line = failure_hint or "Use a short fantasy-language line only; do not reply in English."
        language_block = f"""

LANGUAGE BARRIER (still waiting on DM ruling):
Until a matching confirmed Success ruling exists, do NOT treat the speaker as understood.
Do NOT respond in English. {failure_line}"""
    return f"""

NPC-AUTHORED DM CHECK TAGS (still required until the DM rules):
This NPC's llm_rules declare the exact checks below. Prefer these tag wordings over a
global paraphrase for the same situation. Until DM channel context shows a matching
confirmed (Success) or denied (Failure) ruling for each claim, end your reply with
those exact tags (one per line, after spoken dialogue / stage directions). Follow any
NPC [DM Ruling: Success] / [DM Ruling: Failure] instructions after a matching ruling.
Do not invent DM rulings yourself.{language_block}
{lines}"""


def ensure_required_dm_check_tags(
    npc: dict[str, Any] | None,
    dm_context: dict[str, Any] | None,
    raw_text: str,
) -> str:
    """
    Append any still-required NPC-authored DM CHECK tags the model omitted.

    Does not invent new claims from player text — only exact templates from llm_rules.
    """
    pending = required_dm_checks(npc, dm_context)
    if not pending:
        return raw_text

    text = (raw_text or "").rstrip()
    existing = {
        _normalize_claim(m.group(1))
        for m in DM_CHECK_IN_RULES_PATTERN.finditer(text)
    }

    missing = [
        claim for claim in pending
        if _normalize_claim(claim) not in existing
        and not any(_claim_matches(claim, existing_claim) for existing_claim in existing)
    ]
    if not missing:
        return text

    suffix = "\n".join(f"[DM CHECK: {claim}]" for claim in missing)
    if text:
        return f"{text}\n{suffix}"
    return suffix


_LLM_PROMPT_LOG_LOCK = threading.Lock()


def log_llm_prompt_to_console(messages: list[dict[str, str]] | None) -> None:
    """
    Print the full NPC-reply prompt to the middleware process console.

    Emits one write under a lock so Flask ``threaded=True`` workers cannot
    interleave headers/body lines from concurrent turns.
    """
    msgs = messages or []
    total_chars = sum(len((m or {}).get("content") or "") for m in msgs)
    parts = [
        f"[LLM-PROMPT] npc-reply messages={len(msgs)} chars={total_chars}",
    ]
    for index, message in enumerate(msgs):
        role = (message or {}).get("role") or "?"
        content = (message or {}).get("content") or ""
        parts.append(f"[LLM-PROMPT] --- [{index}] role={role} chars={len(content)} ---")
        parts.append(content)
    parts.append("[LLM-PROMPT] === end ===")
    block = "\n".join(parts) + "\n"
    with _LLM_PROMPT_LOG_LOCK:
        # One print keeps the whole block contiguous in the console stream.
        print(block, end="", flush=True)


def generate_npc_reply_text(
    npc: dict[str, Any] | None,
    dm_context: dict[str, Any] | None,
    messages: list[dict[str, str]],
    ask_llm_fn: Callable[[list[dict[str, str]]], str],
    *,
    player_text: str | None = None,
    enforce_required_dm_checks: bool = True,
) -> str:
    """
    Call the LLM, then ensure NPC-authored required DM CHECK tags are present.

    ``player_text`` is accepted for call-site compatibility; intent classification is
    left to the model via global/NPC llm_rules.

    Set ``enforce_required_dm_checks=False`` for post-ruling auto-replies so a just-
    resolved claim is not force-appended again.
    """
    del player_text  # LLM decides; we only enforce authored templates.
    log_llm_prompt_to_console(messages)
    raw = ask_llm_fn(messages)
    if not enforce_required_dm_checks:
        return raw
    tagged = ensure_required_dm_check_tags(npc, dm_context, raw)
    return enforce_unruled_check_spoken_behavior(npc, dm_context, tagged)
