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
    for match in DM_CHECK_IN_RULES_PATTERN.finditer(blob):
        claim = match.group(1).strip()
        if not claim:
            continue
        key = _normalize_claim(claim)
        if not key or key in seen:
            continue
        seen.add(key)
        claims.append(claim)
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


def format_required_dm_checks_prompt_block(
    npc: dict[str, Any] | None,
    dm_context: dict[str, Any] | None,
) -> str:
    """Remind the model of NPC-authored tags that are still waiting on a ruling."""
    pending = required_dm_checks(npc, dm_context)
    if not pending:
        return ""

    lines = "\n".join(f"- [DM CHECK: {claim}]" for claim in pending)
    return f"""

NPC-AUTHORED DM CHECK TAGS (still required until the DM rules):
This NPC's llm_rules declare the exact checks below. Prefer these tag wordings over a
global paraphrase for the same situation. Until DM channel context shows a matching
confirmed (Success) or denied (Failure) ruling for each claim, end your reply with
those exact tags (one per line, after spoken dialogue / stage directions). Follow any
NPC [DM Ruling: Success] / [DM Ruling: Failure] instructions after a matching ruling.
Do not invent DM rulings yourself.
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
) -> str:
    """
    Call the LLM, then ensure NPC-authored required DM CHECK tags are present.

    ``player_text`` is accepted for call-site compatibility; intent classification is
    left to the model via global/NPC llm_rules.
    """
    del player_text  # LLM decides; we only enforce authored templates.
    log_llm_prompt_to_console(messages)
    raw = ask_llm_fn(messages)
    return ensure_required_dm_check_tags(npc, dm_context, raw)
