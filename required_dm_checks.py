"""
DM CHECK enforcement from authored rules and platform (server-level) policy.

Modes:
1. Literal ``[DM CHECK: …]`` templates in NPC llm_rules stay required until ruled.
2. Server-level situational checks: when the player's prompt asks the NPC to decide,
   reveal a secret, or presents an unverified / possible-lie claim, append a
   ``[DM CHECK]`` based on that request (even if the model forgets the tag).
"""

from __future__ import annotations

import re
from typing import Any, Callable

DM_CHECK_IN_RULES_PATTERN = re.compile(
    r"\[DM CHECK:\s*([^\]]+?)\]",
    re.IGNORECASE,
)

# Player asks the NPC to make/grant a consequential decision.
DECISION_REQUEST_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(?:can|could|may|will)\s+(?:we|i|you)\b.+\b(?:enter|pass|go|leave|take|have|open|unlock|stand aside|let)\b",
        r"\b(?:let|allow|permit)\s+(?:us|me)\b.+\b(?:enter|pass|go|through|in|inside)\b",
        r"\b(?:give|hand|grant|show)\s+(?:us|me)\b.+\b(?:key|pass|access|permission|entry)\b",
        r"\b(?:yes or no|decide|make (?:a |the )?call|your (?:call|decision|judgment))\b",
        r"\b(?:swear|promise|vow)\s+(?:to|that)\b",
    )
]

# Player probes for hidden / secret information.
SECRET_PROBE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(?:secret|secrets|hiding|hidden|confession|confess)\b",
        r"\bwhat(?:'s| is) really (?:going on|happening)\b",
        r"\b(?:tell|give)\s+(?:me|us)\s+the\s+(?:truth|real (?:reason|story))\b",
        r"\b(?:between us|off the record|don't tell (?:anyone|the))\b",
        r"\bwhat (?:aren't you|are you not) (?:telling|saying)\b",
    )
]

# Soft “possible lie / unverified assertion” cues beyond strong claim detectors.
POSSIBLE_LIE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(?:trust me|believe me|i(?:'m| am) telling the truth|would i lie)\b",
        r"\b(?:honestly|no joke|i swear)\b.+\b(?:have|got|was|were|did)\b",
        r"\b(?:the (?:mayor|king|queen|lord|captain|guild) (?:sent|said|told|ordered))\b",
    )
]


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


def _normalize_claim(text: str) -> str:
    cleaned = re.sub(r"[^\w\s]+", " ", (text or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def extract_dm_check_templates_from_rules(npc: dict[str, Any] | None) -> list[str]:
    """Unique DM CHECK claim strings authored inside llm_rules."""
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


def _claim_matches(rule_claim: str, other: str) -> bool:
    """
    Match a rule template to a DM claim_summary.

    Equality always matches. Substring matches require both sides to look like
    real claim text (not short verdict notes like ``No`` / ``Yes``), so ``no``
    cannot match inside ``know``.
    """
    a = _normalize_claim(rule_claim)
    b = _normalize_claim(other)
    if not a or not b:
        return False
    if a == b:
        return True

    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) < 20:
        return False
    return shorter in longer


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
    """Rule-authored DM CHECK claims that still lack a matching DM ruling."""
    return [
        claim
        for claim in extract_dm_check_templates_from_rules(npc)
        if not _ruling_covers_claim(dm_context, claim)
    ]


def format_required_dm_checks_prompt_block(
    npc: dict[str, Any] | None,
    dm_context: dict[str, Any] | None,
) -> str:
    pending = required_dm_checks(npc, dm_context)
    if not pending:
        return ""

    lines = "\n".join(f"- [DM CHECK: {claim}]" for claim in pending)
    return f"""

REQUIRED DM CHECK TAGS (authoritative until the DM rules):
Your llm_rules declare the checks below. Until DM channel context shows a matching
confirmed (Success) or denied (Failure) ruling for each claim, you MUST end your reply
with those exact tags (one per line, after any spoken dialogue or stage directions).
Follow your other NPC-specific rules for how to behave while waiting and after
Success/Failure rulings. Do not invent DM rulings yourself.
{lines}"""


def _player_probes_npc_secrets(player_text: str, npc: dict[str, Any] | None) -> bool:
    secrets = (npc or {}).get("secrets") or []
    if not secrets:
        return False
    blob = " ".join(str(item) for item in secrets if item).lower()
    if not blob.strip():
        return False
    words = [w for w in _normalize_claim(player_text).split() if len(w) >= 5]
    hits = sum(1 for w in words if w in blob)
    return hits >= 2


def detect_server_dm_check_need(
    player_text: str | None,
    npc: dict[str, Any] | None = None,
) -> dict[str, str] | None:
    """
    Server-level policy: does this player prompt need a [DM CHECK]?

    Reasons: decision | secret | unverified_claim
    """
    from dm_channel import (
        detect_player_claim,
        infer_claim_summary,
        looks_like_authorization_exchange,
    )

    text = (player_text or "").strip()
    if not text:
        return None

    if detect_player_claim(text) or looks_like_authorization_exchange(text):
        return {
            "reason": "unverified_claim",
            "claim_summary": infer_claim_summary(text),
        }

    if any(p.search(text) for p in POSSIBLE_LIE_PATTERNS):
        return {
            "reason": "unverified_claim",
            "claim_summary": infer_claim_summary(text),
        }

    if any(p.search(text) for p in DECISION_REQUEST_PATTERNS):
        return {
            "reason": "decision",
            "claim_summary": infer_claim_summary(text) or "player is asking the NPC to decide or grant something",
        }

    if any(p.search(text) for p in SECRET_PROBE_PATTERNS) or _player_probes_npc_secrets(text, npc):
        return {
            "reason": "secret",
            "claim_summary": infer_claim_summary(text) or "player is probing for secret or hidden information",
        }

    return None


def format_server_dm_check_prompt_block(
    npc: dict[str, Any] | None,
    player_text: str | None,
) -> str:
    """Remind the model when server-level policy says this turn needs a DM CHECK."""
    need = detect_server_dm_check_need(player_text, npc)
    if not need:
        return ""

    reason = need["reason"]
    if reason == "decision":
        why = "asks you to make or grant a consequential decision"
    elif reason == "secret":
        why = "asks you to reveal a secret or hidden fact"
    else:
        why = "presents an unverified claim or possible lie"

    return f"""

SERVER DM CHECK POLICY (authoritative for this turn):
The player's latest message {why}. Do not settle it on your own.
Stay in character, withhold final acceptance/revelation, and end your reply with:
[DM CHECK: {need['claim_summary']}]
Do not invent a DM ruling."""


# Backward-compatible alias used by older call sites / tests.
def format_situational_deferral_prompt_block(
    npc: dict[str, Any] | None,
    player_text: str | None,
) -> str:
    return format_server_dm_check_prompt_block(npc, player_text)


def ensure_required_dm_check_tags(
    npc: dict[str, Any] | None,
    dm_context: dict[str, Any] | None,
    raw_text: str,
) -> str:
    """
    Append any still-required rule DM CHECK tags the model omitted.

    Does not rewrite spoken content — success/failure behavior stays in llm_rules.
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
    ]
    if not missing:
        return text

    suffix = "\n".join(f"[DM CHECK: {claim}]" for claim in missing)
    if text:
        return f"{text}\n{suffix}"
    return suffix


def ensure_situational_dm_check_tag(
    npc: dict[str, Any] | None,
    dm_context: dict[str, Any] | None,
    raw_text: str,
    player_text: str | None,
) -> str:
    """
    Server-level: append a [DM CHECK] when the player prompt needs DM confirmation
    and the model omitted the tag.
    """
    need = detect_server_dm_check_need(player_text, npc)
    if not need:
        return raw_text

    claim = (need.get("claim_summary") or "").strip()
    if not claim:
        return raw_text
    if _ruling_covers_claim(dm_context, claim):
        return raw_text

    text = (raw_text or "").rstrip()
    existing = [
        m.group(1).strip()
        for m in DM_CHECK_IN_RULES_PATTERN.finditer(text)
    ]
    if any(_claim_matches(claim, existing_claim) for existing_claim in existing):
        return text

    tag = f"[DM CHECK: {claim}]"
    if text:
        return f"{text}\n{tag}"
    return tag


def generate_npc_reply_text(
    npc: dict[str, Any] | None,
    dm_context: dict[str, Any] | None,
    messages: list[dict[str, str]],
    ask_llm_fn: Callable[[list[dict[str, str]]], str],
    *,
    player_text: str | None = None,
) -> str:
    """Call the LLM, then ensure required and server situational DM CHECK tags are present."""
    raw = ask_llm_fn(messages)
    raw = ensure_required_dm_check_tags(npc, dm_context, raw)
    return ensure_situational_dm_check_tag(npc, dm_context, raw, player_text)
