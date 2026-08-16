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

_CONTENT_STOPWORDS = frozenset(
    {
        "the", "a", "an", "to", "of", "and", "or", "for", "is", "are", "we", "i", "you",
        "us", "me", "that", "this", "with", "from", "have", "has", "had", "do", "does",
        "did", "can", "could", "may", "will", "would", "be", "been", "being", "in", "on",
        "at", "as", "it", "my", "our", "your", "they", "them", "their", "about", "into",
        "just", "like", "what", "when", "where", "which", "who", "how", "then", "than",
        "there", "here", "some", "any", "all", "not", "but", "if", "so", "too", "very",
    }
)

# Consequential access / judgment asks — not everyday "can/could/may/will … have/go/open".
DECISION_REQUEST_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(?:can|could|may)\s+(?:we|i)\b.+\b(?:enter|pass(?:\s+through)?|come\s+inside|stand\s+aside)\b",
        r"\b(?:will|would)\s+you\b.+\b(?:let|allow|permit)\s+(?:us|me)\b",
        r"\b(?:let|allow|permit)\s+(?:us|me)\b.+\b(?:enter|pass|through|inside)\b",
        r"\b(?:give|hand|grant)\s+(?:us|me)\b.+\b(?:key|pass|access|permission|entry|writ|badge)\b",
        r"\byes or no\b",
        r"\bmake (?:a |the )?(?:ruling|judgment|call)\b",
        r"\byour (?:ruling|judgment|decision)\b.+\b(?:on|about|whether)\b",
    )
]

# Explicit secret-probing phrasing — not every mention of "secret" / "hidden".
SECRET_PROBE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(?:tell|give|spill)\s+(?:me|us)\s+(?:your |the )?(?:secret|secrets)\b",
        r"\bwhat (?:secret|secrets) (?:are you|do you)\b",
        r"\bare you hiding (?:something|a secret)\b",
        r"\bconfess\b.+\b(?:secret|truth|what)\b",
        r"\bwhat(?:'s| is) really (?:going on|happening)\b",
        r"\b(?:tell|give)\s+(?:me|us)\s+the\s+(?:truth|real (?:reason|story))\b",
        r"\boff the record\b",
        r"\bwhat (?:aren't you|are you not) (?:telling|saying)\b",
    )
]

# Strong "I'm lying / trust this unverified assertion" cues — not reported speech or fillers.
POSSIBLE_LIE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(?:trust me|believe me|would i lie)\b",
        r"\bi(?:'m| am) telling the truth\b",
        r"\bi swear (?:to (?:you|god)|on)\b",
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


def _content_words(text: str) -> set[str]:
    return {
        w for w in _normalize_claim(text).split()
        if len(w) >= 4 and w not in _CONTENT_STOPWORDS
    }


def _brief_claim_summary(text: str, *, fallback: str, max_len: int = 90) -> str:
    """Prefer a short DM-facing summary over dumping the full player utterance."""
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return fallback
    # First clause / sentence.
    parts = re.split(r"[.!?]|[,;]\s+", cleaned, maxsplit=1)
    brief = (parts[0] if parts else cleaned).strip(" ,;:-")
    if len(brief) > max_len:
        brief = brief[: max_len - 1].rstrip() + "…"
    return brief or fallback


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


def _existing_tag_covers_request(
    server_claim: str,
    existing_claim: str,
    player_text: str,
) -> bool:
    """True when an already-written [DM CHECK] is about the same player request."""
    if _claim_matches(server_claim, existing_claim):
        return True

    player_words = _content_words(player_text)
    existing_words = _content_words(existing_claim)
    if not player_words or not existing_words:
        return False

    overlap = player_words & existing_words
    if len(overlap) >= 2:
        return True

    # Short model summaries often keep one distinctive token (mayor, writ, aqueduct…).
    distinctive = {w for w in player_words if len(w) >= 6}
    return bool(distinctive & existing_words)


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
    """
    True only when the player both cues a probe and overlaps distinctive secret tokens
    as whole words (not loose substring hits of common 5-letter words).
    """
    if not re.search(
        r"\b(secret|secrets|hiding|hidden|confess|tell me about|what about|know about)\b",
        player_text or "",
        flags=re.IGNORECASE,
    ):
        return False

    secrets = (npc or {}).get("secrets") or []
    if not secrets:
        return False

    secret_tokens: set[str] = set()
    for item in secrets:
        for word in _normalize_claim(str(item)).split():
            if len(word) >= 6 and word not in _CONTENT_STOPWORDS:
                secret_tokens.add(word)

    player_tokens = {
        w for w in _normalize_claim(player_text).split()
        if len(w) >= 6 and w not in _CONTENT_STOPWORDS
    }
    return len(secret_tokens & player_tokens) >= 2


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
        summary = infer_claim_summary(text)
        return {
            "reason": "unverified_claim",
            "claim_summary": _brief_claim_summary(summary, fallback=summary or "unverified player claim"),
        }

    if any(p.search(text) for p in POSSIBLE_LIE_PATTERNS):
        summary = infer_claim_summary(text)
        return {
            "reason": "unverified_claim",
            "claim_summary": _brief_claim_summary(
                summary,
                fallback="player asserts something that may be untrue",
            ),
        }

    if any(p.search(text) for p in DECISION_REQUEST_PATTERNS):
        return {
            "reason": "decision",
            "claim_summary": _brief_claim_summary(
                text,
                fallback="player is asking the NPC to decide or grant access",
            ),
        }

    if any(p.search(text) for p in SECRET_PROBE_PATTERNS) or _player_probes_npc_secrets(text, npc):
        return {
            "reason": "secret",
            "claim_summary": _brief_claim_summary(
                text,
                fallback="player is probing for secret or hidden information",
            ),
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
Stay in character, withhold final acceptance/revelation, and end your reply with a brief:
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
    and the model omitted a tag for that same request.
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
    if any(
        _existing_tag_covers_request(claim, existing_claim, player_text or "")
        for existing_claim in existing
    ):
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
