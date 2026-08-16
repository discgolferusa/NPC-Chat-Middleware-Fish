"""
Generic DM CHECK enforcement from authored NPC llm_rules.

Two modes:
1. Literal ``[DM CHECK: …]`` templates stay required until a matching ruling.
2. Vague deferral directives (e.g. "If someone claims official authority, defer to
   the DM") trigger a situational ``[DM CHECK]`` when the player's latest message
   looks like a claim that needs confirmation.
"""

from __future__ import annotations

import re
from typing import Any, Callable

DM_CHECK_IN_RULES_PATTERN = re.compile(
    r"\[DM CHECK:\s*([^\]]+?)\]",
    re.IGNORECASE,
)

# Vague authored directives that mean "don't settle this yourself — ask the DM".
# Keep these specific: bare "until the DM" / "official authority" overlap explicit
# [DM CHECK] template scaffolding (e.g. "Until the DM rules…") and must not match.
DM_DEFERRAL_DIRECTIVE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"defer to (?:the )?(?:dm|dungeon master|gm)",
        r"rather than accepting (?:it )?yourself",
        r"do not (?:simply )?accept .{0,40}\byourself\b",
        r"withhold (?:final )?approval",
        r"if someone claims (?:official )?authority",
        r"claims? (?:official )?authority.{0,120}(?:defer|dm|gm|dungeon master)",
        r"do not (?:grant|accept|believe) .{0,60}\b(?:on their word|without (?:the )?dm|alone)\b",
        r"let (?:the )?(?:dm|gm) (?:confirm|verify|rule|decide)",
        r"(?:ask|prompt)\s+(?:the )?(?:dm|dungeon master|gm)\s+to\s+(?:confirm|verify|rule)",
        r"need(?:s)?\s+(?:the )?(?:dm|dungeon master|gm)\s+to\s+(?:confirm|verify|rule)",
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


def npc_has_dm_deferral_directive(npc: dict[str, Any] | None) -> bool:
    """True when llm_rules include a vague 'defer to the DM' style directive."""
    blob = _npc_rules_blob(npc)
    if not blob.strip():
        return False
    # Ignore machine tags / ruling labels so template scaffolding cannot qualify.
    blob = DM_CHECK_IN_RULES_PATTERN.sub(" ", blob)
    blob = re.sub(r"\[DM Ruling:\s*[^\]]*\]", " ", blob, flags=re.IGNORECASE)
    return any(pattern.search(blob) for pattern in DM_DEFERRAL_DIRECTIVE_PATTERNS)


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
    # Short strings are verdict crumbs / single words — never substring-match them.
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


def format_situational_deferral_prompt_block(
    npc: dict[str, Any] | None,
    player_text: str | None,
) -> str:
    """Remind the model when vague deferral rules apply to this player turn."""
    from dm_channel import detect_player_claim, looks_like_authorization_exchange

    if not npc_has_dm_deferral_directive(npc):
        return ""
    if not (detect_player_claim(player_text) or looks_like_authorization_exchange(player_text)):
        return ""

    return """

SITUATIONAL DM DEFERRAL (from your NPC-specific rules):
The player's latest message looks like a claim of authority, proof, credentials, or
permission. Your rules say to defer to the DM rather than accepting it yourself.
Stay in character, do not grant the claim on their word, and end your reply with:
[DM CHECK: brief summary of what the player claims needs DM confirmation]
Do not invent a DM ruling."""


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
    When vague deferral directives apply and the player made a claim-like request,
    ensure a [DM CHECK] is present even if the model forgot the tag.
    """
    from dm_channel import (
        detect_npc_deferral,
        detect_player_claim,
        infer_claim_summary,
        looks_like_authorization_exchange,
    )

    if not (detect_player_claim(player_text) or looks_like_authorization_exchange(player_text)):
        return raw_text

    # Fire when NPC rules defer vaguely, or the spoken reply already defers.
    # Inspect speech only — required [DM CHECK] tags must not count as deferral.
    speech_only = DM_CHECK_IN_RULES_PATTERN.sub(" ", raw_text or "")
    if not (npc_has_dm_deferral_directive(npc) or detect_npc_deferral(speech_only)):
        return raw_text

    claim = infer_claim_summary(player_text or "", raw_text)
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
    """Call the LLM, then ensure required and situational DM CHECK tags are present."""
    raw = ask_llm_fn(messages)
    raw = ensure_required_dm_check_tags(npc, dm_context, raw)
    return ensure_situational_dm_check_tag(npc, dm_context, raw, player_text)
