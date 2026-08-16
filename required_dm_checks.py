"""
Generic DM CHECK enforcement from authored NPC llm_rules.

If rules embed a literal ``[DM CHECK: …]`` template, keep prompting for that check
until the DM has a matching confirmed/denied ruling. No domain-specific language
logic — success/failure behavior stays in the GM's other llm_rules + DM rulings.
"""

from __future__ import annotations

import re
from typing import Any, Callable

DM_CHECK_IN_RULES_PATTERN = re.compile(
    r"\[DM CHECK:\s*([^\]]+?)\]",
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
    a = _normalize_claim(rule_claim)
    b = _normalize_claim(other)
    if not a or not b:
        return False
    if a == b:
        return True
    # Allow light wording drift between rule template and DM claim_summary.
    if len(a) >= 12 and (a in b or b in a):
        return True
    return False


def _ruling_covers_claim(dm_context: dict[str, Any] | None, claim: str) -> bool:
    for ruling in (dm_context or {}).get("rulings") or []:
        verdict = str(ruling.get("verdict") or "").strip().lower()
        if verdict not in ("confirmed", "denied"):
            continue
        summary = str(ruling.get("claim_summary") or "")
        notes = str(ruling.get("ruling_text") or "")
        if _claim_matches(claim, summary) or _claim_matches(claim, notes):
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


def generate_npc_reply_text(
    npc: dict[str, Any] | None,
    dm_context: dict[str, Any] | None,
    messages: list[dict[str, str]],
    ask_llm_fn: Callable[[list[dict[str, str]]], str],
) -> str:
    """Call the LLM, then ensure rule-declared DM CHECK tags are present when needed."""
    raw = ask_llm_fn(messages)
    return ensure_required_dm_check_tags(npc, dm_context, raw)
