"""
Strip LLM leakage that must never reach transcripts or Piper.

Discord / browser UIs add the speaker label themselves (e.g. ``**Mateo:**``).
This module only cleans the spoken body so Piper does not say the name and so
meta checklist text like ``Check for DM tag none`` never appears on any surface.
"""

from __future__ import annotations

import re
from typing import Any

# Model self-checks about whether a [DM CHECK] is needed (not a real machine tag).
_DM_META_LEAK_PATTERN = re.compile(
    r"(?:"
    r"check\s+for\s+dm\s+tags?\s*(?:[:\-]?\s*(?:none|n/?a|no|not\s+needed|not\s+required))?"
    r"|dm\s+tags?\s*(?:[:\-]\s*)?(?:none|n/?a|no|not\s+needed|not\s+required)"
    r"|no\s+dm\s+(?:check|tag)s?(?:\s+(?:needed|required))?"
    r"|dm\s+check\s*(?:[:\-]\s*)?(?:none|n/?a|no)"
    r")"
    r"\.?",
    re.IGNORECASE,
)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+|\n+")


def _npc_name_aliases(npc: dict[str, Any] | None) -> list[str]:
    """Display name plus first token (Mateo from Mateo Rios), longest first."""
    raw = str((npc or {}).get("name") or "").strip()
    if not raw:
        return []
    aliases = [raw]
    first = raw.split()[0].strip(".,;:'\"")
    if first and first.casefold() != raw.casefold():
        aliases.append(first)
    # Longest first so "Mateo Rios:" wins over "Mateo:".
    return sorted(set(aliases), key=len, reverse=True)


def _strip_speaker_prefixes(text: str, aliases: list[str]) -> str:
    """Remove script-style ``Name:`` labels the model invents; UI/Discord re-add them."""
    if not text or not aliases:
        return text
    out = text
    for alias in aliases:
        escaped = re.escape(alias)
        # Start of reply or after a newline / sentence end.
        pattern = re.compile(
            rf"(^|(?<=\n)|(?<=[.!?…])\s*){escaped}\s*:\s*",
            re.IGNORECASE,
        )
        out = pattern.sub(r"\1", out)
    return out


def _strip_dm_meta_leaks(text: str) -> str:
    cleaned = _DM_META_LEAK_PATTERN.sub(" ", text or "")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _collapse_exact_duplicate_body(text: str) -> str:
    """
    If the spoken body is the same dialogue twice (often after a meta leak), keep one.

    Compares whitespace-normalized halves / repeated prefixes so glued duplicates
    like ``stay.Mateo: Mercy…`` still collapse after prefix stripping.
    """
    compact = re.sub(r"\s+", " ", (text or "")).strip()
    if len(compact) < 8:
        return (text or "").strip()

    # Exact doubled body: A + A (including short lines like "Stay put. Stay put.").
    for cut in range(len(compact) // 2, 3, -1):
        left = compact[:cut].strip()
        right = compact[cut:].strip()
        if len(left) < 4:
            break
        if left == right:
            return left
        if right.rstrip(".!?…") == left.rstrip(".!?…") and len(left) >= 4:
            return left

    # Longest sentence-ended prefix that appears again later.
    for match in reversed(list(re.finditer(r"[.!?…]", compact))):
        end = match.end()
        if end < 8 or end > len(compact) // 2 + 5:
            continue
        chunk = compact[:end].strip()
        rest = compact[end:].lstrip()
        if rest.startswith(chunk):
            return chunk
        if rest.rstrip(".!?…") == chunk.rstrip(".!?…"):
            return chunk

    return (text or "").strip()


def sanitize_npc_spoken_body(text: str, npc: dict[str, Any] | None = None) -> str:
    """
    Clean spoken dialogue only (no ``[DM CHECK:…]`` tags).

    Safe to run on ``spoken_text`` after tag extraction, or on TTS segment bodies.
    """
    body = (text or "").strip()
    if not body:
        return ""

    aliases = _npc_name_aliases(npc)
    body = _strip_dm_meta_leaks(body)
    body = _strip_speaker_prefixes(body, aliases)
    # Meta removal can leave a second ``Name:`` copy; strip again then dedupe.
    body = _strip_speaker_prefixes(body, aliases)
    body = _collapse_exact_duplicate_body(body)
    body = re.sub(r"[ \t]{2,}", " ", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def sanitize_npc_reply_text(text: str, npc: dict[str, Any] | None = None) -> str:
    """
    Clean a full NPC reply while preserving trailing ``[DM CHECK:…]`` machine tags.

    Speaker labels for Discord captions / browser transcripts are applied by those
    clients from ``npc_name`` — they are not part of this string.
    """
    from dm_channel import extract_dm_check_tags

    raw = text or ""
    spoken, claims = extract_dm_check_tags(raw)
    cleaned = sanitize_npc_spoken_body(spoken, npc)
    if not claims:
        return cleaned
    suffix = "\n".join(f"[DM CHECK: {claim}]" for claim in claims)
    if cleaned:
        return f"{cleaned}\n{suffix}"
    return suffix
